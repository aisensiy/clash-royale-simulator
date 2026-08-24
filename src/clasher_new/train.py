import argparse
import os
import time

from environment import CREnv, random_strategy, entity_names
from selfplay import OpponentPool, PooledOpponent
from winrate import make_rusher

from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
import torch.nn as nn
import torch.nn.functional as F
import torch


class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.embedding_dim = 8
        self.entity_embedding = nn.Embedding(len(entity_names), self.embedding_dim)
        # The rich observation adds a handful of scalars (clock, crowns, opponent elixir,
        # weakest towers, card-counting belief). Concatenating 16 numbers straight onto a
        # ~2300-wide CNN feature would let them be ignored; a small MLP gives them a
        # comparable share of the head's input.
        extra = 0
        spaces_ = observation_space.spaces
        if "context" in spaces_:
            extra = spaces_["context"].shape[0] + spaces_["opp_hand"].shape[0]
        self.context_dim = 64 if extra else 0
        self.context_mlp = (nn.Sequential(nn.Linear(extra, 64), nn.ReLU())
                            if extra else None)
        self.in_channels = 13 + self.embedding_dim + 4
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 32, 18)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out + 5 * self.embedding_dim + 1 + self.context_dim,
                            features_dim)

    def forward(self, observation):
        """
        Gets the observation, use the embedding (dim=8) to expand the channels, then use one-hot to further expand the channels.
        The code is ugly but should do the work.
        """
        grid = observation['grid']  # (B, 32, 18, 15)
        hand = observation['hand'].long()  # (B, 5)
        elixir = observation['elixir']

        card_ids = grid[..., 0].long()
        card_vecs = self.entity_embedding(card_ids)

        rest = grid[..., 1:]  # (B, 32, 18, 14)
        x = torch.cat([rest, card_vecs], dim=-1)  # (B, 32, 18, 14+EMBED)
        card_type = x[..., 0].long()  # (B, 32, 18)
        card_type_oh = F.one_hot(card_type, num_classes=4).float()  # (B, 32, 18, 4)
        rest = x[..., 1:]
        x = torch.cat([rest, card_type_oh], dim=-1)
        x = x.permute(0, 3, 1, 2).float()  # (B, C, 32, 18)

        grid_feat = self.cnn(x)

        hand_feat = self.entity_embedding(hand).flatten(1)  # (B, 5*EMBED)
        parts = [grid_feat, hand_feat, elixir.float()]
        if self.context_mlp is not None:
            ctx = torch.cat([observation['context'].float(),
                             observation['opp_hand'].float()], dim=1)
            parts.append(self.context_mlp(ctx))
        return torch.relu(self.fc(torch.cat(parts, dim=1)))


class ThroughputCallback(BaseCallback):
    """Log where wall-clock actually goes: stepping environments vs. the gradient update.

    This is the number that decides whether a GPU box would help at all -- if `train_s`
    stays a small fraction of `rollout_s`, the run is bound by simulating games on CPU
    and more GPU would buy nothing.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._rollout_start = None
        self._train_start = None

    def _on_rollout_start(self):
        now = time.perf_counter()
        if self._train_start is not None:
            self.logger.record("time/train_s", now - self._train_start)
        self._rollout_start = now

    def _on_rollout_end(self):
        self._train_start = time.perf_counter()
        self.logger.record("time/rollout_s", self._train_start - self._rollout_start)

    def _on_step(self):
        return True


class SnapshotCallback(BaseCallback):
    """Drop a copy of the current policy into the opponent pool at a fixed cadence."""

    def __init__(self, pool, every, verbose=0):
        super().__init__(verbose)
        self.pool = pool
        self.every = every
        self._next = None

    def _on_training_start(self):
        # Anchor the schedule to where this run actually starts. Resuming from a 6M-step
        # checkpoint with a counter that begins at 0 makes every threshold below 6M fire
        # at once, filling the pool with dozens of copies of the same policy.
        self._next = self.num_timesteps + self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            path = self.pool.add(self.model, self.num_timesteps)
            if self.verbose:
                print(f"snapshot -> {path}", flush=True)
        return True


class OpponentOutcomeCallback(BaseCallback):
    """Win rate broken down by opponent type, read straight off finished episodes.

    The `script:*` rows are the ones to watch. Self-play win rate sits near 50% by
    construction whether both sides are improving or both are decaying; the scripts
    never change, so only they can tell those two apart.
    """

    def __init__(self, window=400, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.history = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "outcome" not in info:
                continue
            # Track the side as well. The agent now plays both, so a persistent gap
            # between the two rows is what remains of the arena's own bias.
            keys = (info["opponent"], info["opponent"].split(":")[0],
                    f"as_{'blue' if info['learner_player'] == 0 else 'red'}")
            for key in keys:
                buf = self.history.setdefault(key, [])
                buf.append(info["outcome"])
                if len(buf) > self.window:
                    del buf[:-self.window]
        return True

    def _on_rollout_end(self):
        for key, buf in self.history.items():
            if not buf:
                continue
            self.logger.record(f"opponent/{key}_winrate", sum(v == 1 for v in buf) / len(buf))
            self.logger.record(f"opponent/{key}_games", len(buf))


class RandomEvalCallback(BaseCallback):
    """Win rate against the fixed random opponent -- the same yardstick as the old script."""

    # Evaluation runs episodes in a single un-vectorised env, so each call costs roughly
    # `n_episodes * 370` sequential env steps. At 100k/20 that was eating about as much
    # wall clock as the training it was measuring; 500k/10 brings it under 10%.
    def __init__(self, every=500_000, n_episodes=10, use_masking=True, legacy_obs=False,
                 rich_obs=False, verbose=0):
        super().__init__(verbose)
        self.every = every
        self.n_episodes = n_episodes
        self.use_masking = use_masking
        self.legacy_obs = legacy_obs
        self.rich_obs = rich_obs
        self._next = None

    def _on_training_start(self):
        self._next = self.num_timesteps + self.every

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        eval_env = DummyVecEnv([lambda: CREnv(opponent_model=random_strategy,
                                              legacy_obs=self.legacy_obs,
                                              rich_obs=self.rich_obs)])
        if self.use_masking:
            mean_reward, _ = maskable_evaluate_policy(
                self.model, eval_env, n_eval_episodes=self.n_episodes,
                use_masking=True, warn=False)
        else:
            mean_reward, _ = evaluate_policy(
                self.model, eval_env, n_eval_episodes=self.n_episodes, warn=False)
        self.logger.record("eval/mean_reward_vs_random", mean_reward)
        eval_env.close()
        return True


def make_env(seed, legacy_obs, pool_dir=None, masked=False, refresh_every=10,
             record_path=None, record_every=20, rich_obs=False, dmg_scale=1.0):
    def _init():
        # Each worker simulates games in pure Python; a private BLAS thread pool per worker
        # would oversubscribe the box without speeding anything up. This matters twice over
        # under self-play, where the worker also runs the opponent policy itself.
        torch.set_num_threads(1)
        if pool_dir is None:
            opponent = random_strategy
        else:
            algo = MaskablePPO if masked else PPO
            scripts = {"random": random_strategy, "rusher": make_rusher(seed)}
            opponent = PooledOpponent(OpponentPool(pool_dir), scripts, algo,
                                      refresh_every=refresh_every, seed=seed)
        env = CREnv(opponent_model=opponent, legacy_obs=legacy_obs,
                    record_path=record_path, record_every=record_every,
                    rich_obs=rich_obs, dmg_scale=dmg_scale)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=int(os.environ.get("CR_N_ENVS", 96)))
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    # Upstream settled on lr=1e-4, n_epochs=4, target_kl=0.03 to stabilise training
    # ("Modify hyperparameters to stabilize training"). Our own late-run Elo swung
    # 1038 -> 905 -> 1012, which is the same symptom. Exposed as flags rather than new
    # defaults so a run can be compared against the ones already measured.
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--torch-threads", type=int, default=16,
                        help="threads for the gradient update; the rollout forward pass is a\n"
                             "tiny batch, so a large pool here costs more than it buys")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="masked")
    # Masking is off by default: measured over 6M steps it did not beat plain PPO
    # (92%/86% vs 92%/96% win rate against the random and rusher scripts), and a failed
    # deploy already behaves exactly like choosing not to play a card.
    parser.add_argument("--mask", action="store_true",
                        help="use MaskablePPO and forbid unaffordable or illegal actions")
    parser.add_argument("--legacy-obs", action="store_true",
                        help="ablation: reproduce the mirrored-observation bug")
    # On by default from here: without the clock, the crowns and the opponent's elixir,
    # "hold elixir and counter-push" cannot be represented at all, and every checkpoint so
    # far measured exactly 0% on the elixir-management probe.
    parser.add_argument("--no-rich-obs", dest="rich_obs", action="store_false",
                        help="ablation: drop the clock/crowns/opponent-elixir/card-count inputs")
    parser.set_defaults(rich_obs=True)
    # Full tower damage is worth ~10.9 of shaping reward per game against a terminal win
    # bonus of 10, so trading badly still pays as long as something is being hit. Quartered,
    # the shaping keeps its role as a dense learning signal without outweighing the result.
    parser.add_argument("--dmg-scale", type=float, default=0.25,
                        help="multiplier on both tower-damage shaping terms (1.0 = legacy)")
    parser.add_argument("--init-from", type=str, default=None,
                        help="continue from an existing checkpoint instead of a fresh policy")
    parser.add_argument("--self-play", action="store_true",
                        help="train against a pool of past snapshots instead of a fixed script")
    parser.add_argument("--snapshot-every", type=int, default=500_000,
                        help="timesteps between adding the current policy to the pool")
    parser.add_argument("--opponent-refresh", type=int, default=10,
                        help="episodes a worker keeps one opponent before re-sampling")
    parser.add_argument("--record-every", type=int, default=0,
                        help="save one episode in N for exact replay later (0 = off)")
    parser.add_argument("--log-dir", type=str, default="/output/cr_logs")
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)

    pool_dir = os.path.join(args.log_dir, f"{args.run_name}_pool") if args.self_play else None
    record_path = (os.path.join(args.log_dir, f"{args.run_name}_episodes")
                   if args.record_every else None)
    env_fns = [make_env(i, args.legacy_obs, pool_dir, args.mask, args.opponent_refresh,
                        record_path, args.record_every or 1,
                        rich_obs=args.rich_obs, dmg_scale=args.dmg_scale)
               for i in range(args.n_envs)]
    env = VecMonitor(SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns))

    algo = MaskablePPO if args.mask else PPO
    if args.init_from:
        # Self-play from a random policy throws away the run that already reached 96%
        # against the rusher script. Load those weights and keep the same env and
        # hyperparameters; only the opponent changes.
        model = algo.load(args.init_from, env=env, device=args.device,
                          n_steps=args.n_steps, batch_size=args.batch_size,
                          learning_rate=args.learning_rate, n_epochs=args.n_epochs,
                          target_kl=args.target_kl, tensorboard_log=args.log_dir)
        print(f"resumed from {args.init_from}", flush=True)
    else:
        model = algo(
            "MultiInputPolicy", env,
            policy_kwargs={"features_extractor_class": CRFeatureExtractor},
            n_steps=args.n_steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, n_epochs=args.n_epochs,
            target_kl=args.target_kl,
            verbose=1, tensorboard_log=args.log_dir, device=args.device, seed=0,
        )
    model.verbose = 1
    callbacks = [
        CheckpointCallback(save_freq=max(1, 500_000 // args.n_envs),
                           save_path=args.log_dir, name_prefix=args.run_name),
        ThroughputCallback(),
    ]
    if args.self_play:
        pool = OpponentPool(pool_dir)
        # Seed the pool immediately: until the first snapshot lands every worker can only
        # draw scripted opponents, which is not self-play at all.
        pool.add(model, 0)
        callbacks += [SnapshotCallback(pool, args.snapshot_every, verbose=1),
                      OpponentOutcomeCallback()]
    else:
        callbacks.append(RandomEvalCallback(use_masking=args.mask, legacy_obs=args.legacy_obs,
                                            rich_obs=args.rich_obs))
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                    tb_log_name=args.run_name, reset_num_timesteps=False)
    finally:
        print('Saving model.')
        model.save(os.path.join(args.log_dir, f'{args.run_name}_final'))


if __name__ == '__main__':
    main()
