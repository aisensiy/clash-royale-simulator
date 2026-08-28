import argparse
import os
import time

from agents import make_rusher
from scripts_defender import make_defender
from scripts_sniper import make_sniper
from environment import (ARENA_H, ARENA_W, CREnv, N_FLAT_ACTIONS, N_SLOTS,
                         entity_names, grid_layout, random_strategy)
from selfplay import OpponentPool, PooledOpponent

from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
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
        # Two of the grid channels are consumed rather than fed in: the entity id becomes
        # an embedding and the card type a one-hot. Read the width off the space so that
        # adding a channel to the observation does not need an edit here as well.
        # A stacked grid is N copies of one frame's channels laid end to end, newest
        # first. Each frame gets the same treatment -- its entity id embedded, its card
        # type one-hot -- so the width per frame is what it always was, times the stack.
        _, self.frames = grid_layout(spaces_["grid"].shape[-1])
        self.frame_channels = spaces_["grid"].shape[-1] // self.frames
        self.in_channels = self.frames * ((self.frame_channels - 2) + self.embedding_dim + 4)
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
        grid = observation['grid']  # (B, 32, 18, C * frames)
        hand = observation['hand'].long()  # (B, 5)
        elixir = observation['elixir']

        planes = []
        for f in range(self.frames):
            frame = grid[..., f * self.frame_channels:(f + 1) * self.frame_channels]
            card_ids = frame[..., 0].long()
            card_vecs = self.entity_embedding(card_ids)

            rest = frame[..., 1:]  # (B, 32, 18, C-1)
            x = torch.cat([rest, card_vecs], dim=-1)  # (B, 32, 18, C-1+EMBED)
            card_type = x[..., 0].long()  # (B, 32, 18)
            card_type_oh = F.one_hot(card_type, num_classes=4).float()  # (B, 32, 18, 4)
            rest = x[..., 1:]
            planes.append(torch.cat([rest, card_type_oh], dim=-1))
        x = torch.cat(planes, dim=-1).permute(0, 3, 1, 2).float()  # (B, C, 32, 18)
        # Kept for `SpatialActionHead`, which needs the arena at full resolution and runs
        # in the same forward pass. SB3 calls this extractor once per pass and shares it
        # between the policy and the value function, so there is exactly one live value.
        self.last_planes = x

        grid_feat = self.cnn(x)

        hand_feat = self.entity_embedding(hand).flatten(1)  # (B, 5*EMBED)
        parts = [grid_feat, hand_feat, elixir.float()]
        if self.context_mlp is not None:
            ctx = torch.cat([observation['context'].float(),
                             observation['opp_hand'].float()], dim=1)
            parts.append(self.context_mlp(ctx))
        return torch.relu(self.fc(torch.cat(parts, dim=1)))


class SpatialActionHead(nn.Module):
    """The joint action space in two levels: play or wait, and then what and where.

    Cells share weights. A plain `Linear(latent, N_FLAT_ACTIONS)` gives each of the 576
    cells its own parameters and no way to carry anything learned on one cell over to the
    one beside it, which is a lot to ask of the same rollouts that also have to learn what
    to play. This runs a small convolution over the arena at full resolution, conditioned
    on a projection of the trunk's output, and reads off one plane of logits per card
    slot.

    Waiting does not share that softmax, and the first run of this head is why. With one
    "do nothing" outcome against 2304 placements, an untrained policy waits with
    probability about 1/2304 instead of the 1/5 the factorised head started from. PPO
    cannot learn a behaviour it never samples, so the arm spent twelve million steps
    dumping cards the moment it could afford them -- 62 a game against the factorised
    arm's 33, broke on 85% of its decisions -- and lost the head-to-head 17%. Holding
    elixir is the hardest habit in this game to learn and the easiest to price out of the
    distribution by accident.

    So "play or wait" is its own sigmoid, and the placement softmax is conditioned on
    having decided to play. The two levels are composed here into log-probabilities over
    the same 2305 outcomes, which a categorical distribution reproduces exactly
    (`log_softmax` of a log-probability vector is itself). Nothing downstream changes: the
    action space, the codec, the mask and every checkpoint stay as they are, and the
    sampling and entropy maths remain sb3-contrib's rather than ours.

    Masking then renormalises across both levels, which is the behaviour we want -- with
    few placements legal, waiting deserves more of the probability, and with none legal it
    takes all of it. It also means the effective opening probability of waiting is well
    above the 1/2 the sigmoid starts at, since the placements it competes with are only
    the legal ones. Erring toward patience is the safe side of this particular mistake.

    The planes come from the features extractor rather than being recomputed: SB3 calls
    the extractor exactly once per forward pass, immediately before this head, and shares
    it between the policy and the value function.
    """

    def __init__(self, latent_dim, extractor, cond_dim=32, width=48):
        super().__init__()
        # In a list so the extractor is not registered as a submodule of the head as well
        # as of the policy. It would still be optimised once either way, but it would
        # appear twice in the module tree and in anything that walks it.
        self._extractor = [extractor]
        self.cond = nn.Linear(latent_dim, cond_dim)
        self.tower = nn.Sequential(
            nn.Conv2d(extractor.in_channels, width, 3, padding=1), nn.ReLU())
        self.mix = nn.Sequential(
            nn.Conv2d(width + cond_dim, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, N_SLOTS - 1, 1))
        self.play = nn.Linear(latent_dim, 1)
        # SB3 initialises its action head at gain 0.01 so that the opening policy is close
        # to uniform and exploration is not decided by whatever the last layer happened to
        # be born as. The head super() built with that treatment is the one we replaced.
        # On `play` the same treatment means an opening coin flip between playing and
        # waiting, before the mask tilts it further toward waiting.
        for layer in (self.mix[-1], self.play):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, latent):
        planes = self._extractor[0].last_planes
        cond = self.cond(latent)[:, :, None, None].expand(-1, -1, ARENA_H, ARENA_W)
        cells = self.mix(torch.cat([self.tower(planes), cond], dim=1))
        play = self.play(latent)
        # Log-probabilities, not free logits: index 0 carries P(wait) whole, and each
        # placement carries P(play) times its share of the placement softmax. Slot-major,
        # then row-major within a slot -- exactly how `decode_flat_action` reads an index
        # back apart.
        return torch.cat([F.logsigmoid(-play),
                          F.logsigmoid(play) + F.log_softmax(cells.flatten(1), dim=1)],
                         dim=1)


class CRSpatialPolicy(MaskableActorCriticPolicy):
    """`MaskableActorCriticPolicy` with the flat action space read off the arena.

    Only the action head differs from the stock policy: the distribution stays the
    ordinary masked categorical over `N_FLAT_ACTIONS` outcomes, so none of the sampling,
    log-probability or entropy maths is ours to get wrong.
    """

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = SpatialActionHead(self.mlp_extractor.latent_dim_pi,
                                            self.features_extractor)
        # The optimizer super() built closed over the head it replaced.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)


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
                 rich_obs=False, count_obs=False, flat_action=False, frames=1,
                 verbose=0):
        super().__init__(verbose)
        self.every = every
        self.n_episodes = n_episodes
        self.use_masking = use_masking
        self.legacy_obs = legacy_obs
        self.rich_obs = rich_obs
        self.count_obs = count_obs
        self.flat_action = flat_action
        self.frames = frames
        self._next = None

    def _on_training_start(self):
        self._next = self.num_timesteps + self.every

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        eval_env = DummyVecEnv([lambda: CREnv(opponent_model=random_strategy,
                                              legacy_obs=self.legacy_obs,
                                              rich_obs=self.rich_obs,
                                              count_obs=self.count_obs,
                                              flat_action=self.flat_action,
                                              frames=self.frames)])
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
             record_path=None, record_every=20, rich_obs=False, count_obs=False,
             flat_action=False, frames=1, dmg_scale=1.0,
             elixir_scale=0.0, script_names=("random", "rusher"), script_weights=None,
             script_share=0.15):
    def _init():
        # Each worker simulates games in pure Python; a private BLAS thread pool per worker
        # would oversubscribe the box without speeding anything up. This matters twice over
        # under self-play, where the worker also runs the opponent policy itself.
        torch.set_num_threads(1)
        if pool_dir is None:
            opponent = random_strategy
        else:
            algo = MaskablePPO if masked else PPO
            # `anchor` and `counter` are never available here. `anchor` is the defender's
            # held-out twin, and `counter` is the ruler: rating against something that was
            # in the pool measures drill, not strength.
            builders = {"random": lambda s: random_strategy, "rusher": make_rusher,
                        "defender": make_defender, "sniper": make_sniper}
            scripts = {name: builders[name](seed) for name in script_names}
            # The snapshot share gives way to the scripts, not the latest-vs-history split:
            # sampling the newest snapshot less often is what makes self-play chase itself.
            history = max(0.0, 1.0 - script_share - 0.45)
            pool = OpponentPool(pool_dir, script_names=script_names,
                                script_weights=script_weights,
                                p_latest=1.0 - script_share - history,
                                p_history=history, p_script=script_share)
            opponent = PooledOpponent(pool, scripts, algo,
                                      refresh_every=refresh_every, seed=seed,
                                      masked=masked, flat_action=flat_action)
        env = CREnv(opponent_model=opponent, legacy_obs=legacy_obs,
                    record_path=record_path, record_every=record_every,
                    rich_obs=rich_obs, count_obs=count_obs,
                    flat_action=flat_action, frames=frames, dmg_scale=dmg_scale,
                    elixir_scale=elixir_scale)
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
    # On by default for the same reason: `CREnv.observe` writes each unit into `obs[y][x]`,
    # so three Minions on one cell read back as one Minion. How many bodies are in a push
    # is not merely hard to learn from the old grid -- it is not in it.
    parser.add_argument("--no-count-obs", dest="count_obs", action="store_false",
                        help="ablation: drop the per-cell unit-count channels")
    parser.set_defaults(count_obs=True)
    # Off by default after two 12M-step runs lost the head-to-head 17% each. The exact
    # per-cell mask it enables is a real win and is kept unconditionally; the encoding
    # itself is not, and the run that would tell us whether the joint space or the head
    # that came with it was at fault has not been done. See `CRSpatialPolicy`.
    parser.add_argument("--flat-action", dest="flat_action", action="store_true",
                        help="model the placement as one joint choice over 2305 outcomes")
    parser.set_defaults(flat_action=False)
    parser.add_argument("--frames", type=int, default=1,
                        help="stack this many consecutive grids into one observation, "
                             "newest first; 1 is a single snapshot with no history")
    # Full tower damage is worth ~10.9 of shaping reward per game against a terminal win
    # bonus of 10, so trading badly still pays as long as something is being hit. Quartered,
    # the shaping keeps its role as a dense learning signal without outweighing the result.
    parser.add_argument("--dmg-scale", type=float, default=0.25,
                        help="multiplier on both tower-damage shaping terms (1.0 = legacy)")
    # Tower HP alone reports a perfect defence -- killing a 9 elixir push with 6 and
    # losing nothing -- as 0.000 reward, identical to nothing having happened. This pays
    # for the change in how much elixir each side still owns, bank plus board, which is
    # where every defensive trade and every combined push shows up.
    # The scripted defender is opt-in. Changing the opponent distribution and the
    # reward in the same run would make neither result attributable.
    parser.add_argument("--scripts", type=str, default="random,rusher",
                        help="scripted opponents in the pool, comma separated, each "
                             "optionally weighted as name:weight -- `defender` holds elixir "
                             "and answers pushes near its tower, `sniper` answers every "
                             "crossing with the cheapest body in range of its own tower and "
                             "beats every checkpoint measured so far")
    parser.add_argument("--script-share", type=float, default=0.15,
                        help="fraction of episodes played against a script rather than a "
                             "snapshot; the rest of the pool loses the difference")
    parser.add_argument("--elixir-scale", type=float, default=0.0,
                        help="weight on the elixir-differential shaping (0 = off)")
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

    # "random,rusher" and "sniper:4,rusher:1" are both accepted; an unweighted name is 1.
    parsed = [part.split(":") for part in args.scripts.split(",") if part]
    script_names = tuple(p[0] for p in parsed)
    script_weights = [float(p[1]) if len(p) > 1 else 1.0 for p in parsed]

    pool_dir = os.path.join(args.log_dir, f"{args.run_name}_pool") if args.self_play else None
    record_path = (os.path.join(args.log_dir, f"{args.run_name}_episodes")
                   if args.record_every else None)
    env_fns = [make_env(i, args.legacy_obs, pool_dir, args.mask, args.opponent_refresh,
                        record_path, args.record_every or 1,
                        rich_obs=args.rich_obs, count_obs=args.count_obs,
                        flat_action=args.flat_action, frames=args.frames,
                        dmg_scale=args.dmg_scale,
                        elixir_scale=args.elixir_scale,
                        script_names=script_names, script_weights=script_weights,
                        script_share=args.script_share)
               for i in range(args.n_envs)]
    env = VecMonitor(SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns))

    algo = MaskablePPO if args.mask else PPO
    if args.flat_action and not args.mask:
        # The joint action space is worth having mostly because the mask can then be
        # exact per cell, and the spatial head is written against the masked policy.
        raise SystemExit("--flat-action requires --mask (or pass --no-flat-action)")
    # Only the head differs; see `CRSpatialPolicy`.
    policy = CRSpatialPolicy if args.flat_action else "MultiInputPolicy"
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
            policy, env,
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
                                            rich_obs=args.rich_obs,
                                            count_obs=args.count_obs,
                                            flat_action=args.flat_action,
                                            frames=args.frames))
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                    tb_log_name=args.run_name, reset_num_timesteps=False)
    finally:
        print('Saving model.')
        model.save(os.path.join(args.log_dir, f'{args.run_name}_final'))


if __name__ == '__main__':
    main()
