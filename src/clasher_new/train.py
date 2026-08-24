import argparse
import os
import time

from environment import CREnv, random_strategy, entity_names

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
        self.fc = nn.Linear(cnn_out + 5 * self.embedding_dim + 1, features_dim)

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
        combined = torch.cat([grid_feat, hand_feat, elixir.float()], dim=1)
        return torch.relu(self.fc(combined))


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


class RandomEvalCallback(BaseCallback):
    """Win rate against the fixed random opponent -- the same yardstick as the old script."""

    # Evaluation runs episodes in a single un-vectorised env, so each call costs roughly
    # `n_episodes * 370` sequential env steps. At 100k/20 that was eating about as much
    # wall clock as the training it was measuring; 500k/10 brings it under 10%.
    def __init__(self, every=500_000, n_episodes=10, use_masking=True, legacy_obs=False, verbose=0):
        super().__init__(verbose)
        self.every = every
        self.n_episodes = n_episodes
        self.use_masking = use_masking
        self.legacy_obs = legacy_obs
        self._next = every

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        eval_env = DummyVecEnv([lambda: CREnv(opponent_model=random_strategy,
                                              legacy_obs=self.legacy_obs)])
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


def make_env(seed, legacy_obs):
    def _init():
        # Each worker simulates games in pure Python; a private BLAS thread pool per worker
        # would oversubscribe the box without speeding anything up.
        torch.set_num_threads(1)
        env = CREnv(opponent_model=random_strategy, legacy_obs=legacy_obs)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=int(os.environ.get("CR_N_ENVS", 96)))
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--torch-threads", type=int, default=16,
                        help="threads for the gradient update; the rollout forward pass is a\n"
                             "tiny batch, so a large pool here costs more than it buys")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="masked")
    parser.add_argument("--no-mask", action="store_true",
                        help="ablation: plain PPO, invalid actions left samplable")
    parser.add_argument("--legacy-obs", action="store_true",
                        help="ablation: reproduce the mirrored-observation bug")
    parser.add_argument("--log-dir", type=str, default="/output/cr_logs")
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)

    env_fns = [make_env(i, args.legacy_obs) for i in range(args.n_envs)]
    env = VecMonitor(SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns))

    algo = PPO if args.no_mask else MaskablePPO
    model = algo(
        "MultiInputPolicy", env,
        policy_kwargs={"features_extractor_class": CRFeatureExtractor},
        n_steps=args.n_steps, batch_size=args.batch_size,
        verbose=1, tensorboard_log=args.log_dir, device=args.device, seed=0,
    )
    callbacks = [
        CheckpointCallback(save_freq=max(1, 500_000 // args.n_envs),
                           save_path=args.log_dir, name_prefix=args.run_name),
        ThroughputCallback(),
        RandomEvalCallback(use_masking=not args.no_mask, legacy_obs=args.legacy_obs),
    ]
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                    tb_log_name=args.run_name, reset_num_timesteps=False)
    finally:
        print('Saving model.')
        model.save(os.path.join(args.log_dir, f'{args.run_name}_final'))


if __name__ == '__main__':
    main()
