"""Measure win rate, not reward, for each trained checkpoint.

Reward is hard to read across configurations -- it mixes crowns, tower damage and the
terminal bonus. Win rate against a fixed opponent is the number that means something.

Usage (from src/clasher_new):
    python3 winrate.py /output/ablation --episodes 100
"""
import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from card_utils import Card
from core import Position
from environment import ARENA_H, ARENA_W, CREnv, N_SLOTS, random_strategy, rich_obs_for

RUNS = {
    "legacy_nomask": dict(legacy_obs=True, masked=False),
    "legacy_mask": dict(legacy_obs=True, masked=True),
    "fixed_nomask": dict(legacy_obs=False, masked=False),
    "fixed_mask": dict(legacy_obs=False, masked=True),
}


def idle_strategy(observation):
    """Never plays a card. The true floor: anything that loses to this is broken."""
    return 0, 0, 0


def make_rusher(seed=0):
    """Dumps the cheapest affordable card down one lane as soon as it can afford it.

    This is the 'always attacking, never defending' script the reference plan describes:
    a low bar, but a much harder one than random because it actually applies pressure.
    """
    rng = random.Random(seed)

    def strategy(observation):
        elixir = float(observation['elixir'][0])
        if elixir < 4.0:
            return 0, 0, 0
        slot = rng.randint(1, N_SLOTS - 1)
        return slot, rng.randint(10, 13), rng.choice([4, 5, 12, 13])

    return strategy


OPPONENTS = {"idle": idle_strategy, "random": random_strategy, "rusher": make_rusher()}


def play(args):
    run, model_path, opp_name, deterministic, n_episodes, seed = args
    # Without this every worker's torch grabs the whole box. 40 workers x 64 threads
    # spend all their time fighting over cores: measured 50x slower than 1 thread each.
    import torch
    torch.set_num_threads(1)
    cfg = RUNS[run]
    if cfg["masked"]:
        from sb3_contrib import MaskablePPO as Algo
    else:
        from stable_baselines3 import PPO as Algo
    model = Algo.load(model_path, device="cpu")

    # Sides alternate: with an arena that favours red, a blue-only measurement reads
    # several points low and is not comparable to anything measured the other way.
    env = CREnv(opponent_model=OPPONENTS[opp_name], legacy_obs=cfg["legacy_obs"],
                rich_obs=rich_obs_for(model))
    wins = losses = draws = 0
    for ep in range(n_episodes):
        env.learner_player = ep % 2
        obs, _ = env.reset(seed=seed * 1000 + ep)
        done = False
        while not done:
            if cfg["masked"]:
                action, _ = model.predict(obs, deterministic=deterministic,
                                          action_masks=env.action_masks())
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, _, _ = env.step(action)
        me, foe = env.learner, 1 - env.learner
        players = env.battle.players
        # Compare crowns first so a timeout with a crown lead still counts as a win.
        mine, theirs = players[me].get_crown_count(), players[foe].get_crown_count()
        if theirs > mine:
            wins += 1
        elif mine > theirs:
            losses += 1
        elif env.battle.winner == me:
            wins += 1
        elif env.battle.winner == foe:
            losses += 1
        else:
            draws += 1
    return run, opp_name, deterministic, wins, losses, draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                    help="evaluate an arbitrary checkpoint (fixed observation, plain PPO); "
                         "repeatable, and replaces the built-in ablation grid when given")
    args = ap.parse_args()

    if args.model:
        RUNS.clear()
        paths = {}
        for spec in args.model:
            name, _, path = spec.partition("=")
            RUNS[name] = dict(legacy_obs=False, masked=False)
            paths[name] = path
    else:
        paths = None

    jobs = []
    for run in RUNS:
        path = (paths[run] if paths is not None
                else os.path.join(args.root, f"{run}_final.zip"))
        if not os.path.exists(path):
            print(f"跳过 {run}: 找不到 {path}")
            continue
        for opp in OPPONENTS:
            for det in (True, False):
                per = max(1, args.episodes // 4)
                for shard in range(4):
                    jobs.append((run, path, opp, det, per, shard))

    tally = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for run, opp, det, w, l, d in ex.map(play, jobs):
            key = (run, opp, det)
            a, b, c = tally.get(key, (0, 0, 0))
            tally[key] = (a + w, b + l, c + d)
            done += 1
            print(f"[{done}/{len(jobs)}] {run} vs {opp} "
                  f"{'det' if det else 'sample'}: {w}胜 {l}负 {d}平", flush=True)

    for det in (False, True):
        print(f"\n{'=== 采样策略（训练时的行为）' if not det else '=== 确定性策略（每步取最优）'} ===")
        print(f"{'配置':<16}" + "".join(f"{o:>22}" for o in OPPONENTS))
        for run in RUNS:
            row = f"{run:<16}"
            for opp in OPPONENTS:
                if (run, opp, det) not in tally:
                    row += f"{'-':>22}"
                    continue
                w, l, d = tally[(run, opp, det)]
                n = w + l + d
                row += f"{f'{w/n:.0%} 胜 {d/n:.0%} 平':>22}"
            print(row)


if __name__ == "__main__":
    main()
