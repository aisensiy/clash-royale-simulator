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
from agents import SCRIPTS, decide, load_agent
from environment import ARENA_H, ARENA_W, CREnv, N_SLOTS, random_strategy, rich_obs_for

# Whether a checkpoint was trained with action masking is read out of the file itself;
# only the observation layout has to be declared, because the legacy runs predate the
# flag that records it.
RUNS = {
    "legacy_nomask": dict(legacy_obs=True),
    "legacy_mask": dict(legacy_obs=True),
    "fixed_nomask": dict(legacy_obs=False),
    "fixed_mask": dict(legacy_obs=False),
}


# `anchor` is the held-out scripted defender: it never appears in the opponent pool, so
# a win rate against it is a win rate against play the agent was not drilled on.
OPPONENT_NAMES = ("idle", "random", "rusher", "anchor")


def play(args):
    run, model_path, opp_name, deterministic, n_episodes, seed = args
    # Without this every worker's torch grabs the whole box. 40 workers x 64 threads
    # spend all their time fighting over cores: measured 50x slower than 1 thread each.
    import torch
    torch.set_num_threads(1)
    cfg = RUNS[run]
    agent = load_agent(model_path, deterministic=deterministic)

    # Sides alternate: with an arena that favours red, a blue-only measurement reads
    # several points low and is not comparable to anything measured the other way.
    # A fresh instance per worker: the defender scripts keep state between decisions.
    env = CREnv(opponent_model=SCRIPTS[opp_name](seed), legacy_obs=cfg["legacy_obs"],
                rich_obs=agent.rich_obs, count_obs=agent.count_obs,
                flat_action=agent.flat_action, frames=agent.frames)
    wins = losses = draws = 0
    for ep in range(n_episodes):
        env.learner_player = ep % 2
        obs, _ = env.reset(seed=seed * 1000 + ep)
        done = False
        while not done:
            obs, _, done, _, _ = env.step(decide(agent, obs, env, env.learner))
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
                    help="evaluate an arbitrary checkpoint (masking and observation "
                         "layout are read from the file); repeatable, and replaces the "
                         "built-in ablation grid when given")
    args = ap.parse_args()

    if args.model:
        RUNS.clear()
        paths = {}
        for spec in args.model:
            name, _, path = spec.partition("=")
            RUNS[name] = dict(legacy_obs=False)
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
        for opp in OPPONENT_NAMES:
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
        print(f"{'配置':<16}" + "".join(f"{o:>22}" for o in OPPONENT_NAMES))
        for run in RUNS:
            row = f"{run:<16}"
            for opp in OPPONENT_NAMES:
                if (run, opp, det) not in tally:
                    row += f"{'-':>22}"
                    continue
                w, l, d = tally[(run, opp, det)]
                n = w + l + d
                row += f"{f'{w/n:.0%} 胜 {d/n:.0%} 平':>22}"
            print(row)


if __name__ == "__main__":
    main()
