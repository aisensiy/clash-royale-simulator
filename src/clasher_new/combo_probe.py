"""Does the agent ever put two cards down together?

A tank in front of support deals roughly twice the damage the same two cards deal played
apart, and that is the whole reason a player saves elixir. But the payoff only exists if
the agent ever holds enough to buy both and puts them down close together in time and
space. This counts how often that actually happens.

A play joins the combo before it if it lands within `WINDOW` seconds and `RADIUS` tiles
of the previous one -- close enough that the two units travel and fight as one group.

    python3 combo_probe.py /output/ab_elixir/trade_final.zip --games 20
"""
import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from agents import decide, load_agent, make_rusher
from card_utils import Card
from environment import CREnv, action_triple

WINDOW = 2.0     # seconds; four decisions
RADIUS = 4.0     # tiles


def shard(args):
    path, games, seed = args
    import torch
    torch.set_num_threads(1)

    agent = load_agent(path)
    env = CREnv(opponent_model=make_rusher(seed), rich_obs=agent.rich_obs,
                count_obs=agent.count_obs, flat_action=agent.flat_action)
    sizes, peak_elixir, played_games = Counter(), 0.0, 0
    for i in range(games):
        env.learner_player = i % 2
        obs, _ = env.reset(seed=seed * 1000 + i)
        group, last = [], None
        done = False
        while not done:
            me = env.battle.players[env.learner]
            before = me.elixir
            peak_elixir = max(peak_elixir, before)
            action = decide(agent, obs, env, env.learner)
            slot, row, col = action_triple(action, agent.flat_action)
            card = me.cycle[slot - 1] if slot else None
            obs, _, done, _, _ = env.step(action)
            if card is None or env.battle.players[env.learner].elixir >= before:
                continue                       # nothing was actually placed
            now = env.battle.time
            here = np.array([col, row], dtype=float)
            if (last is not None and now - last[0] <= WINDOW
                    and np.linalg.norm(here - last[1]) <= RADIUS):
                group.append(Card(card).elixir)
            else:
                if group:
                    sizes[len(group)] += 1
                group = [Card(card).elixir]
            last = (now, here)
        if group:
            sizes[len(group)] += 1
        played_games += 1
    return sizes, peak_elixir, played_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print(f"一次「组合」= {WINDOW:.0f} 秒内、{RADIUS:.0f} 格内连续落下的牌\n")
    for path in args.models:
        per = max(1, args.games // args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            out = list(ex.map(shard, [(path, per, s) for s in range(args.workers)]))
        sizes = Counter()
        for s, _, _ in out:
            sizes.update(s)
        peak = max(p for _, p, _ in out)
        games = sum(g for _, _, g in out)
        cards = sum(n * c for n, c in sizes.items())
        grouped = sum(n * c for n, c in sizes.items() if n >= 2)

        print(f"{os.path.basename(path)}  ({games} 局)")
        print(f"  每局出牌 {cards / games:.0f} 张，其中 {100 * grouped / max(cards, 1):.0f}% "
              f"是和别的牌一起下的")
        for n in sorted(sizes):
            print(f"    {n} 张一组: {sizes[n] / games:>6.1f} 次/局")
        print(f"  整个测试里见过的最高圣水 {peak:.1f}\n")


if __name__ == "__main__":
    main()
