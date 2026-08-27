"""What does the agent actually do with its elixir?

`behaviour.py` only records elixir at the moment of a successful play, which cannot tell
"spends the instant it can" apart from "sits at the cap unable to place anything". This
samples every decision instead, and separates the three ways a decision passes without a
card going down: chose to wait, could not afford it, or picked an illegal cell.

That distinction inverted the diagnosis once already. The elixir-at-play numbers looked
like an agent that refuses to save; sampling every decision showed one that is never
above 7 and spends four decisions in five reaching for a card it cannot pay for.

    python3 elixir_probe.py /output/mask/mask_final.zip --games 40
"""
import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from agents import decide, load_agent, make_rusher
from card_utils import Card
from environment import CREnv, action_triple

STARTING_ELIXIR = 5.0

# Seconds per elixir, by the tier boundaries in `BattleState.step`.
TIERS = ((120.0, 2.8), (240.0, 1.4), (float("inf"), 2.8 / 3))


def elixir_generated(duration):
    """How much elixir a player earns in a game of `duration` seconds.

    Comparing what was spent against a full 300s game's worth is what made the earlier
    read of these numbers wrong: the agents end their games around 160s, so a
    300s budget makes a player who spends everything look like a hoarder.
    """
    total, start = 0.0, 0.0
    for end, seconds_each in TIERS:
        total += max(0.0, min(duration, end) - start) / seconds_each
        start = end
        if duration <= end:
            break
    return total


BUCKETS = ("chose to wait", "cannot afford", "tried to place", "placed", "illegal cell")


def shard(args):
    path, games, seed = args
    import torch
    torch.set_num_threads(1)

    agent = load_agent(path)
    env = CREnv(opponent_model=make_rusher(seed), rich_obs=agent.rich_obs,
                count_obs=agent.count_obs, flat_action=agent.flat_action)
    elixir, tally, spent, played, seconds = [], Counter(), 0.0, 0, 0.0
    for i in range(games):
        env.learner_player = i % 2
        obs, _ = env.reset(seed=seed * 1000 + i)
        done = False
        while not done:
            me = env.battle.players[env.learner]
            before = me.elixir
            elixir.append(before)
            action = decide(agent, obs, env, env.learner)
            slot, _, _ = action_triple(action, agent.flat_action)
            if slot == 0:
                tally["chose to wait"] += 1
                obs, _, done, _, _ = env.step(action)
                continue
            card = me.cycle[slot - 1]
            affordable = me.can_play_card(card)
            tally["tried to place" if affordable else "cannot afford"] += 1
            obs, _, done, _, _ = env.step(action)
            if env.battle.players[env.learner].elixir < before:
                tally["placed"] += 1
                spent += Card(card).elixir
            elif affordable:
                tally["illegal cell"] += 1
        played += 1
        seconds += env.battle.time
    return elixir, tally, spent, played, seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    for path in args.models:
        per = max(1, args.games // args.workers)
        jobs = [(path, per, s) for s in range(args.workers)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            out = list(ex.map(shard, jobs))
        elixir = np.array([e for r in out for e in r[0]])
        tally = Counter()
        for r in out:
            tally.update(r[1])
        spent = sum(r[2] for r in out)
        games = sum(r[3] for r in out)
        seconds = sum(r[4] for r in out)
        earned = elixir_generated(seconds / games) + STARTING_ELIXIR

        print(f"\n{os.path.basename(path)}  ({games} 局, {len(elixir)} 个决策点)")
        print(f"  圣水均值 {elixir.mean():.2f}   中位数 {np.median(elixir):.2f}")
        print(f"  处于满圣水(>=9.5)的决策占比 {100 * (elixir >= 9.5).mean():.1f}%")
        print(f"  处于 >=7 的占比 {100 * (elixir >= 7).mean():.1f}%"
              f"   <=2 的占比 {100 * (elixir <= 2).mean():.1f}%")
        # "placed" and "illegal cell" break down "tried to place", so the column does not
        # sum to 100%.
        total = tally["chose to wait"] + tally["cannot afford"] + tally["tried to place"]
        for key in BUCKETS:
            print(f"  {key:<16} {tally[key]:>7}  ({100 * tally[key] / total:.1f}% of decisions)")
        print(f"  一局平均 {seconds / games:.0f} 秒，这么长的一局能挣到 {earned:.0f} 圣水")
        print(f"  实际花掉 {spent / games:.0f}/局 —— 挣到的 {100 * spent / games / earned:.0f}%")


if __name__ == "__main__":
    main()
