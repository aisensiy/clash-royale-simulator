"""Measure how a policy actually plays, not just whether it wins.

Win rate hides the difference between "learned the game" and "found one trick". The
replay of the 6M model showed both symptoms of the latter: nearly every card on one
tile, and elixir spent the instant it arrives.
"""
import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from agents import SCRIPTS, decide, load_agent
from card_utils import Card
from environment import ARENA_W, CREnv, action_triple


def _shard(args):
    model_path, opponent_spec, games, seed = args
    import torch
    torch.set_num_threads(1)

    act = load_agent(model_path)
    foe = load_agent(opponent_spec)
    env = CREnv(opponent_model=foe, rich_obs=act.rich_obs,
                opponent_rich_obs=foe.rich_obs, count_obs=act.count_obs,
                opponent_count_obs=foe.count_obs, flat_action=act.flat_action,
                opponent_flat_action=foe.flat_action, frames=act.frames,
                opponent_frames=foe.frames)
    columns = Counter()
    elixir_left, plays, wins, total = [], 0, 0, 0
    for i in range(games):
        env.learner_player = i % 2
        obs, _ = env.reset(seed=seed * 1000 + i)
        done = False
        while not done:
            action = decide(act, obs, env, env.learner)
            slot, y, x = action_triple(action, act.flat_action)
            if slot != 0:
                card = env.battle.players[env.learner].cycle[slot - 1]
                before = env.battle.players[env.learner].elixir
                if before >= Card(card).elixir:
                    columns[x] += 1
                    plays += 1
                    elixir_left.append(before - Card(card).elixir)
            obs, _, done, _, info = env.step(action)
        total += 1
        wins += info.get("outcome") == 1
    return columns, plays, elixir_left, wins, total


def summarise(name, columns, plays, elixir_left, wins, total):
    top = columns.most_common(1)[0] if columns else (None, 0)
    spread = sum(1 for c in columns.values() if c >= plays * 0.02)
    mean_left = sum(elixir_left) / len(elixir_left) if elixir_left else 0.0
    held = sum(1 for e in elixir_left if e >= 4.0) / len(elixir_left) if elixir_left else 0.0
    print(f"{name:<26}{wins/total:>7.0%}{plays/total:>9.0f}"
          f"{top[1]/plays:>12.0%}{spread:>8}{mean_left:>12.2f}{held:>12.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--opponent", default="rusher")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    print(f"对手 = {args.opponent}，每个模型 {args.games} 局，两边轮流坐\n")
    print(f"{'模型':<26}{'胜率':>7}{'每局出牌':>9}{'最常用列占比':>12}{'常用列数':>8}"
          f"{'出牌后剩余圣水':>12}{'留4圣水以上':>12}")
    print("-" * 90)
    for path in args.models:
        shard = max(1, args.games // args.workers)
        jobs = [(path, args.opponent, shard, s) for s in range(args.workers)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            out = list(ex.map(_shard, jobs))
        columns = Counter()
        plays = wins = total = 0
        elixir_left = []
        for c, p, e, w, t in out:
            columns.update(c); plays += p; elixir_left += e; wins += w; total += t
        summarise(os.path.basename(path), columns, plays, elixir_left, wins, total)


if __name__ == "__main__":
    main()
