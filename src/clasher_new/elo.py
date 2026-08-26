"""Rate every snapshot in a pool on one ladder, anchored by the scripted opponents.

Win rate against the current opponent is not a progress signal under self-play: it sits
near 50% whether both sides improve or both decay. Elo fixes that by rating everyone on
a common scale, and pinning the scripts -- which never change -- keeps the whole scale
from drifting.

    python3 elo.py /output/sp/sp_pool --games 40 --workers 32
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

from agents import decide, load_agent
from environment import CREnv

# Fixed reference points. The spread is a guess at the gap between doing nothing and
# applying constant pressure; only the differences between rated players matter, and
# holding these still is what makes ratings comparable across training runs.
ANCHORS = {"idle": 0.0, "random": 400.0, "rusher": 700.0}
K = 24


def play_match(args):
    """One pairing, `games` episodes, blue = first name. Returns points for blue."""
    blue_spec, red_spec, games, seed = args
    import torch
    torch.set_num_threads(1)

    # Whether each side is masked and which observation it wants comes out of its own
    # checkpoint, so a masked run and an unmasked run can be rated on the same ladder.
    blue = load_agent(blue_spec)
    red = load_agent(red_spec)
    env = CREnv(opponent_model=red, rich_obs=blue.rich_obs,
                opponent_rich_obs=red.rich_obs, count_obs=blue.count_obs,
                opponent_count_obs=red.count_obs)
    score = 0.0
    for i in range(games):
        # Half the games from each side, so the arena's own bias cancels instead of
        # being credited to whoever happened to be drawn as blue.
        env.learner_player = i % 2
        obs, _ = env.reset(seed=seed * 1000 + i)
        done = False
        while not done:
            obs, _, done, _, _ = env.step(decide(blue, obs, env, env.learner))
        winner = env.battle.winner
        score += 1.0 if winner == env.learner else 0.0 if winner is not None else 0.5
    return blue_spec, red_spec, score, games


def expected(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", help="directory of snapshot_*.zip")
    ap.add_argument("--games", type=int, default=40, help="episodes per ordered pairing")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--extra", action="append", default=[], metavar="SCRIPT",
                    help="rate a named script alongside the snapshots; unlike the "
                         "anchors its rating is solved for rather than pinned")
    args = ap.parse_args()

    snapshots = sorted(os.path.join(args.pool, f)
                       for f in os.listdir(args.pool) if f.endswith(".zip"))
    if not snapshots:
        raise SystemExit(f"no snapshots in {args.pool}")
    players = list(ANCHORS) + args.extra + snapshots

    # Every snapshot plays every anchor and its neighbours. A full round robin over a
    # large pool costs more than it tells us -- what matters is each snapshot's position
    # against the fixed anchors and against the versions either side of it.
    rated = args.extra + snapshots
    pairings = [(s, a) for s in rated for a in ANCHORS]
    pairings += [(a, b) for a, b in zip(rated, rated[1:])]
    pairings += [(rated[-1], s) for s in rated[:-2]]

    jobs = [(b, r, args.games, i) for i, (b, r) in enumerate(pairings)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, out in enumerate(ex.map(play_match, jobs), 1):
            results.append(out)
            print(f"[{i}/{len(jobs)}] {os.path.basename(out[0])} vs "
                  f"{os.path.basename(out[1])}: {out[2]}/{out[3]}", flush=True)

    ratings = {p: ANCHORS.get(p, 700.0) for p in players}
    # Several passes so ratings settle; anchors are held fixed throughout.
    for _ in range(30):
        for blue, red, score, games in results:
            exp = expected(ratings[blue], ratings[red]) * games
            delta = K * (score - exp) / games
            if blue not in ANCHORS:
                ratings[blue] += delta
            if red not in ANCHORS:
                ratings[red] -= delta

    print(f"\n{'选手':<34}{'Elo':>8}")
    print("-" * 44)
    for name in sorted(players, key=lambda p: -ratings[p]):
        label = name if name in ANCHORS else os.path.basename(name)
        pin = "  (锚点，固定)" if name in ANCHORS else ""
        print(f"{label:<34}{ratings[name]:>8.0f}{pin}")


if __name__ == "__main__":
    main()
