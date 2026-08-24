"""Summarise the 2x2 ablation: read every run's tensorboard scalars and print a table.

Usage:  python3 compare_ablation.py /output/ablation [/output/ablation_from_other_box]
"""
import argparse
import glob
import os
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = ["legacy_nomask", "legacy_mask", "fixed_nomask", "fixed_mask"]
LABEL = {
    "legacy_nomask": "老观测 + 无屏蔽（Jason 现状）",
    "legacy_mask":   "老观测 + 有屏蔽",
    "fixed_nomask":  "新观测 + 无屏蔽",
    "fixed_mask":    "新观测 + 有屏蔽",
}
TAGS = ["rollout/ep_rew_mean", "eval/mean_reward_vs_random", "rollout/ep_len_mean"]


def load(root):
    """tag -> run -> [(step, value)], merged across every event file under `root`."""
    out = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(root, "*", "events.out.tfevents.*"))):
        run = os.path.basename(os.path.dirname(path))
        run = run.rsplit("_", 1)[0] if run.rsplit("_", 1)[-1].isdigit() else run
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            if tag in TAGS:
                out[tag][run].extend((e.step, e.value) for e in acc.Scalars(tag))
    for tag in out:
        for run in out[tag]:
            out[tag][run].sort()
    return out


def tail_mean(series, frac=0.1):
    """Average of the last `frac` of points -- less noisy than the final value alone."""
    if not series:
        return None
    n = max(1, int(len(series) * frac))
    return sum(v for _, v in series[-n:]) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    args = ap.parse_args()

    merged = defaultdict(lambda: defaultdict(list))
    for root in args.roots:
        for tag, runs in load(root).items():
            for run, pts in runs.items():
                merged[tag][run].extend(pts)

    print(f"{'配置':<28}{'训练回报(末10%)':>16}{'对随机对手评测':>16}{'平均局长':>10}{'步数':>12}")
    print("-" * 84)
    base = None
    for run in RUNS:
        train = tail_mean(merged["rollout/ep_rew_mean"].get(run, []))
        ev = tail_mean(merged["eval/mean_reward_vs_random"].get(run, []), frac=0.34)
        ln = tail_mean(merged["rollout/ep_len_mean"].get(run, []))
        steps = merged["rollout/ep_rew_mean"].get(run, [])
        last = steps[-1][0] if steps else 0
        if run == "legacy_nomask":
            base = train
        fmt = lambda v, w: f"{v:>{w}.2f}" if v is not None else f"{'-':>{w}}"
        print(f"{LABEL[run]:<28}{fmt(train,16)}{fmt(ev,16)}{fmt(ln,10)}{last:>12,}")

    if base is not None:
        print("\n相对基准（老观测+无屏蔽）的训练回报变化：")
        for run in RUNS[1:]:
            v = tail_mean(merged["rollout/ep_rew_mean"].get(run, []))
            if v is not None:
                print(f"  {LABEL[run]:<28}{v - base:+.2f}")


if __name__ == "__main__":
    main()
