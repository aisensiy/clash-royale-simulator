"""Tests for the opponent pool. No torch involved: the pool only handles paths."""
import os
import random
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selfplay import HISTORY, LATEST, SCRIPT, OpponentPool


def make_pool(tmp_path, n_snapshots=0, **kwargs):
    pool = OpponentPool(str(tmp_path), **kwargs)
    for i in range(n_snapshots):
        open(os.path.join(str(tmp_path), f"snapshot_{i:012d}.zip"), "w").close()
    return pool


def test_shares_must_sum_to_one(tmp_path):
    with pytest.raises(ValueError):
        OpponentPool(str(tmp_path), p_latest=0.5, p_history=0.4, p_script=0.2)


def test_empty_pool_falls_back_to_scripts(tmp_path):
    pool = make_pool(tmp_path)
    for _ in range(20):
        kind, label, target = pool.sample(random.Random(0))
        assert kind == SCRIPT
        assert label.startswith("script:")


def test_snapshots_are_ordered_by_step_not_by_string(tmp_path):
    pool = make_pool(tmp_path)
    for step in (2_000_000, 500_000, 10_000_000):
        open(os.path.join(str(tmp_path), f"snapshot_{step:012d}.zip"), "w").close()
    steps = [int(os.path.basename(p)[len("snapshot_"):-len(".zip")]) for p in pool.snapshot_paths()]
    assert steps == sorted(steps)


def test_sampling_follows_the_configured_shares(tmp_path):
    pool = make_pool(tmp_path, n_snapshots=6, p_latest=0.45, p_history=0.40, p_script=0.15)
    rng = random.Random(7)
    kinds = Counter(pool.sample(rng)[0] for _ in range(4000))
    assert kinds[SCRIPT] / 4000 == pytest.approx(0.15, abs=0.03)
    assert kinds[LATEST] / 4000 == pytest.approx(0.45, abs=0.03)
    assert kinds[HISTORY] / 4000 == pytest.approx(0.40, abs=0.03)


def test_history_never_returns_the_latest_snapshot(tmp_path):
    pool = make_pool(tmp_path, n_snapshots=5)
    latest = pool.snapshot_paths()[-1]
    rng = random.Random(3)
    for _ in range(500):
        kind, _, target = pool.sample(rng)
        if kind == HISTORY:
            assert target != latest


def test_single_snapshot_is_always_the_latest(tmp_path):
    pool = make_pool(tmp_path, n_snapshots=1)
    rng = random.Random(1)
    kinds = {pool.sample(rng)[0] for _ in range(200)}
    assert HISTORY not in kinds


def test_pruning_keeps_the_oldest_and_newest_and_spreads_the_rest(tmp_path):
    pool = make_pool(tmp_path, max_snapshots=4)
    paths = [f"snapshot_{i:012d}.zip" for i in range(20)]
    dropped = set(pool.prune_list(paths))
    kept = [p for p in paths if p not in dropped]
    assert len(kept) == 4
    assert kept[0] == paths[0], "oldest snapshot must survive"
    assert kept[-1] == paths[-1], "newest snapshot must survive"


def test_pruning_is_a_no_op_below_the_cap(tmp_path):
    pool = make_pool(tmp_path, max_snapshots=8)
    assert pool.prune_list([f"snapshot_{i:012d}.zip" for i in range(8)]) == []


class FakeModel:
    """Just enough of an SB3 model for the pool to save something."""

    def save(self, path):
        open(path, "w").close()


def test_snapshot_schedule_anchors_to_where_the_run_starts(tmp_path):
    """Resuming from a 6M-step checkpoint must not backfill every missed threshold."""
    import train

    pool = OpponentPool(str(tmp_path))
    cb = train.SnapshotCallback(pool, every=100_000)
    cb.model = FakeModel()

    cb.num_timesteps = 6_000_000
    cb._on_training_start()
    for step in range(6_000_000, 6_050_000, 10_000):
        cb.num_timesteps = step
        cb._on_step()
    assert pool.snapshot_paths() == [], "nothing is due yet in the first 50k steps"

    cb.num_timesteps = 6_100_000
    cb._on_step()
    assert len(pool.snapshot_paths()) == 1
