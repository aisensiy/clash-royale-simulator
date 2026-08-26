"""The grid has to say how many bodies are on a cell, not just what the last one was.

`CREnv.observe` writes a whole row of numbers into `obs[y][x]` per unit, so three Minions
from one card -- they land close enough to share a cell -- read back as a single Minion.
A push of six bodies and a push of two look the same on the input, which means "is this
push worth answering" is not a hard question to learn: it is an unanswerable one.

These pin the two count channels that fix it, and the plumbing around them: the two
encodings are different input widths, so every place that builds an environment for a
checkpoint has to serve it the width it was trained on.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Position
from environment import (ARENA_H, ARENA_W, CH_ENEMY_COUNT, CH_OWN_COUNT, CREnv, DECK,
                         N_COUNT_CHANNELS, N_UNIT_CHANNELS, count_obs_for,
                         random_strategy)

CH_PLAYER, CH_COST = 1, 2


def board(count_obs=True, opponent_count_obs=None, enemy_at=None, own_at=None,
          learner=0, settle=120):
    """A battle with units placed by hand, and the observation both sides get from it."""
    import battle
    import player
    enemy_at, own_at = enemy_at or [], own_at or []

    def deck_with(names):
        return list(names) + [c for c in DECK if c not in names]

    env = CREnv(opponent_model=lambda obs: (0, 0, 0), count_obs=count_obs,
                opponent_count_obs=opponent_count_obs)
    decks = {learner: deck_with([n for n, _ in own_at]),
             1 - learner: deck_with([n for n, _ in enemy_at])}
    env.battle = battle.BattleState(player.PlayerState(0, decks[0][:], 10.0),
                                    player.PlayerState(1, decks[1][:], 10.0))
    env.learner = learner
    for pid, spots in ((1 - learner, enemy_at), (learner, own_at)):
        for name, pos in spots:
            assert env.battle.deploy_card(pid, name, pos), f"{name} at {pos.x},{pos.y}"
    for _ in range(settle):                  # let the deploy timers run out
        env.battle.step(1 / 60)
    return env


def occupied(grid):
    """The cells holding an enemy unit, as (row, col) pairs. Towers cost nothing."""
    rows, cols = np.nonzero((grid[:, :, CH_PLAYER] > 0.5) & (grid[:, :, CH_COST] > 0))
    return list(zip(rows.tolist(), cols.tolist()))


def bodies(grid, channel):
    """How many deployed units are counted, ignoring the three towers each side starts
    with -- they are entities like any other and are counted too, which is right for the
    policy and only noise for these tests."""
    return float(grid[:, :, channel][grid[:, :, CH_COST] > 0].sum())


# --------------------------------------------------------------------- counting

def test_three_bodies_from_one_card_are_counted_as_three():
    """The whole point. One Minions card puts three units on the board, and before this
    they collapsed into whichever one was written last."""
    env = board(enemy_at=[("Minions", Position(3.5, 19.0))])
    grid = env.observe(0)["grid"]
    total = bodies(grid, CH_ENEMY_COUNT)
    assert total == 3.0, f"three Minions counted as {total}"
    stacked = [c for c in occupied(grid) if grid[c[0], c[1], CH_ENEMY_COUNT] > 1]
    assert stacked, "they never shared a cell, so this board proves nothing"


def test_the_two_sides_are_counted_apart():
    """A cell can hold units from both players -- they meet at the bridge every game. One
    combined count would say two bodies without saying whose."""
    env = board(enemy_at=[("Knight", Position(3.5, 19.0))],
                own_at=[("Knight", Position(3.5, 12.0))])
    grid = env.observe(0)["grid"]
    assert bodies(grid, CH_ENEMY_COUNT) == 1.0
    assert bodies(grid, CH_OWN_COUNT) == 1.0
    # And each knight is counted on exactly one side of the ledger, not both.
    assert not np.any((grid[:, :, CH_OWN_COUNT] > 0) & (grid[:, :, CH_ENEMY_COUNT] > 0))


def test_an_empty_cell_counts_nothing():
    env = board()
    grid = env.observe(0)["grid"]
    empty = (grid[:, :, CH_OWN_COUNT] + grid[:, :, CH_ENEMY_COUNT]) == 0
    assert np.all(grid[:, :, :N_UNIT_CHANNELS][empty] == 0), "an empty cell described a unit"


# --------------------------------------------------------------------- the old grid

def test_the_first_fifteen_channels_are_untouched():
    """Whatever the counts change, they change by addition: a checkpoint reading channel
    9 for hit points still finds hit points there, and so does every script."""
    placed = [("Giant", Position(3.5, 19.0)), ("Musketeer", Position(4.5, 19.0))]
    counted = board(count_obs=True, enemy_at=placed).observe(0)["grid"]
    plain = board(count_obs=False, enemy_at=placed).observe(0)["grid"]
    assert plain.shape[-1] == N_UNIT_CHANNELS
    assert counted.shape[-1] == N_UNIT_CHANNELS + N_COUNT_CHANNELS
    assert np.array_equal(counted[:, :, :N_UNIT_CHANNELS], plain)


@pytest.mark.parametrize("count_obs", [False, True])
def test_the_observation_matches_the_declared_space(count_obs):
    env = CREnv(opponent_model=random_strategy, count_obs=count_obs)
    obs, _ = env.reset(seed=1)
    assert env.observation_space["grid"].shape == obs["grid"].shape
    assert obs["grid"].shape == (ARENA_H, ARENA_W,
                                N_UNIT_CHANNELS + (N_COUNT_CHANNELS if count_obs else 0))
    assert env.observation_space.contains(obs)


def test_each_side_gets_the_width_it_was_trained_on():
    """A run with the counts and a run without have to be able to play each other, which
    is the only way to know whether the channels were worth adding."""
    env = board(count_obs=True, opponent_count_obs=False,
                enemy_at=[("Knight", Position(3.5, 19.0))])
    assert env.observe(0)["grid"].shape[-1] == N_UNIT_CHANNELS + N_COUNT_CHANNELS
    assert env.observe(1)["grid"].shape[-1] == N_UNIT_CHANNELS


# --------------------------------------------------------------------- plumbing

class FakeModel:
    """Just enough of a loaded checkpoint for the width check to read."""

    def __init__(self, channels):
        import gymnasium as gym
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                   shape=(ARENA_H, ARENA_W, channels)),
        })


def test_the_width_is_read_off_the_checkpoint():
    """Nothing declares this on the command line: the eval tools load checkpoints from
    several runs at once, and getting it wrong is a shape error at best."""
    assert count_obs_for(FakeModel(N_UNIT_CHANNELS + N_COUNT_CHANNELS))
    assert not count_obs_for(FakeModel(N_UNIT_CHANNELS))


def test_a_script_opponent_stays_on_the_old_grid():
    """A ruler that changes between rounds is not a ruler. The scripts read fixed channel
    indices and were calibrated against the 15-channel grid, so they keep it."""
    from agents import load_agent
    assert load_agent("counter").count_obs is False


def test_the_feature_extractor_follows_the_grid_width():
    """The extractor used to hardcode 13 input planes, which is 15 minus the entity id and
    the card type. Adding a channel to the observation must not need an edit there."""
    torch = pytest.importorskip("torch")
    from train import CRFeatureExtractor
    plain = CREnv(opponent_model=random_strategy, count_obs=False).observation_space
    counted = CREnv(opponent_model=random_strategy, count_obs=True).observation_space
    assert (CRFeatureExtractor(counted).in_channels
            - CRFeatureExtractor(plain).in_channels) == N_COUNT_CHANNELS
    extractor = CRFeatureExtractor(counted)
    env = CREnv(opponent_model=random_strategy, count_obs=True)
    obs, _ = env.reset(seed=1)
    batch = {k: torch.as_tensor(np.asarray(v)[None]) for k, v in obs.items()}
    assert extractor(batch).shape == (1, extractor.features_dim)


def test_a_counted_env_runs_a_whole_game():
    env = CREnv(opponent_model=random_strategy, count_obs=True, learner_player=0)
    env.reset(seed=3)
    done, steps = False, 0
    while not done and steps < 700:
        _, _, done, _, _ = env.step((0, 0, 0))
        steps += 1
    assert done, "the game never ended"
