"""One snapshot cannot say what is moving, and this game is about what is moving.

The grid gives every unit's position and health at an instant. A Giant walking at our
tower and one that was just pulled back read identically; so does a tower that lost 40% of
its health this second and one that lost it ten seconds ago. Whether a push arrives before
we can answer it is a question about rates, and rates are not in a single frame.

These pin the stack that fixes it: N consecutive grids laid end to end, newest first, and
the plumbing that lets a stacked run and an unstacked one still play each other.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Position
from environment import (ARENA_H, ARENA_W, CREnv, DECK, N_COUNT_CHANNELS, N_UNIT_CHANNELS,
                         frames_for, grid_layout, random_strategy)

COUNTED = N_UNIT_CHANNELS + N_COUNT_CHANNELS


def running(frames=3, opponent_frames=None, count_obs=True, seed=5):
    env = CREnv(opponent_model=lambda obs: (0, 0, 0), count_obs=count_obs,
                frames=frames, opponent_frames=opponent_frames, learner_player=0)
    env.reset(seed=seed)
    for player in env.battle.players:
        player.elixir = 10.0
    return env


# --------------------------------------------------------------------- the layout

@pytest.mark.parametrize("counted", [False, True])
@pytest.mark.parametrize("frames", [1, 2, 3, 4, 8])
def test_a_width_says_how_it_was_built(counted, frames):
    """Every eval tool reads this off a checkpoint instead of being told, so a width that
    decodes two ways would silently serve the wrong observation."""
    base = COUNTED if counted else N_UNIT_CHANNELS
    assert grid_layout(base * frames) == (counted, frames)


def test_a_width_that_is_not_whole_frames_is_an_error():
    with pytest.raises(ValueError):
        grid_layout(16)


# --------------------------------------------------------------------- the stack

def test_the_grid_is_as_wide_as_the_stack():
    env = running(frames=3)
    assert env.observation_space["grid"].shape == (ARENA_H, ARENA_W, COUNTED * 3)
    obs = env.observe(0)
    assert obs["grid"].shape == env.observation_space["grid"].shape
    assert env.observation_space.contains(obs)


def test_the_newest_frame_is_first():
    """Both the scripts and the feature extractor read fixed channel indices. Putting the
    current frame anywhere but the front changes what every one of them means."""
    env = running(frames=3)
    stacked = env.observe(0)["grid"]
    current = env._frame(0)["grid"]
    assert np.array_equal(stacked[..., :COUNTED], current)


def test_the_opening_decision_repeats_itself_rather_than_inventing_an_empty_arena():
    """There is no past at the first decision. Zeros would say the arena was empty a
    second ago, which is a push that never happened."""
    env = running(frames=3)
    grid = env.observe(0)["grid"]
    for f in range(1, 3):
        assert np.array_equal(grid[..., :COUNTED], grid[..., f * COUNTED:(f + 1) * COUNTED])


def test_a_unit_that_moved_shows_up_as_a_difference_between_frames():
    """The whole point: after a few decisions the newest frame and the oldest disagree,
    and the disagreement is where the walking is."""
    import battle
    import player
    env = running(frames=3)
    # Only the first four cards of a cycle are in hand, so the Giant has to be dealt one.
    red = ["Giant"] + [c for c in DECK if c != "Giant"]
    env.battle = battle.BattleState(player.PlayerState(0, DECK[:], 10.0),
                                    player.PlayerState(1, red, 10.0))
    assert env.battle.deploy_card(1, "Giant", Position(3.5, 19.0))
    for _ in range(200):                      # let it walk
        env.battle.step(1 / 60)
    env.observe(0)
    moved = False
    for step in range(4):
        env._step_index += 1
        for _ in range(30):
            env.battle.step(1 / 60)
        grid = env.observe(0)["grid"]
        newest = grid[..., :COUNTED]
        oldest = grid[..., 2 * COUNTED:]
        moved = moved or not np.array_equal(newest, oldest)
    assert moved, "four decisions of a walking Giant left no trace between frames"


def test_reading_the_observation_twice_does_not_shift_history():
    """A tool that looks at the observation without stepping -- every probe does -- must
    not age the stack under the policy that is about to act on it."""
    env = running(frames=3)
    first = env.observe(0)["grid"].copy()
    again = env.observe(0)["grid"]
    assert np.array_equal(first, again)
    assert len(env._grid_history[0]) == 3


def test_the_history_starts_empty_again_on_reset():
    """Or the last game's final push is the new game's opening frame."""
    env = running(frames=3)
    env.observe(0)
    env.reset(seed=9)
    grid = env.observe(0)["grid"]
    assert np.array_equal(grid[..., :COUNTED], grid[..., COUNTED:2 * COUNTED])


# --------------------------------------------------------------------- per side

def test_each_side_gets_the_stack_it_was_trained_on():
    env = running(frames=3, opponent_frames=1)
    assert env.observe(0)["grid"].shape[-1] == COUNTED * 3
    assert env.observe(1)["grid"].shape[-1] == COUNTED


def test_a_stacked_env_plays_a_whole_game():
    env = CREnv(opponent_model=random_strategy, count_obs=True, frames=3, learner_player=0)
    env.reset(seed=3)
    done, steps = False, 0
    while not done and steps < 700:
        _, _, done, _, _ = env.step((0, 0, 0))
        steps += 1
    assert done, "the game never ended"


# --------------------------------------------------------------------- plumbing

class FakeModel:
    def __init__(self, channels):
        import gymnasium as gym
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                   shape=(ARENA_H, ARENA_W, channels)),
        })


def test_the_stack_is_read_off_the_checkpoint():
    assert frames_for(FakeModel(COUNTED)) == 1
    assert frames_for(FakeModel(COUNTED * 3)) == 3
    assert frames_for(FakeModel(N_UNIT_CHANNELS)) == 1


def test_a_script_stays_on_a_single_frame():
    """A ruler that changes between rounds is not a ruler."""
    from agents import load_agent
    assert load_agent("counter").frames == 1


def test_the_feature_extractor_widens_by_exactly_the_stack():
    """Each frame gets the same treatment -- entity id embedded, card type one-hot -- so
    the input planes are one frame's worth, times the stack, and nothing else moves."""
    torch = pytest.importorskip("torch")
    from train import CRFeatureExtractor
    one = CRFeatureExtractor(CREnv(opponent_model=random_strategy, count_obs=True,
                                   frames=1).observation_space)
    three = CRFeatureExtractor(CREnv(opponent_model=random_strategy, count_obs=True,
                                     frames=3).observation_space)
    assert three.in_channels == 3 * one.in_channels
    env = CREnv(opponent_model=random_strategy, count_obs=True, frames=3)
    obs, _ = env.reset(seed=1)
    batch = {k: torch.as_tensor(np.asarray(v)[None]) for k, v in obs.items()}
    assert three(batch).shape == (1, three.features_dim)
