"""Action masking, and the plumbing that decides who gets handed a mask.

The bug these guard against is silent in both directions. A masked policy asked to act
without a mask samples over actions it was never trained to score, so as a self-play
opponent it plays close to randomly -- the learner then trains against sandbags and the
run looks better than it is. A masked checkpoint *evaluated* without a mask measures far
weaker than it is, which is how "masking does not help" could be concluded from a
measurement that never used the masks.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import battle
import player
from card_utils import Card
from environment import ARENA_H, ARENA_W, CREnv, DECK, N_SLOTS, random_strategy
from selfplay import OpponentPool, PooledOpponent


def attach(elixir=10.0, opponent=random_strategy):
    env = CREnv(opponent_model=opponent)
    env.battle = battle.BattleState(player.PlayerState(0, DECK[:], elixir),
                                    player.PlayerState(1, DECK[:], elixir))
    return env


def split(mask):
    """The concatenated mask back into its three per-dimension parts."""
    return mask[:N_SLOTS], mask[N_SLOTS:N_SLOTS + ARENA_H], mask[N_SLOTS + ARENA_H:]


# ------------------------------------------------------------------- the mask itself

def test_waiting_is_always_allowed():
    """Slot 0 must survive every mask, or a broke agent has no legal action at all."""
    for elixir in (0.0, 1.0, 5.0, 10.0):
        slots, _, _ = split(attach(elixir).action_masks(0))
        assert slots[0]


def test_slot_mask_admits_exactly_the_affordable_cards():
    env = attach(4.0)
    slots, _, _ = split(env.action_masks(0))
    for i in range(1, N_SLOTS):
        card = env.battle.players[0].cycle[i - 1]
        assert bool(slots[i]) == (Card(card).elixir <= 4.0), card


def test_a_broke_agent_may_only_wait():
    slots, _, _ = split(attach(0.0).action_masks(0))
    assert slots[0] and not slots[1:].any()


def test_each_side_gets_its_own_mask():
    """The mask depends on that player's own elixir, not the learner's."""
    env = attach(10.0)
    env.battle.players[1].elixir = 0.0
    assert split(env.action_masks(0))[0][1:].any()
    assert not split(env.action_masks(1))[0][1:].any()


def test_mask_defaults_to_the_learner():
    env = attach(10.0)
    env.learner = 1
    env.battle.players[1].elixir = 0.0
    assert not split(env.action_masks())[0][1:].any()


# ------------------------------------------------------- who gets handed a mask

class Spy:
    """An opponent that records whether it was given a mask, and for which side."""

    def __init__(self, masked):
        self.masked = masked
        self.calls = []

    def __call__(self, observation, action_masks=None):
        self.calls.append(action_masks)
        return 0, 0, 0


def test_a_masked_opponent_is_handed_the_mask_for_its_own_side():
    spy = Spy(masked=True)
    env = attach(10.0, opponent=spy)
    env.learner = 0
    env.battle.players[1].elixir = 0.0   # only the opponent is broke
    env.opponent_action()
    assert len(spy.calls) == 1
    slots, _, _ = split(spy.calls[0])
    assert slots[0] and not slots[1:].any(), "got the learner's mask, not its own"


def test_an_unmasked_opponent_is_not_handed_one():
    spy = Spy(masked=False)
    env = attach(10.0, opponent=spy)
    env.opponent_action()
    assert spy.calls == [None]


def test_a_scripted_opponent_still_works():
    """Scripts take one argument; nothing may try to pass them a mask."""
    env = attach(10.0, opponent=random_strategy)
    env.opponent_action()  # must not raise


# ------------------------------------------------------- the self-play opponent

class FakePolicy:
    def __init__(self):
        self.masks = []

    def predict(self, observation, deterministic=False, action_masks=None):
        self.masks.append(action_masks)
        return (0, 0, 0), None


def make_pooled(tmp_path, masked, n_snapshots=1):
    for i in range(n_snapshots):
        open(os.path.join(str(tmp_path), f"snapshot_{i:012d}.zip"), "w").close()
    pool = OpponentPool(str(tmp_path), p_latest=1.0, p_history=0.0, p_script=0.0)
    algo = type("Algo", (), {"load": staticmethod(lambda path, device: FakePolicy())})
    return PooledOpponent(pool, {"random": random_strategy}, algo, seed=0, masked=masked)


def test_pooled_opponent_forwards_the_mask(tmp_path):
    opponent = make_pooled(tmp_path, masked=True)
    assert opponent.masked
    mask = np.ones(N_SLOTS + ARENA_H + ARENA_W, dtype=bool)
    opponent({}, mask)
    assert opponent._policy.masks[0] is mask


def test_an_unmasked_pool_never_asks_for_a_mask(tmp_path):
    opponent = make_pooled(tmp_path, masked=False)
    assert not opponent.masked
    opponent({}, None)
    assert opponent._policy.masks == [None]


def test_a_script_from_the_pool_wants_no_mask(tmp_path):
    """Even in a masked run: `random_strategy` takes one argument and would raise."""
    pool = OpponentPool(str(tmp_path), p_latest=0.0, p_history=0.0, p_script=1.0)
    algo = type("Algo", (), {"load": staticmethod(lambda path, device: FakePolicy())})
    opponent = PooledOpponent(pool, {"random": random_strategy, "rusher": random_strategy},
                              algo, seed=0, masked=True)
    assert not opponent.masked
    opponent({"elixir": np.array([5.0], dtype=np.float32)})


# --------------------------------------------------- reading a checkpoint's algorithm

def train_tiny(masked, path):
    """A 1-step checkpoint of the right class. Uses MlpPolicy on a toy env: the point is
    the file format, not the network."""
    import gymnasium as gym
    if masked:
        from sb3_contrib import MaskablePPO as Algo
    else:
        from stable_baselines3 import PPO as Algo
    env = gym.make("CartPole-v1")
    if masked:
        from sb3_contrib.common.wrappers import ActionMasker
        env = ActionMasker(env, lambda e: np.ones(2, dtype=bool))
    Algo("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu").save(path)


@pytest.mark.parametrize("masked", [True, False])
def test_a_checkpoint_reports_its_own_algorithm(tmp_path, masked):
    pytest.importorskip("sb3_contrib")
    from agents import is_masked_checkpoint
    path = os.path.join(str(tmp_path), "cp.zip")
    train_tiny(masked, path)
    assert is_masked_checkpoint(path) is masked


def test_scripts_load_as_agents_that_ignore_masks():
    from agents import decide, load_agent
    agent = load_agent("rusher")
    assert not agent.masked and not agent.rich_obs
    env = attach(10.0)
    assert decide(agent, env.observe(0), env, 0) is not None
