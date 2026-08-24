"""The AI-vs-human server path, without sockets or a display.

What matters here is that the live server and training see the same game: the agent's
observation comes from the same CREnv, its actions land through the same egocentric
transform, and the human's plays reach its card counter. A fake policy stands in for a
checkpoint so the test needs no trained model on disk.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import battle
import player
from environment import DECK
from server import AI_DECISION_TICKS, GameServer


class FakePolicy:
    """Stands in for a loaded PPO. `plan` is a queue of (slot, y, x) actions."""

    def __init__(self, plan, rich=False):
        self.plan = list(plan)
        self.seen = []
        spaces = {"grid": None, "hand": None, "elixir": None}
        if rich:
            spaces.update({"context": None, "opp_hand": None})
        self.observation_space = type("Space", (), {"spaces": spaces})()

    def predict(self, obs, deterministic=False):
        self.seen.append(obs)
        return (self.plan.pop(0) if self.plan else (0, 0, 0)), None


def make_server(plan=(), rich=False):
    srv = GameServer(ai_checkpoint="fake.zip")
    srv.battle = battle.BattleState(player.PlayerState(0, DECK[:], 10.0),
                                    player.PlayerState(1, DECK[:], 10.0))
    srv.attach_ai(FakePolicy(plan, rich=rich))
    return srv


def test_ai_mode_waits_for_one_human():
    assert GameServer(ai_checkpoint="fake.zip").n_players == 1
    assert GameServer().n_players == 2


def test_the_agent_and_the_server_share_one_battle():
    """Not a copy: a stale battle would make the agent play a game nobody is watching."""
    srv = make_server()
    assert srv.ai_env.battle is srv.battle
    assert srv.ai_env.learner == 1


def test_agent_actions_land_in_its_own_half():
    """Row 0 is the agent's near edge, so player 1's deploys must be reflected."""
    srv = make_server(plan=[(1, 3, 4)])
    before = set(srv.battle.entities)
    srv.ai_step()
    new = [srv.battle.entities[i] for i in set(srv.battle.entities) - before]
    assert len(new) == 1, "the agent's card was not placed"
    unit = new[0]
    assert unit.player == 1
    # Egocentric (3, 4) for red is absolute (18 - 4.5, 32 - 3.5) -- the top half.
    assert unit.position.y > 16


def test_slot_zero_places_nothing():
    srv = make_server(plan=[(0, 5, 5)])
    before = set(srv.battle.entities)
    srv.ai_step()
    assert set(srv.battle.entities) == before


def test_the_agent_counts_the_humans_cards():
    """Without note_external_play the human's cycle stays unknown for the whole game."""
    srv = make_server(rich=True)
    assert np.allclose(srv.ai_env._opp_hand_belief(1), 0.5)
    for card in srv.battle.players[0].cycle[:4]:
        srv.battle.players[0].elixir = 10.0
        assert srv.battle.deploy_card(0, card, __import__("core").Position(9.5, 5.5))
        srv.ai_env.note_external_play(0, card)
    belief = srv.ai_env._opp_hand_belief(1)
    hand = srv.battle.players[0].cycle[:4]
    assert np.array_equal(belief, np.array([1.0 if c in hand else 0.0 for c in DECK],
                                           dtype=np.float32))


def test_the_agent_is_asked_twice_a_second():
    """30 ticks at 60Hz is the same decision period the policy was trained with."""
    assert AI_DECISION_TICKS == 30


def test_a_rich_checkpoint_gets_a_rich_observation():
    srv = make_server(rich=True)
    assert srv.ai_env.rich_obs
    srv.ai_step()
    assert set(srv.ai_model.seen[0]) == {"grid", "hand", "elixir", "context", "opp_hand"}


def test_a_plain_checkpoint_gets_the_plain_observation():
    srv = make_server(rich=False)
    assert not srv.ai_env.rich_obs
    srv.ai_step()
    assert set(srv.ai_model.seen[0]) == {"grid", "hand", "elixir"}
