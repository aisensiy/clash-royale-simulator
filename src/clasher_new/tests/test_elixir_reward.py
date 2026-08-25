"""The elixir-differential shaping.

Tower HP alone reports a perfect defence as 0.000 reward -- identical to a stretch where
nothing happened. This term pays for the change in how much elixir each side still owns,
counting the bank and the board together, so a trade shows up the moment it is made.

The tests that matter are the ones that keep it from being farmed: playing a card must
be worth nothing by itself, and a card that summons three bodies must not be worth three
times its price.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import battle
import player
from card_utils import Card
from core import Position
from agents import make_rusher
from environment import CREnv, DECK, KILL_SHARE, UNIT_VALUE, random_strategy


def attach(elixir=10.0, scale=1.0, deck=None):
    env = CREnv(opponent_model=random_strategy, elixir_scale=scale)
    deck = deck or DECK
    env.battle = battle.BattleState(player.PlayerState(0, deck[:], elixir),
                                    player.PlayerState(1, deck[:], elixir))
    env.learner = 0
    return env


def deck_with(*first):
    return list(first) + [c for c in DECK if c not in first]


# ------------------------------------------------------------------ what a unit is worth

def test_a_multi_body_card_splits_its_price():
    """Minions costs 3 and puts down three bodies; each body is worth 1, not 3."""
    assert UNIT_VALUE["Minions"] == pytest.approx(Card("Minions").elixir / 3)
    assert UNIT_VALUE["Archer"] == pytest.approx(Card("Archer").elixir / 2)
    assert UNIT_VALUE["Knight"] == pytest.approx(Card("Knight").elixir)


def test_spells_have_no_standing_value():
    """A Fireball leaves no body behind, so it must not appear in the table at all."""
    for name in DECK:
        if Card(name).type == "spell":
            assert name not in UNIT_VALUE


def test_towers_are_not_counted_as_army():
    """Nobody bought the towers. A fresh board is worth nothing to either side."""
    env = attach()
    assert env._army_value(0) == 0.0
    assert env._army_value(1) == 0.0


# ------------------------------------------------------------------ the potential

def test_playing_a_card_moves_value_but_does_not_create_it():
    """The whole point: elixir goes from the bank to the board and the edge is unchanged.

    If this failed the agent could farm the shaping by dumping cards, which is exactly
    the behaviour we are trying to move it away from.
    """
    env = attach()
    before = env._elixir_edge()
    assert env.battle.deploy_card(0, "Knight", Position(3.5, 10.0))
    assert env._elixir_edge() == pytest.approx(before)


def settle(env, seconds=2.0):
    """Cards do not become bodies until their deploy time has elapsed."""
    for _ in range(int(seconds * 60)):
        env.battle.step(1 / 60)


def test_a_full_health_army_is_worth_what_it_cost():
    """Three bodies at full health add back up to the one price that was paid."""
    env = attach(deck=deck_with("Minions"))
    assert env.battle.deploy_card(0, "Minions", Position(3.5, 10.0))
    settle(env)
    assert env._army_value(0) == pytest.approx(Card("Minions").elixir)


def damaged_knight(env, health):
    knight = next(e for e in env.battle.entities.values()
                  if e.player == 0 and e.name == "Knight" and e.is_alive)
    knight.hp = knight.data.hp * health
    return knight


def test_a_damaged_unit_is_worth_less():
    """Value falls with health, but only across the share that damage can claim."""
    env = attach(deck=deck_with("Knight"))
    assert env.battle.deploy_card(0, "Knight", Position(3.5, 10.0))
    settle(env)
    damaged_knight(env, 0.25)
    expected = Card("Knight").elixir * (KILL_SHARE + (1 - KILL_SHARE) * 0.25)
    assert env._army_value(0) == pytest.approx(expected)


def test_chip_damage_can_never_claim_more_than_half_a_unit():
    """A Giant on one hit point still deals full damage. Paying out its whole price for
    damage alone would reward harassing a push rather than killing it."""
    env = attach(deck=deck_with("Knight"))
    assert env.battle.deploy_card(0, "Knight", Position(3.5, 10.0))
    settle(env)
    full = env._army_value(0)
    damaged_knight(env, 0.001)
    assert env._army_value(0) == pytest.approx(full * KILL_SHARE, rel=1e-2)


def test_the_kill_is_worth_more_than_every_hit_that_led_to_it():
    """The whole point of the split: finishing it beats softening it."""
    env = attach(deck=deck_with("Knight"))
    assert env.battle.deploy_card(0, "Knight", Position(3.5, 10.0))
    settle(env)
    full = env._army_value(0)
    knight = damaged_knight(env, 0.001)
    from_chipping = full - env._army_value(0)
    before_kill = env._army_value(0)
    knight.hp = 0
    knight.is_alive = False
    from_the_kill = before_kill - env._army_value(0)
    assert from_the_kill > from_chipping


def test_a_dead_unit_is_worth_nothing():
    env = attach(deck=deck_with("Knight"))
    assert env.battle.deploy_card(0, "Knight", Position(3.5, 10.0))
    settle(env)
    knight = next(e for e in env.battle.entities.values()
                  if e.player == 0 and e.name == "Knight" and e.is_alive)
    knight.hp = 0
    knight.is_alive = False
    assert env._army_value(0) == 0.0


def test_the_edge_is_measured_from_the_learner():
    """Pinning the other side must flip the sign, not keep the blue-eyed view."""
    env = attach(deck=deck_with("Knight"))
    assert env.battle.deploy_card(0, "Knight", Position(3.5, 10.0))
    settle(env)
    env.battle.players[0].elixir = 0.0     # blue converted its bank into a body
    env.learner = 0
    mine = env._elixir_edge()
    env.learner = 1
    assert env._elixir_edge() == pytest.approx(-mine)


def test_killing_their_unit_is_worth_what_it_cost_them():
    env = attach(deck=deck_with("Knight"))
    assert env.battle.deploy_card(1, "Knight", Position(3.5, 22.0))
    settle(env)
    before = env._elixir_edge()
    knight = next(e for e in env.battle.entities.values()
                  if e.player == 1 and e.name == "Knight" and e.is_alive)
    knight.hp = 0
    knight.is_alive = False
    assert env._elixir_edge() - before == pytest.approx(Card("Knight").elixir)


# ------------------------------------------------------------------ in the step reward

def test_the_term_is_off_by_default():
    env = CREnv(opponent_model=random_strategy)
    assert env.elixir_scale == 0.0


def test_being_at_the_elixir_cap_counts_against_you():
    """A player sitting at 10 regenerates nothing while the other keeps earning.

    This falls out of the potential rather than being coded, and it is the right sign:
    elixir you cannot store is elixir you are throwing away.
    """
    env = attach(elixir=10.0, scale=1.0)
    env.opponent = lambda obs: (0, 0, 0)
    env.battle.players[1].elixir = 5.0
    before = env._elixir_edge()
    env.step((0, 0, 0))
    assert env._elixir_edge() < before


def test_a_trade_in_our_favour_pays():
    """Red loses a unit, no tower is touched, and the step reward is positive.

    Both banks start away from the cap so the two sides regenerate at the same rate and
    cancel; the only thing that moves is the Knight dying.
    """
    env = attach(elixir=5.0, scale=1.0, deck=deck_with("Knight"))
    env.opponent = lambda obs: (0, 0, 0)
    assert env.battle.deploy_card(1, "Knight", Position(3.5, 22.0))
    settle(env)
    red = next(e for e in env.battle.entities.values()
               if e.player == 1 and e.name == "Knight" and e.is_alive)
    # Walk it into blue's tower range on one hit point, so it dies inside this step.
    red.position = Position(3.5, 9.0)
    red.hp = 1
    env.battle.players[0].elixir = env.battle.players[1].elixir = 5.0

    def towers():
        return sum(env.battle.players[p].king_tower_hp
                   + env.battle.players[p].left_tower_hp
                   + env.battle.players[p].right_tower_hp for p in (0, 1))

    before = towers()
    _, reward, _, _, _ = env.step((0, 0, 0))
    assert towers() == before, "a tower moved; this no longer isolates the term"
    assert not [e for e in env.battle.entities.values()
                if e.player == 1 and e.name == "Knight" and e.is_alive], "it survived"
    assert reward > 0


def test_scaling_the_term_scales_only_that_term():
    """Two identical games, different weights: the gap is exactly the shaping."""
    def play(scale):
        # `make_rusher` carries its own Random(seed); a strategy drawing from the global
        # module would make the two runs different games and the comparison meaningless.
        env = CREnv(opponent_model=make_rusher(0), elixir_scale=scale, learner_player=0)
        env.reset(seed=7)
        total = 0.0
        for _ in range(40):
            _, r, done, _, _ = env.step((0, 0, 0))
            total += r
            if done:
                break
        return total

    zero, one = play(0.0), play(1.0)
    assert zero != one
