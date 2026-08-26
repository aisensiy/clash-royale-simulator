"""The held-out ruler has to be a different player, not the defender in a hat.

Its whole job is to measure a checkpoint against play the checkpoint was never drilled
on. If it makes the same decisions as `scripts_defender`, a rating against it says the
same thing the anchor already said, and both are contaminated by what sits in the
training pool. So these tests pin two separate things: that it plays the way its
docstring claims, and that on the same board it does not agree with the defender.
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
from environment import ARENA_W, CREnv, DECK, N_SLOTS, entity_names
from scripts_defender import FIRST_ROW, LAST_ROW, make_defender
from scripts_sniper import BRIDGE_ROW, LANES, POST_ROW, make_sniper


def deck_with(*first):
    """Only the first four cards of a cycle are in hand."""
    return list(first) + [c for c in DECK if c not in first]


def observation(elixir=10.0, hand=("Fireball", "Knight", "Archer", "Musketeer"),
                enemy_at=None, learner=0, settle=120):
    """A board with the enemy units placed by hand, seen from `learner`'s own frame.

    Unlike the defender's helper this one takes the observer's hand explicitly: a script
    that leads with spells cannot be tested on a hand that happens not to hold one.
    """
    enemy_at = enemy_at or []
    env = CREnv(opponent_model=lambda obs: (0, 0, 0))
    decks = {learner: deck_with(*hand),
             1 - learner: deck_with(*[n for n, _ in enemy_at])}
    env.battle = battle.BattleState(player.PlayerState(0, decks[0][:], 10.0),
                                    player.PlayerState(1, decks[1][:], 10.0))
    env.learner = learner
    for name, pos in enemy_at:
        assert env.battle.deploy_card(1 - learner, name, pos), f"{name} at {pos.x},{pos.y}"
    # `settle` is how long the placed units are given to walk. Two seconds is enough to
    # clear the deploy timers; a defence case needs longer, because a unit dropped on the
    # far bank has to actually reach our side of the river before there is anything to
    # answer.
    for _ in range(settle):
        env.battle.step(1 / 60)
    # After the settle loop, or regeneration would hand the script more than it should have.
    env.battle.players[learner].elixir = elixir
    return env.observe(learner)


def played(action):
    return int(action[0]), int(action[1]), int(action[2])


def name_of(obs, slot):
    return entity_names[int(obs["hand"][slot - 1])]


# Deep in red's half, so a red unit is still walking down towards us.
def far(col):
    return Position(col + 0.5, 19.0)


# --------------------------------------------------------------------- spells first

def test_it_fireballs_a_clump():
    """Three bodies standing together are worth more than the spell that kills them."""
    obs = observation(enemy_at=[("Giant", far(3)), ("Musketeer", Position(4.5, 20.0))])
    slot, row, col = played(make_sniper()(obs))
    assert slot != 0, "left a 9 elixir clump alone"
    assert Card(name_of(obs, slot)).type == "spell", f"answered with {name_of(obs, slot)}"
    assert abs(col - 4) <= 3 and row > LAST_ROW, f"cast at ({row},{col}), nowhere near it"


def test_it_does_not_spend_a_spell_on_one_cheap_body():
    """A 4 elixir Fireball on a 3 elixir Knight is a losing trade, and the script has to
    know that or it is just `rusher` with a splash."""
    obs = observation(elixir=6.0, enemy_at=[("Knight", far(3))])
    slot, _, _ = played(make_sniper()(obs))
    if slot != 0:
        assert Card(name_of(obs, slot)).type != "spell"


def test_it_casts_across_the_river_too():
    """Spells reach the whole arena, and a clump massing on their side is just as dead.
    The defender can never do this: it only ever plays in rows 9..14."""
    obs = observation(enemy_at=[("Giant", Position(3.5, 25.0)),
                                ("Musketeer", Position(4.5, 25.0))])
    slot, row, _ = played(make_sniper()(obs))
    assert slot != 0 and Card(name_of(obs, slot)).type == "spell"
    assert row > LAST_ROW, "a spell that only ever lands in our own half is not a spell"


# --------------------------------------------------------------------- defending

def test_it_defends_on_a_fixed_post_with_the_cheapest_body():
    """No spell in hand, so it has to put a body down -- the cheapest one, on the post."""
    obs = observation(elixir=6.0, hand=("Knight", "Musketeer", "Giant", "MiniPekka"),
                      enemy_at=[("Giant", far(3)), ("Musketeer", Position(4.5, 19.0))],
                      settle=400)
    slot, row, col = played(make_sniper()(obs))
    assert slot != 0, "a 9 elixir push went unanswered"
    affordable = [name_of(obs, s) for s in range(1, N_SLOTS)
                  if Card(name_of(obs, s)).elixir <= 6.0]
    assert name_of(obs, slot) == min(affordable, key=lambda n: Card(n).elixir)
    assert (row, col) == (POST_ROW, LANES[0]), f"defended at ({row},{col})"


def test_the_post_follows_the_lane_the_push_is_in():
    obs = observation(elixir=6.0, hand=("Knight", "Musketeer", "Giant", "MiniPekka"),
                      enemy_at=[("Giant", far(14)), ("Musketeer", Position(13.5, 19.0))],
                      settle=400)
    slot, _, col = played(make_sniper()(obs))
    assert slot != 0
    assert col == LANES[1]


def test_an_empty_board_and_a_low_bank_is_a_pass():
    assert played(make_sniper()(observation(elixir=5.0)))[0] == 0


# --------------------------------------------------------------------- chipping

def test_it_chips_with_a_single_card_out_of_a_full_bank():
    obs = observation(elixir=10.0, hand=("Knight", "Musketeer", "Giant", "MiniPekka"))
    slot, row, _ = played(make_sniper()(obs))
    assert slot != 0 and row == BRIDGE_ROW


def test_it_never_builds_a_push():
    """Stateless on purpose: the same board twice gives the same card in the same place,
    never a tank followed by its support. Whatever this ruler rewards, it is not the
    habit the pool defender was put there to teach."""
    act = make_sniper()
    board = dict(elixir=10.0, hand=("Knight", "Musketeer", "Giant", "MiniPekka"))
    assert played(act(observation(**board))) == played(act(observation(**board)))


# --------------------------------------------------------------------- independence

def test_it_disagrees_with_the_defender_on_the_same_board():
    """The reason it exists. Same hand, same board, different decision."""
    board = dict(elixir=8.0, hand=("Fireball", "Knight", "Archer", "Musketeer"),
                 enemy_at=[("Giant", far(3)), ("Musketeer", Position(4.5, 20.0))])
    obs = observation(**board)
    assert played(make_sniper()(obs)) != played(make_defender()(obs))


def test_it_is_available_as_a_training_opponent():
    """It started out held out of training, and stopped being a ruler the moment it was
    promoted: it beats every checkpoint measured, which makes it worth far more as a
    teacher than as a yardstick. `scripts_counter` took over as the held-out one."""
    import train
    source = open(train.__file__).read()
    assert '"sniper": make_sniper' in source


# --------------------------------------------------------------------- plumbing

def test_it_only_ever_names_a_card_it_can_pay_for():
    act = make_sniper()
    for elixir in (0.0, 1.0, 2.5, 4.0, 7.0, 10.0):
        for enemy in ([], [("Knight", far(3))], [("Giant", far(3)), ("Musketeer", far(4))]):
            obs = observation(elixir=elixir, enemy_at=enemy)
            slot, _, _ = played(act(obs))
            if slot == 0:
                continue
            assert Card(name_of(obs, slot)).elixir <= elixir


def test_bodies_stay_inside_the_legal_band():
    """Spells may go anywhere; a body outside rows 9..14 is a refused deploy."""
    act = make_sniper()
    for elixir in (4.0, 6.0, 8.0, 10.0):
        for enemy in ([], [("Knight", far(3))], [("Giant", far(14)), ("Musketeer", far(13))]):
            obs = observation(elixir=elixir, hand=("Knight", "Musketeer", "Giant", "MiniPekka"),
                              enemy_at=enemy)
            slot, row, col = played(act(obs))
            if slot == 0:
                continue
            assert FIRST_ROW <= row <= LAST_ROW, f"row {row}"
            assert 0 <= col < ARENA_W


def test_it_survives_a_whole_game():
    from agents import load_agent
    env = CREnv(opponent_model=load_agent("sniper"), learner_player=0)
    env.reset(seed=3)
    done, steps = False, 0
    while not done and steps < 700:
        _, _, done, _, _ = env.step((0, 0, 0))
        steps += 1
    assert done, "the game never ended"
