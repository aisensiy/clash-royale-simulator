"""The ruler currently held out of training, and the reason it is a third script.

`scripts_defender` ranks cards by hit points per elixir and `scripts_sniper` by price.
Neither reads what the thing coming at it actually is, so both answer a flying swarm and
a lone Giant with the same card. This one picks by matchup, which is the axis the other
two share no part of -- and that is what makes a rating against it independent of them.
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
from scripts_counter import CENTRE, FIRST_ROW, LAST_ROW, make_counter
from scripts_defender import make_defender
from scripts_sniper import make_sniper


def deck_with(*first):
    return list(first) + [c for c in DECK if c not in first]


def observation(elixir=10.0, hand=("Musketeer", "Knight", "MiniPekka", "Arrows"),
                enemy_at=None, learner=0, settle=400):
    """A board with the enemy units placed by hand, seen from `learner`'s own frame.

    `settle` defaults high: a defence case needs the enemy to have actually crossed the
    river, which takes several seconds of walking, not the two the deploy timers need.
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
    for _ in range(settle):
        env.battle.step(1 / 60)
    env.battle.players[learner].elixir = elixir
    return env.observe(learner)


def played(action):
    return int(action[0]), int(action[1]), int(action[2])


def name_of(obs, slot):
    return entity_names[int(obs["hand"][slot - 1])]


def far(col):
    return Position(col + 0.5, 19.0)


# --------------------------------------------------------------------- matchup

def test_air_is_answered_by_something_that_can_shoot_air():
    """Minions fly. A Knight in front of them is 3 elixir thrown away, however good its
    hit points per elixir look."""
    obs = observation(elixir=10.0, hand=("Knight", "Musketeer", "MiniPekka", "Giant"),
                      enemy_at=[("Minions", far(3))])
    slot, _, _ = played(make_counter()(obs))
    assert slot != 0, "a flying push went unanswered"
    assert Card(name_of(obs, slot)).attack_air, f"answered Minions with {name_of(obs, slot)}"


def test_a_ground_tank_gets_the_fastest_killer_in_hand():
    """Not the cheapest and not the beefiest: the one that removes it soonest."""
    obs = observation(elixir=10.0, hand=("Knight", "MiniPekka", "Archer", "Giant"),
                      enemy_at=[("Giant", far(3))])
    slot, _, _ = played(make_counter()(obs))
    picked = name_of(obs, slot)
    fastest = max(("Knight", "MiniPekka", "Archer", "Giant"),
                  key=lambda n: Card(n).damage / Card(n).hit_speed)
    assert picked == fastest, f"answered a Giant with {picked}"


def test_a_swarm_gets_a_spell():
    """Several bodies at once is what one splash is for."""
    obs = observation(elixir=10.0, hand=("Arrows", "Knight", "Musketeer", "MiniPekka"),
                      enemy_at=[("Minions", far(2)), ("Archer", far(4)), ("Knight", far(6))])
    slot, _, _ = played(make_counter()(obs))
    assert Card(name_of(obs, slot)).type == "spell", f"answered a swarm with {name_of(obs, slot)}"


def test_the_grid_cannot_count_bodies_standing_on_one_cell():
    """Pins a limit of the observation, not of this script.

    `CREnv.observe` writes each unit into `obs[y][x]`, so units sharing a cell overwrite
    each other and only the last one survives. Three Minions from one card stand close
    enough to land on a single cell, and read back as one Minion -- which is why the swarm
    rule above counts occupied cells rather than bodies, and why it cannot fire on a stack.
    The count channels added in `test_count_channels.py` are what fixes this, but they are
    served only to checkpoints trained on them: the scripts stay on the 15-channel grid so
    that a rating this ruler gave last round still means the same thing."""
    obs = observation(elixir=10.0, enemy_at=[("Minions", far(3))])
    grid = obs["grid"]
    cells = np.nonzero((grid[:, :, 1] > 0.5) & (grid[:, :, 2] > 0))[0]
    assert len(cells) == 1, "the encoding changed -- the swarm rule can be sharpened"


def test_a_lone_cheap_body_is_left_to_the_tower():
    obs = observation(elixir=10.0, enemy_at=[], settle=120)
    assert played(make_counter()(obs))[0] == 0


# --------------------------------------------------------------------- placement

def test_a_ground_answer_is_pulled_toward_the_middle():
    """Standing in front of the tower lets the push keep walking at the tower. Placing
    toward the centre turns it away, which is the point of the placement rule."""
    obs = observation(elixir=10.0, hand=("Knight", "MiniPekka", "Archer", "Giant"),
                      enemy_at=[("Giant", far(3))])
    _, row, col = played(make_counter()(obs))
    assert FIRST_ROW <= row <= LAST_ROW
    assert 3 < col <= CENTRE, f"placed at column {col}, no closer to the middle than the push"


def test_it_attacks_only_off_a_defence_that_left_units_standing():
    """No bank threshold anywhere in it: with a full bank and an empty board it passes."""
    assert played(make_counter()(observation(elixir=10.0, settle=120)))[0] == 0


# --------------------------------------------------------------------- independence

@pytest.mark.parametrize("other", [make_defender, make_sniper])
def test_it_disagrees_with_both_of_the_other_scripts(other):
    """A ruler that makes the same decisions as a training opponent measures drill."""
    obs = observation(elixir=8.0, hand=("Musketeer", "Knight", "MiniPekka", "Giant"),
                      enemy_at=[("Minions", far(3))])
    assert played(make_counter()(obs)) != played(other()(obs))


def test_it_is_not_in_the_training_pool():
    """The invariant that `sniper` used to carry, moved here when sniper was promoted."""
    import train
    source = open(train.__file__).read()
    assert '"counter"' not in source, "counter reached train.py; the held-out rating is gone"


# --------------------------------------------------------------------- plumbing

def test_it_only_ever_names_a_card_it_can_pay_for():
    act = make_counter()
    for elixir in (0.0, 1.0, 2.5, 4.0, 7.0, 10.0):
        for enemy in ([], [("Knight", far(3))], [("Giant", far(3)), ("Minions", far(4))]):
            obs = observation(elixir=elixir, enemy_at=enemy)
            slot, _, _ = played(act(obs))
            if slot == 0:
                continue
            assert Card(name_of(obs, slot)).elixir <= elixir


def test_bodies_stay_inside_the_legal_band():
    act = make_counter()
    for elixir in (4.0, 7.0, 10.0):
        for enemy in ([], [("Knight", far(3))], [("Giant", far(14)), ("Musketeer", far(13))]):
            obs = observation(elixir=elixir, hand=("Knight", "Musketeer", "MiniPekka", "Giant"),
                              enemy_at=enemy)
            slot, row, col = played(act(obs))
            if slot == 0:
                continue
            assert FIRST_ROW <= row <= LAST_ROW, f"row {row}"
            assert 0 <= col < ARENA_W


def test_it_survives_a_whole_game():
    from agents import load_agent
    env = CREnv(opponent_model=load_agent("counter"), learner_player=0)
    env.reset(seed=3)
    done, steps = False, 0
    while not done and steps < 700:
        _, _, done, _, _ = env.step((0, 0, 0))
        steps += 1
    assert done, "the game never ended"
