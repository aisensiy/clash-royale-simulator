"""The three reasons a Clash Royale player saves elixir, pinned as properties.

Each of these is something a human takes for granted and the simulator has to actually
implement, because the whole case for teaching an agent to hold elixir rests on them:

  - a tank in front of support is worth more than the two cards played apart
  - a fight held inside your own tower's range costs you less
  - an attacker crossing the bridge spends seconds being shot before it can hurt anything

They are also the properties most likely to be broken silently by a change to pathing,
targeting or tower placement, which is why they are assertions and not a report.

The last two tests are about the reward rather than the simulator: a perfect defence --
the best play available in the position -- is worth exactly 0.000 under tower HP alone.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import battle
import player
from card_utils import Card
from core import Position
from environment import CREnv, DECK, UNIT_VALUE, random_strategy

BLUE_TOWER_Y = 6.5
TOWER_RANGE = 7.5


def deck_with(*first):
    """Only the first four cards of a cycle are in hand."""
    return list(first) + [c for c in DECK if c not in first]


def fresh(blue_hand=(), red_hand=(), elixir=10.0):
    return battle.BattleState(player.PlayerState(0, deck_with(*blue_hand), elixir),
                              player.PlayerState(1, deck_with(*red_hand), elixir))


def run(b, seconds):
    for _ in range(int(seconds * 60)):
        b.step(1 / 60)


def alive(b, name, pid):
    return [e for e in b.entities.values()
            if e.is_alive and e.player == pid and e.name == name and e.id > 6]


def tower_hp(b, pid):
    p = b.players[pid]
    return p.left_tower_hp + p.right_tower_hp + p.king_tower_hp


# ------------------------------------------------------- a tank in front of support

def damage_from_push(cards, seconds=40.0):
    """Blue attacks red's left tower with `cards`, played back to front half a second
    apart, and we count what red's towers lose."""
    b = fresh(blue_hand=tuple(name for name, _ in cards))
    for i, (name, pos) in enumerate(cards):
        b.players[0].elixir = 10.0     # isolate the mechanic from the elixir budget
        assert b.deploy_card(0, name, pos), f"{name} refused"
        if i + 1 < len(cards):
            run(b, 0.5)
    before = tower_hp(b, 1)
    run(b, seconds)
    return before - tower_hp(b, 1)


TANK_POS = Position(3.5, 12.0)
SUPPORT_POS = Position(3.5, 10.5)


def test_a_tank_in_front_beats_the_two_cards_played_apart():
    """The reason 6 elixir held is worth more than two lots of 3 spent as they arrive.

    Red's tower shoots whatever is nearest, so a Knight in front keeps the Archer alive
    long enough to matter. Measured at roughly 2x when this was written; the assertion
    is deliberately loose, since the claim is that it is superlinear at all.
    """
    together = damage_from_push([("Knight", TANK_POS), ("Archer", SUPPORT_POS)])
    apart = (damage_from_push([("Knight", TANK_POS)])
             + damage_from_push([("Archer", SUPPORT_POS)]))
    assert together > 1.5 * apart, f"{together} vs {apart} apart"


def test_the_best_value_push_costs_more_than_the_agent_has_ever_held():
    """A standing measurement, not a rule: the elixir probe has never seen either arm
    above 7, and the strongest push in this deck costs 9."""
    assert Card("Giant").elixir + Card("Musketeer").elixir == 9


# ------------------------------------------------------- fighting near your own tower

def contested_fight(fight_y, attacker="Musketeer", defender="Knight"):
    """Hold the same fight at a chosen distance from blue's own tower.

    Red may only place cards in its own half, so the attacker is dropped at the back and
    the defence is held until it has walked down to where the fight should happen.
    """
    b = fresh(blue_hand=(defender,), red_hand=(attacker,))
    assert b.deploy_card(1, attacker, Position(3.5, 18.0))
    waited = 0.0
    while waited < 40.0:
        units = alive(b, attacker, 1)
        if units and units[0].position.y <= fight_y + 3.0:
            break
        b.step(1 / 60)
        waited += 1 / 60
    assert b.deploy_card(0, defender, Position(3.5, fight_y))
    elapsed = 0.0
    while alive(b, attacker, 1) and alive(b, defender, 0) and elapsed < 40.0:
        b.step(1 / 60)
        elapsed += 1 / 60
    survivor = sum(u.hp for u in alive(b, defender, 0))
    return survivor / Card(defender).hp


def test_defending_close_to_your_tower_keeps_the_defender_alive():
    """The tower's damage is added to yours, so the same trade costs you less.

    This is the mechanic behind an elixir advantage: the survivor is the counter-push.
    """
    near = contested_fight(9.0)
    far = contested_fight(BLUE_TOWER_Y + TOWER_RANGE + 0.5)
    assert near > far + 0.25, f"near {near:.0%} vs far {far:.0%}"


# ------------------------------------------------------- the walk from the bridge

def test_an_attacker_is_shot_for_seconds_before_it_can_hurt_you():
    """Crossing the bridge is not arriving. The gap is the window a defender adds
    ranged support into, and it has to be long enough to be worth playing into."""
    b = fresh(red_hand=("Giant",))
    assert b.deploy_card(1, "Giant", Position(3.5, 17.5))
    start_hp = Card("Giant").hp
    before = tower_hp(b, 0)
    seconds = 0.0
    while tower_hp(b, 0) == before and alive(b, "Giant", 1) and seconds < 60.0:
        b.step(1 / 60)
        seconds += 1 / 60
    units = alive(b, "Giant", 1)
    chipped = (start_hp - (units[0].hp if units else 0)) / start_hp
    assert seconds > 5.0, f"only {seconds:.1f}s"
    assert chipped > 0.15, f"only {chipped:.0%} of it gone"


# ------------------------------------------------------- what the reward pays for it

DMG_SCALE = 0.25


def defend_or_race(mode, elixir_scale=0.0, seconds=45.0):
    """Red commits a 9 elixir push; blue spends 6 either in front of it or on the other
    lane. Nobody is topped up, so the banks stay honest and the edge means something."""
    env = CREnv(opponent_model=lambda obs: (0, 0, 0), learner_player=0,
                dmg_scale=DMG_SCALE, elixir_scale=elixir_scale)
    env.battle = fresh(blue_hand=("Knight", "Archer"), red_hand=("Giant", "Musketeer"))
    env.learner = 0
    b = env.battle

    assert b.deploy_card(1, "Giant", Position(3.5, 17.5))
    run(b, 0.5)
    assert b.deploy_card(1, "Musketeer", Position(3.5, 19.0)), "red could not afford it"
    run(b, 4.0)

    lane_x = 3.5 if mode == "defend" else 14.5
    tank_y, support_y = (11.0, 9.0) if mode == "defend" else (12.0, 10.5)
    assert b.deploy_card(0, "Knight", Position(lane_x, tank_y))
    run(b, 0.5)
    assert b.deploy_card(0, "Archer", Position(lane_x, support_y)), "blue could not afford it"

    blue_before, red_before = tower_hp(b, 0), tower_hp(b, 1)
    crowns_before = (b.players[1].get_crown_count(), b.players[0].get_crown_count())
    edge_before = env._elixir_edge()
    run(b, seconds)

    blue_lost = blue_before - tower_hp(b, 0)
    red_lost = red_before - tower_hp(b, 1)
    taken = b.players[1].get_crown_count() - crowns_before[0]
    given = b.players[0].get_crown_count() - crowns_before[1]
    reward = (5 * taken - 5 * given
              + DMG_SCALE * (0.001 * red_lost - 0.0012 * blue_lost)
              + elixir_scale * (env._elixir_edge() - edge_before))
    return {"reward": reward, "blue tower lost": blue_lost}


def test_today_a_perfect_defence_is_worth_exactly_nothing():
    """Blue kills a 9 elixir push with 6 and loses no tower HP. Tower HP is all the
    reward can see, so the best play in the position scores the same as an empty board."""
    defend = defend_or_race("defend")
    assert defend["blue tower lost"] == 0, "the push got through; this is not the case"
    assert defend["reward"] == pytest.approx(0.0)


def test_the_elixir_term_pays_for_the_defence_that_tower_hp_cannot_see():
    """The same position, with the shaping on: defending is now worth something."""
    assert defend_or_race("defend", elixir_scale=0.05)["reward"] > 0


def test_ignoring_the_push_is_punished_either_way():
    """A sanity check on the sign: the shaping must not make racing look good."""
    for scale in (0.0, 0.05):
        assert defend_or_race("race", elixir_scale=scale)["reward"] < 0
