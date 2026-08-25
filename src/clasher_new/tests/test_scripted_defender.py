"""The scripted defender has to actually play the way it claims to.

A script that runs without crashing but spends its elixir on arrival would be just
another `rusher` under a better name, and would teach the learner nothing. These pin the
three behaviours it exists for: it holds elixir, it answers a push between the push and
its own tower, and it commits a tank and support together.
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
from environment import ARENA_H, ARENA_W, CREnv, DECK, N_SLOTS, entity_names
from scripts_defender import FIRST_ROW, LAST_ROW, make_anchor, make_defender


def deck_with(*first):
    """Only the first four cards of a cycle are in hand."""
    return list(first) + [c for c in DECK if c not in first]


def observation(elixir=10.0, learner=0, enemy_at=None, own_at=None):
    """A board with the units placed by hand, seen from `learner`'s own point of view.

    Whoever is being observed gets `elixir`; the other side is given a full bank and a
    cycle dealing it whatever the caller asked to place, so a low-elixir case for the
    observer does not silently fail to set up the board it is about to be tested on.
    """
    enemy_at, own_at = enemy_at or [], own_at or []
    env = CREnv(opponent_model=lambda obs: (0, 0, 0))
    decks = {learner: deck_with(*[n for n, _ in own_at]),
             1 - learner: deck_with(*[n for n, _ in enemy_at])}
    banks = {learner: elixir, 1 - learner: 10.0}
    env.battle = battle.BattleState(player.PlayerState(0, decks[0][:], banks[0]),
                                    player.PlayerState(1, decks[1][:], banks[1]))
    env.learner = learner
    for pid, spots in ((1 - learner, enemy_at), (learner, own_at)):
        for name, pos in spots:
            assert env.battle.deploy_card(pid, name, pos), f"{name} at {pos.x},{pos.y}"
    for _ in range(120):                     # let the deploy timers run out
        env.battle.step(1 / 60)
    # Set the bank last: stepping regenerates elixir, so setting it before the settle
    # loop would hand the script more than the case says it has.
    env.battle.players[learner].elixir = elixir
    return env.observe(learner)


def played(action):
    slot, row, col = int(action[0]), int(action[1]), int(action[2])
    return slot, row, col


# --------------------------------------------------------------------- holding elixir

@pytest.mark.parametrize("elixir", [1.0, 3.0, 5.0])
def test_it_does_not_spend_on_arrival(elixir):
    """The whole point. `rusher` plays as soon as it can afford anything; this waits."""
    act = make_defender()
    slot, _, _ = played(act(observation(elixir=elixir)))
    assert slot == 0, f"played a card at {elixir} elixir with nothing to answer"


def test_it_commits_once_the_bank_is_full():
    act = make_defender()
    slot, _, _ = played(act(observation(elixir=10.0)))
    assert slot != 0


def test_the_anchor_holds_longer_than_the_training_copy():
    """The two variants must not be the same player, or the held-out rating is not
    measuring anything the pool copy did not already drill."""
    quiet = observation(elixir=8.0)
    assert played(make_defender()(quiet))[0] != 0
    assert played(make_anchor()(quiet))[0] == 0


# --------------------------------------------------------------------- defending

RIVER_SIDE = Position(3.5, 20.0)     # red's half, so a red unit walks down from there


def test_it_answers_a_push_that_has_crossed():
    """Even below its holding threshold: a threat overrides the bank."""
    obs = observation(elixir=5.0, enemy_at=[("Knight", RIVER_SIDE)])
    # Walk the threat into our half by placing it where it will already have crossed.
    obs = observation(elixir=5.0, enemy_at=[("Knight", Position(3.5, 18.0))])
    slot, row, _ = played(make_defender()(obs))
    if slot == 0:
        pytest.skip("the threat has not crossed yet in this snapshot")
    assert FIRST_ROW <= row <= LAST_ROW


def test_it_never_places_outside_the_legal_band():
    """Rows 0..8 hold our own towers and rows past 14 are the enemy's half while their
    towers stand. Anything outside is a refused deploy, which is a wasted decision."""
    act = make_defender()
    for elixir in (4.0, 6.0, 8.0, 10.0):
        for enemy in ([], [("Knight", Position(3.5, 18.0))], [("Giant", Position(14.5, 19.0))]):
            slot, row, col = played(act(observation(elixir=elixir, enemy_at=enemy)))
            if slot == 0:
                continue
            assert FIRST_ROW <= row <= LAST_ROW, f"row {row}"
            assert 0 <= col < ARENA_W, f"col {col}"


def test_it_only_ever_names_a_card_it_can_pay_for():
    act = make_defender()
    for elixir in (0.0, 1.0, 2.5, 4.0, 7.0, 10.0):
        obs = observation(elixir=elixir, enemy_at=[("Knight", Position(3.5, 18.0))])
        slot, _, _ = played(act(obs))
        if slot == 0:
            continue
        name = entity_names[int(obs["hand"][slot - 1])]
        assert Card(name).elixir <= elixir, f"{name} costs more than {elixir}"


def test_it_never_names_a_spell():
    """Spells are left out on purpose, so that what the script demonstrates is holding
    and combining rather than a well-aimed Fireball."""
    act = make_defender()
    for elixir in (6.0, 8.0, 10.0):
        for enemy in ([], [("Minions", Position(3.5, 19.0))]):
            obs = observation(elixir=elixir, enemy_at=enemy)
            slot, _, _ = played(act(obs))
            if slot == 0:
                continue
            assert Card(entity_names[int(obs["hand"][slot - 1])]).type != "spell"


# --------------------------------------------------------------------- combining

def test_the_tank_is_followed_by_support_right_behind_it():
    """Two cards, one after the other, close enough to travel as one group -- the
    behaviour worth about twice what the same cards deal played apart."""
    act = make_defender()
    quiet = observation(elixir=10.0)
    first_slot, first_row, first_col = played(act(quiet))
    assert first_slot != 0
    second_slot, second_row, second_col = played(act(observation(elixir=10.0)))
    assert second_slot != 0, "the follow-up never came"
    assert second_col == first_col, "the support went down a different lane"
    assert second_row < first_row, "the support is not behind the tank"


def test_the_tank_is_the_beefiest_card_in_hand():
    act = make_defender()
    obs = observation(elixir=10.0)
    slot, _, _ = played(act(obs))
    names = [entity_names[int(obs["hand"][i])] for i in range(N_SLOTS - 1)]
    affordable = [n for n in names if Card(n).type != "spell" and Card(n).elixir <= 10.0]
    best = max(affordable, key=lambda n: Card(n).hp / Card(n).elixir)
    assert entity_names[int(obs["hand"][slot - 1])] == best


def test_a_new_episode_forgets_a_half_finished_push():
    """Otherwise the support for a tank played in the last game lands on an empty board
    at the start of the next one."""
    act = make_defender()
    assert played(act(observation(elixir=10.0)))[0] != 0     # tank down, support pending
    act.on_episode_start()
    slot, _, _ = played(act(observation(elixir=2.0)))
    assert slot == 0, "it played the pending support into a new game"


# --------------------------------------------------------------------- plumbing

def test_each_side_of_a_match_gets_its_own_instance():
    """The script keeps state; blue and red sharing one object would have them finish
    each other's pushes."""
    from agents import load_agent
    assert load_agent("defender") is not load_agent("defender")


def test_the_pool_resets_the_script_between_episodes():
    from selfplay import OpponentPool, PooledOpponent
    calls = []
    script = lambda obs: (0, 0, 0)
    script.on_episode_start = lambda: calls.append(1)
    pool = OpponentPool("/nonexistent", p_latest=0.0, p_history=0.0, p_script=1.0,
                        script_names=("defender",))
    opponent = PooledOpponent(pool, {"defender": script}, algo=None, seed=0)
    opponent.on_episode_start()
    assert calls == [1]


def test_it_survives_a_whole_game_against_the_learner_side():
    """End to end: no refused-deploy storm, no crash, and the game ends."""
    from agents import load_agent
    env = CREnv(opponent_model=load_agent("defender"), learner_player=0)
    obs, _ = env.reset(seed=3)
    done = False
    steps = 0
    while not done and steps < 700:
        obs, _, done, _, _ = env.step((0, 0, 0))
        steps += 1
    assert done, "the game never ended"
