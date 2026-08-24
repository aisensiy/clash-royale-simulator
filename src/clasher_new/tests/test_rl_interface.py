"""Regression tests for the RL-facing parts of the simulator.

Run from `src/clasher_new`:  python3 -m pytest tests/ -q
"""
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import battle
import player
from card_utils import Card
from core import Position
from environment import ARENA_H, ARENA_W, CREnv, DECK, N_SLOTS, entity_names, random_strategy


def make_battle(elixir=10.0):
    return battle.BattleState(player.PlayerState(0, DECK[:], elixir),
                              player.PlayerState(1, DECK[:], elixir))


def attach(elixir=10.0):
    env = CREnv(opponent_model=random_strategy)
    env.battle = make_battle(elixir)
    return env


# --------------------------------------------------------------------------- observation

def test_enemy_units_are_not_mirrored_into_own_half():
    """A red unit sitting in red's half must appear in the far rows of blue's view."""
    env = attach()
    assert env.battle.deploy_card(1, 'Knight', Position(5.5, 20.5))
    for _ in range(60):
        env.battle.step(1 / 60)

    knight = next(e for e in env.battle.entities.values()
                  if e.name == 'Knight' and e.is_alive)
    grid = env.observe(0)['grid']
    kid = entity_names.index('Knight')
    cells = [(y, x) for y in range(ARENA_H) for x in range(ARENA_W)
             if int(grid[y][x][0]) == kid]

    assert cells == [(int(knight.position.y), int(knight.position.x))]
    assert cells[0][0] >= 16, "enemy unit in enemy half must land in the far rows"


def test_observation_is_egocentric_for_both_players():
    """Both players see themselves near row 0, and their own units flagged as `own`."""
    env = attach()
    assert env.battle.deploy_card(0, 'Knight', Position(8.5, 5.5))
    assert env.battle.deploy_card(1, 'MiniPekka', Position(9.5, 26.5))
    for _ in range(30):
        env.battle.step(1 / 60)

    for pid in (0, 1):
        grid = env.observe(pid)['grid']
        own = [(y, x) for y in range(ARENA_H) for x in range(ARENA_W)
               if grid[y][x][0] != 0 and grid[y][x][1] == 0]
        enemy = [(y, x) for y in range(ARENA_H) for x in range(ARENA_W)
                 if grid[y][x][0] != 0 and grid[y][x][1] == 1]
        assert own and enemy
        assert max(y for y, _ in own) < min(y for y, _ in enemy), \
            f"player {pid}: own units must sit below enemy units in its own frame"


# --------------------------------------------------------------------------- action masks

def split_mask(mask):
    return mask[:N_SLOTS], mask[N_SLOTS:N_SLOTS + ARENA_H], mask[N_SLOTS + ARENA_H:]


def test_mask_has_right_shape_and_never_empty():
    env = attach(elixir=0.0)
    mask = env.action_masks()
    assert mask.shape == (N_SLOTS + ARENA_H + ARENA_W,)
    slot, ys, xs = split_mask(mask)
    assert slot[0], "doing nothing must always stay available"
    assert ys.any() and xs.any(), "an all-false dimension would break the sampler"


def test_slot_mask_matches_affordability():
    for elixir in (0.0, 2.0, 3.0, 5.0, 10.0):
        env = attach(elixir=elixir)
        p0 = env.battle.players[0]
        slot, _, _ = split_mask(env.action_masks())
        for i in range(1, N_SLOTS):
            card = p0.cycle[i - 1]
            assert bool(slot[i]) == (elixir >= Card(card).elixir), \
                f"slot {i} ({card}, {Card(card).elixir} elixir) at {elixir} elixir"


def test_mask_never_forbids_a_legal_action():
    """False negatives are worse than false positives: they remove real options."""
    rng = random.Random(7)
    for game in range(3):
        env = attach(elixir=5.0)
        b = env.battle
        steps = 0
        while not b.game_over and steps < 200:
            slot_m, y_m, x_m = split_mask(env.action_masks())
            p0 = b.players[0]
            for slot in range(1, N_SLOTS):
                card = p0.cycle[slot - 1]
                if not p0.can_play_card(card):
                    continue
                assert slot_m[slot], f"{card} is playable but its slot was masked out"
                for y in range(ARENA_H):
                    for x in range(ARENA_W):
                        probe = battle.BattleState.deploy_card
                        legal = _would_deploy(b, card, y, x)
                        if legal:
                            assert y_m[y] and x_m[x], \
                                f"{card} is legal at ({y},{x}) but the mask forbids it"
            s, y, x = rng.randint(0, 4), rng.randint(0, ARENA_H - 1), rng.randint(0, ARENA_W - 1)
            if s:
                b.deploy_card(0, p0.cycle[s - 1], Position(x + 0.5, y + 0.5))
            for _ in range(30):
                if b.game_over:
                    break
                b.step(1 / 60)
            steps += 1


def _would_deploy(b, card_name, y, x):
    """Geometry half of `deploy_card`'s acceptance test, without mutating the battle."""
    pos = Position(x + 0.5, y + 0.5)
    if Card(card_name).type == 'spell':
        return True
    if b.is_position_occupied_by_building(pos, 0):
        return False
    if pos.y <= 1.0 and (pos.x <= 6.0 or pos.x > 12.0):
        return False
    if pos.y >= 21.0:
        return False
    if pos.y >= 15.0:
        tower = b.players[1].left_tower_hp if pos.x <= 9 else b.players[1].right_tower_hp
        return tower <= 0
    return True


def test_masking_removes_most_rejected_actions():
    """The point of the whole change: sampling under the mask should mostly succeed."""
    rng = random.Random(0)
    ok = attempts = 0
    for game in range(3):
        env = attach(elixir=5.0)
        b = env.battle
        steps = 0
        while not b.game_over and steps < 300:
            slot_m, y_m, x_m = split_mask(env.action_masks())
            slot = rng.choice([i for i in range(N_SLOTS) if slot_m[i]])
            y = rng.choice([i for i in range(ARENA_H) if y_m[i]])
            x = rng.choice([i for i in range(ARENA_W) if x_m[i]])
            if slot:
                attempts += 1
                ok += bool(b.deploy_card(0, b.players[0].cycle[slot - 1], Position(x + 0.5, y + 0.5)))
            for _ in range(30):
                if b.game_over:
                    break
                b.step(1 / 60)
            steps += 1
    rate = ok / max(attempts, 1)
    print(f"\nmasked deploy success rate: {rate:.1%} ({ok}/{attempts})")
    assert rate > 0.75, f"masked actions still fail {1-rate:.1%} of the time"


# --------------------------------------------------------------------------- env contract

def test_terminated_and_truncated_are_distinct():
    env = CREnv(opponent_model=random_strategy)
    env.reset(seed=0)
    for _ in range(2000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert not truncated, "this MDP has no external time limit; truncation must stay False"
        if terminated:
            break
    else:
        pytest.fail("game never ended within 2000 steps")


def test_observation_survives_units_pushed_past_the_arena_edge():
    """Collision resolution can shove a unit a hair outside the grid; that must not crash."""
    env = attach()
    assert env.battle.deploy_card(0, 'Knight', Position(0.5, 5.5))
    knight = next(e for e in env.battle.entities.values()
                  if e.name == 'Knight' and e.is_alive)
    for pos in (Position(18.04, 5.5), Position(-0.2, 5.5), Position(8.5, 32.01)):
        knight.position = pos
        for pid in (0, 1):
            env.observe(pid)  # must not raise IndexError


def test_untouched_towers_at_the_time_limit_are_a_draw():
    """Both sides at full HP means both lowest towers fall together -- neither player wins."""
    env = attach()
    env.battle.time = 300.0
    env.battle.step(1 / 60)
    assert env.battle.game_over
    assert env.battle.winner is None


# Entity ids of the princess towers, from `BattleState.update_player_hp`.
LEFT_TOWER_ENTITY = {0: 3, 1: 1}


def test_lowest_tower_loses_the_tiebreak():
    """The single lowest-HP tower on the board decides it, whichever side holds it.

    Damage the tower entity, not `PlayerState.left_tower_hp`: every step starts with
    `update_player_hp()`, which copies the entities back over the player fields.
    """
    for damaged, expected_winner in ((0, 1), (1, 0)):
        env = attach()
        env.battle.entities[LEFT_TOWER_ENTITY[damaged]].hp -= 100
        env.battle.time = 300.0
        env.battle.step(1 / 60)
        assert env.battle.winner == expected_winner, \
            f"player {damaged} holds the weakest tower and must lose"
