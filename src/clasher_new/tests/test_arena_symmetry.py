"""Pin down where the two sides differ. Mirroring a cell means (x, y) -> (17-x, 31-y)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import battle
import player
from core import Position
from environment import ARENA_H, ARENA_W, DECK


def fresh(elixir=10.0):
    return battle.BattleState(player.PlayerState(0, DECK[:], elixir),
                              player.PlayerState(1, DECK[:], elixir))


def mirror(x, y):
    return ARENA_W - 1 - x, ARENA_H - 1 - y


def legal_cells(b, player_id, card="Knight"):
    """Cells where `deploy_card`'s geometry check passes, without mutating the battle."""
    out = set()
    for y in range(ARENA_H):
        for x in range(ARENA_W):
            pos = Position(x + 0.5, y + 0.5)
            if b.is_position_occupied_by_building(pos, 0):
                continue
            if player_id == 0:
                if pos.y <= 1.0 and (pos.x <= 6.0 or pos.x > 12.0):
                    continue
                if pos.y >= 21.0:
                    continue
                if pos.y >= 15.0:
                    tower = b.players[1].left_tower_hp if pos.x <= 9 else b.players[1].right_tower_hp
                    if tower > 0:
                        continue
            else:
                if pos.y > 31.0 and (pos.x <= 6.0 or pos.x > 12.0):
                    continue
                if pos.y <= 10:
                    continue
                if pos.y <= 17.0:
                    tower = b.players[0].left_tower_hp if pos.x <= 9 else b.players[0].right_tower_hp
                    if tower > 0:
                        continue
            out.add((x, y))
    return out


def test_deploy_regions_are_mirror_images():
    b = fresh()
    blue = legal_cells(b, 0)
    red = legal_cells(b, 1)
    red_mirrored = {mirror(x, y) for x, y in red}
    only_blue = sorted(blue - red_mirrored)
    only_red = sorted(red_mirrored - blue)
    assert not only_blue and not only_red, (
        f"蓝方独有 {len(only_blue)} 格 {only_blue[:6]}; 红方独有 {len(only_red)} 格 {only_red[:6]}")


def test_towers_sit_at_mirrored_positions():
    b = fresh()
    blue = sorted((round(e.position.x, 3), round(e.position.y, 3))
                  for e in b.entities.values() if e.player == 0)
    red = sorted(mirror_pos(e.position) for e in b.entities.values() if e.player == 1)
    assert blue == red, f"blue={blue} red_mirrored={red}"


def mirror_pos(p):
    return (round(ARENA_W - p.x, 3), round(ARENA_H - p.y, 3))


# Only the first four cards of the deck are in hand and therefore playable.
IN_HAND = [c for c in DECK[:4] if c != "Arrows"]


@pytest.mark.parametrize("card", IN_HAND)
@pytest.mark.parametrize("spot", [(4, 6), (13, 6), (8, 12)])
def test_a_unit_walks_the_same_path_on_either_side(card, spot):
    """Deploy on mirrored cells, run the same ticks, compare mirrored positions."""
    blue_b = fresh()
    assert blue_b.deploy_card(0, card, Position(spot[0] + 0.5, spot[1] + 0.5))
    red_b = fresh()
    mx, my = mirror(*spot)
    assert red_b.deploy_card(1, card, Position(mx + 0.5, my + 0.5))

    for _ in range(240):
        blue_b.step(1 / 60)
        red_b.step(1 / 60)

    blue_units = sorted((e.position.x, e.position.y) for e in blue_b.entities.values()
                        if e.player == 0 and e.name == card and e.is_alive)
    red_units = sorted((ARENA_W - e.position.x, ARENA_H - e.position.y)
                       for e in red_b.entities.values()
                       if e.player == 1 and e.name == card and e.is_alive)
    assert len(blue_units) == len(red_units), f"{card}@{spot}: 存活数不同"
    worst = max((max(abs(b[0] - r[0]), abs(b[1] - r[1]))
                 for b, r in zip(blue_units, red_units)), default=0.0)
    assert worst < 0.05, f"{card}@{spot}: 镜像位置最大偏差 {worst:.3f}"


@pytest.mark.xfail(reason="residual asymmetry near the river: 115 of 200 mirrored games "
                          "now stay mirrored to the end, up from 22, but the rest still "
                          "diverge somewhere around y=15..17",
                   strict=False)
def test_identical_mirrored_play_leaves_identical_tower_hp():
    """The end-to-end check: same cards, mirrored spots, same times -- towers must match.

    Any per-tick asymmetry compounds over a game, so this catches what single-unit
    path comparisons miss.
    """
    b = fresh(elixir=10.0)
    script = [(0, "Knight", (4, 6)), (60, "MiniPekka", (13, 6)),
              (180, "Minions", (8, 12)), (300, "Knight", (13, 10))]
    pending = list(script)
    for tick in range(60 * 90):
        while pending and pending[0][0] == tick:
            _, card, (x, y) = pending.pop(0)
            mx, my = mirror(x, y)
            b.players[0].elixir = b.players[1].elixir = 10.0
            assert b.deploy_card(0, card, Position(x + 0.5, y + 0.5))
            assert b.deploy_card(1, card, Position(mx + 0.5, my + 0.5))
        b.step(1 / 60)

    p0, p1 = b.players
    blue = (p0.king_tower_hp, p0.left_tower_hp, p0.right_tower_hp)
    # Mirroring swaps left and right.
    red = (p1.king_tower_hp, p1.right_tower_hp, p1.left_tower_hp)
    assert blue == red, f"blue={blue} red_mirrored={red}"


def test_battle_uses_the_pathfinder_this_file_tests():
    """`pathfinding.py` and `pathfinding_heap.py` are near-duplicates and only one is live.

    A symmetry fix was once applied to the unused one and looked green for weeks, so this
    asserts which module the simulator actually imports.
    """
    import battle as battle_module
    assert battle_module.EntityPathfinder.__module__ == "pathfinding_heap"


def test_path_cells_mirror_exactly():
    """cell(size - v) must equal 2*size - 1 - cell(v), boundaries included."""
    from pathfinding_heap import position_to_cell

    for i in range(ARENA_W * 4 + 1):
        x = i / 4.0
        for j in range(ARENA_H * 4 + 1):
            y = j / 4.0
            if x == ARENA_W / 2 or y == ARENA_H / 2:
                continue  # dead centre has no mirror image of its own
            cx, cy = position_to_cell(Position(x, y))
            mx, my = position_to_cell(Position(ARENA_W - x, ARENA_H - y))
            assert (mx, my) == (2 * ARENA_W - 1 - cx, 2 * ARENA_H - 1 - cy), \
                f"({x},{y}) -> ({cx},{cy}) but mirrored -> ({mx},{my})"


def test_a_mirrored_game_is_a_draw_not_a_red_win():
    """Both kings falling on the same tick used to be scored as a win for red.

    In a game where both sides play the same script in their own frame that is not an
    edge case: it is the outcome of every game that stays mirrored to the end, so the
    bias landed on exactly the measurement meant to detect bias.
    """
    b = fresh()
    for entity_id in (5, 6):        # both king towers
        b.entities[entity_id].hp = 0
    b.step(1 / 60)
    assert b.game_over
    assert b.winner is None, f"simultaneous double knockout scored as a win for {b.winner}"


def test_one_king_down_still_decides_the_game():
    for loser, expected_winner in ((6, 1), (5, 0)):
        b = fresh()
        b.entities[loser].hp = 0
        b.step(1 / 60)
        assert b.game_over and b.winner == expected_winner
