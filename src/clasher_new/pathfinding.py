"""Unused. `battle.py` imports `pathfinding_heap`, which is the live implementation.

Kept because it is upstream's file, but nothing in the simulator loads it -- a symmetry
fix applied here once looked green for weeks while the real pathfinder was untouched.
Change `pathfinding_heap.py` instead.
"""
from pathlib import Path
import math
from core import Position

grid_path = Path(__file__).with_name('tilemap_lane_grid.txt')
with grid_path.open('r') as f:
    contents = [list(each) for each in f.read().splitlines()]

cell_cache = {}

ARENA_W, ARENA_H = 18.0, 32.0


def _axis_to_cell(value, size):
    """Half-cell index that mirrors correctly about the centre of the arena.

    Plain `floor(2*v)` is not symmetric under `v -> size - v` when `2*v` is an exact
    integer: the boundary belongs to the cell above it on one side and the cell below
    on the other. Every deploy lands on `n + 0.5`, i.e. exactly such a boundary, so the
    two players' units entered different path-finding cells from mirrored spots and
    walked measurably different routes. Snapping boundary values toward the centre
    makes `cell(size - v) == 2*size - 1 - cell(v)` hold.
    """
    scaled = 2 * value
    if scaled == int(scaled) and value > size / 2:
        return int(scaled) - 1
    return math.floor(scaled)


def position_to_cell(position: Position):
    return (_axis_to_cell(position.x, ARENA_W), _axis_to_cell(position.y, ARENA_H))

def cell_to_position(cell):
    if cell not in cell_cache:
        x, y = cell
        cell_cache[cell] = Position((x+0.5)/2, (y+0.5)/2)
    return cell_cache[cell]

def heuristic(current, goal):
    return 10 * max(abs(current.x - goal.x), abs(current.y - goal.y))

def get_neighboring_points(x, y):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            new_x, new_y = x+dx, y+dy
            if dx == dy == 0: continue
            if new_x < 0 or new_y < 0 or new_x >= 36 or new_y >= 64: continue
            yield new_x, new_y


class EntityPathfinder:
    def __init__(self, entity, target, battle_state):
        self.start_position = Position(entity.position.x, entity.position.y)
        self.target_position = Position(target.position.x, target.position.y)
        self.target = target
        self.entity = entity
        self.start_cell = position_to_cell(self.start_position)
        self.battle = battle_state
        self.goals = set()

    def heuristic(self, cell):
        x, y = cell
        return min(
            10 * max(abs(x - gx), abs(y - gy))
            for gx, gy in self.goals
        )

    def calculate(self):
        self.goals = set()

        radius = self.target.data.collision_radius + self.entity.data.range
        # The first step is to calculate some viable cells that is in attack position.

        target_cell = position_to_cell(self.target_position)
        scan_radius = math.ceil(radius*2) + 1
        for x in range(target_cell[0]-scan_radius, target_cell[0]+scan_radius):
            for y in range(target_cell[1]-scan_radius, target_cell[1]+scan_radius):
                distance = cell_to_position((x, y)).distance_to(self.target_position)
                # I added 0.375 to radius so that short-ranged troops like lumberjack can reach the tower instead of leering to the side
                if distance < radius+0.375 and self.battle.ground_walkable(cell_to_position((x, y)), self.entity.data.collision_radius):
                    self.goals.add((x, y))
        # The second step is to filter goals, only keep the closest one.
        self.goals = {min(self.goals, key=lambda c: cell_to_position(c).distance_to(self.target_position))}

        g = {}
        f = {}
        parent = {}
        open_set = {self.start_cell}
        closed_set = set()
        g[self.start_cell] = 0
        f[self.start_cell] = self.heuristic(self.start_cell)

        while open_set:
            current = min(open_set, key=lambda node: f[node])
            if current in self.goals:
                break
            open_set.remove(current)
            closed_set.add(current)
            for neighbor in get_neighboring_points(current[0], current[1]):
                if neighbor in closed_set: continue
                neighbor_position = cell_to_position(neighbor)
                if not self.battle.ground_walkable(neighbor_position, self.entity.data.collision_radius):
                    continue
                nx, ny = neighbor
                px, py = current
                tile_char = contents[63-ny][nx]
                if tile_char == 'W':
                    tile_cost = 800 if not self.entity.data.is_air_unit else 7
                elif tile_char == '.':
                    tile_cost = 8
                else:
                    tile_cost = 5
                if nx != px and ny != py:
                    geo_cost = 14
                else:
                    geo_cost = 10
                step_cost = tile_cost * geo_cost
                tentative_g = g[current] + step_cost
                if neighbor not in g or tentative_g < g[neighbor]:
                    g[neighbor] = tentative_g
                    parent[neighbor] = current
                    f[neighbor] = g[neighbor] + self.heuristic((nx, ny))
                    open_set.add(neighbor)
        path = [current]
        while path[-1] != self.start_cell:
            path.append(parent[path[-1]])
        path.reverse()

        positions = [cell_to_position(each) for each in path]
        return positions

if __name__ == '__main__':
    from battle import BattleState
    from player import PlayerState

    player_0_deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
    player_1_deck = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']
    battle = BattleState(PlayerState(0, player_0_deck, 10), PlayerState(1, player_1_deck, 10))
    battle.deploy_card(0, 'Knight', Position(9.5, 0.5))

    pathfind = EntityPathfinder(battle.entities[7], battle.entities[2], battle)
    print(pathfind.calculate())






