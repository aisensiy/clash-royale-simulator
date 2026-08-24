import battle, player
from new_visualization import Visualizer
from card_utils import Card
from core import Position

import gymnasium as gym
from random import shuffle, randint
import time
import numpy as np

DECK = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']

entity_names = ['None', 'Knight', 'MiniPekka', 'Arrows', 'Minions', 'Archer',
                'Musketeer', 'Fireball', 'Giant', 'King_PrincessTowers',
                'KingTower', 'ArrowsSpell', 'FireballSpell']
# The agent has to learn that it can only deploy fireball and arrows, and the entities that actually appear are
# the arrows/fireball+spells thingy.

card_types = ['troop', 'character', 'spell', 'building']
# Troop mean princess tower, short for tower troop.
# Actual troops are represented as "characters".
speed_types = [0, 0.75, 1.0, 1.5]

ARENA_H, ARENA_W = 32, 18
N_SLOTS = 5  # slot 0 is "do nothing", slots 1..4 are the four cards in hand


def _base_troop_legality():
    """Cells where a troop may be deployed by player 0, ignoring buildings and tower state.

    Mirrors the geometry checks in `BattleState.deploy_card`, which are evaluated on the
    cell centre `Position(x + 0.5, y + 0.5)`. Returns (always_legal, needs_left_tower_down,
    needs_right_tower_down) as boolean grids.
    """
    always = np.zeros((ARENA_H, ARENA_W), dtype=bool)
    left = np.zeros((ARENA_H, ARENA_W), dtype=bool)
    right = np.zeros((ARENA_H, ARENA_W), dtype=bool)
    for y in range(ARENA_H):
        for x in range(ARENA_W):
            py, px = y + 0.5, x + 0.5
            if py <= 1.0 and (px <= 6.0 or px > 12.0):
                continue  # behind the king tower
            if py >= 21.0:
                continue  # never reachable, even with both princess towers down
            if py >= 15.0:
                if px <= 9:
                    left[y][x] = True
                else:
                    right[y][x] = True
            else:
                always[y][x] = True
    return always, left, right


_TROOP_ALWAYS, _TROOP_NEEDS_LEFT, _TROOP_NEEDS_RIGHT = _base_troop_legality()
_ALL_CELLS = np.ones((ARENA_H, ARENA_W), dtype=bool)


class CREnv(gym.Env):
    def __init__(self, opponent_model=None, visualize=False, speed=1.0, legacy_obs=False):
        super().__init__()
        self.opponent = opponent_model
        # `legacy_obs` reproduces the pre-fix encoding on purpose so the two can be
        # compared under identical conditions. Do not turn it on for real training.
        self.legacy_obs = legacy_obs
        self.battle: battle.BattleState = None
        self.speed = speed
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(ARENA_H, ARENA_W, 15), dtype=np.float32),
            "hand": gym.spaces.Box(low=0, high=len(entity_names) - 1, shape=(5,), dtype=np.int32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32)
        })
        self.action_space = gym.spaces.MultiDiscrete([N_SLOTS, ARENA_H, ARENA_W])

        self.visualize = visualize
        self.visualizer = None
        self._deck_0 = DECK[:]
        self._deck_1 = DECK[:]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        shuffle(self._deck_0)
        shuffle(self._deck_1)
        self.battle = battle.BattleState(player.PlayerState(0, self._deck_0[:], 5.0),
                                         player.PlayerState(1, self._deck_1[:], 5.0))
        if self.visualize:
            self.visualizer = Visualizer(self.battle)
        # Now return initial observation
        return self.observe(0), {}

    def opponent_action(self):
        obs1 = self.observe(1)
        opponent_action = self.opponent(obs1)
        slot, y, x = opponent_action
        p1 = self.battle.players[1]
        if slot != 0:
            card_name = p1.cycle[slot - 1]
            self.battle.deploy_card(1, card_name, Position(18-(x+0.5), 32-(y+0.5)))
            # Yes, this transformation seems weird, but it should be correct

    def action_masks(self, player_id=0):
        """Boolean mask over the three action dimensions, concatenated: [slot | y | x].

        MultiDiscrete masks are per-dimension, so a joint constraint ("this card may go
        here") cannot be expressed exactly -- the position mask is the union over every
        currently playable card. Elixir, which accounts for the overwhelming majority of
        rejected actions, *is* masked exactly because it depends only on the slot.
        """
        p = self.battle.players[player_id]
        enemy = self.battle.players[1 - player_id]

        slot_mask = np.zeros(N_SLOTS, dtype=bool)
        slot_mask[0] = True  # doing nothing is always available
        playable = []
        for i in range(1, N_SLOTS):
            card_name = p.cycle[i - 1]
            if p.can_play_card(card_name):
                slot_mask[i] = True
                playable.append(card_name)

        if not playable:
            return np.concatenate([slot_mask,
                                   np.ones(ARENA_H, dtype=bool),
                                   np.ones(ARENA_W, dtype=bool)])

        if any(Card(name).type == 'spell' for name in playable):
            legal = _ALL_CELLS  # spells may target the whole arena
        else:
            legal = _TROOP_ALWAYS.copy()
            if enemy.left_tower_hp <= 0:
                legal |= _TROOP_NEEDS_LEFT
            if enemy.right_tower_hp <= 0:
                legal |= _TROOP_NEEDS_RIGHT
            legal = legal & ~self._building_cells()

        return np.concatenate([slot_mask, legal.any(axis=1), legal.any(axis=0)])

    def _building_cells(self):
        """Cells whose centre overlaps a live building footprint (towers included)."""
        blocked = np.zeros((ARENA_H, ARENA_W), dtype=bool)
        ys = np.arange(ARENA_H, dtype=np.float32)[:, None] + 0.5
        xs = np.arange(ARENA_W, dtype=np.float32)[None, :] + 0.5
        for bx, by, r in self.battle.building_positions:
            blocked |= (xs - bx) ** 2 + (ys - by) ** 2 < r ** 2
        return blocked

    def step(self, action):
        """
        The action is a tuple with three values: (slot, y, x). When slot=0, no action is performed. Else deploy card on
        slot to the corresponding position on the arena.
        A decision is made every 30 frames (which is half a second). The reward is calculated by the damage dealt/taken,
        destroyed tower/lost tower and won game/lose game.
        The opponent is a function that takes in the observation and outputs the action tuple.
        """

        p0, p1 = self.battle.players
        blue_hps_old = p0.king_tower_hp+p0.left_tower_hp+p0.right_tower_hp
        red_hps_old = p1.king_tower_hp+p1.left_tower_hp+p1.right_tower_hp
        blue_left = 3-p0.get_crown_count()
        red_left = 3-p1.get_crown_count()

        slot, y, x = action
        if slot != 0:
            card_name = p0.cycle[slot-1]
            self.battle.deploy_card(0, card_name, Position(x+0.5, y+0.5))

        self.opponent_action()
        # only make decisions per half second
        for i in range(30):
            if self.battle.game_over:
                break
            for j in range(int(self.speed)):
                self.battle.step(1/60)
            if self.visualizer:
                self.visualizer.render_frame()
                time.sleep(1/60)
        blue_hps_new = p0.king_tower_hp+p0.left_tower_hp+p0.right_tower_hp
        red_hps_new = p1.king_tower_hp+p1.left_tower_hp+p1.right_tower_hp
        blue_left_new = 3-p0.get_crown_count()
        red_left_new = 3-p1.get_crown_count()

        reward = 5*(red_left-red_left_new)-5*(blue_left-blue_left_new)+0.001*(red_hps_old-red_hps_new)-0.0012*(blue_hps_old-blue_hps_new)
        if self.battle.game_over:
            if self.battle.winner == 0:
                reward += 10
            else:
                reward -= 10

        # Every way this game ends -- king tower down, sudden death, or the 300s rule -- is a
        # real terminal state with a decided outcome, so the value function must not bootstrap
        # past it. `truncated` stays False; it is not a time limit imposed from outside the MDP.
        return self.observe(0), reward, self.battle.game_over, False, {}

    def observe(self, player_id_observe=0):
        """Gives an egocentric representation of the game state.

        The grid is always drawn from `player_id_observe`'s point of view: own units occupy
        the low rows, the enemy the high rows, and channel 1 is 0 for own units and 1 for the
        enemy. Player 1 therefore sees the arena point-reflected, which matches the action
        transform in `opponent_action`.
        """
        obs = np.zeros((ARENA_H, ARENA_W, 15), dtype=np.float32)
        mirror = (player_id_observe == 1)
        for id, each in self.battle.entities.items():
            if not each.is_alive: continue
            if isinstance(each, battle.Projectile): continue
            entity_id = entity_names.index(each.name)
            card_type = card_types.index(each.data.type)
            relative_player = each.player if self.legacy_obs else int(each.player != player_id_observe)
            elixir = each.data.elixir
            is_air = int(each.data.is_air_unit)
            attacks_ground, attacks_air = int(each.data.attack_ground), int(each.data.attack_air)

            speed = each.data.speed
            hp_left = np.log(each.hp) / 10
            hp_percentage = each.hp / each.data.hp if each.data.hp != 0 else 0
            hit_speed = each.data.hit_speed
            attack_range = each.data.range / 3
            sight_range = each.data.sight_range / 3
            damage = each.data.damage / 200
            projectile_damage = each.data.projectile_data.damage / 200

            # Collision resolution can push a unit a fraction past the arena edge, which
            # then indexes off the end of the grid. Clamp instead of crashing; the unit is
            # at the boundary either way.
            x = min(max(int(each.position.x), 0), ARENA_W - 1)
            y = min(max(int(each.position.y), 0), ARENA_H - 1)
            # Legacy: reflect every red entity, whoever is looking -- which drops enemy
            # units into the viewer's own half. Fixed: reflect the whole arena for red's
            # own view, so each player sees itself near row 0.
            if (each.player == 1) if self.legacy_obs else mirror:
                x = ARENA_W - 1 - x
                y = ARENA_H - 1 - y
            obs_arr = np.array([entity_id, relative_player, elixir, card_type, speed, is_air, attacks_ground, attacks_air,
                                hp_left, hp_percentage, hit_speed, attack_range, sight_range, damage, projectile_damage])
            obs[y][x] = obs_arr.copy()

        hand = np.array([entity_names.index(each) for each in self.battle.players[player_id_observe].cycle[:5]],
                        dtype=np.int32)

        return {
            'grid': obs,
            'hand': hand,
            'elixir': np.array([self.battle.players[player_id_observe].elixir], dtype=np.float32)
        }


def random_strategy(observation):
    slot = randint(0, N_SLOTS - 1)
    y = randint(0, ARENA_H - 1)
    x = randint(0, ARENA_W - 1)
    return slot, y, x


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env
    env = CREnv(random_strategy, visualize=True)
    check_env(env)
