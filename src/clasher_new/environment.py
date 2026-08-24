import battle, player
from new_visualization import Visualizer
from card_utils import Card
from core import Position

import gymnasium as gym
import json
import os
from random import randint
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
N_CONTEXT = 8  # see CREnv.observe: the scalars a human reads off the screen
KING_TOWER_HP = 4824


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
    def __init__(self, opponent_model=None, visualize=False, speed=1.0, legacy_obs=False,
                 realtime=True, learner_player=None,
                 record_path=None, record_every=20,
                 rich_obs=False, dmg_scale=1.0):
        super().__init__()
        self.opponent = opponent_model
        # `legacy_obs` reproduces the pre-fix encoding on purpose so the two can be
        # compared under identical conditions. Do not turn it on for real training.
        self.legacy_obs = legacy_obs
        self.battle: battle.BattleState = None
        self.speed = speed
        # `rich_obs` adds everything a human sees but the original encoding left out:
        # the opponent's elixir, the clock (which also sets the elixir rate), both crown
        # counts, both sides' weakest tower (the 300s tiebreak compares exactly that), and
        # a card-counting belief over the opponent's hand. Without these, "hold elixir and
        # punish" is not merely hard to learn -- it is not expressible from the input.
        self.rich_obs = rich_obs
        # Weight on the tower-damage shaping terms. At 1.0 the shaping available over a
        # full game (~10.9) rivals the terminal win bonus (10), which pays for constant
        # output regardless of whether the trade was good.
        self.dmg_scale = dmg_scale
        spaces = {
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(ARENA_H, ARENA_W, 15), dtype=np.float32),
            "hand": gym.spaces.Box(low=0, high=len(entity_names) - 1, shape=(5,), dtype=np.int32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
        }
        if rich_obs:
            spaces["context"] = gym.spaces.Box(low=0.0, high=1.0, shape=(N_CONTEXT,), dtype=np.float32)
            spaces["opp_hand"] = gym.spaces.Box(low=0.0, high=1.0, shape=(len(DECK),), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.MultiDiscrete([N_SLOTS, ARENA_H, ARENA_W])

        self.visualize = visualize
        self.visualizer = None
        # Watching live wants wall-clock pacing; recording a file does not.
        self.realtime = realtime
        # Which side the agent plays. None means a fresh draw each episode: the arena is
        # not perfectly symmetric (an identical policy wins only 40% as blue), so an agent
        # pinned to one side both learns half the game and has that bias baked into every
        # win rate it reports.
        self.learner_player = learner_player
        self.learner = 0 if learner_player is None else learner_player
        # Episode recording. The simulator itself is deterministic -- no module in
        # battle.py, card_mechanics.py or arena.py draws a random number -- so a game is
        # fully reproducible from the two decks, which side the learner took, and both
        # players' actions. Seeds would not be enough: the learner's actions come from one
        # sampling stream shared across every env in the main process.
        self.record_path = record_path
        self.record_every = record_every
        self._episode_index = -1
        self._recording = None
        self._deck_0 = DECK[:]
        self._deck_1 = DECK[:]
        # Cards each player has been *seen* playing, in order. This is public information:
        # a card that is played goes to the back of its owner's cycle, so every play pins
        # down one more position. See `_opp_hand_belief`.
        self._plays = [[], []]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if self.learner_player is None:
            self.learner = int(self.np_random.integers(2))
        # Shuffle through the env's own generator, not the `random` module: otherwise
        # `reset(seed=...)` does not actually determine the episode and anything that
        # depends on the opening hand is quietly irreproducible.
        self._deck_0 = [DECK[i] for i in self.np_random.permutation(len(DECK))]
        self._deck_1 = [DECK[i] for i in self.np_random.permutation(len(DECK))]
        self._episode_index += 1
        self._plays = [[], []]
        self._recording = None
        if self.record_path and self._episode_index % self.record_every == 0:
            self._recording = {"deck_0": self._deck_0[:], "deck_1": self._deck_1[:],
                               "learner": self.learner, "actions": []}
        self.battle = battle.BattleState(player.PlayerState(0, self._deck_0[:], 5.0),
                                         player.PlayerState(1, self._deck_1[:], 5.0))
        if self.visualize:
            self.visualizer = Visualizer(self.battle)
        # Self-play opponents rotate between episodes; scripted ones have no hook.
        if hasattr(self.opponent, "on_episode_start"):
            self.opponent.on_episode_start()
        return self.observe(self.learner), {}

    def deploy(self, player_id, action):
        """Place a card from `player_id`'s own point of view.

        Actions are egocentric, matching the observation: row 0 is always the near edge
        of the acting player's half. For player 1 that means reflecting through the
        centre of the arena to get absolute coordinates.
        """
        slot, y, x = action
        if slot == 0:
            return False
        card_name = self.battle.players[player_id].cycle[slot - 1]
        if player_id == 0:
            pos = Position(x + 0.5, y + 0.5)
        else:
            pos = Position(ARENA_W - (x + 0.5), ARENA_H - (y + 0.5))
        played = self.battle.deploy_card(player_id, card_name, pos)
        if played:
            self._plays[player_id].append(card_name)
        return played

    def opponent_action(self):
        opponent = 1 - self.learner
        action = self.opponent(self.observe(opponent))
        self.deploy(opponent, action)
        return action

    def action_masks(self, player_id=None):
        """Boolean mask over the three action dimensions, concatenated: [slot | y | x].

        MultiDiscrete masks are per-dimension, so a joint constraint ("this card may go
        here") cannot be expressed exactly -- the position mask is the union over every
        currently playable card. Elixir, which accounts for the overwhelming majority of
        rejected actions, *is* masked exactly because it depends only on the slot.
        """
        if player_id is None:
            player_id = self.learner
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
            blocked = self._building_cells()
            if player_id == 1:
                blocked = blocked[::-1, ::-1]  # the mask is built in player 0's frame
            legal = legal & ~blocked

        return np.concatenate([slot_mask, legal.any(axis=1), legal.any(axis=0)])

    def _known_cycle_tail(self, player_id):
        """The suffix of `player_id`'s cycle that a spectator can pin down exactly.

        Playing a card sends it to the back of the cycle, so the cards that have been
        played, ordered by their most recent play, occupy the last positions -- and every
        card never played is somewhere ahead of all of them.
        """
        tail = []
        for name in self._plays[player_id]:
            if name in tail:
                tail.remove(name)
            tail.append(name)
        return tail

    def _opp_hand_belief(self, observer):
        """P(card is in the opponent's hand right now), from public information only.

        This is card counting, and it is what a human actually knows: at the start every
        one of the eight cards is equally likely to be among the four in hand (0.5), and
        each play the opponent makes fixes one more position until the hand is known
        exactly. Cards the opponent has never played stay ambiguous -- the belief is
        uniform over their unknown ordering, which ignores the (small) information leaked
        by *which* card the opponent chose to play. That is the same approximation a
        human makes.
        """
        opponent = 1 - observer
        tail = self._known_cycle_tail(opponent)
        belief = np.zeros(len(DECK), dtype=np.float32)
        unseen = [c for c in DECK if c not in tail]
        if len(unseen) >= 4:
            # The hand is four of the `unseen` cards, whose order is unknown.
            p = 4.0 / len(unseen)
            for c in unseen:
                belief[DECK.index(c)] = p
        else:
            # Fewer than four cards left unplayed: all of them are in hand, and the rest
            # of the hand is the front of the known tail.
            for c in unseen:
                belief[DECK.index(c)] = 1.0
            for c in tail[:4 - len(unseen)]:
                belief[DECK.index(c)] = 1.0
        return belief

    def _context(self, observer):
        me = self.battle.players[observer]
        foe = self.battle.players[1 - observer]
        t = self.battle.time
        # Elixir generation triples over the course of a game; the agent cannot pace its
        # spending without knowing which phase it is in.
        rate_tier = 0.0 if t < 120 else 0.5 if t < 240 else 1.0
        # Between 180s and 300s a one-crown lead ends the game immediately.
        sudden_death = 1.0 if 180 <= t < 300 else 0.0

        def weakest(p):
            live = [h for h in (p.king_tower_hp, p.left_tower_hp, p.right_tower_hp) if h > 0]
            return (min(live) / KING_TOWER_HP) if live else 0.0

        return np.array([
            foe.elixir / 10.0,
            min(t / 300.0, 1.0),
            rate_tier,
            sudden_death,
            me.get_crown_count() / 3.0,
            foe.get_crown_count() / 3.0,
            # At 300s every surviving tower drains at once and the single weakest one
            # falls first, so these two numbers *are* the tiebreak.
            weakest(me),
            weakest(foe),
        ], dtype=np.float32)

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

        me = self.battle.players[self.learner]
        foe = self.battle.players[1 - self.learner]
        my_hps_old = me.king_tower_hp+me.left_tower_hp+me.right_tower_hp
        foe_hps_old = foe.king_tower_hp+foe.left_tower_hp+foe.right_tower_hp
        my_left = 3-me.get_crown_count()
        foe_left = 3-foe.get_crown_count()

        self.deploy(self.learner, action)
        opponent_action = self.opponent_action()
        if self._recording is not None:
            self._recording["actions"].append(
                [int(a) for a in action] + [int(a) for a in opponent_action])
        # only make decisions per half second
        for i in range(30):
            if self.battle.game_over:
                break
            for j in range(int(self.speed)):
                self.battle.step(1/60)
            if self.visualizer:
                self.visualizer.render_frame()
                if self.realtime:
                    time.sleep(1/60)
        my_hps_new = me.king_tower_hp+me.left_tower_hp+me.right_tower_hp
        foe_hps_new = foe.king_tower_hp+foe.left_tower_hp+foe.right_tower_hp
        my_left_new = 3-me.get_crown_count()
        foe_left_new = 3-foe.get_crown_count()

        reward = (5*(foe_left-foe_left_new) - 5*(my_left-my_left_new)
                  + self.dmg_scale*(0.001*(foe_hps_old-foe_hps_new)
                                    - 0.0012*(my_hps_old-my_hps_new)))
        if self.battle.game_over:
            if self.battle.winner == self.learner:
                reward += 10
            elif self.battle.winner is not None:
                reward -= 10
            # winner is None on an exact tiebreak draw: no terminal bonus either way.

        info = {}
        if self.battle.game_over:
            # Which opponent this episode was against, so win rate can be broken down by
            # opponent type. Against the scripts it is the only non-circular progress signal.
            info["opponent"] = getattr(self.opponent, "label", "script:fixed")
            if self._recording is not None:
                self._write_record(info["opponent"])
            info["learner_player"] = self.learner
            info["outcome"] = (0 if self.battle.winner is None else
                               1 if self.battle.winner == self.learner else -1)

        # Every way this game ends -- king tower down, sudden death, or the 300s rule -- is a
        # real terminal state with a decided outcome, so the value function must not bootstrap
        # past it. `truncated` stays False; it is not a time limit imposed from outside the MDP.
        return self.observe(self.learner), reward, self.battle.game_over, False, info

    def _write_record(self, opponent_label):
        """Append one finished episode. Each env writes its own file to avoid contention."""
        self._recording["opponent"] = opponent_label
        self._recording["winner"] = self.battle.winner
        self._recording["outcome"] = (0 if self.battle.winner is None else
                                      1 if self.battle.winner == self.learner else -1)
        os.makedirs(self.record_path, exist_ok=True)
        name = f"episodes_{os.getpid()}.jsonl"
        with open(os.path.join(self.record_path, name), "a") as fh:
            fh.write(json.dumps(self._recording, separators=(",", ":")) + "\n")
        self._recording = None

    def replay_record(self, record, frame_hook=None):
        """Re-run a recorded episode. Returns the finished BattleState.

        Deterministic by construction: the decks, the side and every action are taken
        from the record, and nothing else in the simulator draws a random number.
        """
        self.learner = record["learner"]
        self._plays = [[], []]
        self.battle = battle.BattleState(player.PlayerState(0, record["deck_0"][:], 5.0),
                                         player.PlayerState(1, record["deck_1"][:], 5.0))
        if self.visualize:
            self.visualizer = Visualizer(self.battle)
            if frame_hook is not None:
                # The visualizer only exists once the battle does, so the capture hook
                # has to be attached here rather than by the caller beforehand.
                frame_hook(self.visualizer)
        opponent = 1 - self.learner
        for row in record["actions"]:
            if self.battle.game_over:
                break
            self.deploy(self.learner, row[:3])
            self.deploy(opponent, row[3:])
            for _ in range(30):
                if self.battle.game_over:
                    break
                self.battle.step(1 / 60)
                if self.visualizer:
                    self.visualizer.render_frame()
        return self.battle

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

        out = {
            'grid': obs,
            'hand': hand,
            'elixir': np.array([self.battle.players[player_id_observe].elixir], dtype=np.float32)
        }
        if self.rich_obs:
            out['context'] = self._context(player_id_observe)
            out['opp_hand'] = self._opp_hand_belief(player_id_observe)
        return out


def rich_obs_for(model):
    """Whether an env has to produce the rich observation for this loaded checkpoint.

    Read off the saved observation space rather than passed around by hand: the eval
    tools load checkpoints from several different runs and getting this wrong is a shape
    error at best and a silently wrong measurement at worst.
    """
    space = getattr(model, "observation_space", None)
    return "context" in getattr(space, "spaces", {})


def random_strategy(observation):
    slot = randint(0, N_SLOTS - 1)
    y = randint(0, ARENA_H - 1)
    x = randint(0, ARENA_W - 1)
    return slot, y, x


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env
    env = CREnv(random_strategy, visualize=True)
    check_env(env)
