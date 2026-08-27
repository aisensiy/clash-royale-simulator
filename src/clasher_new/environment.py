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
# Per-cell channels. The first 15 describe *a* unit standing on the cell; because they are
# written one unit at a time into the same slot, a cell holding three bodies keeps only the
# last of them. The two count channels say how many are actually there -- see CREnv.observe.
N_UNIT_CHANNELS = 15
N_COUNT_CHANNELS = 2
CH_OWN_COUNT, CH_ENEMY_COUNT = N_UNIT_CHANNELS, N_UNIT_CHANNELS + 1
# The flat action space: "do nothing", then one outcome per (card, cell) pair.
# `MultiDiscrete([N_SLOTS, ARENA_H, ARENA_W])` factorises a placement into an
# independent row and an independent column, and the policy's distribution over cells is
# then their outer product. It cannot say "either (3,5) or (20,12)" without also putting
# weight on (3,12) and (20,5); and the mask, which sb3-contrib takes per dimension, can
# only say "some cell in row 3 is legal", never "this card may go on this cell". 2305
# outcomes is small enough to model jointly, which removes both problems at once.
N_CELLS = ARENA_H * ARENA_W
N_FLAT_ACTIONS = 1 + (N_SLOTS - 1) * N_CELLS
KING_TOWER_HP = 4824


def decode_flat_action(action):
    """A single index back into the `(slot, y, x)` triple the simulator deploys."""
    index = int(action)
    if index <= 0:
        return 0, 0, 0
    index -= 1
    slot, cell = divmod(index, N_CELLS)
    y, x = divmod(cell, ARENA_W)
    return slot + 1, y, x


def encode_flat_action(slot, y, x):
    """The inverse, for tests and for replaying a recorded triple through a flat policy."""
    if int(slot) == 0:
        return 0
    return 1 + (int(slot) - 1) * N_CELLS + int(y) * ARENA_W + int(x)


def action_triple(action, flat_action):
    """Whatever the acting side's policy produced, as the `(slot, y, x)` a deploy needs.

    The two action encodings meet here and nowhere else, which is also why the eval tools
    that break an action apart call this rather than indexing it.
    """
    if flat_action:
        return decode_flat_action(action)
    return tuple(int(a) for a in action)


def flat_action_for(model):
    """Whether this checkpoint was trained on the joint action space.

    Read off the saved action space for the same reason as `count_obs_for`: the eval
    tools load checkpoints from several runs at once, and a side handed the wrong action
    encoding does not crash -- it deploys the wrong card in the wrong place.
    """
    space = getattr(model, "action_space", None)
    return bool(getattr(space, "n", None) == N_FLAT_ACTIONS)


# What one unit of each card is worth, in elixir. A card that summons several bodies
# splits its cost between them, or a Minions deploy would read as 9 elixir of value for
# a 3 elixir card.
UNIT_VALUE = {name: Card(name).elixir / Card(name).spawn_number
              for name in DECK if Card(name).type != 'spell'}

# How much of a unit's price is only collected by finishing it off. Chip damage is worth
# real but limited credit: a Giant on one hit point still deals full damage, so paying
# out its whole value for damage alone would reward harassing a push instead of killing
# it. At 0.5 a kill is always worth more than every point of chip damage before it.
KILL_SHARE = 0.5


def _base_troop_legality(player_id):
    """Cells where a troop may be deployed, in `player_id`'s own egocentric frame.

    Mirrors the geometry checks in `BattleState.deploy_card`, which are evaluated on the
    absolute position `CREnv.deploy` builds -- and the two players' checks there are *not*
    reflections of one another. Red's forward band is one row deeper than blue's, and the
    tower that opens a lane is named in absolute coordinates, so red's egocentric left is
    gated by the enemy's right tower. Building the grids per player off the absolute
    position is the only way to keep that straight; a mirrored copy of blue's grids opens
    the wrong lane, which is exactly the bug this replaced. Returns (always_legal,
    needs_enemy_left_tower_down, needs_enemy_right_tower_down) as boolean grids.
    """
    always = np.zeros((ARENA_H, ARENA_W), dtype=bool)
    left = np.zeros((ARENA_H, ARENA_W), dtype=bool)
    right = np.zeros((ARENA_H, ARENA_W), dtype=bool)
    for y in range(ARENA_H):
        for x in range(ARENA_W):
            if player_id == 0:
                py, px = y + 0.5, x + 0.5
                if py <= 1.0 and (px <= 6.0 or px > 12.0):
                    continue  # behind the king tower
                if py >= 21.0:
                    continue  # never reachable, even with both princess towers down
                forward = py >= 15.0
            else:
                py, px = ARENA_H - (y + 0.5), ARENA_W - (x + 0.5)
                if py > 31.0 and (px <= 6.0 or px > 12.0):
                    continue
                if py <= 10.0:
                    continue
                forward = py <= 17.0
            if not forward:
                always[y][x] = True
            elif px <= 9:
                left[y][x] = True
            else:
                right[y][x] = True
    return always, left, right


_TROOP_LEGALITY = {p: _base_troop_legality(p) for p in (0, 1)}
_ALL_CELLS = np.ones((ARENA_H, ARENA_W), dtype=bool)


class CREnv(gym.Env):
    def __init__(self, opponent_model=None, visualize=False, speed=1.0, legacy_obs=False,
                 realtime=True, learner_player=None,
                 record_path=None, record_every=20,
                 rich_obs=False, opponent_rich_obs=None, dmg_scale=1.0,
                 elixir_scale=0.0, count_obs=False, opponent_count_obs=None,
                 flat_action=False, opponent_flat_action=None):
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
        # A checkpoint trained without the rich observation cannot be handed one, so a
        # match between the two encodings needs each side served its own. Follows the
        # learner rather than the player id, since which side the learner takes changes
        # between episodes.
        self.rich_obs_opponent = rich_obs if opponent_rich_obs is None else opponent_rich_obs
        # `count_obs` appends the two count channels. Without them a cell holding three
        # Minions reads back as one Minion -- how many bodies are in a push is simply not
        # in the input, so no amount of training can teach what a push is worth.
        self.count_obs = count_obs
        # Same reason the rich observation is served per side: the two encodings are
        # different input widths, so a match between them has to hand each side its own.
        self.count_obs_opponent = count_obs if opponent_count_obs is None else opponent_count_obs
        # Weight on the tower-damage shaping terms. At 1.0 the shaping available over a
        # full game (~10.9) rivals the terminal win bonus (10), which pays for constant
        # output regardless of whether the trade was good.
        # `flat_action` swaps the factorised action space for the joint one. See
        # `N_FLAT_ACTIONS`: it is what lets the mask be exact per cell.
        self.flat_action = flat_action
        # Per side, for the same reason as the observation flags -- a run on the joint
        # action space and a run on the factorised one have to be able to play each other,
        # which is the only way to know whether the change was worth making. Unlike those
        # flags the default is *not* to mirror the learner: an observation is something
        # this env produces, so serving the opponent whatever the learner gets is at worst
        # useless, while an action is something it consumes, and reading a `(slot, y, x)`
        # triple as an index deploys a different card somewhere else without erroring.
        # Every opponent that does speak the joint space says so -- see
        # `opponent_flat_action` -- and everything hand-written answers in triples.
        self.flat_action_opponent = bool(opponent_flat_action)
        self.dmg_scale = dmg_scale
        self.elixir_scale = elixir_scale
        spaces = {
            "grid": gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                   shape=(ARENA_H, ARENA_W, self.n_grid_channels)),
            "hand": gym.spaces.Box(low=0, high=len(entity_names) - 1, shape=(5,), dtype=np.int32),
            "elixir": gym.spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
        }
        if rich_obs:
            spaces["context"] = gym.spaces.Box(low=0.0, high=1.0, shape=(N_CONTEXT,), dtype=np.float32)
            spaces["opp_hand"] = gym.spaces.Box(low=0.0, high=1.0, shape=(len(DECK),), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = (gym.spaces.Discrete(N_FLAT_ACTIONS) if flat_action
                             else gym.spaces.MultiDiscrete([N_SLOTS, ARENA_H, ARENA_W]))

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
        else:
            # The eval tools pin the side per episode (`env.learner_player = i % 2`) and
            # then reset. Without this, that assignment did nothing and the side stayed
            # whatever __init__ set it to, so "half the games from each side" was not
            # actually happening.
            self.learner = self.learner_player
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

    @property
    def opponent_flat_action(self):
        """Which action encoding the opponent is playing right now.

        A self-play pool holds checkpoints and scripts side by side, and a script emits a
        `(slot, y, x)` triple whatever encoding the run itself uses. So an opponent that
        knows its own answer gets to give it, exactly like `masked`; the constructor value
        is the fallback for the eval tools, which load one fixed opponent.
        """
        return getattr(self.opponent, "flat_action", self.flat_action_opponent)

    def opponent_action(self):
        opponent = 1 - self.learner
        observation = self.observe(opponent)
        # A masked policy has to be handed the mask for the side it is actually playing.
        # Without it, sb3-contrib samples over actions the policy never learned to score
        # and the opponent plays close to randomly -- which under self-play means the
        # learner spends the whole run beating up sandbags.
        if getattr(self.opponent, "masked", False):
            action = self.opponent(observation, self.action_masks(opponent))
        else:
            action = self.opponent(observation)
        action = action_triple(action, self.opponent_flat_action)
        self.deploy(opponent, action)
        return action

    def _legal_cells(self, player_id, card_name, blocked):
        """Where this one card may be placed, in the acting player's own frame."""
        if Card(card_name).type == 'spell':
            return _ALL_CELLS  # spells may target the whole arena
        enemy = self.battle.players[1 - player_id]
        always, needs_left, needs_right = _TROOP_LEGALITY[player_id]
        legal = always.copy()
        if enemy.left_tower_hp <= 0:
            legal |= needs_left
        if enemy.right_tower_hp <= 0:
            legal |= needs_right
        return legal & ~blocked

    def action_masks(self, player_id=None):
        """Which actions are legal right now, in whichever encoding this side plays.

        On the joint action space the mask is exact: one bool per `(card, cell)` pair, so
        a troop is never offered a cell across the river and a spell is never denied one.
        On the factorised space it cannot be -- sb3-contrib takes one mask per dimension,
        so a joint constraint ("this card may go here") has no way to be expressed and the
        position mask is the union over every currently playable card. Elixir, which
        accounts for the overwhelming majority of rejected actions, is exact either way
        because it depends only on the slot.
        """
        if player_id is None:
            player_id = self.learner
        flat = self.flat_action if player_id == self.learner else self.opponent_flat_action
        p = self.battle.players[player_id]

        slot_mask = np.zeros(N_SLOTS, dtype=bool)
        slot_mask[0] = True  # doing nothing is always available
        playable = {}
        for i in range(1, N_SLOTS):
            card_name = p.cycle[i - 1]
            if p.can_play_card(card_name):
                slot_mask[i] = True
                playable[i] = card_name

        # The mask is built in player 0's frame, so red's view of it is point-reflected.
        blocked = self._building_cells()
        if player_id == 1:
            blocked = blocked[::-1, ::-1]

        if flat:
            mask = np.zeros(N_FLAT_ACTIONS, dtype=bool)
            mask[0] = True
            for slot, card_name in playable.items():
                start = 1 + (slot - 1) * N_CELLS
                mask[start:start + N_CELLS] = self._legal_cells(
                    player_id, card_name, blocked).ravel()
            return mask

        if not playable:
            # Every position is offered rather than none: a dimension with no legal value
            # at all is not something the masked distribution can be asked to sample from.
            return np.concatenate([slot_mask,
                                   np.ones(ARENA_H, dtype=bool),
                                   np.ones(ARENA_W, dtype=bool)])

        legal = np.zeros((ARENA_H, ARENA_W), dtype=bool)
        for slot, card_name in playable.items():
            legal |= self._legal_cells(player_id, card_name, blocked)
        return np.concatenate([slot_mask, legal.any(axis=1), legal.any(axis=0)])

    def _army_value(self, player_id):
        """The elixir standing on the board for one side, discounted by damage taken.

        A unit at full health is worth exactly what it cost -- which is what keeps
        playing a card worth nothing by itself -- and it keeps `KILL_SHARE` of that price
        until the moment it dies. Towers are not included: they were never bought, and
        `UNIT_VALUE` has no entry for them.
        """
        total = 0.0
        for entity in self.battle.entities.values():
            if entity.player != player_id or not entity.is_alive:
                continue
            value = UNIT_VALUE.get(entity.name)
            if value is None or not isinstance(entity, battle.Troop):
                continue
            health = max(0.0, entity.hp) / entity.data.hp
            total += value * (KILL_SHARE + (1.0 - KILL_SHARE) * health)
        return total

    def _elixir_edge(self):
        """How far ahead the learner is, counting the bank and the board together.

        Playing a card moves elixir from one to the other and leaves this unchanged, so
        the shaping cannot be farmed by dumping cards; only trades move it.
        """
        me, foe = self.learner, 1 - self.learner
        return ((self.battle.players[me].elixir + self._army_value(me))
                - (self.battle.players[foe].elixir + self._army_value(foe)))

    def note_external_play(self, player_id, card_name):
        """Record a play that did not go through `deploy`.

        The live server owns its own BattleState and applies the human player's cards
        itself, so without this the agent would card-count only its own plays and read
        its opponent's hand as unknown for the whole game.
        """
        self._plays[player_id].append(card_name)

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

        action = action_triple(action, self.flat_action)
        me = self.battle.players[self.learner]
        foe = self.battle.players[1 - self.learner]
        my_hps_old = me.king_tower_hp+me.left_tower_hp+me.right_tower_hp
        foe_hps_old = foe.king_tower_hp+foe.left_tower_hp+foe.right_tower_hp
        my_left = 3-me.get_crown_count()
        foe_left = 3-foe.get_crown_count()
        edge_old = self._elixir_edge() if self.elixir_scale else 0.0

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
        if self.elixir_scale and not self.battle.game_over:
            # A potential over the elixir each side still owns: what is in the bank plus
            # what is standing on the board. Rewarding the change in it makes an elixir
            # trade visible the moment it happens -- killing a 5 elixir push with a 3
            # elixir answer is +2 -- where tower HP alone reports a perfect defence as
            # nothing having happened at all.
            #
            # Skipped on the terminal step: the winner's towers falling clears the board,
            # and a potential that jumps to zero would pay out a spurious lump sum that
            # has nothing to do with the trade.
            reward += self.elixir_scale * (self._elixir_edge() - edge_old)
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

    @property
    def n_grid_channels(self):
        return N_UNIT_CHANNELS + (N_COUNT_CHANNELS if self.count_obs else 0)

    def observe(self, player_id_observe=0):
        """Gives an egocentric representation of the game state.

        The grid is always drawn from `player_id_observe`'s point of view: own units occupy
        the low rows, the enemy the high rows, and channel 1 is 0 for own units and 1 for the
        enemy. Player 1 therefore sees the arena point-reflected, which matches the action
        transform in `opponent_action`.
        """
        counted = self.count_obs if player_id_observe == self.learner else self.count_obs_opponent
        channels = N_UNIT_CHANNELS + (N_COUNT_CHANNELS if counted else 0)
        obs = np.zeros((ARENA_H, ARENA_W, channels), dtype=np.float32)
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
            # The unit channels are written whole, so whoever lands on this cell last is
            # the one described. The counts are added up instead, which is the only part of
            # a stack of bodies that survives the overwrite.
            obs[y][x][:N_UNIT_CHANNELS] = obs_arr
            if counted:
                obs[y][x][CH_ENEMY_COUNT if each.player != player_id_observe
                          else CH_OWN_COUNT] += 1

        hand = np.array([entity_names.index(each) for each in self.battle.players[player_id_observe].cycle[:5]],
                        dtype=np.int32)

        out = {
            'grid': obs,
            'hand': hand,
            'elixir': np.array([self.battle.players[player_id_observe].elixir], dtype=np.float32)
        }
        rich = self.rich_obs if player_id_observe == self.learner else self.rich_obs_opponent
        if rich:
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


def count_obs_for(model):
    """Whether this checkpoint was trained with the per-cell count channels.

    Read off the width of the saved grid rather than passed around by hand, for the same
    reason as `rich_obs_for`: handing a 15-channel checkpoint a 17-channel observation is
    a shape error, and the reverse is a silently wrong measurement.
    """
    space = getattr(model, "observation_space", None)
    grid = getattr(space, "spaces", {}).get("grid")
    return bool(grid is not None and grid.shape[-1] >= N_UNIT_CHANNELS + N_COUNT_CHANNELS)


def random_strategy(observation):
    slot = randint(0, N_SLOTS - 1)
    y = randint(0, ARENA_H - 1)
    x = randint(0, ARENA_W - 1)
    return slot, y, x


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env
    env = CREnv(random_strategy, visualize=True)
    check_env(env)
