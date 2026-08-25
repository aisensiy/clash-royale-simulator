"""A hand-written opponent that plays the way a human describes the game.

Every opponent in the pool so far spends elixir the instant it arrives -- `random`
throws cards anywhere, `rusher` sends single cards up one lane -- so nothing in the
training distribution has ever punished the learner for feeding cards one at a time.
Self-play does not fix that: both sides share the habit, so neither is made to pay for
it. This script is the missing opponent. It does three things the measurements in
`tests/test_game_mechanics.py` show actually pay off in this simulator:

  - it holds elixir instead of spending it as it arrives
  - it answers a push inside its own tower's range, where the tower's damage is added
    to its own, rather than meeting it at the river
  - it commits a tank and support together, which deals about twice what the same two
    cards deal played apart

It sees only the observation, exactly like a policy: no peeking at the battle state.

Two variants exist on purpose. `make_defender` is meant for the opponent pool, and
`make_anchor` is held out of training so that a rating measured against it is not a
rating against something the agent was drilled on.
"""
import numpy as np

from card_utils import Card
from environment import ARENA_H, ARENA_W, N_SLOTS, entity_names

# Grid channels, from `CREnv.observe`.
CH_PLAYER = 1     # 0 = mine, 1 = theirs, in the acting player's own frame
CH_COST = 2       # the card's elixir price; towers cost nothing, which is how they
                  # are told apart from units without naming them

# The near edge of the arena is rows 0..15 in the acting player's own frame. Cards may
# not be placed past row 14 while the enemy's towers stand, and rows 0..8 are where our
# own towers sit, so every placement this script makes lives in between.
FIRST_ROW, LAST_ROW = 9, 14
OWN_HALF = 17     # anything at or below this row is across the river and coming
LANES = (3, 14)   # the two princess tower columns


def _playable(hand, elixir):
    """(slot, name) for each card in hand that can be paid for, spells excluded.

    Spells are left out of this script deliberately: it is meant to demonstrate holding
    elixir for a combined push and defending in range, and a script that also throws
    Fireballs would make it harder to say which behaviour did the work.
    """
    out = []
    for slot in range(1, N_SLOTS):
        name = entity_names[int(hand[slot - 1])]
        card = Card(name)
        if card.type == 'spell' or card.elixir > elixir:
            continue
        out.append((slot, name))
    return out


def _tank_value(name):
    """Hit points per elixir: what a card is worth standing in front."""
    return Card(name).hp / Card(name).elixir


def _support_value(name):
    """Damage per second per elixir: what a card is worth standing behind."""
    card = Card(name)
    return (card.damage / card.hit_speed) / card.elixir if card.hit_speed else 0.0


def _threats(grid):
    """Rows and columns of every enemy unit that has crossed into our half."""
    enemy = (grid[:, :, CH_PLAYER] > 0.5) & (grid[:, :, CH_COST] > 0)
    rows, cols = np.nonzero(enemy)
    keep = rows <= OWN_HALF
    return rows[keep], cols[keep]


def _own_units(grid):
    mine = (grid[:, :, CH_PLAYER] < 0.5) & (grid[:, :, CH_COST] > 0)
    return np.nonzero(mine)


def make_defender(seed=0, hold=6.0, push_at=8.0, support_gap=2):
    """Hold elixir, answer pushes near your own tower, commit tank and support together.

    `hold` is the bank it will not spend below unless something is coming; `push_at` is
    what it waits for before starting an attack of its own. `seed` is accepted so this
    is interchangeable with the other script factories, which need one.
    """
    state = {"support": None}

    def on_episode_start():
        state["support"] = None

    def strategy(observation):
        grid = observation['grid']
        hand = observation['hand']
        elixir = float(observation['elixir'][0])
        playable = _playable(hand, elixir)
        if not playable:
            return 0, 0, 0

        # A tank went down last decision; put the support in behind it while the two
        # will still travel together.
        pending = state["support"]
        if pending is not None:
            state["support"] = None
            row, col = pending
            slot, _ = max(playable, key=lambda p: _support_value(p[1]))
            return slot, max(FIRST_ROW, row - support_gap), col

        rows, cols = _threats(grid)
        if len(rows):
            # Answer the deepest one -- the closest to our own tower -- and stand between
            # it and the tower rather than running out to meet it, so the tower's damage
            # is added to whatever we put down.
            i = int(np.argmin(rows))
            row = int(np.clip(rows[i] - 1, FIRST_ROW, LAST_ROW))
            col = int(np.clip(cols[i], 0, ARENA_W - 1))
            near = _own_units(grid)[0]
            already_defending = len(near) and int(np.min(near)) <= row + 3
            if already_defending:
                slot, _ = max(playable, key=lambda p: _support_value(p[1]))
                return slot, max(FIRST_ROW, row - support_gap), col
            slot, _ = max(playable, key=lambda p: _tank_value(p[1]))
            return slot, row, col

        # Nothing to answer. Spend only out of a full bank, and lead with the tank so
        # the support that follows next decision has something to hide behind.
        if elixir < max(hold, push_at):
            return 0, 0, 0
        slot, name = max(playable, key=lambda p: _tank_value(p[1]))
        col = LANES[int(elixir * 10) % 2]     # alternate lanes without needing a rng
        state["support"] = (LAST_ROW, col)
        return slot, LAST_ROW, col

    strategy.on_episode_start = on_episode_start
    return strategy


def make_anchor(seed=0):
    """The held-out variant: same idea, different thresholds and a tighter defence.

    Kept out of the opponent pool so that a rating measured against it is a rating
    against play the agent was never drilled on. Beating the training copy of a script
    and beating the idea it stands for are not the same thing.
    """
    return make_defender(seed=seed, hold=7.0, push_at=9.0, support_gap=3)
