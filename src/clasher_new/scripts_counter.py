"""The third hand-written opponent, and the only one held out of training.

`scripts_sniper` was the ruler that showed the pool-defender run was a real improvement
and not just drill against one family of scripts. It is now strong enough to be worth
training against -- it beats every checkpoint measured so far -- and the moment it goes
into the pool it stops being able to measure anything. So it is promoted to teacher and
this takes over as ruler.

Neither of the other two reads what a unit *is*. The defender ranks cards by hit points
per elixir, the sniper by price; both would answer a Minions swarm and a Giant with the
same card. This one picks by matchup:

  - anything flying has to be answered by a card that can shoot air, whatever it costs
  - a swarm of small bodies gets a spell, because one splash covers all of them
  - a single big body gets the highest damage per second in hand, not the cheapest and
    not the beefiest

and it defends by pulling toward the middle of the arena rather than standing in front
of a tower, so a push aimed at one tower spends its walk crossing the map instead.

It attacks only with what survived a defence -- there is no bank threshold at which it
starts a push of its own. That makes it the only one of the three whose offence is a
consequence of its defence.

Held out of training on purpose. A test fails if its name reaches `train.py`.
"""
import numpy as np

from card_utils import Card
from environment import ARENA_W, N_SLOTS, entity_names

# Grid channels, from `CREnv.observe`.
CH_PLAYER = 1
CH_COST = 2
CH_AIR = 5        # 1 if the unit flies
CH_HEALTH = 9

FIRST_ROW, LAST_ROW = 9, 14
OWN_HALF = 17
CENTRE = 8        # the column between the two lanes
LANES = (3, 14)


def _dps(name):
    card = Card(name)
    return (card.damage / card.hit_speed) if card.hit_speed else 0.0


def _playable(observation, elixir):
    hand = observation['hand']
    return [(slot, entity_names[int(hand[slot - 1])]) for slot in range(1, N_SLOTS)
            if Card(entity_names[int(hand[slot - 1])]).elixir <= elixir]


def _threats(grid):
    """Enemy units on our side of the river: rows, columns, elixir value, and whether
    each one flies."""
    enemy = (grid[:, :, CH_PLAYER] > 0.5) & (grid[:, :, CH_COST] > 0)
    rows, cols = np.nonzero(enemy)
    keep = rows <= OWN_HALF
    rows, cols = rows[keep], cols[keep]
    value = grid[rows, cols, CH_COST] * grid[rows, cols, CH_HEALTH]
    flying = grid[rows, cols, CH_AIR] > 0.5
    return rows, cols, value, flying


def _own_rows_cols(grid):
    mine = (grid[:, :, CH_PLAYER] < 0.5) & (grid[:, :, CH_COST] > 0)
    return np.nonzero(mine)


def make_counter(seed=0, answer_at=2.0, swarm_at=3, push_with=1):
    """Answer by matchup, defend toward the middle, attack only off a won defence.

    `answer_at` is how much enemy elixir has to be on our side before a card is worth
    spending; `swarm_at` is how many separate bodies count as a swarm worth a spell;
    `push_with` is how many of our own units have to have survived before it sends one
    more card after them. `seed` is accepted for interface compatibility -- nothing here
    is random, so the same board always gets the same answer.

    The defaults come from an 18-point sweep against a checkpoint from an older run --
    never against the run this will be used to rate -- and are the setting that lands
    nearest an even match (47%), which is where a rating separates players best. The same
    sweep ranges from 6% to 58%, so these three numbers are not incidental.
    """
    def strategy(observation):
        grid = observation['grid']
        elixir = float(observation['elixir'][0])
        playable = _playable(observation, elixir)
        if not playable:
            return 0, 0, 0
        spells = [(s, n) for s, n in playable if Card(n).type == 'spell']
        bodies = [(s, n) for s, n in playable if Card(n).type != 'spell']

        rows, cols, value, flying = _threats(grid)
        if len(rows) and float(np.sum(value)) >= answer_at:
            # A swarm is several small bodies, which is exactly what one spell is for.
            if len(rows) >= swarm_at and spells:
                slot, _ = max(spells, key=lambda p: Card(p[1]).elixir)
                return slot, int(np.median(rows)), int(np.median(cols))
            if np.any(flying):
                # Nothing else in hand can touch it, so cost does not come into it.
                shooters = [(s, n) for s, n in bodies if Card(n).attack_air]
                if shooters:
                    i = int(np.argmin(np.where(flying, rows, 99)))
                    slot, _ = max(shooters, key=lambda p: _dps(p[1]))
                    row = int(np.clip(rows[i] - 1, FIRST_ROW, LAST_ROW))
                    return slot, row, int(np.clip(cols[i], 0, ARENA_W - 1))
                if spells:      # a spell is the last thing in hand that reaches the air
                    slot, _ = max(spells, key=lambda p: Card(p[1]).elixir)
                    return slot, int(np.median(rows)), int(np.median(cols))
            if bodies:
                # One big body on the ground: kill it fastest, and place toward the middle
                # so it turns away from the tower it was walking at and spends its walk
                # crossing the arena instead.
                i = int(np.argmin(rows))
                slot, _ = max(bodies, key=lambda p: _dps(p[1]))
                row = int(np.clip(rows[i] - 1, FIRST_ROW, LAST_ROW))
                col = int(round((int(cols[i]) + CENTRE) / 2))
                return slot, row, col

        # Nothing to answer. Attack only off a defence that left units standing: whatever
        # survived is already walking, so one more card behind it is a push that cost the
        # opponent more than it cost us.
        own_rows, own_cols = _own_rows_cols(grid)
        if len(own_rows) >= push_with and bodies:
            slot, _ = max(bodies, key=lambda p: _dps(p[1]))
            col = int(np.clip(int(np.median(own_cols)), 0, ARENA_W - 1))
            row = int(np.clip(int(np.max(own_rows)) - 1, FIRST_ROW, LAST_ROW))
            return slot, row, col
        return 0, 0, 0

    return strategy
