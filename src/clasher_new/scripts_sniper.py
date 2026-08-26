"""A second hand-written opponent, built to disagree with the first one.

`scripts_defender.py` is in the training pool, and `make_anchor` -- the variant held out
of it -- is the same algorithm with three constants changed. An agent drilled against the
pool copy has effectively been drilled against the anchor too, so "it beats the anchor"
cannot be read as "it got stronger": it may only mean it learned to beat this one family
of scripts. A ruler has to be independent of the thing it measures.

So this script shares no decision with the defender family. Where the defender picks the
highest hit points per elixir and stands it between the threat and its own tower, this one:

  - leads with spells, which the defender never plays at all, and picks their target by
    searching for the clump of enemy bodies worth the most elixir inside the blast radius
  - defends with the *cheapest* affordable card, dropped on a fixed post in front of the
    threatened tower, without tracking the threat's row
  - attacks with single cards at the bridge and never builds a combined push
  - keeps no state between decisions, so there is no follow-up card to predict

An agent that has learned to mass bodies into one push -- the behaviour the defender is
supposed to teach -- is exactly what a spell punishes. If a checkpoint beats the anchor
but not this, the gain was against the defender's habits rather than against the game.

Held out of training on purpose: `train.py` cannot put it in the pool, and it should not
be added there. Once an agent trains against it, it stops being a ruler.
"""
import numpy as np

from card_utils import Card
from environment import ARENA_H, ARENA_W, N_SLOTS, entity_names

# Grid channels, from `CREnv.observe`, same as scripts_defender.
CH_PLAYER = 1
CH_COST = 2
CH_HEALTH = 9

BRIDGE_ROW = 14   # the last row a card may be placed on while both enemy towers stand
POST_ROW = 11     # the fixed defensive post: in front of our own towers, in tower range
LANES = (3, 14)   # the two princess tower columns
OWN_HALF = 17     # at or below this row means it is on our side and coming


def _blast_radius(name):
    """The spell's damage radius in tiles.

    Card data stores it in thousandths, like every other distance in `card_utils`, and
    leaves it out entirely for cards that do not splash -- hence the fallback.
    """
    raw = Card(name).projectile_damage_radius
    return (raw / 1000.0) if raw else 2.0


def _hand(observation):
    """(slot, name) for all four cards in hand, whether affordable or not."""
    hand = observation['hand']
    return [(slot, entity_names[int(hand[slot - 1])]) for slot in range(1, N_SLOTS)]


def _enemy_cells(grid):
    """(rows, cols, value) of enemy units -- towers cost nothing, so they drop out."""
    enemy = (grid[:, :, CH_PLAYER] > 0.5) & (grid[:, :, CH_COST] > 0)
    rows, cols = np.nonzero(enemy)
    value = grid[rows, cols, CH_COST] * grid[rows, cols, CH_HEALTH]
    return rows, cols, value


def _best_blast(rows, cols, value, radius):
    """The cell whose blast radius covers the most enemy elixir, and how much that is.

    Only cells that already hold a unit are considered as centres: the best circle over a
    set of points can always be slid until a point sits at its centre, and searching 576
    grid cells per decision instead of the handful occupied would cost far more for the
    same answer.
    """
    best, best_at = 0.0, None
    for i in range(len(rows)):
        near = (np.abs(rows - rows[i]) <= radius) & (np.abs(cols - cols[i]) <= radius)
        total = float(np.sum(value[near]))
        if total > best:
            best, best_at = total, (int(rows[i]), int(cols[i]))
    return best, best_at


def make_sniper(seed=0, spell_at=5.0, defend_at=3.0, chip_at=8.0):
    """Punish clumps with spells, defend on a fixed post, chip with single cards.

    `spell_at` is how much enemy elixir has to be standing inside one blast before a
    spell is worth casting; `defend_at` is the same threshold for answering with a body;
    `chip_at` is the bank above which it sends a card across on its own. `seed` is taken
    for interface compatibility with the other script factories -- nothing here is random,
    which is itself deliberate: a ruler that varies between measurements is a bad ruler.

    The defaults are set for *resolution*, not for strength. A 27-point sweep against a
    checkpoint from a different run than the ones being rated -- so that the ruler is not
    tuned on what it measures -- put the script anywhere from 6% to 67% depending on these
    three numbers, and (5, 3, 8) is the setting that lands nearest an even match, which is
    where a rating separates players best.

    Strength is available on request and is worth knowing about: at (8, 2, 9) the script
    takes 24 of 24 games off the strongest checkpoint. Nothing here is clever -- at
    `spell_at=99`, meaning it never casts a spell at all, it still wins 96%. What beats
    the agent is holding to nearly a full bank, answering every crossing with the cheapest
    body in range of its own tower, and chipping one card at a time.
    """
    def strategy(observation):
        grid = observation['grid']
        elixir = float(observation['elixir'][0])
        rows, cols, value = _enemy_cells(grid)

        affordable = [(slot, name) for slot, name in _hand(observation)
                      if Card(name).elixir <= elixir]
        if not affordable:
            return 0, 0, 0
        spells = [(s, n) for s, n in affordable if Card(n).type == 'spell']
        bodies = [(s, n) for s, n in affordable if Card(n).type != 'spell']

        # A spell is worth casting wherever the bodies are, not only in our own half:
        # a clump walking up the far side is just as dead. Take the biggest blast on
        # offer and require it to pay for itself.
        if len(rows) and spells:
            slot, name = max(spells, key=lambda p: _blast_radius(p[1]))
            total, at = _best_blast(rows, cols, value, _blast_radius(name))
            if at is not None and total >= max(spell_at, Card(name).elixir):
                return slot, at[0], at[1]

        # Something is on our side of the river. Answer it with the cheapest body we can
        # play, on the post in front of whichever tower it is walking at -- the post never
        # moves, so unlike the defender there is nothing to bait out of position.
        incoming = rows <= OWN_HALF
        if np.any(incoming) and bodies:
            threat_value = float(np.sum(value[incoming]))
            if threat_value >= defend_at:
                col = LANES[0] if np.mean(cols[incoming]) < ARENA_W / 2 else LANES[1]
                slot, _ = min(bodies, key=lambda p: Card(p[1]).elixir)
                return slot, POST_ROW, col

        # Nothing to punish and nothing to answer. Send one card over -- one, never a
        # pair: this script exists to be different from the one that combines.
        if elixir >= chip_at and bodies:
            slot, _ = min(bodies, key=lambda p: Card(p[1]).elixir)
            col = LANES[len(rows) % 2]
            return slot, BRIDGE_ROW, col
        return 0, 0, 0

    return strategy
