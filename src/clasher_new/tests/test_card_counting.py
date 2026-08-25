"""The opponent-hand belief and the new context scalars.

The claim under test is a soundness one: the belief is built only from cards the
opponent has been *seen* playing, and it never asserts something a spectator could not
have worked out. A test that only checked "it converges to the true hand" would pass for
an implementation that simply read the opponent's cycle.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ARENA_H, ARENA_W, CREnv, DECK, N_CONTEXT, random_strategy
from agents import make_rusher


def rich_env(**kwargs):
    env = CREnv(opponent_model=random_strategy, rich_obs=True, **kwargs)
    env.reset(seed=7)
    return env


def force_play(env, player_id, card_name):
    """Put `card_name` at the front of the player's hand and play it.

    Returns True if the deploy went through. The point is to drive the cycle, not to
    exercise placement rules, so the card goes somewhere always legal.
    """
    p = env.battle.players[player_id]
    p.cycle.remove(card_name)
    p.cycle.insert(0, card_name)
    p.elixir = 10.0
    return env.deploy(player_id, (1, 8, 9))


def true_hand_mask(env, player_id):
    hand = env.battle.players[player_id].cycle[:4]
    return np.array([1.0 if c in hand else 0.0 for c in DECK], dtype=np.float32)


def test_opening_belief_is_uniform():
    env = rich_env()
    for observer in (0, 1):
        belief = env._opp_hand_belief(observer)
        assert np.allclose(belief, 0.5)
        # Four of the eight cards are in hand, whichever four they are.
        assert belief.sum() == pytest.approx(4.0)


@pytest.mark.parametrize("n_played", [1, 2, 3, 4, 5, 6, 7, 8])
def test_belief_is_sound_after_n_plays(n_played):
    """Certainty is only ever asserted when it is deducible, and it is always right."""
    env = rich_env()
    opponent = 1
    for card in DECK[:n_played]:
        assert force_play(env, opponent, card)
    belief = env._opp_hand_belief(observer=0)
    truth = true_hand_mask(env, opponent)

    assert belief.sum() == pytest.approx(4.0)
    for i, card in enumerate(DECK):
        if belief[i] == 1.0:
            assert truth[i] == 1.0, f"claimed {card} is in hand but it is not"
        if belief[i] == 0.0:
            assert truth[i] == 0.0, f"ruled out {card} but it is in hand"

    unseen = 8 - n_played
    if unseen >= 4:
        # Nothing is decided yet: every card the opponent has not played is equally likely.
        assert np.allclose(np.unique(belief), [0.0, 4.0 / unseen])
    else:
        # Every position is pinned down, so the belief is the hand itself.
        assert np.array_equal(belief, truth)


def test_belief_stays_uncertain_while_cards_are_unseen():
    """Three plays is not enough to know the hand -- and the belief must not pretend."""
    env = rich_env()
    for card in DECK[:3]:
        assert force_play(env, 1, card)
    belief = env._opp_hand_belief(observer=0)
    assert ((belief > 0) & (belief < 1)).sum() == 5


def test_replaying_a_card_moves_it_back_to_the_end():
    """A card played twice is at the back, not still where its first play left it."""
    env = rich_env()
    for card in DECK[:5]:
        assert force_play(env, 1, card)
    assert force_play(env, 1, DECK[0])
    assert env._known_cycle_tail(1) == DECK[1:5] + [DECK[0]]
    assert np.array_equal(env._opp_hand_belief(0), true_hand_mask(env, 1))


def test_each_side_only_counts_the_other_sides_plays():
    env = rich_env()
    for card in DECK[:4]:
        assert force_play(env, 0, card)
    # Player 1 has watched four plays and knows player 0's hand exactly...
    assert np.array_equal(env._opp_hand_belief(1), true_hand_mask(env, 0))
    # ...while player 0 has seen nothing and still knows nothing.
    assert np.allclose(env._opp_hand_belief(0), 0.5)


def test_plays_are_forgotten_between_episodes():
    env = rich_env()
    for card in DECK[:4]:
        assert force_play(env, 1, card)
    env.reset(seed=8)
    assert env._plays == [[], []]
    assert np.allclose(env._opp_hand_belief(0), 0.5)


def test_failed_deploys_leak_nothing():
    """A card the opponent could not afford was never revealed and must not be counted."""
    env = rich_env()
    p = env.battle.players[1]
    p.elixir = 0.0
    assert env.deploy(1, (1, 8, 9)) is False
    assert env._plays[1] == []
    assert np.allclose(env._opp_hand_belief(0), 0.5)


def test_context_reports_the_opponent_not_the_observer():
    env = rich_env()
    env.battle.players[0].elixir = 2.0
    env.battle.players[1].elixir = 9.0
    assert env._context(0)[0] == pytest.approx(0.9)
    assert env._context(1)[0] == pytest.approx(0.2)


def test_context_tracks_the_clock_and_the_elixir_tier():
    env = rich_env()
    for t, tier, sudden in [(0.0, 0.0, 0.0), (150.0, 0.5, 0.0),
                            (200.0, 0.5, 1.0), (250.0, 1.0, 1.0)]:
        env.battle.time = t
        ctx = env._context(0)
        assert ctx[1] == pytest.approx(min(t / 300.0, 1.0))
        assert ctx[2] == pytest.approx(tier)
        assert ctx[3] == pytest.approx(sudden)


def test_context_reports_crowns_and_the_weakest_tower():
    env = rich_env()
    foe = env.battle.players[1]
    # update_player_hp() copies tower HP back off the entities every step, so the damage
    # has to be done to the entity, not to the player's mirror of it.
    env.battle.entities[1].hp = 0
    env.battle.entities[2].hp = 500
    env.battle.step(1 / 60)
    ctx = env._context(0)
    assert foe.get_crown_count() == 1
    assert ctx[5] == pytest.approx(1 / 3)
    assert ctx[7] == pytest.approx(500 / 4824, abs=1e-4)


def test_rich_observation_matches_the_declared_space():
    env = rich_env()
    obs, _ = env.reset(seed=3)
    assert set(obs) == {"grid", "hand", "elixir", "context", "opp_hand"}
    assert obs["context"].shape == (N_CONTEXT,)
    assert obs["opp_hand"].shape == (len(DECK),)
    for key in ("context", "opp_hand"):
        space = env.observation_space[key]
        assert space.contains(obs[key]), f"{key} outside its declared box"


def test_plain_observation_is_unchanged_by_default():
    env = CREnv(opponent_model=random_strategy)
    obs, _ = env.reset(seed=3)
    assert set(obs) == {"grid", "hand", "elixir"}
    assert set(env.observation_space.spaces) == {"grid", "hand", "elixir"}


def test_dmg_scale_only_scales_the_shaping_terms():
    """Same game, two weights: the damage part moves, the crown and win parts do not."""
    rewards = {}
    for scale in (1.0, 0.25):
        # A fresh rusher each time: it carries its own Random(seed), so both runs see
        # exactly the same game. `random_strategy` draws from the global module and would
        # make the two totals incomparable.
        env = CREnv(opponent_model=make_rusher(0), dmg_scale=scale)
        env.reset(seed=11)
        total = 0.0
        done = False
        while not done:
            _, r, done, _, _ = env.step((0, 0, 0))
            total += r
        rewards[scale] = total
    # Doing nothing against a random opponent takes damage and deals none, so the
    # shaping is strictly negative and quartering it must move the total up.
    assert rewards[0.25] > rewards[1.0]


def test_each_side_can_be_served_a_different_observation():
    """A rich checkpoint has to be able to face one trained without the extra inputs."""
    env = CREnv(opponent_model=random_strategy, learner_player=0,
                rich_obs=True, opponent_rich_obs=False)
    env.reset(seed=5)
    assert set(env.observe(0)) == {"grid", "hand", "elixir", "context", "opp_hand"}
    assert set(env.observe(1)) == {"grid", "hand", "elixir"}
    # It follows the learner, not the player id: swapping sides swaps who gets what.
    env.learner_player = 1
    env.reset(seed=5)
    assert set(env.observe(1)) == {"grid", "hand", "elixir", "context", "opp_hand"}
    assert set(env.observe(0)) == {"grid", "hand", "elixir"}


def test_pinning_the_side_between_episodes_takes_effect():
    """`env.learner_player = k` then reset is how every eval tool alternates sides."""
    env = CREnv(opponent_model=random_strategy)
    for side in (1, 0, 1):
        env.learner_player = side
        env.reset(seed=2)
        assert env.learner == side
    # Back to None means a fresh draw each episode, which is what training uses.
    env.learner_player = None
    sides = set()
    for seed in range(30):
        env.reset(seed=seed)
        sides.add(env.learner)
    assert sides == {0, 1}
