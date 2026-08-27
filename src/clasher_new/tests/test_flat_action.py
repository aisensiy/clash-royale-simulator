"""A placement is one decision, so the policy has to be able to score it as one.

`MultiDiscrete([N_SLOTS, ARENA_H, ARENA_W])` makes the row and the column two
independent draws, and the distribution over cells is their outer product. Wanting either
(3,5) or (20,12) therefore also means wanting (3,12) and (20,5), and no amount of
training removes that -- it is the shape of the output, not the weights in it. The mask
has the same disease: sb3-contrib takes one mask per dimension, so the most it can say is
"some cell in row 3 is legal".

These pin the joint encoding that fixes both, and the plumbing around it: the two action
spaces are different shapes, so every place that builds an environment for a checkpoint
has to serve it the one it was trained on.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Position
from environment import (ARENA_H, ARENA_W, CREnv, DECK, N_CELLS, N_FLAT_ACTIONS, N_SLOTS,
                         action_triple, decode_flat_action, encode_flat_action,
                         flat_action_for, random_strategy)


def every_action():
    yield 0, 0, 0
    for slot in range(1, N_SLOTS):
        for y in range(ARENA_H):
            for x in range(ARENA_W):
                yield slot, y, x


def started(flat_action=True, opponent_flat_action=False, elixir=10.0, learner=0):
    """A started game whose opponent passes. The opponent here is a bare lambda that
    answers with a triple, so it is told so: a callable with no opinion of its own
    inherits the learner's encoding, which is right for a self-play snapshot and wrong
    for anything hand-written."""
    env = CREnv(opponent_model=lambda obs: (0, 0, 0), flat_action=flat_action,
                opponent_flat_action=opponent_flat_action, learner_player=learner)
    env.reset(seed=7)
    for player in env.battle.players:
        player.elixir = elixir
    return env


# --------------------------------------------------------------------- the codec

def test_every_action_survives_the_round_trip():
    """2305 outcomes, and an off-by-one anywhere in the packing silently deploys the
    wrong card somewhere else -- nothing crashes, the agent just plays nonsense."""
    seen = set()
    for slot, y, x in every_action():
        index = encode_flat_action(slot, y, x)
        assert 0 <= index < N_FLAT_ACTIONS
        assert index not in seen, f"({slot},{y},{x}) collided at {index}"
        seen.add(index)
        assert decode_flat_action(index) == (slot, y, x)
    assert len(seen) == N_FLAT_ACTIONS


def test_doing_nothing_is_index_zero():
    """Slot 0 means "no card", so the row and column that came with it mean nothing.
    All of them have to land on the same index or the policy has 576 separate ways to
    spell one decision and splits its credit between them."""
    assert encode_flat_action(0, 0, 0) == 0
    assert encode_flat_action(0, 17, 9) == 0
    assert decode_flat_action(0) == (0, 0, 0)


def test_the_layout_is_slot_major_then_row_major():
    """The action head reads its logits off `(B, 4, ARENA_H, ARENA_W)` planes and
    flattens them. That flatten order and `decode_flat_action` are the same convention
    written twice, so this pins them together."""
    cells = np.arange((N_SLOTS - 1) * N_CELLS).reshape(N_SLOTS - 1, ARENA_H, ARENA_W)
    logits = np.concatenate([[-1.0], cells.reshape(-1)])
    for slot, y, x in [(1, 0, 0), (1, 5, 9), (2, 31, 17), (4, 12, 3)]:
        assert logits[encode_flat_action(slot, y, x)] == cells[slot - 1, y, x]


def test_action_triple_passes_a_factorised_action_through():
    assert action_triple(np.array([2, 11, 4]), flat_action=False) == (2, 11, 4)
    assert action_triple(encode_flat_action(2, 11, 4), flat_action=True) == (2, 11, 4)


# --------------------------------------------------------------------- the env

def test_the_action_space_follows_the_encoding():
    assert started(flat_action=True).action_space.n == N_FLAT_ACTIONS
    assert list(started(flat_action=False).action_space.nvec) == [N_SLOTS, ARENA_H, ARENA_W]


def test_the_two_encodings_deploy_the_same_card_in_the_same_place():
    """The point of the change is the shape of the policy's output, not the game. If a
    game plays differently under the two encodings, the A/B measures the wrong thing."""
    flat, factorised = started(flat_action=True), started(flat_action=False)
    slot, y, x = 1, 11, 5
    flat.step(encode_flat_action(slot, y, x))
    factorised.step((slot, y, x))
    placed = [(e.name, round(e.position.x, 3), round(e.position.y, 3))
              for env in (flat, factorised)
              for e in env.battle.entities.values()]
    half = len(placed) // 2
    assert placed[:half] == placed[half:]


def test_a_flat_env_plays_a_whole_game():
    env = CREnv(opponent_model=random_strategy, flat_action=True, learner_player=0)
    env.reset(seed=3)
    done, steps = False, 0
    while not done and steps < 700:
        _, _, done, _, _ = env.step(0)
        steps += 1
    assert done, "the game never ended"


# --------------------------------------------------------------------- the mask

def legal_by_simulation(env, player_id, slot, y, x):
    """Whether the simulator actually accepts this placement, asked by trying it."""
    import copy
    probe = copy.deepcopy(env.battle)
    name = probe.players[player_id].cycle[slot - 1]
    if player_id == 0:
        pos = Position(x + 0.5, y + 0.5)
    else:
        pos = Position(ARENA_W - (x + 0.5), ARENA_H - (y + 0.5))
    return bool(probe.deploy_card(player_id, name, pos))


def test_the_joint_mask_offers_nothing_the_simulator_would_refuse():
    """Exactness is the whole reason for the joint space. Every action the mask leaves
    open has to be one the game accepts -- checked against the simulator itself, not
    against the geometry this module used to build the mask."""
    env = started(elixir=10.0)
    mask = env.action_masks(0)
    assert mask.shape == (N_FLAT_ACTIONS,)
    assert mask[0], "doing nothing has to stay available or sampling can have no answer"
    offered = [decode_flat_action(i) for i in np.nonzero(mask)[0] if i]
    assert offered, "an env with ten elixir offered no placement at all"
    refused = [(s, y, x) for s, y, x in offered
               if not legal_by_simulation(env, 0, s, y, x)]
    assert not refused, f"{len(refused)} illegal cells offered, e.g. {refused[:3]}"


def test_the_joint_mask_hides_nothing_the_simulator_would_accept():
    """The other direction: a mask that is merely conservative would quietly forbid the
    agent from ever playing a legal cell."""
    env = started(elixir=10.0)
    mask = env.action_masks(0)
    missed = [(s, y, x) for s, y, x in every_action()
              if s and not mask[encode_flat_action(s, y, x)]
              and legal_by_simulation(env, 0, s, y, x)]
    assert not missed, f"{len(missed)} legal cells hidden, e.g. {missed[:3]}"


def test_the_factorised_mask_cannot_be_exact_and_the_joint_one_is():
    """The defect this round removes, pinned from both sides. A building's footprint is a
    cell whose row holds legal cells and whose column holds legal cells, so no pair of
    per-dimension masks can exclude it while keeping them."""
    joint, factorised = started(flat_action=True), started(flat_action=False)
    for env in (joint, factorised):
        for player in env.battle.players:
            player.elixir = 10.0
    old = factorised.action_masks(0)
    rows, cols = old[N_SLOTS:N_SLOTS + ARENA_H], old[N_SLOTS + ARENA_H:]
    new = joint.action_masks(0)

    troop_slots = [s for s in range(1, N_SLOTS)
                   if joint.battle.players[0].cycle[s - 1] not in ("Arrows", "Fireball")]
    assert troop_slots, "this hand is all spells, so it proves nothing"
    slot = troop_slots[0]
    leaks = [(y, x) for y in range(ARENA_H) for x in range(ARENA_W)
             if rows[y] and cols[x] and not legal_by_simulation(joint, 0, slot, y, x)]
    assert leaks, "no cell to demonstrate on -- the board geometry changed"
    for y, x in leaks:
        assert not new[encode_flat_action(slot, y, x)], f"({y},{x}) still offered"


@pytest.mark.parametrize("learner", [0, 1])
@pytest.mark.parametrize("towers", [(1, 1), (0, 1), (1, 0), (0, 0)])
def test_the_mask_agrees_with_the_simulator_cell_by_cell(learner, towers):
    """Every card, every cell, both sides, every tower state, against the simulator.

    Red's deploy checks are not a reflection of blue's -- its forward band reaches a row
    deeper, and the tower that opens a lane is named in absolute coordinates, so red's
    egocentric left is gated by the enemy's *right* tower. Sharing one grid between the
    players opened the wrong lane for red the moment a princess tower fell, which cost
    2.5% of its placements and went unseen for four rounds because the per-dimension mask
    was too coarse to show it.
    """
    env = started(elixir=10.0, learner=learner)
    foe = env.battle.players[1 - learner]
    foe.left_tower_hp = 1000.0 if towers[0] else 0.0
    foe.right_tower_hp = 1000.0 if towers[1] else 0.0
    mask = env.action_masks(learner)
    wrong = [(s, y, x) for s, y, x in every_action() if s
             and bool(mask[encode_flat_action(s, y, x)])
             != legal_by_simulation(env, learner, s, y, x)]
    assert not wrong, (f"{len(wrong)} cells disagree as player {learner} with towers "
                       f"{towers}, e.g. {wrong[:4]}")


def test_a_spell_may_go_where_a_troop_may_not():
    """Two cards, two different sets of legal cells, one mask. The factorised encoding
    has to take their union and hand both cards the larger one."""
    env = started(elixir=10.0)
    hand = env.battle.players[0].cycle[:N_SLOTS - 1]
    spells = [i + 1 for i, name in enumerate(hand) if name in ("Arrows", "Fireball")]
    troops = [i + 1 for i, name in enumerate(hand) if name not in ("Arrows", "Fireball")]
    if not spells or not troops:
        pytest.skip("this hand does not hold one of each")
    mask = env.action_masks(0)
    far_side = encode_flat_action(spells[0], ARENA_H - 2, ARENA_W // 2)
    assert mask[far_side], "a spell was refused the opponent's half"
    assert not mask[encode_flat_action(troops[0], ARENA_H - 2, ARENA_W // 2)]


def test_a_player_who_can_afford_nothing_still_has_a_move():
    env = started(elixir=0.0)
    mask = env.action_masks(0)
    assert mask[0] and mask.sum() == 1


# --------------------------------------------------------------------- per side

def test_each_side_gets_the_encoding_it_was_trained_on():
    """A run on the joint space and a run on the factorised one have to be able to play
    each other -- that match is the measurement."""
    env = started(flat_action=True, opponent_flat_action=False)
    env.learner = 0
    assert env.action_masks(0).shape == (N_FLAT_ACTIONS,)
    assert env.action_masks(1).shape == (N_SLOTS + ARENA_H + ARENA_W,)


def test_a_script_opponent_keeps_its_triples_in_a_flat_run():
    """Scripts answer with `(slot, y, x)` whatever the run trains on. Decoding one of
    those as a flat index would deploy a different card somewhere else, and nothing would
    report an error."""
    played = []

    def script(observation):
        return 1, 11, 5

    env = CREnv(opponent_model=script, flat_action=True, learner_player=0)
    env.reset(seed=5)
    assert env.opponent_flat_action is False
    for player in env.battle.players:
        player.elixir = 10.0
    env.step(0)
    deployed = [e for e in env.battle.entities.values()
                if e.player == 1 and e.name in DECK]
    assert deployed, "the script's placement was dropped"


def test_the_pool_opponent_answers_for_whoever_is_loaded():
    """The self-play pool holds snapshots and scripts side by side. A snapshot speaks the
    run's own encoding; a script never does."""
    from selfplay import PooledOpponent

    opponent = PooledOpponent.__new__(PooledOpponent)
    opponent._flat_action = True
    opponent._script = None
    assert opponent.flat_action is True
    opponent._script = lambda obs: (0, 0, 0)
    assert opponent.flat_action is False


# --------------------------------------------------------------------- plumbing

class FakeModel:
    def __init__(self, space):
        self.action_space = space


def test_the_encoding_is_read_off_the_checkpoint():
    import gymnasium as gym
    assert flat_action_for(FakeModel(gym.spaces.Discrete(N_FLAT_ACTIONS)))
    assert not flat_action_for(
        FakeModel(gym.spaces.MultiDiscrete([N_SLOTS, ARENA_H, ARENA_W])))
    assert not flat_action_for(FakeModel(gym.spaces.Discrete(7)))


def test_a_script_is_not_on_the_joint_space():
    from agents import load_agent
    assert load_agent("counter").flat_action is False


def head_and_logits():
    torch = pytest.importorskip("torch")
    from train import CRFeatureExtractor, SpatialActionHead
    env = CREnv(opponent_model=random_strategy, flat_action=True, count_obs=True)
    obs, _ = env.reset(seed=1)
    extractor = CRFeatureExtractor(env.observation_space)
    batch = {k: torch.as_tensor(np.asarray(v)[None]) for k, v in obs.items()}
    head = SpatialActionHead(extractor.features_dim, extractor)
    return torch, head, head(extractor(batch)), batch


def test_the_spatial_head_outputs_one_logit_per_action():
    """And it has to run off the same planes the trunk sees, in the same forward pass."""
    torch, _, logits, _ = head_and_logits()
    assert logits.shape == (1, N_FLAT_ACTIONS)
    assert torch.isfinite(logits).all()


def test_the_head_emits_log_probabilities():
    """The two levels are composed into one vector over the same 2305 outcomes, which
    only reproduces the intended distribution if that vector is already normalised --
    `log_softmax` of a log-probability vector is itself."""
    torch, _, logits, _ = head_and_logits()
    assert torch.logsumexp(logits, dim=1).abs().max() < 1e-4


def test_waiting_starts_at_a_coin_flip_and_not_at_one_in_2305():
    """The defect that cost the first run of this head. With "do nothing" as one outcome
    among 2304 placements, an untrained policy waits about 0.04% of the time, never
    samples patience, and so cannot learn it."""
    torch, _, logits, _ = head_and_logits()
    p_wait = float(logits[0, 0].exp())
    assert 0.3 < p_wait < 0.7, f"opening wait probability {p_wait:.4f}"


def test_the_two_levels_are_independent_of_each_other():
    """Where to play is decided as if the decision to play had already been taken, so
    moving the play/wait split must not reorder placements or change their relative
    weights."""
    torch, head, before, batch = head_and_logits()
    ratio_before = before[0, 1:] - before[0, 1:].logsumexp(0)
    with torch.no_grad():
        head.play.bias.add_(3.0)          # much keener to play
    after = head(head._extractor[0](batch))
    assert float(after[0, 0].exp()) < float(before[0, 0].exp()), "the split did not move"
    ratio_after = after[0, 1:] - after[0, 1:].logsumexp(0)
    assert torch.allclose(ratio_before, ratio_after, atol=1e-5)


def test_masking_moves_probability_onto_waiting():
    """Renormalising across both levels is the behaviour we want: with few placements
    legal, waiting deserves more of the distribution, and with none legal it takes all."""
    torch, _, logits, _ = head_and_logits()
    row = logits[0]

    def wait_probability(mask):
        masked = torch.where(torch.as_tensor(mask), row, torch.full_like(row, -np.inf))
        return float(torch.softmax(masked, dim=0)[0])

    everything = np.ones(N_FLAT_ACTIONS, dtype=bool)
    a_few = np.zeros(N_FLAT_ACTIONS, dtype=bool)
    a_few[0] = True
    a_few[1:20] = True
    nothing_but_waiting = np.zeros(N_FLAT_ACTIONS, dtype=bool)
    nothing_but_waiting[0] = True
    assert wait_probability(everything) < wait_probability(a_few)
    assert wait_probability(nothing_but_waiting) == pytest.approx(1.0)
