"""Loading an opponent -- a checkpoint or a script -- without being told what it is.

Every evaluation tool used to take a `--masked` flag that applied to both sides at once,
which makes a masked run and an unmasked run impossible to play against each other. The
checkpoint already records which algorithm wrote it and which observation keys its
network has input layers for, so nothing has to be declared on the command line.

Agents returned here are callables `act(observation, masks=None)`. They carry the
attributes the caller needs in order to build a matching environment:

    act.rich_obs     this side wants the clock/crowns/opponent-elixir/card-count inputs
    act.count_obs    this side wants the two per-cell unit-count channels
    act.flat_action  this side speaks the joint action space, not `(slot, y, x)`
    act.frames       how many stacked frames of history its grid carries
    act.masked       this side must be handed `env.action_masks(player)` every decision
"""
import json
import os
import random
import zipfile

from environment import (N_SLOTS, count_obs_for, flat_action_for, frames_for,
                         random_strategy, rich_obs_for)
from scripts_defender import make_anchor, make_defender
from scripts_counter import make_counter
from scripts_sniper import make_sniper


def idle_strategy(observation):
    """Never plays a card. The true floor: anything that loses to this is broken."""
    return 0, 0, 0


def make_rusher(seed=0):
    """Dumps the cheapest affordable card down one lane as soon as it can afford it.

    This is the 'always attacking, never defending' script the reference plan describes:
    a low bar, but a much harder one than random because it actually applies pressure.
    """
    rng = random.Random(seed)

    def strategy(observation):
        elixir = float(observation['elixir'][0])
        if elixir < 4.0:
            return 0, 0, 0
        slot = rng.randint(1, N_SLOTS - 1)
        return slot, rng.randint(10, 13), rng.choice([4, 5, 12, 13])

    return strategy


# Factories, not instances. `make_defender` keeps state between decisions, and elo.py
# loads both sides of a match by name -- handing the same object to blue and red would
# have them share it.
SCRIPTS = {
    "idle": lambda seed=0: idle_strategy,
    "random": lambda seed=0: random_strategy,
    "rusher": make_rusher,
    "defender": make_defender,
    "anchor": make_anchor,
    "sniper": make_sniper,
    # The held-out ruler -- see scripts_counter.py. Do not add it to the builders in
    # train.py; `sniper` was held out until it was promoted to a training opponent, and
    # this is what replaced it.
    "counter": make_counter,
}


def is_masked_checkpoint(path):
    """True if this zip was written by MaskablePPO.

    Loading it to find out costs a full policy construction, and getting it wrong is
    silent: MaskablePPO.predict without `action_masks` samples freely over actions the
    policy was never trained to score, so a masked checkpoint read as unmasked measures
    as a much weaker player than it is.
    """
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("data").decode())
    return "maskable" in json.dumps(data.get("policy_class", "")).lower()


def load_agent(spec, deterministic=False):
    """A scripted opponent by name, or a checkpoint by path."""
    if spec in SCRIPTS:
        act = SCRIPTS[spec]()
        wrapped = lambda obs, masks=None: act(obs)
        wrapped.rich_obs = False
        # Scripts read fixed channel indices and were tuned against the 15-channel grid.
        # Leaving them on it keeps a ruler comparable with the ratings it gave last round.
        wrapped.count_obs = False
        # Scripts answer with a `(slot, y, x)` triple whatever the run trains on.
        wrapped.flat_action = False
        # They read fixed channel indices, which a stacked grid keeps pointing at the
        # current frame -- but a ruler that changes between rounds is not a ruler.
        wrapped.frames = 1
        wrapped.masked = False
        wrapped.name = spec
        # A stateful script has to be told when a new game starts, or it carries the last
        # one's half-finished push into the next.
        if hasattr(act, "on_episode_start"):
            wrapped.on_episode_start = act.on_episode_start
        return wrapped

    masked = is_masked_checkpoint(spec)
    if masked:
        from sb3_contrib import MaskablePPO as Algo
    else:
        from stable_baselines3 import PPO as Algo
    model = Algo.load(spec, device="cpu")

    if masked:
        def act(obs, masks=None):
            return model.predict(obs, deterministic=deterministic,
                                 action_masks=masks)[0]
    else:
        def act(obs, masks=None):
            return model.predict(obs, deterministic=deterministic)[0]

    act.rich_obs = rich_obs_for(model)
    act.count_obs = count_obs_for(model)
    act.flat_action = flat_action_for(model)
    act.frames = frames_for(model)
    act.masked = masked
    act.name = os.path.basename(spec)
    act.model = model
    return act


def decide(agent, observation, env, player_id):
    """Call `agent`, handing it a mask only if it is a masked policy.

    Building the mask costs a grid operation, so it is skipped for the agents that would
    throw it away -- which is every script and every plain-PPO checkpoint.
    """
    if getattr(agent, "masked", False):
        return agent(observation, env.action_masks(player_id))
    return agent(observation)
