"""Loading an opponent -- a checkpoint or a script -- without being told what it is.

Every evaluation tool used to take a `--masked` flag that applied to both sides at once,
which makes a masked run and an unmasked run impossible to play against each other. The
checkpoint already records which algorithm wrote it and which observation keys its
network has input layers for, so nothing has to be declared on the command line.

Agents returned here are callables `act(observation, masks=None)`. They carry two
attributes the caller needs in order to build a matching environment:

    act.rich_obs   this side wants the clock/crowns/opponent-elixir/card-count inputs
    act.masked     this side must be handed `env.action_masks(player)` every decision
"""
import json
import os
import random
import zipfile

from environment import N_SLOTS, random_strategy, rich_obs_for


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


SCRIPTS = {"idle": idle_strategy, "random": random_strategy, "rusher": make_rusher()}


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
        act = SCRIPTS[spec]
        wrapped = lambda obs, masks=None: act(obs)
        wrapped.rich_obs = False
        wrapped.masked = False
        wrapped.name = spec
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
