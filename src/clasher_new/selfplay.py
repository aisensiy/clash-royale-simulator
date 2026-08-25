"""Self-play opponents: a pool of past snapshots plus a few fixed scripts.

The pool deals in file paths and labels only. Loading a checkpoint into a policy is
deferred to a `loader` callable so the sampling and pruning rules can be tested without
torch, and so each environment worker controls when it pays the load cost.
"""
import os
import random
import re

SNAPSHOT_RE = re.compile(r"^snapshot_(\d+)\.zip$")

LATEST = "latest"
HISTORY = "history"
SCRIPT = "script"


class OpponentPool:
    """Snapshots of past versions of the agent, plus the scripted opponents.

    Sampling mixes three sources. The scripted share matters most: scripts never get
    better or worse, so the win rate against them is the only signal that tells
    "both sides improved" apart from "both sides decayed together".
    """

    def __init__(self, directory, max_snapshots=8,
                 p_latest=0.45, p_history=0.40, p_script=0.15, script_names=("random", "rusher")):
        total = p_latest + p_history + p_script
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"sampling shares must sum to 1, got {total}")
        if max_snapshots < 2:
            raise ValueError("max_snapshots must leave room for a latest and a history entry")
        self.directory = directory
        self.max_snapshots = max_snapshots
        self.p_latest = p_latest
        self.p_history = p_history
        self.p_script = p_script
        self.script_names = tuple(script_names)

    # ---------------------------------------------------------------- snapshots

    def snapshot_paths(self):
        """Every snapshot on disk, oldest first."""
        if not os.path.isdir(self.directory):
            return []
        found = []
        for name in os.listdir(self.directory):
            m = SNAPSHOT_RE.match(name)
            if m:
                found.append((int(m.group(1)), os.path.join(self.directory, name)))
        return [path for _, path in sorted(found)]

    def add(self, model, step):
        """Save the current policy as a snapshot, then prune back to `max_snapshots`."""
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"snapshot_{step:012d}.zip")
        model.save(path)
        for stale in self.prune_list(self.snapshot_paths()):
            os.remove(stale)
        return path

    def prune_list(self, paths):
        """Which snapshots to delete so that what remains spans the whole history.

        Keeping only the most recent K would leave the agent training against a narrow
        band of near-identical versions, which is how self-play starts going in circles.
        The newest is always kept (it is the main training partner) and so is the oldest
        (it is the cheapest check that we have not drifted somewhere strictly worse).
        """
        if len(paths) <= self.max_snapshots:
            return []
        keep = {0, len(paths) - 1}
        middle_slots = self.max_snapshots - len(keep)
        if middle_slots > 0:
            span = len(paths) - 2
            for i in range(middle_slots):
                keep.add(1 + round(i * (span - 1) / max(middle_slots - 1, 1)))
        return [p for i, p in enumerate(paths) if i not in keep]

    # ---------------------------------------------------------------- sampling

    def sample(self, rng=random):
        """Pick an opponent. Returns (kind, label, path_or_script_name).

        Falls back to scripts while the pool is still empty, which is the whole of the
        first snapshot interval.
        """
        paths = self.snapshot_paths()
        if not paths:
            return SCRIPT, "script:" + self._pick_script(rng), self._last_script

        roll = rng.random()
        if roll < self.p_script:
            name = self._pick_script(rng)
            return SCRIPT, f"script:{name}", name
        if roll < self.p_script + self.p_latest or len(paths) == 1:
            return LATEST, "latest", paths[-1]
        return HISTORY, "history", rng.choice(paths[:-1])

    def _pick_script(self, rng):
        self._last_script = rng.choice(self.script_names)
        return self._last_script


class PooledOpponent:
    """The callable an environment uses as its opponent, backed by the pool.

    Holds exactly one loaded policy at a time and only re-samples every
    `refresh_every` episodes. With 128 workers each sampling independently there is
    plenty of variety on the board at any moment, while memory stays at one model per
    worker instead of the whole pool per worker.
    """

    def __init__(self, pool, scripts, algo, refresh_every=10, seed=0, device="cpu",
                 masked=False):
        self.pool = pool
        self.scripts = scripts          # name -> callable(observation) -> action
        self.algo = algo                # PPO or MaskablePPO
        self._masked = masked
        self.refresh_every = refresh_every
        self.device = device
        self.rng = random.Random(seed)
        self.label = "script:none"
        self._episodes = 0
        self._policy = None
        self._policy_path = None
        self._script = None
        self._resample()

    @property
    def masked(self):
        """Whether the environment should hand this opponent an action mask.

        A MaskablePPO policy scores invalid actions with weights that were never
        trained, so sampling it without a mask produces near-noise -- under self-play
        that turns the whole opponent pool into sandbags. Scripts want no mask even in a
        masked run, so this follows whichever opponent is currently loaded.
        """
        return self._masked and self._script is None

    def on_episode_start(self):
        # Scripts may carry state between decisions; forward the hook so a half-finished
        # push does not leak into the next game.
        if self._script is not None and hasattr(self._script, "on_episode_start"):
            self._script.on_episode_start()
        self._episodes += 1
        if self._episodes % self.refresh_every == 0:
            self._resample()

    def _resample(self):
        kind, label, target = self.pool.sample(self.rng)
        self.label = label
        if kind == SCRIPT:
            self._script = self.scripts[target]
            return
        self._script = None
        if target != self._policy_path:
            self._policy = self.algo.load(target, device=self.device)
            self._policy_path = target

    def __call__(self, observation, action_masks=None):
        if self._script is not None:
            return self._script(observation)
        if self._masked:
            action, _ = self._policy.predict(observation, deterministic=False,
                                             action_masks=action_masks)
        else:
            action, _ = self._policy.predict(observation, deterministic=False)
        return action
