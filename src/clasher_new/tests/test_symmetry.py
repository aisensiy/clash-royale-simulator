"""The two sides must be interchangeable, or self-play optimises the wrong thing.

The learner is always player 0. If the arena, the deploy rules or the tiebreak favour
either side, a self-play win rate reads as "the agent got worse" when nothing about the
agent changed. Playing one frozen policy against a copy of itself isolates that.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mirror_shard(args):
    model_path, games, seed = args
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from environment import CREnv

    model = PPO.load(model_path, device="cpu")
    act = lambda obs: model.predict(obs, deterministic=False)[0]
    env = CREnv(opponent_model=act)
    blue = red = draw = 0
    for i in range(games):
        obs, _ = env.reset(seed=seed * 10_000 + i)
        done = False
        while not done:
            obs, _, done, _, _ = env.step(act(obs))
        winner = env.battle.winner
        blue += winner == 0
        red += winner == 1
        draw += winner is None
    return blue, red, draw


def mirror_match(model_path, games=200, workers=20):
    shard = max(1, games // workers)
    jobs = [(model_path, shard, s) for s in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_mirror_shard, jobs))
    return (sum(b for b, _, _ in out), sum(r for _, r, _ in out), sum(d for _, _, d in out))


if __name__ == "__main__":
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    blue, red, draw = mirror_match(path, games=n)
    total = blue + red + draw
    print(f"同一个模型自己打自己 {total} 局：蓝 {blue} 胜 / 红 {red} 胜 / {draw} 平")
    print(f"蓝方胜率 {blue/total:.1%}  —— 对称的话应该在 50% 附近")
