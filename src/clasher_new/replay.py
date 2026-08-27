"""Play one game between two agents and record it as a video plus a text play-by-play.

Training happens on a headless box, so the pygame renderer draws to an offscreen
surface and the frames are written to a file instead of a window.

    python3 replay.py --blue /output/pool/snapshot_000020000000.zip \
                      --red  /output/pool/snapshot_000005000000.zip \
                      --out /output/replays/late_vs_early.mp4
"""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from agents import decide, load_agent
from environment import CREnv
CARD_SHORT = {"MiniPekka": "迷你皮卡", "Musketeer": "火枪手", "Minions": "亡灵",
              "Archer": "弓箭手", "Knight": "骑士", "Giant": "巨人",
              "Fireball": "火球", "Arrows": "箭雨"}


def replay_recorded(args, imageio):
    """Re-render a game that actually happened during training.

    No models are loaded: the record carries the decks, the side and every action, and
    nothing in the simulator draws a random number, so this is the same game frame for
    frame.
    """
    import json

    with open(args.from_record) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    record = records[args.index]

    env = CREnv(opponent_model=lambda obs: (0, 0, 0), visualize=True, realtime=False)
    frames, tick = [], [0]
    env.replay_record(record,
                      frame_hook=lambda vis: attach_capture(vis, frames, tick, args.every))

    outcome = {1: "学习方胜", -1: "学习方负", 0: "平局"}[record["outcome"]]
    side = "蓝" if record["learner"] == 0 else "红"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"重放第 {args.index} 局：学习方坐{side}方，对手 {record['opponent']}，{outcome}")
    print(f"共 {len(record['actions'])} 个决策点\n视频 {args.out}（{len(frames)} 帧）")


def attach_capture(visualizer, frames, tick, every):
    """Draw and keep one tick in `every`; painting discarded frames dominated the run."""
    raw_render = visualizer.render_frame

    def render_and_capture():
        if tick[0] % every == 0:
            raw_render()
            frames.append(np.transpose(pygame.surfarray.array3d(visualizer.screen), (1, 0, 2)))
        tick[0] += 1

    visualizer.render_frame = render_and_capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue", help="checkpoint path, or random/idle/rusher")
    ap.add_argument("--red", help="checkpoint path, or random/idle/rusher")
    ap.add_argument("--from-record", help="a .jsonl written by --record-every")
    ap.add_argument("--index", type=int, default=0, help="which episode in that file")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--every", type=int, default=10,
                    help="draw and keep one tick in N; the sim runs 60 ticks a second")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import imageio.v2 as imageio

    if args.from_record:
        return replay_recorded(args, imageio)
    if not (args.blue and args.red):
        raise SystemExit("需要 --blue 和 --red，或者 --from-record")

    blue, red = load_agent(args.blue), load_agent(args.red)
    blue_name, red_name = blue.name, red.name

    # Pin the sides so `--blue` really is the blue player in the video and the log.
    # Each side gets the observation its own checkpoint was trained on; a model handed
    # keys its network has no input layer for dies on the first prediction.
    env = CREnv(opponent_model=red, visualize=True, realtime=False, learner_player=0,
                rich_obs=blue.rich_obs, opponent_rich_obs=red.rich_obs,
                count_obs=blue.count_obs, opponent_count_obs=red.count_obs,
                flat_action=blue.flat_action, opponent_flat_action=red.flat_action)
    obs, _ = env.reset(seed=args.seed)

    frames, log = [], []

    # The 30 ticks between decisions are simulated inside `env.step`, so frames have to
    # be grabbed from the renderer itself rather than from this loop.
    raw_render = env.visualizer.render_frame
    tick = [0]

    def render_and_capture():
        # Only draw on ticks we actually keep. Drawing costs ~13ms and the simulation
        # renders 30 ticks per decision, so painting frames we then discard dominated
        # the whole run.
        if tick[0] % args.every == 0:
            raw_render()
            frames.append(np.transpose(pygame.surfarray.array3d(env.visualizer.screen), (1, 0, 2)))
        tick[0] += 1

    env.visualizer.render_frame = render_and_capture
    hooked = env.battle.deploy_card

    def record_deploy(player_id, card_name, position):
        ok = hooked(player_id, card_name, position)
        if ok:
            t = env.battle.time
            side = "蓝" if player_id == 0 else "红"
            card = CARD_SHORT.get(card_name, card_name)
            log.append(f"{int(t)//60}:{int(t)%60:02d}  {side} {card:<5} → "
                       f"({int(position.x)}, {int(position.y)})  "
                       f"圣水 {env.battle.players[player_id].elixir:.1f}")
        return ok

    env.battle.deploy_card = record_deploy

    crowns = (0, 0)
    step = 0
    done = False
    while not done:
        obs, _, done, _, _ = env.step(decide(blue, obs, env, env.learner))
        # `deploy_card` is rebound on the battle object, which reset() replaces.
        new_crowns = (env.battle.players[1].get_crown_count(),
                      env.battle.players[0].get_crown_count())
        if new_crowns != crowns:
            t = env.battle.time
            log.append(f"{int(t)//60}:{int(t)%60:02d}  ** 皇冠 蓝 {new_crowns[0]} : {new_crowns[1]} 红 **")
            crowns = new_crowns
        step += 1

    winner = env.battle.winner
    verdict = "蓝方胜" if winner == 0 else "红方胜" if winner == 1 else "平局"
    log.append(f"结束：{verdict}（蓝={blue_name} 红={red_name}）")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    text_path = os.path.splitext(args.out)[0] + ".txt"
    with open(text_path, "w") as fh:
        fh.write("\n".join(log) + "\n")

    print("\n".join(log))
    print(f"\n视频 {args.out}（{len(frames)} 帧）\n战报 {text_path}")


if __name__ == "__main__":
    main()
