import argparse, socket, json, threading, time
from battle import BattleState, Building
from player import PlayerState
from arena import Position

TICK_RATE = 60
DT = 1.0 / TICK_RATE
AI_DECISION_TICKS = 30  # the agent decides twice a second, exactly as in training

class GameServer:
    def __init__(self, host='10.235.130.132', port=9999, ai_checkpoint=None, deterministic=False):
        # With a checkpoint the server needs only one human: it plays player 1 itself,
        # driving the same CREnv used for training so the observation the agent sees here
        # is the one it was trained on rather than a re-implementation of it.
        self.ai_checkpoint = ai_checkpoint
        self.deterministic = deterministic
        self.ai_model = None
        self.ai_env = None
        self.n_players = 1 if ai_checkpoint else 2
        self.host, self.port = host, port
        self.clients = []      # list of (conn, player_id)
        self.inputs = [[], []] # pending inputs per player
        self.lock = threading.Lock()
        self.battle = None
        self.decks = [None, None]

    def send(self, conn, msg):
        data = (json.dumps(msg) + '\n').encode()
        conn.sendall(data)

    def broadcast(self, msg):
        for conn, _ in self.clients:
            try: self.send(conn, msg)
            except: pass

    def get_state(self):
        return {
            'time': self.battle.time,
            'game_over': self.battle.game_over,
            'winner': self.battle.winner,
            'elixir': [p.elixir for p in self.battle.players],
            'entities': [
                e.to_dict()
                for e in self.battle.entities.values() if e.is_alive
            ],
            'hands': [
                self.battle.players[0].cycle,
                self.battle.players[1].cycle
            ]
        }

    def setup_ai(self):
        """Load the checkpoint and build the env wrapper that produces its observation."""
        from stable_baselines3 import PPO
        self.attach_ai(PPO.load(self.ai_checkpoint, device="cpu"))
        print(f"AI loaded from {self.ai_checkpoint} "
              f"(rich observation: {self.ai_env.rich_obs})", flush=True)

    def attach_ai(self, model):
        """Point a loaded policy at this server's battle.

        The env is the training env with its own battle replaced, so the agent sees the
        observation it was trained on rather than a second implementation of it that
        could drift.
        """
        from environment import CREnv, count_obs_for, rich_obs_for
        self.ai_model = model
        self.ai_env = CREnv(opponent_model=None, learner_player=1,
                            rich_obs=rich_obs_for(model),
                            count_obs=count_obs_for(model))
        self.ai_env.battle = self.battle
        self.ai_env.learner = 1
        self.ai_env._plays = [[], []]

    def ai_step(self):
        obs = self.ai_env.observe(1)
        action, _ = self.ai_model.predict(obs, deterministic=self.deterministic)
        self.ai_env.deploy(1, action)

    def handle_client(self, conn, player_id):
        buf = ''
        while True:
            try:
                buf += conn.recv(4096).decode()
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    msg = json.loads(line)
                    if msg['type'] == 'deck':
                        self.decks[player_id] = msg['cards']
                        continue
                    with self.lock:
                        self.inputs[player_id].append(msg)
            except socket.timeout:
                continue
            except:
                break

    def run(self):
        # Wait for 2 players
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((self.host, self.port))
        srv.listen(2)
        print(f"Waiting for {self.n_players} player(s) on {self.host}:{self.port}...", flush=True)
        while len(self.clients) < self.n_players:
            conn, addr = srv.accept()
            conn.settimeout(0.01)
            pid = len(self.clients)
            self.clients.append((conn, pid))
            self.send(conn, {'type': 'hello', 'player_id': pid})
            threading.Thread(target=self.handle_client, args=(conn, pid), daemon=True).start()
            print(f"Player {pid} connected from {addr}", flush=True)

        # Start game
        while not all(self.decks[:self.n_players]):
            time.sleep(0.05)

        if self.ai_checkpoint:
            # The agent only knows the eight cards it was trained on, and its hand
            # observation indexes a fixed name table, so an arbitrary deck would either
            # crash it or be meaningless. Both sides play the training deck; the client
            # renders whatever hand the server sends, so the selection screen's choice is
            # simply overridden.
            from environment import DECK
            self.decks = [DECK[:], DECK[:]]
            print(f"Deck overridden for both sides: {' '.join(DECK)}", flush=True)

        self.battle = BattleState(PlayerState(0, self.decks[0], 5), PlayerState(1, self.decks[1], 5))
        if self.ai_checkpoint:
            self.setup_ai()
        self.broadcast({'type': 'start'})
        print("Game started!", flush=True)

        # Main loop
        tick = 0
        while not self.battle.game_over:
            t0 = time.perf_counter()
            with self.lock:
                for pid, input_list in enumerate(self.inputs):
                    for inp in input_list:
                        if inp['type'] == 'deploy':
                            print(inp['card'], pid, int(inp['x'])+0.5, int(inp['y'])+0.5, self.battle.time, flush=True)
                            ok = self.battle.deploy_card(pid, inp['card'],
                                                         Position(int(inp['x'])+0.5, int(inp['y'])+0.5))
                            if ok and self.ai_env is not None:
                                # Otherwise the agent card-counts only itself and treats
                                # the human's cycle as unknown for the whole game.
                                self.ai_env.note_external_play(pid, inp['card'])
                    self.inputs[pid] = []
            if self.ai_model is not None and tick % AI_DECISION_TICKS == 0:
                self.ai_step()
            self.battle.step(DT)
            tick += 1
            self.broadcast({'type': 'state', **self.get_state()})
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, DT - elapsed))

        self.broadcast({'type': 'state', **self.get_state()})
        print(f"Game over! Winner: Player {self.battle.winner}", flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Host a game. With --ai the server plays player 1 from a checkpoint "
                    "and waits for a single human client.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--ai", default=None, metavar="CHECKPOINT",
                    help="path to a .zip; the agent takes player 1 (red, top of the arena)")
    ap.add_argument("--deterministic", action="store_true",
                    help="agent always takes its best action instead of sampling")
    a = ap.parse_args()
    GameServer(a.host, a.port, ai_checkpoint=a.ai, deterministic=a.deterministic).run()
