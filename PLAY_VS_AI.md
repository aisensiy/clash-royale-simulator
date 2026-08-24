# 和训练出来的智能体对战

服务端跑模拟器和智能体，客户端是本地的 pygame 窗口，两者用一条 TCP 连接通信。
智能体接的是训练时那套 `CREnv`（只是把里面的 battle 换成服务端这一局），所以它在这里
看到的观测和训练时逐字节一致，不存在「线上一套线下一套」的偏差。

## 一、启动服务端（在有 checkpoint 的机器上）

```bash
cd src/clasher_new
python3 server.py --host 0.0.0.0 --port 9999 --ai /output/rich/rich_final.zip
```

- 智能体坐 **1 号位（红方，屏幕上方）**，只等 **1 个**人类连进来。
- 加 `--deterministic` 让它每步取最优动作；默认是采样，和训练时的行为一致。
- **双方牌组会被强制改成训练用的 8 张**：`Knight MiniPekka Arrows Minions Musketeer
  Fireball Giant Archer`。智能体的手牌编码只认这 8 张，换别的牌它要么崩要么在瞎打。

## 二、如果服务端在容器里，先开隧道

OpenBayes 只对外暴露 HTTP 端口，裸 TCP 要走 SSH 隧道。在**本地**执行：

```bash
ssh -N -L 9999:127.0.0.1:9999 -p <容器的ssh端口> root@ssh.openbayes.com
```

保持这个窗口开着，之后客户端连 `127.0.0.1:9999` 就行。

## 三、启动客户端（本地）

客户端只要 `pygame numpy fastcore` 三个包，**不需要 torch**：

```bash
python3 -m venv ~/.venvs/cr && ~/.venvs/cr/bin/pip install pygame numpy fastcore
cd src/clasher_new/client_side
~/.venvs/cr/bin/python client.py --host 127.0.0.1 --port 9999 --deck training
```

`--deck training` 直接用智能体认识的那 8 张牌，跳过选牌界面。不加 `--deck` 会打开选牌
界面，但对战 `--ai` 时选了也会被服务端覆盖掉。不加 `--host` 会弹一个输入 IP 的界面。

操作：从下方手牌**拖到场地**放牌。圣水不够的牌拖不动。

## 四、一局是怎么跑的

- 模拟器按真实时间 60 帧/秒推进，一局最多 300 秒。
- 智能体**每 30 帧（半秒）**决策一次，和训练时的决策频率相同。
- 你出的每一张牌都会喂进它的数牌器（`note_external_play`），所以它知道你打过什么、
  你的循环转到哪儿了——和一个会数牌的人类掌握的信息一样多，不多也不少。

## 五、已知的限制

- 只有这 8 张牌，没有法术之外的建筑、没有升级等级差。
- 竞技场本身对红方（智能体这一侧）有残余的偏向，见 `tests/test_arena_symmetry.py` 里
  7 个标记为 xfail 的用例。也就是说你输了不全是技不如人。
- 服务端一局打完就退出，要再来一局得重启服务端。
