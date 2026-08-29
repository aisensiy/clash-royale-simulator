# 参照物（基准 checkpoint）

**当前参照物：`reference.zip` = `era5_flat_plain_final.zip`**（联合动作空间轮的
plain 对照臂 final，12M steps）

来源：OpenBayes `clash-royale-rl/2` job `/output/flat_ab/plain_final.zip`。
编码：MaskablePPO + rich obs + count channels + factorised 动作空间，单帧。

2026-08-30 绝对棋力天梯第一（1076 Elo，锚定 rusher=700），见
`ideas/absolute-ladder-2026-08-30.md`。

## 规则

1. `ladder/best.zip`（旧"冠军"，天梯实测仅 752 Elo）**已退役**，不再作为
   进步与否的参照物。
2. 以后判断"这轮训练有没有进步"，看新模型在 `elo.py` 天梯上对
   `reference.zip` **及三个锚点**的绝对分，而不是 head-to-head 胜率——
   同水平模型互打胜负参半是噪声，不是退步。
3. 若某轮 final 在天梯上稳定超过 `reference.zip` 50 分以上（约需
   每对阵 100 局确认），将参照物更新为该轮模型，并在此记录更换时间与
   两代分数。
4. 天梯锚点（idle/random/rusher）永远固定，不要动。

## 更新历史

| 日期 | 参照物 | Elo | 备注 |
|---|---|---|---|
| 2026-08-30 | era5_flat_plain_final（本文件 reference.zip） | 1076 | 首次设立；取代 ladder/best.zip（752） |
