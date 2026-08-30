# 参照物（基准 checkpoint）

**当前参照物：`reference.zip` = `teachpool_final.zip`**（脚本老师对手池轮，12M steps）

来源：OpenBayes `clash-royale-rl/2` job，`/output/cr_logs/teachpool_final.zip`。
编码：MaskablePPO + rich obs + count channels + factorised 动作空间，单帧；
自博弈池含加权脚本（defender:3, sniper:3, rusher:1, random:1，脚本采样占 35%）。

2026-08-30 验收天梯（每对阵 100 局）：对旧参照物 65/100（p≈0.001），
天梯分 1066 vs 985（+80）。行为画像无塔缩（常用列 8 vs 11，出牌后剩余圣水
略更保守 0.68 vs 0.44）。详见 `ideas/teachpool-ladder-2026-08-30.md`。

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
| 2026-08-30 | era5_flat_plain_final | 1076 | 首次设立；取代 ladder/best.zip（752） |
| 2026-08-30 | teachpool_final（本文件 reference.zip） | +80 vs 前任（同梯内） | 对前任 65/100；天梯曲线 6M→9M→final 仍在上升，后续值得延长训练 |
