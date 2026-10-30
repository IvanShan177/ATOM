# DSV4.1 owner attention：本机完整 C256 A/B/A/B

2026-10-30，mi355-gpu-41。保留 FP8 owner / retained-prefill 实现后，A0 去复制加受限 H64 的合并输出吞吐增加 **5.966%**。两对均正向，但幅度分别为8.623%和3.439%；本次是性能样本，不是稳定性能或整网精度验收。

| 顺序 | Attention | 输出 tok/s，四卡合计 | 正式秒数 | 平均 TPOT ms | P99 TPOT ms |
|---|---|---:|---:|---:|---:|
| A1 | native | 6273.750 | 375.288 | 38.384 | 70.295 |
| B1 | A0 + 受限 H64 | 6814.736 | 345.496 | 35.396 | 65.285 |
| A2 | native | 6263.668 | 375.892 | 38.510 | 70.009 |
| B2 | A0 + 受限 H64 | 6479.089 | 363.394 | 37.347 | 71.115 |

合并按总输出量/总正式秒数计算：**6268.705 → 6642.675 tok/s，1.059657×**；平均TPOT 38.447→36.371ms，降低5.398%。A重复漂移−0.161%，B重复漂移−4.925%，后者原因尚未定位。B2相对A2的P99 TPOT增加1.580%，尾延迟未一致改善。单次GPU频率快照不足以解释服务波动。

配方与公开 ATOM random dashboard C256 测点一致：TP4/EP1/DP1，FP8 KV / FP4 index，真实DSpark5，level3/FULL，max-seqs512、batched-tokens16384、attention chunk16384、memory .9、prefix cache。random8192/1024、range .8、C256、seed0、chat template、ignore-eos、request-rate inf。每组512次重复首prompt预热和2560条正式请求，四组10240条正式请求全部成功，逐请求输入/要求输出/实际输出token数一致；每组正式输入18,926,022、输出2,354,462。

每组额外执行8条128/32短请求及512条C256/8192/256图准备请求，不计入正式时间。这会预热图和部分prefix cache，四组协议相同。四次启动独立服务并复用编译缓存；正式日志扫描未发现新增编译/捕获标记。B1在512次配方预热尾部补捕获M256/M128，发生于正式首请求之前。日志扫描不覆盖所有潜在JIT路径。MTP统计为含准备及预热的累计值，不能称正式专属AL。

`ATOM_DSV41_OWNER_FP8_ATTENTION` 默认仍为 `native`；仅显式设为 `a0` 或 `h64` 时启用候选：

- `a0` 保留四个H16及其split策略，直接传Q/scale切片，并写入同一个BF16输出的切片，删除复制和拼接。
- `h64` 仅在本地行数221/384使用现有原生H64 OPUS入口，其他行数走A0。本次服务实际命中M384/H64，未观察到M221。
- 四rank每个40层合同一致；原路径M384/H16主机绑定320次/每rank，候选M384/H64为80次。这是eager/capture绑定数，不是GPU replay计数。

同时收录 owner FFN 的分派与输出合同：TP partial先加shared再RS；完整TP all-reduce输出只截取owner行，避免重复加shared/归约。调用融合后端前等待shared生产完成，缺少bucket保持原路径；绑定BF16融合后端不取消独立native预量化入口的资格。当前所有实际加载层 `comm_fused=False`，此次性能差异没有启用通信融合。FP8 attention也不再提前导入当前AITER已移除依赖的旧BF16辅助路径。

采样源码为 ATOM `d957f46dd87acceb3d90fb011b72c54df2c0bf5e` 加本次收录的六份源码；AITER `c141da6bd10300dfc22da7777c23efc6c7b88712`。源文件哈希、配方、各组结果、实际合同、退出码和证据摘要见 [机器可读记录](dsv41_results_20261030/attention_abab/summary.json)。完整原始证据在本服务器 `../validation/dashboard_abab_20261030/`，未将大型日志加入Git。四组有效AITER CSV一致，原有412份CSV保持原样。

历史 **1.191939×** 是旧时间点的主线5269.811与owner6281.294之比。本次没有重跑当前主线，不能把两次倍率相乘并称为当前实测总加速。局部真实权重验证曾使用确定性投影调度；本次服务使用默认投影，仍未做整网数值等价或GSM8K验收。

本地提交检查：8项owner FFN合同通过，修改源码Black/Ruff和语法检查通过，accuracy catalog schema通过。完整CPU脚本已执行，但分支已有的 `tests/models/deepseek_v41/test_row_sharded_mhc.py` 在无AITER环境导入时报collection error；该文件不存在于主线基点124c3b0d1，不归因为主线问题。全库Black另有10个与本次修改无关、和提交前HEAD字节一致的格式问题；全库Ruff输出426条诊断，本次修改源码无诊断。以上不记为完整CI通过；本地检查原始日志及清单放在 `../validation/commit_plan_20261030/`，当前无关联PR，未触发远端CI。
