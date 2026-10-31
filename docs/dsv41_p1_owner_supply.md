# P1 owner 一次供料与 FFN Mega-MHC 量化

本轮在 `0653b59a7` 上新增 owner FFN 供料候选，基线为已有 A0/H64 路径。**完整 C256 四臂合并 −3.02%，未通过稳定整网收益验收；保持默认关闭。**

## 使用与范围

`ATOM_DSV41_OWNER_FFN_SUPPLY` 默认 `native`，可选 `once` 或 `mhc`：

- `once`：owner 本地量化一次，汇集 FP8、E8M0 scales 和 gate logits，供 shared/routed FFN 共用。
- `mhc`：由 FFN Mega-MHC 生成同一份量化值和 scales，同时保留 router 使用的 BF16。

当前只启用 owner M384（全局 M1536），要求 TP4/EP1/DP1、native W4A8 group32、H5120、384 experts、top-6，以及已验证的全局 router kernel。hash/image routing、其他容量、未知 router 调度或绑定 BF16 通信 backend 时走原路径。`mhc` 在没有 HIP packed mHC 权重时使用独立量化。

本轮实测专家中间维为有效 576 / 物理 640。完整模型服务固定 `ATOM_DSV41_OWNER_FP8_ATTENTION=h64`，该开关与本次供料开关独立。

## 数值与完整窗口

真实 layer8 FFN/mHC 和 layer9 attention/mHC 权重，三组构造输入、四 rank。窗口包括 `prepare_ffn`、mHC/norm、量化/AG/router、shared/routed、combine/RS、`finish_ffn`、下一层 delayed mHC/norm/QKV/qk_norm。

三次独立进程，每组输入 ABBA、每批 128 次 graph replay，按最慢 rank 计时：

| 候选 | 三次完整窗口速度比 |
|---|---|
| 一次供料 | 1.04567× / 1.04572× / 1.04113× |
| 一次供料 + FFN Mega-MHC 量化 | 1.05847× / 1.05165× / 1.05598× |

最后一轮 156 项输入/路由/mHC 比较全部逐字节一致；84 项 G1 输入、有效排序 scales/IDs/routes 比较全部逐字节一致。三臂均保持 native G1/G2 调度：

- G1：`flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w3_bnt0_gui`
- G2：`flydsl_moe2_afp8_wfp4_bf16_t64x128x128_atomic`

最终 RS 输出仍有 native atomic 自比较波动：最后一轮 native-self relative L2 最大 0.000958，候选 graph 对 native 最大 0.000987；下一层 Q-lora 分别为 0.001830 / 0.002000。不能据此宣称默认整网精度通过。排序缓冲区未写入的容量尾部不参与有效数值比较。

## 短服务 A/B

两臂相同当前源码、H64、C256、8K 输入、256 输出、256 warmups、512 正式请求、真实 DSpark5 和 FULL graph。各 512 条正式请求全部成功。

| 指标 | A：native 供料 | B：Mega-MHC 供料 |
|---|---:|---:|
| 输出吞吐 tok/s | 2661.133 | 2726.293 |
| 总 token 吞吐 tok/s | 88220.256 | 90380.420 |
| 平均 TPOT ms | 71.0368 | 70.0146 |
| 正式 AL | 2.150114 | 2.142764 |
| owner 全局实际行数 | 330690 | 332730 |
| M384 replay | 215 | 218 |

样本吞吐 **+2.4486%**。B 正式 AL 稍低、owner 实际行数稍多；两臂 scheduler/draft/容量分布仍不同。这是一对短服务筛选，尚不能作为完整 dashboard 配方的稳定整网收益。

四 rank 的 layer0–39 均记录 M384 fused-quant 的 eager/capture 调用；B 正式区间有 218 次对应 owner graph replay，0 次新捕获。其他容量回退。A 供料调用记录为空。

后续完整配方结果见下一节。当前候选未通过稳定整网收益验收，不触发完整 GSM8K；默认数值自比较和整网精度仍保留限制。

## 完整 C256 A/B/A/B：未通过稳定收益验收

用户继续授权后，完成四次独立重启的完整 dashboard 配方：8K 输入、1K 输出、C256、2560 正式请求、512 warmups、random range 0.8、seed 0、真实 DSpark5、FULL graph。四臂共 10240 条正式请求全部成功，input/output token 数完全相同。两臂固定既有 H64，B 只增加 P1 `mhc` 供料。

| 臂 | 输出吞吐 tok/s | 正式耗时 s | 正式 AL | owner 实际全局行数 | M384 replay | 正式新捕获 |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 6753.933 | 348.6061 | 2.467321 | 5736972 | 3728 | 0 |
| B1 | 6835.311 | 344.4557 | 2.488065 | 5692560 | 3733 | 1 |
| A2 | 6747.229 | 348.9524 | 2.469649 | 5734068 | 3742 | 0 |
| B2 | 6281.154 | 374.8454 | 2.470566 | 5733312 | 3728 | 0 |

按两次正式输出 token 总量 / 两次正式耗时合并：**A 6750.579 → B 6546.526 tok/s，−3.0227%**。总 token 吞吐为 61014.190 → 59169.884 tok/s，比例相同。两对独立差异为 **+1.2049% / −6.9076%**。A2/A1 漂移 **−0.0993%**，B2/B1 漂移 **−8.1073%**。

B1 的正式 AL 比 A1 高约 0.84%，并在正式尾部新捕获 owner M16（全局 M64）回退图。512 条常规 warmups 未保证覆盖全部尾部容量，当前 phase 文件保持 warmup，允许此捕获；这是实际测量限制。原始时间完整保留，没有根据日志事后减掉捕获成本。

更关键的是 B2/A2：正式 AL 仅 +0.0372%，owner 实际行数 −0.0132%，容量行数基本相同，双方均无正式新图捕获；B2 平均 TPOT 却 +7.4642%、TTFT +4.9702%。因此当前计数不支持用 AL、owner 执行行数或新图捕获解释该退化。B1/B2 的 AL 变化约 −0.70%、实际行数 +0.72%，也不足以单独确定 −8.11% 吞吐漂移的根因。不能据此反推历史 B1/B2 的漂移原因。

四 rank 都核验了行数分区与全局计数。B1/B2 均有 40 层 M384 fused-quant 的预热/capture 调用，正式区间供料 Python 调用没有增加；其他 owner 容量保持 native。B2 运行中 AMD-SMI 报出的 7 个 host PID 全部对应本容器 API/engine/TP 进程，映射保存在 `full_b2/host_process_map.txt`；该快照不是完整的硬件频率/带宽归因。

**决策：P1 保留为默认关闭的实验候选，不计入已实现的稳定整网收益。** 完整单层收益和短服务 +2.45% 均被保留，但不得替代这轮完整 A/B/A/B 的负结果。暂停扩大 P1 形状和叠加新融合。

下一步优先做同一真实服务轨迹的多层 FFN 连续窗口，记录路由分布、排序后的有效工作量、实际 G1/G2 调度及 AG/RS 生命周期，定位单层获益与完整服务退化之间的差异；同时补齐正式区间前所有 owner 容量的图覆盖检查，利用已有 seal 合同拒绝正式新捕获。修正实验协议后先做短复测，有稳定结果再安排新的完整四臂，避免反复用整网碰运气。

原始证据在 `full_a1`、`full_b1`、`full_a2`、`full_b2`；汇总见 `FULL_SUMMARY.json`，配方一致性见 `FULL_PROTOCOL_AUDIT.json`。

## 证据与后续

库外证据目录为 `validation/p1_20261030/`，含 `RESULTS.md`、`SUMMARY.json`、冻结的运行源码、完整原始计时和分派记录。服务筛选结果和清理状态以该目录报告为准。

修改源码 AST/Black/Ruff、8 项新供料准入合同和 8 项原 owner FFN/RS 合同通过。完整 CPU CI 未在本轮复跑；历史 collection/全库 lint 限制仍在。完整 dashboard A/B/A/B 已完成；GSM8K 未运行。历史 1.19× 和既有 attention 收益不能与本轮结果相乘称为实测总收益。
