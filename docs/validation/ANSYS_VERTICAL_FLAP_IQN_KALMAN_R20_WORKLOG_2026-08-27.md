# ANSYS 竖直薄板 IQN--Kalman r20 研究日志

日期：2026-08-27
状态：可供代码/证据审查的 WIP；未提交、未推送、未合并

本日志是 [`ANSYS_VERTICAL_FLAP_IQN_KALMAN_ACCELERATION_PROTOCOL_2026-08-26.md`](ANSYS_VERTICAL_FLAP_IQN_KALMAN_ACCELERATION_PROTOCOL_2026-08-26.md)
的 r20 证据索引，记录已完成的 17 ms prefix 与未闭合项。它不替代协议中的
预锁定门槛，也不包含大型 JSON 或二进制产物。

## 工作树与 WIP 边界

- branch：`codex/iqn-threshold-kalman-study`。
- base HEAD：`2187530bc6e156a70953d07294f3b3ebb09e1dd5`。
- 工作树有未提交的 solver、runner、case、focused test、工具和文档修改；它们是
  当前研究 WIP，不得以清理、reset 或 mirror 覆盖来“整理”。本日志和 README
  索引也尚未 commit。
- r20 manifest 中记录的 runner/core/IQN/Kalman SHA 与本次 r20 所用源码匹配；
  但当前树仍是 dirty WIP。任何后续源码修改都会使该 runner/source identity
  stale，必须重新做 source-matched fresh preflow，而不能复用 r20 snapshot。

## r20 可复核产物

run identity：`ansys_vf__segment_supported_face_dt5e4__20260827__r20`；全部为
strict-CUDA/f32。fresh preflow output：

`validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/ansys_vf__segment_supported_face_dt5e4__20260827__r20__preflow`

snapshot 目录：

`validation_runs/solver_soaks/ansys_vf__segment_supported_face_dt5e4__20260827__r20__snapshot`

snapshot NPZ：`preflow_state.5469e7ef362f4c87afa24169fbff488e.npz`，SHA256：
`dfeb4bfcdfdbc063e41a1ad3abc4a7b505b797b73aefc4fab4882f65cbfb3119`。

同一 snapshot 的 output identities：

- Q0：`validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/ansys_vf__segment_supported_face_dt5e4__20260827__r20__q0_carry__fsi50`
- H1：`validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/ansys_vf__segment_supported_face_dt5e4__20260827__r20__h1_k3__fsi34`
- H3：`validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/ansys_vf__segment_supported_face_dt5e4__20260827__r20__h3_k3_reuse__fsi34`

三组都从上述 fresh snapshot 启动，而非从 `step_fields/*.npz` 重启。

## Q0 fail-closed 边界

Q0 请求 50 步，accepted `34/50`（`17 ms`）。第 35 步没有 step artifact；
`progress.json` 记录 fail-closed，而不是一个可被用于比较的第 35 步状态。
首个 one-sided direct-geometry witness 的 `count=4`：component face
`(0, 17, 28)`、axis `1`。两个候选 segment 分别为 source row
`(0, 16, 28)`、marker pair `(23, 25)`、region `202`，以及 source row
`(0, 17, 28)`、marker pair `(26, 27)`、region `303`；endpoint-support
ratio 分别为 `1.7190750615891468` 和 `1.0075753568933896`。

当前结论为 `DO_NOT_PATCH`。这是 side--cap seam 的双 endpoint-unsupported
候选，不是可由浮点尾差合理解释的单一容差边界：放宽 tolerance 会接受
out-of-support 的 endpoint 外推，从而改变 strict direct-geometry 语义并掩盖
component/segment ownership 根因。它不是 25 ms 完成结果。

## H1/H3 17 ms A/B

H1 为 Kalman 初值；H3 为相同 Kalman 加 accepted-only previous-step IQN secant
reuse。二者均完成请求的 34 步；H3 在其后续 33 个物理步 reuse history，且
history reset 为零。Q0、H1、H3 的累计 ledger：

| 组 | Coupling trials | Rejected | CG | Momentum | SST | MPM |
|---|---:|---:|---:|---:|---:|---:|
| Q0 | 103 | 69 | 6672 | 4443 | 3314 | 15350 |
| H1 | 103 | 69 | 6688 | 4443 | 3314 | 15350 |
| H3 | 81 | 47 | 5280 | 3487 | 2615 | 12070 |

因此 H3 相对 H1 的 coupling trial work 为 `103 -> 81`（`-21.36%`）。已完成
34 步中，Fluid/Solid accepted physical time 与 macro `dt_s` 的审计、remaining
time、以及 artifact 已记录的有限/残差/pressure-CG health 字段均通过；H1/H3
velocity 序列最大 NRMSE 约 `1.4e-5`。但 step history 没有序列化 explicit OOB，
所以 OOB 未被这批 r20 artifact 独立闭合。

H3/H1 单次 raw wall-time speedup 为 `1.139x`；由 artifact mtime 得出的 warm
proxy 为 `1.299x`。两者都只是单次研究观测（后者还只是 proxy），不能作为正式
性能结论或正式 FSI50 结论。

## 聚焦证据与已知未通过项

- schema gate：`1 pass + 4 subtests`。
- evidence gate：RED 为 `3 fail`；GREEN 先为 `15 pass`，补足后为 `18 pass`。
- threshold gate：RED 为 `1 fail + 8 pass`；GREEN 为 `9 pass`。
- exact-node geometry collection 找到 `1` 个节点，但命令在 `1243.2 s` 超时；它是
  `TIMEOUT`，不是 PASS。
- r19 在 schema freshness gate 失败；r20 因此重新生成 fresh preflow 和 snapshot。

## 可视化审查产物（Windows 本地，不纳入仓库）

- `C:\Users\lizhu\.codex\visualizations\2026\08\24\01a0342f-230b-72d0-a124-2e489284ec43\r20_h1_h3_minus_q0_step34_differences.png`
  SHA256 `7B7A38955575302137CDA3AB600FC692AF07DFB45ED1EDC8DEB832392E75E29B`
- `C:\Users\lizhu\.codex\visualizations\2026\08\24\01a0342f-230b-72d0-a124-2e489284ec43\r20_q0_h1_h3_step34_fluid_speed_pressure_clouds.png`
  SHA256 `1A78E41B4483B2DF62AA64126B9FF1B677AB58DA1046A1E29C851CDC0DC4DA54`
- `C:\Users\lizhu\.codex\visualizations\2026\08\24\01a0342f-230b-72d0-a124-2e489284ec43\r20_q0_h1_h3_step34_iterations_reuse_work.png`
  SHA256 `430CCC334682D7844409BA620D9DFE3DE512E5222145CEAE3536D05219E368CC`
- `C:\Users\lizhu\.codex\visualizations\2026\08\24\01a0342f-230b-72d0-a124-2e489284ec43\r20_q0_h1_h3_step34_solid_displacement_velocity_clouds.png`
  SHA256 `6EB70B32163804BE6A2DA86C05B8A2E976D933343926936594757EED13020D4D`

这些是 step 34 的可视化审查辅助证据，不能替代数值 gates。

## 未完成项与下一步

尚未推送 main，尚未完成 strict FSI50、25 ms 比较或 Fluent comparison，因而没有
Fluent parity 或正式加速主张。下一步是对双 unsupported side--cap seam 做
root-cause investigation；只有得到不改变 strict geometry 语义的最小根因修复后，
才运行 fresh source-matched preflow、FSI1/FSI2/FSI8/FSI50 gates，并重做 matched
Q0/H1/H3 验证。

## 2026-08-30 r51 H3 exact50 延续

本节是后续证据，不改写上面的 r20 历史边界。权威 WSL 工作树为
`/home/zhuohengli/worktrees/HIBM-MPM-r21-validation`，branch
`codex/closure-diagnostic-r23`，本轮起始 HEAD
`deef2f3cdc4274f593eda885298a3a7092f0517f`。方法为逐轴 Kalman iteration-0
首猜、accepted-only previous-step IQN secant reuse、zero preflow、细网格，且
`kalman_writeback_mode=off`、`kalman_modified_physics=false`。

canonical artifact root：

`validation_runs/solver_soaks/ansys_vf__kalman_iqnreuse_nopreflow__fine__20260830__r51__h3`

完成 50 步的 resume attempt：

`validation_runs/solver_soaks/ansys_vf__kalman_iqnreuse_nopreflow__fine__20260830__r51__h3__resume50__attempt1`

结果为 strict-CUDA `50/50`，最终物理时间 `0.025 s`；50 个 `step_fields`、50 个
`step_history`、accepted checkpoint/journal 和双根 provenance 均通过。每步 fluid
和 solid 均完整消费 `dt_s=5e-4 s`。50 步内 pressure failure、CG breakdown、MPM
OOB、deformation clamp、solid retry/reject、invalid marker/pressure row 均为零。
canonical closure 最大 `9.89631758e-7 < 1.1e-6 m/s`，no-slip 最大
`5.03263873e-5 < 1e-4 m/s`，CG 最大 exact relative residual
`9.75780797e-7 < 1e-6`。step 42 的 target/claim/alpha/region conflict 均为零。
这些证据证明当前工况的 50 步稳定完成，不推出任意工况或 5000 步保证。

### Accepted-state Kalman 与 IQN reuse 审计

新增 strict comparator profile
`kalman_iqn_reuse_material_reference_fine50_v2`；最终 contract SHA256 为
`79dcf7de150de618fedff4b9710186a1750003a91d790ca23b4043c039b9cd74`。
它从 raw trial guess/candidate/residual 独立重建 retained/local secant 矩阵，重放
`lstsq`、rank、condition、update limit、fallback、next guess 和第一残差；同时逐步
验证 Kalman accepted-state 计数、warmup 边界、上一接受步 source、retained/imported
pair count 及 initial-residual 链。任何 reset 必须由 raw replay 证明；
`residual_growth_limit` 必须严格满足
`first_residual > 4 * prior_initial_residual`，并优先于 reuse update。profile 还锁定
marker-target closure 的 `1e-4/1.1e-6 m/s` 两级容差、四类 residual 和 invalid-axis
count，不改变 pressure、target tolerance 或 Fluent 诊断阈值。

真实 r51 结果中，Kalman 在 step 1--5 warmup，step 6--50 使用预测，共 45 步；
writeback 全程关闭。previous-step IQN history 在 48 步使用。step 24 因
`0.5574506631 > 4 * 0.1098353843` 安全丢弃复用历史、回退 Picard 后仍收敛接受，
step 25 恢复复用。首猜 RMSE 均值约 `0.0198655 m/s`；NIS 均值约 `73.2825`，范围
`4.1025--225.0764`。高 NIS 表明协方差仍未统计标定，不能把 Kalman 称为已校准。

严格匹配的 FSI8 Q2/H3 仅改变 previous-step IQN reuse：H3 coupling trials
`24 -> 21`（`-12.5%`），rejected trials `16 -> 13`（`-18.75%`），CG work
`-12.40%`，warm wall time 单次观测改善 `13.11%`；对应 8 步主要数值 NRMSE 为
`2.96e-7--3.55e-4`。没有身份完全一致的 Q0/Q2 FSI50，因此不声明正式 50 步
加速比。r51 两次进程的实际 summary 合计约 `40.54 min`；剔除冷 JIT 后 48 个
warm steps 平均约 `40.756 s/步`，5000 步仅按线性 warm 外推仍约 `56.6 h`。

### 锁定 Fluent fresh50 离线诊断

最终输出 bundle：

`validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/kalman_iqnreuse_nopreflow_r51_fsi50_20260830_r3`

`comparison_report.json` 状态为 `diagnostic_complete`；exact50、dual-root checkpoint、
strict pressure semantics、材料参考伴随审计和 H3 v2 方法身份均通过。bundle 的
`CHECKSUMS.sha256` 全部通过。保留的 5% diagnostic gate 失败；按 10%高一致性参考，
主要 waveform/field NRMSE 为下表。r1/r2 是最终 v2 contract 前的中间 bundle，保留
用于审计但不再作为当前结论来源。

| 指标 | r51 H3 相对锁定 Fluent | 10%参考 |
|---|---:|---|
| transverse velocity `v` | `4.28%` | 通过 |
| final speed field | `13.21%` | 未通过 |
| gauge pressure | `19.19%` | 未通过 |
| tip mean-vector displacement | `22.69%` | 未通过 |
| whole-solid max displacement | `24.40%` | 未通过 |
| streamwise interface force | `35.94%` | 未通过 |
| transverse interface force | `57.84%` | 未通过 |
| out-of-plane force leakage | `0.0%` | 通过 |

峰值相位误差并非所有位移分量都接近：tip norm、whole-solid max 和 tip streamwise
均为 `1/50=2%`，但 tip transverse 为 `23/50=46%`。因此只能说前三个特定标量的
峰值步接近，不能泛称“位移振荡相位接近”。wake 速度、压力幅值、位移幅值及力历史
仍有显著差异。Fluent 是 2-D intrinsic structure，本求解器是 3-D-equivalent slab
MPM，并且本次 H3 为 zero preflow；这些是解释差异的模型/初态因素，不是自动豁免。
当前不能声称 10%数值一致、Fluent parity，或用 Fluent 作为绝对真值。

### 本轮测试边界

- H3 v2 contract（含生产 `IqnIlsAccelerator` 生成的 50 步 raw trace fixture）：
  `47 passed in 185.58 s`。
- current-IQN、material-reference、native comparison/hardening/pressure/campaign 和 H3
  七文件矩阵：`283 passed, 3 skipped in 391.04 s`。三个 skip 是可选 coarse/legacy
  Fluent 输入未提供，不是数值失败。
- IQN-ILS 与统一求解器核心：修正 checkpoint-resume 主循环上线后遗留的旧源码字符串
  断言，结果为 `31 passed, 17 subtests passed in 5.85 s`；未改 runner 或数值实现。
- 独立只读审查结论 `ship`，无 P0/P1；确认 raw replay 与生产运算次序一致、测试不是
  手填 update 的假绿路径、closure 与 growth 边界均 fail-closed。

这些是聚焦回归和离线比较证据，不是全仓测试、80%全仓覆盖、5000 步 soak 或 Fluent
parity 声明。
