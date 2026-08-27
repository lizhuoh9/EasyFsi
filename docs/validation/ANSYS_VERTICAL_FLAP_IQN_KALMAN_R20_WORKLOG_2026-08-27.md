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
