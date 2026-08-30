# 文档导航

这里是项目文档的统一入口。请先按任务选择文档，不要递归读取整个
`docs/refactoring/`。

## 代码结构

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md)：依赖方向和兼容策略。
- [`SIMULATION_CORE_USAGE.md`](SIMULATION_CORE_USAGE.md)：可复用求解核心的使用边界与入口。
- [`MODULE_MAP.md`](MODULE_MAP.md)：`simulation_core` 各功能包的职责和修改位置。
- [`VALIDATION.md`](VALIDATION.md)：当前验证命令与 ANSYS vertical-flap 验证矩阵。
- [`POST_REFACTOR_BASELINE.md`](POST_REFACTOR_BASELINE.md)：重构后的结构基线。

## 当前 ANSYS vertical-flap 工作

按下面顺序阅读：

1. [`refactoring/ADAPTIVE_SOLID_SUBSTEPS_PERFORMANCE_KALMAN_THREAD_HANDOFF_2026-08-24.md`](refactoring/ADAPTIVE_SOLID_SUBSTEPS_PERFORMANCE_KALMAN_THREAD_HANDOFF_2026-08-24.md)
   - [`validation/ANSYS_VERTICAL_FLAP_SOLID_SUBSTEP_AB_GATES_2026-08-25.md`](validation/ANSYS_VERTICAL_FLAP_SOLID_SUBSTEP_AB_GATES_2026-08-25.md)：fixed1600/adaptive 结果生成前锁定的门槛与命令。
   - [`validation/ANSYS_VERTICAL_FLAP_KALMAN_FSI50_RESULTS_2026-08-26.md`](validation/ANSYS_VERTICAL_FLAP_KALMAN_FSI50_RESULTS_2026-08-26.md)：五模式 modified-physics FSI50、迭代次数、warm baseline 耗时与发布边界。
   - [`validation/ANSYS_VERTICAL_FLAP_IQN_KALMAN_ACCELERATION_PROTOCOL_2026-08-26.md`](validation/ANSYS_VERTICAL_FLAP_IQN_KALMAN_ACCELERATION_PROTOCOL_2026-08-26.md)：Kalman 只作下一物理步首猜、IQN-ILS 强耦合的结果前锁定协议；与 modified-physics 写回实验分离。
   - [`validation/ANSYS_VERTICAL_FLAP_IQN_KALMAN_R20_WORKLOG_2026-08-27.md`](validation/ANSYS_VERTICAL_FLAP_IQN_KALMAN_R20_WORKLOG_2026-08-27.md)：r20 的 source-matched 17 ms prefix、H1/H3 work ledger、fail-closed seam 证据和未闭合验证边界。
   - [`validation/ANSYS_VERTICAL_FLAP_MARKER_CLOSURE_R28_WORKLOG_2026-08-27.md`](validation/ANSYS_VERTICAL_FLAP_MARKER_CLOSURE_R28_WORKLOG_2026-08-27.md)：r23--r28 closure 预算修复、真实分钟级 profile、exact 36-step PASS 与 41/50 新 target-conflict 边界。
   - [`validation/ANSYS_VERTICAL_FLAP_SEGMENT_AGGREGATION_R29_WORKLOG_2026-08-28.md`](validation/ANSYS_VERTICAL_FLAP_SEGMENT_AGGREGATION_R29_WORKLOG_2026-08-28.md)：从r29 segment aggregation、材料参考/伴随载荷、物理外边界和完整accepted checkpoint重构，记录到r47 source-matched K200。r47以exit 0完成200/200步和0.1 s物理时间；journal与固定资源审计通过，但最坏pressure residual接近1e-6门槛，K5000仍未证明。
   - [`refactoring/ANSYS_VERTICAL_FLAP_CONTINUOUS_EXECUTION_DESIGN_2026-08-28.md`](refactoring/ANSYS_VERTICAL_FLAP_CONTINUOUS_EXECUTION_DESIGN_2026-08-28.md)：材料W/W.T、几何候选、IQN/accepted恢复、attempt/canonical dual-root及同条件Fluent诊断顺序。包含r47 K200、速度云图哈希和exact50比较边界；dual-root前50步还逐步绑定全部460个公共history字段，不宣称Fluent真值、10%通过或5000步保证。
     - r47 exact50 dual-root诊断中，v速度`4.27%`和out-of-plane leakage `0.0`通过5%单项检查；speed `13.33%`、pressure `19.07%`、位移`20.92%/23.23%`及力`37.52%/57.25%`均未满足10%高一致性目标。3D/2D差异不自动豁免这些偏差，内部求解器门槛未放宽。
2. [`refactoring/ANSYS_VERTICAL_FLAP_GENERIC_SOLVER_THREAD_HANDOFF_2026-07-23.md`](refactoring/ANSYS_VERTICAL_FLAP_GENERIC_SOLVER_THREAD_HANDOFF_2026-07-23.md)
3. [`VALIDATION.md`](VALIDATION.md) 中对应的运行与证据说明
4. [`../validation_runs/README.md`](../validation_runs/README.md) 中的验证产物索引

关键实现入口：

- `cases/ansys_vertical_flap_fsi.py`：案例配置和入口。
- `benchmarks/official/solid_mpm_fsi_runner.py`：官方通用 FSI runner。
- `simulation_core/fluids/solver.py`：流体、SST transport 和动量求解。
- `simulation_core/drivers/generic_fsi_solver.py`：通用 FSI 驱动契约。

## 大文件快速定位

不要整文件读取 `simulation_core/fluids/solver.py`、
`benchmarks/official/solid_mpm_fsi_runner.py` 或 `cases/turek_hron_fsi.py`。
先按符号定位，再读取相邻代码：

```powershell
Select-String -Path simulation_core\fluids\solver.py `
  -Pattern '^\s*(def|class)\s+(prepare_sst_wall_distance|advance_sst_transport|predict|_solve_sst_momentum_unsplit_helmholtz)\b'
```

ANSYS 初始化/预流问题优先检查：

- `prepare_sst_wall_distance`：SST wall-distance 准备；固定几何命中严格
  geometry/topology/wall-flag cache key 时复用，任一 obstacle writer 会统一失效。
- `advance_sst_transport`：k/omega transport。
- `predict`：动量预测和 SST 动量 Helmholtz。
- `_solve_sst_momentum_unsplit_helmholtz`：assembly、PCG 和 commit。

runner 中使用同样方式搜索 `preflow`、`prepare_sst_wall_distance`、
`advance_sst_transport` 和 `fluid.predict`，不要从文件开头顺序阅读。

## 其他验证主题

- [`TUREK_HRON_VALIDATION.md`](TUREK_HRON_VALIDATION.md)：Turek-Hron FSI。
- [`ANSYS_VERTICAL_FLAP_2D_TO_3D_SLAB_EQUIVALENCE.md`](ANSYS_VERTICAL_FLAP_2D_TO_3D_SLAB_EQUIVALENCE.md)：2D/3D slab 等价性。
- `validation/`：可复核的验证说明和报告。
- [`validation/HIBM_MPM_PAPER_VS_CODE.md`](validation/HIBM_MPM_PAPER_VS_CODE.md)：论文要求与当前实现的对照表。
- `refactoring/`：按日期保留的目标、审计和线程交接记录。
- [`refactoring/REFACTORING_NOTES.md`](refactoring/REFACTORING_NOTES.md)：重构阶段记录与风险说明。
- `refactoring/FSI_AUDIT_*_GOAL_2026-07-02.md`：已归档的完整审计修复目标；根目录同名文件仅保留旧链接转发。

## 文档命名约定

- 长期说明：`<TOPIC>_<PURPOSE>.md`
- 带时间边界的目标/报告：`<TOPIC>_<PURPOSE>_YYYY-MM-DD.md`
- 线程交接：`<TOPIC>_THREAD_HANDOFF_YYYY-MM-DD.md`

历史文档保留原名，以免破坏已有链接。新文档应使用完整主题名称，避免仅用
`v48`、`final2`、`new` 等无法独立理解的名字。
