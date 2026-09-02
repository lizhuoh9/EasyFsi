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
   - [`validation/ANSYS_VERTICAL_FLAP_IQN_KALMAN_R20_WORKLOG_2026-08-27.md`](validation/ANSYS_VERTICAL_FLAP_IQN_KALMAN_R20_WORKLOG_2026-08-27.md)：r20 的 17 ms 研究前缀及 r51 zero-preflow 细网格 H3 exact50 延续；后者完成 50/50、raw-IQN-replay/closure v2 身份审计和 r3 锁定 Fluent 诊断，但关键误差多数仍超过 10%，不作 parity 或 5000 步保证。
   - [`validation/ANSYS_VERTICAL_FLAP_MARKER_CLOSURE_R28_WORKLOG_2026-08-27.md`](validation/ANSYS_VERTICAL_FLAP_MARKER_CLOSURE_R28_WORKLOG_2026-08-27.md)：r23--r28 closure 预算修复、真实分钟级 profile、exact 36-step PASS 与 41/50 新 target-conflict 边界。
   - [`validation/ANSYS_VERTICAL_FLAP_SEGMENT_AGGREGATION_R29_WORKLOG_2026-08-28.md`](validation/ANSYS_VERTICAL_FLAP_SEGMENT_AGGREGATION_R29_WORKLOG_2026-08-28.md)：从r29 segment aggregation、材料参考/伴随载荷、物理外边界和完整accepted checkpoint重构，记录到r47 source-matched K200。r47以exit 0完成200/200步和0.1 s物理时间；journal与固定资源审计通过，但最坏pressure residual接近1e-6门槛，K5000仍未证明。
   - [`validation/ANSYS_VERTICAL_FLAP_KALMAN_STATISTICAL_CALIBRATION_GOAL_2026-08-31.md`](validation/ANSYS_VERTICAL_FLAP_KALMAN_STATISTICAL_CALIBRATION_GOAL_2026-08-31.md)：R24 的 D0/D1 来源、冻结校准矩阵、统计门槛、不可越过的 R25--R28 边界及完成证据。
   - [`validation/ANSYS_VERTICAL_FLAP_KALMAN_CALIBRATION_REPORT_2026-08-31.md`](validation/ANSYS_VERTICAL_FLAP_KALMAN_CALIBRATION_REPORT_2026-08-31.md)：CPU-only R24 离线回放最终报告；K0 逐步奇偶校验精确通过，但无 Kalman 候选胜过 C0，最终分类为 `FAIL_NO_KALMAN_PREDICTIVE_VALUE`，因此不授权 R25。
   - [`validation/ANSYS_VERTICAL_FLAP_KALMAN_ORACLE_HEADROOM_GOAL_2026-08-31.md`](validation/ANSYS_VERTICAL_FLAP_KALMAN_ORACLE_HEADROOM_GOAL_2026-08-31.md)：R24B 同源 strict-CUDA exact8 Oracle 上限实验的冻结矩阵、联合门禁、alpha 条件分支、事务边界与完成证据。
   - [`validation/ANSYS_VERTICAL_FLAP_KALMAN_ORACLE_HEADROOM_REPORT_2026-08-31.md`](validation/ANSYS_VERTICAL_FLAP_KALMAN_ORACLE_HEADROOM_REPORT_2026-08-31.md)：schema-2 证据由 live runs 自底向上复算后，Q0/Q3 获得 `PASS_ORACLE_HEADROOM`；但 alpha 0.25/0.50/0.75 均未减少 24 次 coupling trials，仅完美 Oracle 降到 8 次，故不直接授权现有 Kalman、K3/GRU、自适应或长跑。
   - [`validation/ANSYS_VERTICAL_FLAP_ORACLE_THRESHOLD_IQN_FIRST_UPDATE_GOAL_2026-08-31.md`](validation/ANSYS_VERTICAL_FLAP_ORACLE_THRESHOLD_IQN_FIRST_UPDATE_GOAL_2026-08-31.md)：R24C displacement-relative audit、离散 carry-to-oracle threshold matrix、conditional IQN reuse 因子实验及严格停止树。
   - [`validation/ANSYS_VERTICAL_FLAP_ORACLE_THRESHOLD_IQN_FIRST_UPDATE_REPORT_2026-08-31.md`](validation/ANSYS_VERTICAL_FLAP_ORACLE_THRESHOLD_IQN_FIRST_UPDATE_REPORT_2026-08-31.md)：R24C 获得三个 PASS、选择安全的 `omega=0.75`，carry reuse 将 exact8 coupling trials 从 24 降到 19；结论仍为 `deployable=false`，不授权 GRU、长跑或 Fluent parity 声明。
   - [`validation/ANSYS_VERTICAL_FLAP_GRU_KALMAN_FEASIBILITY_GOAL_2026-09-02.md`](validation/ANSYS_VERTICAL_FLAP_GRU_KALMAN_FEASIBILITY_GOAL_2026-09-02.md)：R25A CPU-only POD-GRU / Kalman-residual-GRU feasibility harness, frozen split, gates, and holdout safety boundary.
   - [`validation/ANSYS_VERTICAL_FLAP_GRU_KALMAN_FEASIBILITY_REPORT_2026-09-02.md`](validation/ANSYS_VERTICAL_FLAP_GRU_KALMAN_FEASIBILITY_REPORT_2026-09-02.md)：R25A one-shot CPU holdout 得到 G0 fail、GK0 fail、GK1 pass 的 review-required 分类；train-only POD-AR NRMSE `0.0200` 明显优于 G0/GK1，故停止于离线结论，不授权 CUDA、live probes 或部署声明。
   - [`validation/ANSYS_VERTICAL_FLAP_GRU_KALMAN_LIVE_PROBE_GOAL_2026-09-02.md`](validation/ANSYS_VERTICAL_FLAP_GRU_KALMAN_LIVE_PROBE_GOAL_2026-09-02.md)：R25B 容量匹配 G0-M/GDelta-M、冻结 GK1/POD-AR 与 step-7/8 strict-CUDA no-commit matrix；首猜逐字校验、每臂/全 sweep 回滚和离散 work 门槛预先锁定。
   - [`validation/ANSYS_VERTICAL_FLAP_R24C1_CI_ATTESTATION_GOAL_2026-09-01.md`](validation/ANSYS_VERTICAL_FLAP_R24C1_CI_ATTESTATION_GOAL_2026-09-01.md)：R24C.1 修正冻结 runner 的默认 traction predicate，并按新 139-entry source map 完整重建 strict-CUDA preflow、Q0、9 个 rollback-safe probes 与四象限 reuse 证据；seal 额外绑定三份 Q0 compact report 的原始 SHA，只在 runtime 之外容忍遗留非有限诊断，并保留 `deployable=false` 与 clean-final-HEAD green CI 边界。
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
