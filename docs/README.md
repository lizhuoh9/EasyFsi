# 文档导航

这里是项目文档的统一入口。请先按任务选择文档，不要递归读取整个
`docs/refactoring/`。

## 代码结构

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md)：依赖方向和兼容策略。
- [`MODULE_MAP.md`](MODULE_MAP.md)：`simulation_core` 各功能包的职责和修改位置。
- [`VALIDATION.md`](VALIDATION.md)：当前验证命令与 ANSYS vertical-flap 验证矩阵。
- [`POST_REFACTOR_BASELINE.md`](POST_REFACTOR_BASELINE.md)：重构后的结构基线。

## 当前 ANSYS vertical-flap 工作

按下面顺序阅读：

1. [`refactoring/ANSYS_VERTICAL_FLAP_GENERIC_SOLVER_THREAD_HANDOFF_2026-07-23.md`](refactoring/ANSYS_VERTICAL_FLAP_GENERIC_SOLVER_THREAD_HANDOFF_2026-07-23.md)
2. [`VALIDATION.md`](VALIDATION.md) 中对应的运行与证据说明
3. [`../validation_runs/README.md`](../validation_runs/README.md) 中的验证产物索引

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

- `prepare_sst_wall_distance`：SST wall-distance 准备。
- `advance_sst_transport`：k/omega transport。
- `predict`：动量预测和 SST 动量 Helmholtz。
- `_solve_sst_momentum_unsplit_helmholtz`：assembly、PCG 和 commit。

runner 中使用同样方式搜索 `preflow`、`prepare_sst_wall_distance`、
`advance_sst_transport` 和 `fluid.predict`，不要从文件开头顺序阅读。

## 其他验证主题

- [`TUREK_HRON_VALIDATION.md`](TUREK_HRON_VALIDATION.md)：Turek-Hron FSI。
- [`ANSYS_VERTICAL_FLAP_2D_TO_3D_SLAB_EQUIVALENCE.md`](ANSYS_VERTICAL_FLAP_2D_TO_3D_SLAB_EQUIVALENCE.md)：2D/3D slab 等价性。
- `validation/`：可复核的验证说明和报告。
- `refactoring/`：按日期保留的目标、审计和线程交接记录。
- `refactoring/FSI_AUDIT_*_GOAL_2026-07-02.md`：已归档的完整审计修复目标；根目录同名文件仅保留旧链接转发。

## 文档命名约定

- 长期说明：`<TOPIC>_<PURPOSE>.md`
- 带时间边界的目标/报告：`<TOPIC>_<PURPOSE>_YYYY-MM-DD.md`
- 线程交接：`<TOPIC>_THREAD_HANDOFF_YYYY-MM-DD.md`

历史文档保留原名，以免破坏已有链接。新文档应使用完整主题名称，避免仅用
`v48`、`final2`、`new` 等无法独立理解的名字。
