# HIBM-MPM 重构版（refactored/）

本目录是对上级目录 HIBM-MPM 代码的**功能等价重构副本**。原目录代码未被修改。

> 当前同步说明：本仓库已经包含 refactored 副本之后继续推进的 sharp HIBM-MPM / squid FSI 收敛修复与验证工作。最新目标和审查记录见
> [SHARP_HIBM_MPM_CONVERGENCE_FIX_GOAL_2026-06-18.md](docs/refactoring/SHARP_HIBM_MPM_CONVERGENCE_FIX_GOAL_2026-06-18.md)、
> [SQUID_2S_SIMULATION_GOAL_2026-06-17.md](docs/refactoring/SQUID_2S_SIMULATION_GOAL_2026-06-17.md) 和
> [SQUID_JET_FSI_COUPLING_REVIEW.md](SQUID_JET_FSI_COUPLING_REVIEW.md)。

## 目录性质

- 这是一份**保守、可验证**的重构：所有修改都以"行为不变"为第一原则，逐条列在
  [REFACTORING_NOTES.md](REFACTORING_NOTES.md) 中，并按风险分级。
- 大文件（`fluid.py`、`hibm_mpm.py`、`tri_surface.py`、`mooney_shell_mpm.py`、
  `cases/squid_soft_robot/runner.py`）**未做结构性大改**——对 3.5 万行的数值代码做整体改写
  无法在不跑长仿真验证的情况下保证功能不变。对它们的重构按
  [REFACTORING_NOTES.md](REFACTORING_NOTES.md) 中的分阶段蓝图执行。
- 一次性调试脚本（`tools_*.py`、`run_phase0_raw_map_scaling.py`）按审计结论归档到
  `archive/tools/`，不参与测试。

## 已应用的修改（摘要）

| 文件 | 修改 | 类型 |
|---|---|---|
| `simulation_core/runtime.py` | `default_fp` 严格校验（原先拼错会静默落到 f64）；重复初始化时 fp 不一致改为报错；记录已初始化的 arch/fp | bug 修复 |
| `simulation_core/validation.py` | 删除全仓无引用的 `check()` / `ValidationCheck` / `validation_summary()` | 死代码删除 |
| `simulation_core/geometry.py` | 删除全仓无引用的 `vertex_area_weights()`；`orient_faces_outward()` 向量化（语义逐位等价，初始化加速） | 死代码删除 + 性能 |
| `simulation_core/fsi_coupling.py` | 删除全仓无引用的 `_solve_small_linear_system()` 与 `_max_secant_amplification()`；`required_history` 化简为常量 2（原表达式恒等于 2） | 死代码删除 |
| `simulation_core/neo_hookean_mpm.py` | 删除类内无引用的 `_particle_grid_out_of_bounds()` 与 `_read_vector()` | 死代码删除 |
| `simulation_core/projected_ibm.py` | `ProjectedIbmRegionPairStepConfig.__post_init__` 增加 `primary_region_id != secondary_region_id` 校验（原先两 region 相同会被静默接受并双重计力） | 边界条件加固 |

其余文件为原样副本。`simulation_core/__init__.py` 的公开 API 完全不变。

## 如何验证

在本目录（`refactored/`）下运行：

```powershell
& "D:/TOOL/Anaconda/python.exe" -m unittest discover -s tests/contracts -p "test_*.py" -v
& "D:/TOOL/Anaconda/python.exe" -m unittest discover -s tests/integration -p "test_*.py" -v
& "D:/TOOL/Anaconda/python.exe" -m unittest discover -s tests/tools -p "test_*.py" -v
# 需要 CUDA GPU：
& "D:/TOOL/Anaconda/python.exe" -m unittest discover -s tests -p "test_*.py" -v
```

测试目录是随副本一起拷贝的，`test_source_static_contracts.py` 的源码契约检查在本目录内自洽。

## 运行案例

与原目录一致：

```powershell
& "D:/TOOL/Anaconda/python.exe" run_simulation.py squid-soft-robot --steps 8
```

## Repository layout

- `simulation_core/`: reusable solver package. Implementation lives under layered packages; top-level legacy modules are compatibility shims.
- `cases/`: runnable simulation cases registered by `run_simulation.py`.
- `benchmarks/`: official/vendor benchmark adapters and reusable benchmark runners.
- `tools/`: diagnostics, rendering, and post-processing helpers.
- `tests/`: tests grouped by `solvers/`, `cases/`, `benchmarks/`, `tools/`, `integration/`, and `contracts/`.
- `docs/`: architecture, validation, and refactoring records.
- `archive/`: historical one-shot maintenance scripts.

See [ARCHITECTURE.md](ARCHITECTURE.md) for dependency direction and legacy
compatibility policy. See [docs/VALIDATION.md](docs/VALIDATION.md) for the
current structure validation matrix. Detailed refactoring step records live in
`docs/refactoring/`.

Use `python -m tools.diagnostics...` or `python -m tools.rendering...` for helper scripts.
