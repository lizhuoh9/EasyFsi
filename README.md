# HIBM-MPM 求解器与验证仓库

这是当前维护中的三维 HIBM-MPM/MPM/CFD 仿真代码库，不再是原工程的静态“等价副本”。
历史重构记录仅用于追溯；当前行为以源码、测试和 source-matched 数值验证为准。

快速导航：

- [文档索引](docs/README.md)
- [模块地图](docs/MODULE_MAP.md)
- [验证说明](docs/VALIDATION.md)
- [HIBM-MPM 论文与代码对照](docs/validation/HIBM_MPM_PAPER_VS_CODE.md)
- [架构说明](ARCHITECTURE.md)

## 当前求解器边界

`simulation_core/` 保存可复用的流体、固体、sharp HIBM-MPM 耦合、几何、材料与诊断实现。
共享 runtime-adapter 协议位于 `simulation_core/drivers/`；案例只能装配几何、边界条件和
报告，不能复制一套相同物理问题的第二个耦合公式。

当前 HIBM-MPM 案例只保留 canonical sharp formulation。旧的
`legacy_projected_reduced` 模式、ANSYS cell-obstacle backend 和相应兼容 CLI 已退役；
低层 projected-IBM 数学工具仍可供不同物理案例复用，但不是第二个生产工作流。

官方 ANSYS rectangular-solid/vertical-flap 基准使用经过 source-matched CUDA
preflow、FSI1、FSI8 和 FSI50 验证的 direct runner：
`benchmarks.official.solid_mpm_fsi_runner.run_hibm_mpm_fsi`。
`cases.ansys_vertical_flap_fsi.run_ansys_vertical_flap_benchmark` 只添加案例元数据和
官方报告校验。修改任何参与 identity 的源码后，旧 preflow snapshot 必须失效并重建。

Squid 保留 typed `StepLoopContext` 和同一 sharp formulation 的 case-specific fixed-point
装配；Turek-Hron 使用共享 runtime-adapter driver。不同案例可以有不同几何与结构模型，
但不得重新引入同一案例的 parallel legacy/generic 数值分支。

## 如何验证

在本目录（`refactored/`）下运行：

```powershell
$python = if ($env:EASYFSI_PYTHON) { $env:EASYFSI_PYTHON } else { 'python' }
& $python -m unittest discover -s tests/contracts -p "test_*.py" -v
& $python -m unittest discover -s tests/integration -p "test_*.py" -v
& $python -m unittest discover -s tests/tools -p "test_*.py" -v
# 需要 CUDA GPU：
& $python -m unittest discover -s tests -p "test_*.py" -v
```

结构测试、device 测试和正式数值验证是不同门禁；focused/host 通过不能替代 CUDA
FSI50 或 Fluent comparator。长程验证必须使用唯一输出目录，并记录源码、配置和几何 identity。

## 运行案例

Squid 案例必须显式传 `--source-config`，指向一份**已存在**的 GUI 导出
`simulation_config.json`。CLI 的历史默认路径（`_diagnostic_runs/.../simulation_config.json`）
是被忽略的诊断输出，不在仓库内，因此不带该参数直接运行会立刻失败——
runner 现在会在创建任何输出目录/写 `run_process.json` **之前**报
`source config not found` 并保持文件系统不变：

```powershell
& $python run_simulation.py squid-soft-robot --steps 8 `
    --source-config ".\config\simulation_config.json"
```

可运行案例清单以 `run_simulation.py --help` 输出为准（`comsol-*` 两个基准案例
只提供 `run_*_fsi_smoke()` 编程入口，没有 CLI `main()`，不能从该分发器运行）。

## Repository layout

For a two-minute navigation path, start with [docs/README.md](docs/README.md).
Agents should also follow [AGENTS.md](AGENTS.md) to avoid recursively scanning
large historical validation trees.

- `simulation_core/`: reusable solver package. Implementation lives under layered packages; `simulation_core/__init__.py` is the public facade. Removed legacy module names are listed in `docs/MODULE_MAP.md`.
- `cases/`: runnable simulation cases registered by `run_simulation.py`.
- `benchmarks/`: official/vendor benchmark adapters and reusable benchmark runners.
- `tools/`: diagnostics, rendering, and post-processing helpers.
- `tests/`: tests grouped by `solvers/`, `cases/`, `benchmarks/`, `tools/`, `integration/`, and `contracts/`.
- `docs/`: architecture, validation, and refactoring records.
- `archive/`: historical one-shot maintenance scripts and explicitly dated legacy snapshots.

See [ARCHITECTURE.md](ARCHITECTURE.md) for dependency direction and legacy
compatibility policy. See [docs/VALIDATION.md](docs/VALIDATION.md) for the
current structure validation matrix. Detailed refactoring step records live in
`docs/refactoring/`.

Use `python -m tools.diagnostics...` or `python -m tools.rendering...` for helper scripts.
