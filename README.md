# HIBM-MPM 求解器与验证仓库

本目录是当前维护中的 HIBM-MPM/MPM/CFD 仿真代码库。代码已经历模块拆分、入口收敛、
失效参数和死代码清理；历史重构记录仅用于追溯，不能替代当前源码、测试和验证说明。

当前导航入口：

- [模块地图](docs/MODULE_MAP.md)：实现归属、依赖方向和已删除的旧入口；
- [验证说明](docs/VALIDATION.md)：结构检查、短程诊断和正式数值验证的边界；
- [HIBM-MPM 论文与代码对照](HIBM_MPM_PAPER_VS_CODE.md)：sharp 耦合实现与论文要求的对应关系；
- [架构说明](ARCHITECTURE.md)：仓库层级和公共接口约束。

## 当前求解器架构

`simulation_core/` 保存可复用的流体、固体、耦合、材料、几何和诊断实现。旧的根级兼容
模块与通用 FSI driver 已删除；新代码应使用 `docs/MODULE_MAP.md` 列出的功能包路径。

当前案例工作流只保留 canonical sharp HIBM-MPM 耦合模式 `HIBM_MPM_SHARP`，不再提供
`legacy_projected_reduced` 模式选择、别名或兼容 shim。

官方 ANSYS rectangular-solid/vertical-flap 基准的数值入口是
`benchmarks.official.solid_mpm_fsi_runner.run_hibm_mpm_fsi`；
`cases.ansys_vertical_flap_fsi.run_ansys_vertical_flap_benchmark` 只在其外层添加案例元数据和
官方报告校验。

其他物理案例仍保留各自的几何、边界条件和模型装配：

- Squid：`cases/squid_soft_robot/runner.py`；
- Turek-Hron：`cases/turek_hron_fsi.py`；
- COMSOL：`cases/comsol_multibody_mechanism_fsi.py` 和
  `cases/comsol_water_balloon_fsi.py`。

上述收敛只统一耦合模式和官方 ANSYS 数值入口，并不表示所有案例由同一个运行函数驱动。

## 运行案例

Squid 案例必须显式传入已存在的 GUI 导出 `simulation_config.json`。输入不存在时，runner
会在创建输出目录或写入 `run_process.json` 之前 fail closed：

```powershell
$python = if ($env:EASYFSI_PYTHON) { $env:EASYFSI_PYTHON } else { 'python' }
& $python run_simulation.py squid-soft-robot --steps 8 `
    --source-config ".\config\simulation_config.json"
```

可运行的 CLI 案例以 `run_simulation.py --help` 为准。COMSOL 两个案例提供编程式 smoke
入口，不应假定它们都由 `run_simulation.py` 分发。

## 验证边界

先运行与改动相关的 host-only/focused 测试，再按 [docs/VALIDATION.md](docs/VALIDATION.md)
选择结构检查、短程 Taichi 诊断或正式验证。focused 测试通过不等同于完整 GPU 回归、
长时程收敛或 Fluent 一致性验证；相关结论必须引用实际运行产物。

```powershell
& $python -m pytest -q tests/contracts
& $python -m pytest -q tests/cases/test_squid_package_exports.py `
    tests/benchmarks/test_official_benchmark_solver.py
```

只有明确需要数值验证且输出目录已经确定时，才启动 CUDA/Taichi 长任务。

## Repository layout

- `simulation_core/`：可复用求解器实现；
- `cases/`：案例专用装配、边界条件和报告；
- `benchmarks/official/`：官方基准合同与 canonical ANSYS runner；
- `tools/`：诊断、验证、渲染和后处理；
- `tests/`：按 solver、case、benchmark、integration 和 contract 分组的测试；
- `docs/`：当前架构、验证说明和历史重构记录；
- `archive/`：不参与生产导入的一次性历史脚本。

辅助工具应通过 `python -m tools.diagnostics...` 或
`python -m tools.rendering...` 的模块路径调用。
