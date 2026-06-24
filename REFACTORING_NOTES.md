# 重构说明（详细）

日期：2026-06-11。基线：上级目录 HIBM-MPM 工作区（main 分支，含未提交修改）。

原则：**行为不变优先**。只应用"可静态证明等价"或"把静默错误变成显式报错"的修改；
任何会改变数值轨迹的重写一律不做，只写入分阶段蓝图。

---

## 一、已应用的修改（逐条）

### 1. `simulation_core/runtime.py` — bug 修复
- **原 bug 1**：`default_fp` 只判断 `== "f32"`，否则一律静默落到 `ti.f64`。
  传 `"fp32"`、`"float32"` 等拼写错误会把整个仿真切到 f64（显存翻倍、速度减半）而无任何提示。
  → 现在 `default_fp ∉ {"f32","f64"}` 直接 `ValueError`。
- **原 bug 2**：`_INITIALIZED_ARCH` 赋值后从未被读取（写而不读的死状态）；
  第二次以不同 `default_fp` 调用 `init_taichi` 会被静默忽略。
  → 现在记录 `_INITIALIZED_ARCH/_INITIALIZED_FP`，fp 不一致时报错；
  `cuda`/`gpu` 视为兼容（gpu 是 cuda 的别名超集），不报错。
- 校验逻辑移到 `_INITIALIZED` 检查之前的部分保持原顺序（先验证参数再决定是否跳过），
  与原实现一致。

### 2. `simulation_core/validation.py` — 死代码删除
全仓 Grep 验证（simulation_core/、cases/、tests/、根脚本）后删除：
- `check()`（无任何调用方）
- `ValidationCheck`（只被 `check()`/`validation_summary()` 使用）
- `validation_summary()`（无任何调用方）

保留并保持签名不变：`vector_norm`、`finite_field_diagnostics`、`force_nonzero_when_loaded`、
`boundary_drive_compliance_report`、`checks_passed`、`FieldDiagnostic`、
`BoundaryDriveComplianceReport`（后两者为内部/报告结构）。

### 3. `simulation_core/geometry.py` — 死代码删除 + 等价向量化
- 删除 `vertex_area_weights()`：全仓无引用。
- `orient_faces_outward()`：原实现是逐面 Python 循环（O(faces) 解释器开销），
  对 38×105 球面网格约 8k 面循环。向量化版本用同一公式
  `dot(cross(b-a, c-a), centroid-center) < 0` 批量判定并交换 (v1,v2) 列，
  浮点运算与比较逐位等价（同为 float64，同一表达式结构），结果 winding 完全一致。

### 4. `simulation_core/fsi_coupling.py` — 死代码删除 + 恒等式化简
- 删除 `_solve_small_linear_system()`（约 40 行高斯消元，全仓无调用——
  IQN-ILS 实际使用的是 `_least_squares_coefficients()` 的 QR 路径）。
- 删除 `_max_secant_amplification()`（只是 `_secant_amplification_stats(...)[0]` 的包装，无调用方）。
- `_iqn_ils_interface_reaction_guess` 中
  `required_history = min(2, len(current_force_n) + 1)`：
  `current_force_n` 经 `_force_vector` 校验后长度 ≥ 1，故表达式恒等于 2，化简为字面量 2 并加注释。

### 5. `simulation_core/neo_hookean_mpm.py` — 死代码删除
- 删除 `_particle_grid_out_of_bounds()`：本类内核只用 `_particle_grid_stencil_out_of_bounds()`；
  mooney_shell_mpm.py 里的同名方法是各自类内的另一份拷贝，互不引用。
- 删除 `_read_vector()`：本类报告路径走 `report_host_snapshot`，此 staticmethod 无调用方
  （fluid.py 的同名方法是另一份拷贝且有使用，保留）。

### 6. `simulation_core/projected_ibm.py` — 边界条件加固
- `ProjectedIbmRegionPairStepConfig.__post_init__` 增加：
  `primary_region_id == secondary_region_id` 时 `ValueError`。
  原实现静默接受相同 region id，会导致同一组三角面被当作两个区域重复计力/重复施加速度目标，
  且 action-reaction 报告全部双倍——这是一个只会在配置错误时触发、但触发后极难定位的陷阱。
  所有现有调用方（squid 案例与测试）都传不同 id，不受影响。

### 7. 目录级清理
- `tools_*.py`（17 个）+ `run_phase0_raw_map_scaling.py` → `archive/tools/`。
  审计结论：全部为一次性调试脚本（硬编码了特定 run 目录、日期编号 010/041/163 等），
  无任何模块 import 它们；保留归档以备查。
- 根目录的进度/计划 `*.md` 不拷贝（属于会话工件，非代码）。

---

## 二、未应用、按蓝图分阶段执行的重构

> 这些改动会触碰数值热路径或 5k+ 行文件的结构，必须配合"跑同一配置 N 步、
> 对比 history.csv 全字段逐位一致"的等价性验证流程执行，故不在本次副本中直接应用。

### 阶段 A：共享设备端数学库（消除四重复制）
新建 `simulation_core/_device_math.py`，把以下 `@ti.func` 提为模块级共享：
- `_axis_grid_coordinate_device`（4 份：fluid.py:1611、hibm_mpm.py:698/2936、tri_surface.py:311）
- `_grid_coordinate_from_fields`（3 份：hibm_mpm.py:729/2967、tri_surface.py:340）
- `_sample_pressure_trilinear`（4 份）、`_sample_fluid_velocity_trilinear`（3 份）
- `_sample_velocity_gradient`（hibm_mpm/tri_surface 各一份）

Taichi 支持模块级 `@ti.func` 被多个 `@ti.data_oriented` 类的 kernel 调用，
技术上无障碍；预计净删 600–900 行。验证：全测试套 + 1 个 squid 短跑逐位对比。

### 阶段 B：`mooney_shell_mpm.py` 双胞胎类合并
`TriMooneyShellMpmState` 与 `UvMooneyShellMpmState` 有约 20 个方法逐字或近逐字重复
（`_scatter_particle`、`_interpolate_grid_velocity(_delta)`、`_atomic_add_*`、
`_accumulate_mooney_face`、报告打包等），约 700 行冗余。
方案：提取共同基类（`@ti.data_oriented` 继承可用）或组合一个 `_ShellMpmCore`。
注意已发现的**能力漂移**：Tri 有 `save_state/restore_state`，Uv 没有——合并时补齐。

### 阶段 C：`cases/squid_soft_robot/runner.py` 拆分（9608 行 → 包）
`run()` 一个函数 4463 行（4567–9030），`parse_args` 570 行。建议拆为：
```
cases/squid_soft_robot/
├── __init__.py        # main(argv)、run(args) 门面，保持入口不变
├── cli.py             # parse_args（argparse 全部定义）
├── spec.py            # SquidReducedSpec、infer_spec、spec_with_* 系列
├── geometry.py        # 水域几何、喷口锥形、graded grid 构建
├── schedules.py       # pressure_schedule_pa 等
├── checkpointing.py   # write/load_run_checkpoint、fingerprint、resume 校验
├── snapshots.py       # NPZ/VTI 快照、失败工件
├── coupling_step.py   # 每步 FSI 耦合内层（legacy 与 sharp 两条路径）
├── validation.py      # 各 *_passes / *_report 检查函数
└── outputs.py         # history.csv、summary JSON 组装
```
注意：`tests/test_source_static_contracts.py` 与 `tests/test_squid_latest_core_config.py`
钉死了本文件的多段**源码原文**（如 `"def spec_with_nozzle_graded_grid("`、`default=3000`）。
拆分时必须同步把这些契约测试改为指向新模块（契约本身建议改为行为断言而非源码文本断言）。
另外 run() 主循环内有两段完全相同的"数值守卫 + 失败工件"块（5891–5906 与 6921–6936），
以及循环体内定义的闭包函数（每步重建），拆分时合并/上提。

### 阶段 D：`fluid.py` 的 `CartesianFluidSolver` 分层（5257 行单类）
内聚子职责（带行号）：网格/规格装载（679–1056）、HIBM 障碍/可达性（1149–1480、3505–3690）、
状态快照（1481–1556）、预测步与速度约束（1582–2030）、试验场设置（1954–2030，测试用）、
力扩散/体力（2165–2335）、散度与分区报告（2431–2930）、Jacobi/FV-Jacobi（2930–3133）、
FV 算子与 CG（3134–3504）、多重网格（3690–3871）、出口通量报告（3871–3971）、
投影编排 `project()`（4777–5252）。
方案：按上述边界拆 mixin/协作对象，公开 API 不动。`project()` 内部把
"CG 统计归集"（4887–4929 的 nonlocal 闭包）收敛为一个小状态对象。

### 阶段 E：性能（两处算法级热点，见审查报告五）
1. `hibm_mpm.py::_scatter_marker_forces_to_mpm_particles_kernel`：
   O(markers × particles × 2) 暴力双循环 → 用 MPM 背景网格做粒子分桶（spatial hash），
   或按论文路径经网格散布。
2. `hibm_mpm.py::_search_and_classify_kernel`（及 grid_fields 变体）：
   O(全部网格节点 × 全部投影三角形) → 先把三角形写入粗网格桶，节点只查邻近桶；
   或只对 obstacle 边界带内的节点做全量搜索。

两者都改变浮点求和顺序，属于"数值上等价但非逐位一致"的改动，需用物理量收敛性验收。

---

## 三、验证记录（2026-06-11，执行模型 Fable 5 / claude-fable-5[1m]）

### 收尾（GOAL 任务 1）
- `_finalize_copy.py` 重跑：40 个文件全部已就位（copied 0 / skipped 40），
  `simulation_core/` 13 个 .py、`tests/` 16 个 .py，树完整。
- 补齐 2 个被 `tests/test_simulation_core_package.py` 钉死的文档契约：
  `SIMULATION_CORE_USAGE.md`、`HIBM_MPM_PAPER_VS_CODE.md`（首跑第 3 组时 3 个
  `FileNotFoundError`，从原目录原样复制后修复）。两文件已加入
  `_finalize_copy.py::UNCHANGED_FILES` 清单。

### 测试（GOAL 任务 2/3，全部在 refactored/ 下用 `python -m unittest` 执行）
| 组 | 模块 | 结果 |
|---|---|---|
| 1（纯 Python） | test_validation + test_fsi_coupling + test_generic_entrypoint | **51 tests OK**（0.17s） |
| 2（静态契约） | test_source_static_contracts | **13 tests OK**（0.39s） |
| 3（CUDA） | test_simulation_core_package + test_hyperelastic_ecoflex | **30 tests OK**（2.1s，文档补齐后） |
| 4（CUDA 慢档） | test_neo_hookean_mpm + test_projected_ibm + test_fluid | **63 tests OK**（824.9s） |

四组合计 **157 tests 全绿**，无任何代码修复需求。

### 抄写漂移核验（GOAL 任务 3 的预防性执行）
6 个修改模块逐一与原文件做 `git diff --no-index` 全文对比：
差异**仅含**本文档第一节记录的故意修改，无任何非故意漂移。
附核验补充：`validation.py` 删除 `validation_summary` 后 `asdict` 导入仍被
4 处使用（非死导入）；`projected_ibm.py` 新增的 `int()` 强转与原代码全部下游
消费点（`int(config.primary_region_id)` 等 8 处）语义一致且幂等。

### 功能等价冒烟（GOAL 任务 4）
- 命令：`run_simulation.py squid-soft-robot --steps 8 --source-config
  _codex_validation/squid_case_run_render_20260610_005/simulation_config.json
  --projection-divergence-tolerance 1.0`（两棵树参数完全一致）。
- 容差说明：005 配置原本是给上层 `sim_core.py` 求解器的，在本案例默认守卫
  （interior_divergence_l2 ≤ 1e-2）下 step 3 即触发（实测 1.0067e-2，原代码行为，
  与重构无关）。`--projection-divergence-tolerance` 只是事后守卫阈值，不进入
  数值轨迹（CG 停止准则是独立的 `cg_tolerance`），放宽到 1.0 仅为让 8 步跑完。
- 归因方法：原目录跑两次测**运行间噪声底**（Taichi CUDA 原子加法求和顺序非确定），
  再与"原 vs refactored"对比（工具与输出存于 `_smoke_equivalence/`）：

| 指标（summary.json 全部 727 个浮点叶子） | 原 vs 原（噪声底） | 原 vs refactored |
|---|---|---|
| 逐位相同 | 348（47.9%） | 349（48.0%） |
| 中位相对差 | 2.63e-06 | 2.45e-06 |
| p90 相对差 | 2.09e-04 | 1.57e-04 |
| 最大相对差 | 5.0e-01（近零诊断残差 ~1e-8） | 5.8e-01（同类字段） |

- 结论：**原 vs refactored 的差异 ≤ 原代码自身运行间噪声**（中位/p90 更小），
  两次对比的最大差异字段都是绝对值 1e-7~1e-8 的近零诊断比值。整数/布尔/字符串/
  结构字段全部一致（completed_steps=8 等）。**功能等价成立。**

### 清理（GOAL 任务 5）
- `fix_remove_deepseek.py`（一次性环境修复，已执行完毕）→ 移入 `archive/`。
- 冒烟对比工件保留在 `_smoke_equivalence/`（original / original_rerun / refactored
  三组输出 + compare_summaries.py / noise_stats.py 两个对比工具）。

### 可选最后一档（不在验收门槛内）
- test_squid_latest_core_config + test_core_fluid：**251 tests，1079.1s，250 通过 / 1 失败**。
- 唯一失败 `test_default_output_directory_is_gitignored`：环境问题而非代码问题——
  根 `.gitignore` 的 `cases/output_008step/` 模式（含中间斜杠）锚定到仓库根，
  不覆盖 refactored 子树，`git check-ignore` 对 `refactored/cases/output_008step/...`
  返回 1。修复：新建 `refactored/.gitignore`（镜像原工件模式 + `_smoke_equivalence/`），
  单测重跑 **OK**。该文件是副本树专属新文件，不在 `_finalize_copy.py` 清单内。
- 至此全部 18 个测试文件（157 + 251 = 408 个测试）在 refactored/ 下全绿。
