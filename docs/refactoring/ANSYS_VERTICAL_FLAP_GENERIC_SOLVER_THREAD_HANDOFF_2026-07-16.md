# ANSYS vertical-flap 通用求解器线程交接（2026-07-16）

> **2026-07-20 最终冻结说明：**第 1--11 节是历史过程，不能再作为当前执行指令。当前代码、正式运行、已证伪假设、剩余根因和清理状态以第 12 节为准。用户已明确要求分析完成后暂停；本线程没有启动下一轮 A/B 或 checkpoint 实现。

## 1. 当前冻结状态

- 用户已明确要求暂停并交接；本线程不再继续补测试，也没有启动下一次生产门禁。
- 冻结检查时 `Get-Process -Name python,pythonw` 返回空：当前没有 Python/仿真进程。
- 三路只读审查中，测试审查已完成；代码审查和数值审查已被主动中断，避免旧线程继续工作。
- **v8 从未启动**。下面列出的 v8 路径目前均不存在，必须保持为下一线程的全新目录。
- 不要恢复、续跑或误判 v1--v7 的失败目录；失败工件是只读证据。
- 工作树原本就有大量用户修改和验证工件。本线程没有 stage、commit、reset 或清理这些改动；下一线程也必须保留无关修改。

## 2. 最终目标和不可变约束

目标是用当前仓库中的**通用** HIBM/MPM/FV 求解器完成 ANSYS/Fluent vertical-flap 验证，而不是为本算例加入专用分支：

1. 生产规模 1 步门禁；
2. 40 步 fixed-solid preflow 并写入可校验快照；
3. 从快照启动 50 步生产仿真并保存逐步字段；
4. 与最新 Fluent 基准做数值对比；
5. 输出固定色标 `0..31 m/s` 的 GIF，隐藏 HIBM marker/rest positions，非流体/solid 为黑色，且无白色 halo/圆框或青白 marker 边框。

硬约束：

- 任何时刻只能有一个仿真/Python 进程。
- 不修改物理配置、健康门、CG 预算/容差或 component capacity。
- 不加入 case-specific 求解分支，不硬编码本算例结果，不伪造通过或工件。
- 失败后必须先读 artifact，按 RED -> 最小通用修复 -> GREEN -> 只读复审的顺序处理，只能使用全新目录重试。
- 验证命令使用 `$env:EASYFSI_PYTHON` 指定的解释器；未设置时使用当前环境的 `python`。

## 3. 到目前为止完成的工作

### 3.1 较早阶段完成的通用修复

本线程前半段已完成并验证过下列通用合同，后续不能回退：

- 快照 NaN 合同只对白名单内的 9 个可选、未触达的 zmin projection diagnostics 允许 NaN；健康字段、未知 NaN 和 inf 仍 fail-closed。
- direct velocity hard constraint 与 pressure external-normal provenance 分账。
- HIBM-owned rows 保留 direct 语义；face-symmetric 镜像使用 pressure-effective mask；多外边界交线按位合并。
- snapshot schema 4 持久化上述 provenance 并严格校验。
- external-exact pressure provenance 条件式路由：无 external provenance 时保留 direct；有 external provenance 时使用 `direct & (owned | external)`。
- pressure/interface、force/geometry 使用 f64；multigrid 正体积检查；topology cache invalidation；JSON-safe graph failure diagnostics；`pressure_solve_context/phase` 不再隐式字符串化。
- 这些修复不含算例专用分支。此前分阶段回归、`py_compile`、`diff-check` 和独立审查均曾通过；具体旧批次证据保留在本线程日志和对应测试中。

### 3.2 v7 失败的根因已经冻结

v7 目录：

`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\runs\current_solver_preflow1_joint_qp_probe_genericfix_v7_20260716`

事实来源：

- `failure.json`
- `our_solver_config.json`
- `run_manifest.json`

v7 在第一步压力求解前严格失败：

- `PressureSolveConvergenceError`
- stage：`pressure_projection_physical_gate`
- reason：`unreached_component_label_overflow`
- elapsed：`412.85426240001107 s`
- outlet-unreached cells：`1548`
- raw Cartesian physical components：`1296`
- diagnostic capacity / compact count：`768`
- physical overflow：`true`
- active exact interface rows：`1584`
- invalid/overflow rows：`0/0`
- diagonal integral mismatch：`0`
- CG/PCG/BiCGSTAB iterations：`0/0/0`
- 无 summary、无 preflow history、无 snapshot JSON/NPZ。

根因不是 CG 不收敛，而是旧的 Cartesian flood component capacity gate 在 exact positive interface rows 组成真实压力算子图之前提前失败。实际 exact interface rows 可以合并或锚定大量看似独立的 Cartesian pockets，因此 capacity 应当应用于**最终未锚定的 exact-operator roots**，而不是应用于预接口的物理诊断分区。

### 3.3 已实现的当前通用修复

主要修改位于 `simulation_core/fluids/solver.py`：

- 新增 full-grid pressure outlet operator labels/raw-root 字段和 f64 diagonal/scale ledgers。
- 基础 Cartesian 边使用 `_pressure_cells_connected()`；当 row list 非空时以 row list 为 authoritative exact interface graph，旧 coupling 仅作为 row_count=0 的兼容 fallback。
- 只联合有限且严格正的 transmissibility。
- zmin outlet 且 mobility 非零的 roots 被标记为 anchored。
- exact interface rows 从两端 diagonal ledger 扣除；只把真实剩余 diagonal excess 视为 anchor，并使用与 f64 累加误差相称的容差。
- 只有最终未锚定 roots 被压缩到 capacity 768；若最终 roots 仍超过 768，必须结构化 fail-closed，且不能发布 graph。
- `_prepare_pressure_outlet_nullspace_component_graph()` 允许合法的 physical flood overflow 被 exact operator graph 重新分类，但 labels 未收敛、row/provenance 无效、union 未收敛或最终 roots 超容量仍失败。
- `project()` 在最终 hard-mask refresh 之后、旧 physical structural gate 之前建立 graph。
- FV-CG/BiCGSTAB/exact residual confirmation 复用同一个 graph source/context，避免重建漂移。
- 旧 closed-Neumann 路径保持独立。

重要语义核对：`hibm_pressure_reachability_barrier` 只属于旧的出口可达性诊断。实际 `_fv_diagonal_kernel`、`_fv_laplacian_apply_kernel`、`_weighted_dot_kernel` 都只排除 `obstacle`，并不排除 reachability barrier。因此新 operator graph **应包含** barrier cells；把它们排除会与真实 FV operator 不一致。

### 3.4 RED/GREEN 和静态验证证据

核心 RED（旧代码）覆盖：

- 769 个 physical pockets 经 exact rows 合并前，旧逻辑先报 `component_capacity_overflow`；
- `project()` 先报 `unreached_component_label_overflow`，CG 未进入。

核心 GREEN 命令：

```powershell
$python = if ($env:EASYFSI_PYTHON) { $env:EASYFSI_PYTHON } else { 'python' }
& $python -m unittest -v `
  tests.solvers.test_core_fluid.UnreachedSetInterfaceHitObservabilityTests.test_over_capacity_cartesian_pockets_collapse_to_one_operator_root_before_capacity_gate `
  tests.solvers.test_pressure_projection_physical_failure_contract.PressureProjectionPhysicalFailureContracts.test_active_exact_operator_reclassifies_physical_overflow_before_structural_gate
```

结果：`2/2 passed`，约 `156.201 s`。

补充合同：

- row-list edge 连到 outlet-reachable cell 后 final root count 为 0；
- 结构化 mock 反向合同验证 final count=769 时抛 `component_capacity_overflow`、source count=769、graph 未发布。

结果：`2/2 passed`，约 `91.902 s`。

测试 fixture 漂移修复：四个测试原来只写旧 scalar active 标记；现显式写 hard mask=7 并刷新 derived pressure-hard mask。该改动只修复测试，不改变生产语义。结果：`4/4 passed`，约 `191.873 s`。

其他证据：

- 真实 operator 数值合同：`4/4 passed`，约 `206.707 s`。
- 相关三个测试文件：`11/11 passed`，约 `155.660 s`。
- `UnreachedSetInterfaceHitObservabilityTests` 的 37 个方法均有分批绿色证据。一次大批量运行因 30 分钟外层超时/JIT 内存约 7 GB 被终止，并非 assertion failure；随后按小批次补齐。
- 相关文件 `py_compile` 通过。
- `git diff --check` 通过，只有既有 LF -> CRLF warning。

当前涉及本轮修复的文件：

- `simulation_core/fluids/solver.py`
- `tests/solvers/test_core_fluid.py`
- `tests/solvers/test_pressure_projection_physical_failure_contract.py`
- `tests/solvers/test_pressure_nullspace_graph_structured_failures.py`

## 4. 尚未完成的问题（必须先解决，不能直接跑 v8）

测试只读审查已完成，提出以下未闭环项：

1. **High：需要 kernel-level 反向容量合同。** 当前已有一个 mock/host-level 结构化反向测试：`test_outlet_exact_graph_capacity_gate_uses_final_unanchored_roots`。但审查要求再用真实 769-pocket fixture 验证：激活 exact operator 但不连接这些 pockets，最终仍有 769 个未锚定 roots 时，必须抛 `PressureSolveConvergenceError(reason=component_capacity_overflow, count=769)`；graph 不发布，CG 不进入。该项可防止 kernel 内部静默裁剪造成假绿。
2. **Medium：增强 project 集成断言。** `test_active_exact_operator_reclassifies_physical_overflow_before_structural_gate` 目前只记录 solve 被调用，需捕获 kwargs 并断言 `pressure_components_use_operator_graph=True`、`pressure_nullspace_component_count` 和 remove-mean policy；建议覆盖 final count=0 和 1。
3. **Medium：补同图复用链。** 需要覆盖 `project prepare -> CG 不重建 -> BiCGSTAB/exact confirmation` 使用相同 source/final/context，至少包含 zero-root；可在 prepare 后替换重建函数为 forbidden/counter。
4. 代码审查和数值审查因用户要求暂停而被中断，**没有最终结论**。下一线程必须重新进行这两路只读审查；不能把此前“审查进行中”写成通过。

在以上项完成并通过之前，不应启动生产门禁。

## 5. 下一线程的推荐执行顺序

1. 先核实没有 Python 进程，并重读本交接、v7 `failure.json`、当前 diff。
2. 按测试审查意见补 RED 合同；先证明合同能捕获错误/缺口，再做最小通用实现调整（若现实现已满足，则记录为新增防回归合同，不要为了制造变化而改 solver）。
3. 运行新增测试、相关小批测试、`py_compile`、`git diff --check`；避免把所有 Taichi fixture/JIT 聚在一个长进程中，防止 7 GB 级缓存和外层超时。
4. 完成代码+数值两路只读复审，必须 High=0、Medium=0；测试审查也需确认上述缺口已闭环。
5. 再确认无 Python、目标路径全不存在，然后公开通知用户即将启动，才可启动全新 v8 生产级 1 步门禁。
6. v8 通过后才启动全新 40 步 fixed-solid preflow；40 步通过并校验快照后才启动 50 步生产运行。
7. 50 步通过后与 Fluent 数值对比并渲染 GIF；视觉相似不能替代数值一致性。

## 6. v8 预留路径（尚未创建）

运行目录：

`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\runs\current_solver_preflow1_joint_qp_probe_genericfix_v8_20260716`

日志前缀：

`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\launch_logs\current_solver_preflow1_joint_qp_probe_genericfix_v8_20260716`

快照前缀：

`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\preflow_snapshots\current_solver_fine_joint_qp_probe_genericfix_v8_20260716\preflow_state`

启动脚本：

`validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\scripts\run_our_solver_vertical_flap.py`

Windows 启动注意事项：

- 必须把含空格的脚本绝对路径作为一个完整参数传给 `Start-Process`；v6 就是因路径拆分而零步失败。
- 后台窗口用 `-WindowStyle Hidden`。
- 若当前 PowerShell 同时存在 `Path`/`PATH` 导致启动器报重复键，只在该启动 PowerShell 中清理重复项，再设置单一 `Path`。
- anisotropic radius/probe 必须使用当前代码计算出的默认值；不要在命令行用十进制字面量重新表达，以免 snapshot identity 漂移。

## 7. v8 必须保持的精确配置

以 v7 的 `our_solver_config.json` 为 identity 来源；v8 只能改变 run/output/snapshot 路径：

- steps=0, preflow_steps=1, `single_step_legacy`
- grid=`(4,256,320)`
- solid particles=`(1,256,20)`
- marker_count=64，dual physical faces 共 128
- dt=`0.0005 s`
- projection iterations=1080
- post-Dirichlet consistency projections=3
- FV multigrid preconditioner
- CG tolerance=`1e-6`
- predictor substeps=64
- solid substeps=1600
- dynamic solid volume=true
- current-code computed defaults：radius xyz=`(0.0012000000000000001, 0.000390625, 0.00046875)`，probe=`0.0011250000000000001`
- `flow_hibm_sharp_interpolate_velocity_rows=false`
- pressure failure policy=`raise`

一步门禁成功必须满足：正常退出、无 failure/interruption、summary `completed` 且 requested steps=0、1 行健康 preflow history、joint Q/P measured+converged、128 markers 全部 valid/direct、CG/pressure/no-slip/物理门通过、snapshot manifest+generation NPZ integrity+identity 通过。

## 8. 后续 40/50 步和 Fluent 对比

只有 v8 严格通过后才定义新的 40 步目录（不要复用历史 v1--v3）。40 步需要：正常退出、40 行连续有限健康 preflow history、summary completed/requested steps=0、snapshot manifest+generation NPZ 通过当前 loader 的 integrity+identity。

50 步必须从该新快照启动，保留 preflow identity=40，并启用 `--save-step-fields`；需要 summary/progress completed、requested=completed=50、50 行 history、50 个连续 step_fields 和 50 个连续 step_history、所有字段有限、marker/CG/pressure/physical guards 全部通过。

最新 Fluent 基准：

`validation_runs\ansys_vertical_flap_fsi\official_fluent_fine_fsi_valid_2026-07-10\runs\fresh50_20260713_104843\postprocess_compare31_20260713`

最终 GIF 曾复制到 Codex 会话本地的 visualization 目录；该目录不属于仓库，正式证据仍以本节列出的 repo-relative artifact 路径为准。

## 9. 明确不要做的事

- 不要把 v7 物理门失败当作 CG failure。
- 不要降低 0.5 m/s no-slip 健康门或任何现有压力/物理门。
- 不要提高 component capacity 来掩盖 graph 分类错误。
- 不要跳过一步门禁直接跑 40/50 步。
- 不要复用失败目录、旧快照、旧 final fields 或旧 step fields。
- 不要把渲染去掉白色边框当成求解器数值正确；必须完成 Fluent 数值对比。
- 不要在共享脏工作树上 reset/checkout 用户修改，也不要未经要求提交。

## 10. v38u/v38v 连续运行收口结果

本节覆盖本交接文档之后继续完成的工作。结论必须分成两层：

1. **连续运行稳定性问题已用通用几何不变量修复，并通过当前 40+50 步生产流程。**
2. **与 Fluent 的数值 parity 仍未通过；本次完成的是诚实的诊断对比，不得把运行完成或视觉相似写成 parity。**

### 10.1 第 33 步故障根因和通用修复

旧 v38t 已越过早先第 13 步故障并连续完成 32 步，但在第 33 步搜索前触发：

`RuntimeError: invalid projection segment geometry before IB search: count=9`

保存字段表明，3D 固体法向经过 Nanson 更新后会在 2D 挤出算例的非活动 `x` 轴上积累有限漂移。第 32 步最大 `|n_x|=9.81418e-6`，按第 31--32 步外推，第 33 步正好有 9 个相邻 segment 越过旧的 `1e-5` 硬阈值；对应报错的 9 个 segment。与此同时，活动 `y-z` 平面内 segment 长度和法向都健康，因此这不是退化几何。

通用修复位于 `simulation_core/coupling/hibm_mpm/core.py`：

- 对声明了 inactive axis 的 2D/挤出搜索，统一将 marker/segment 法向投影到活动搜索平面并归一化；
- segment 健康检查、segment 插值、point-marker 搜索和压力边界法向使用同一个活动平面几何定义；
- 真正的零活动平面法向或端点抵消仍然 fail closed；
- 3D/triangle 搜索保持原行为；
- 没有提高 `1e-5` 阈值，也没有加入 step、marker id 或本算例专用分支。

回归测试位于 `tests/solvers/test_hibm.py`：

- `test_inactive_axis_segment_search_projects_finite_normal_drift`
- `test_inactive_axis_point_search_projects_normal_before_probe`
- `test_inactive_axis_segment_search_rejects_zero_active_normal`

上述场景先观察到 RED，再实现最小修复。相关验证结果：7 个聚焦测试通过，`HibmMpmIbNodeSearchTests` 22/22 通过，3D triangle 保持测试通过。

### 10.2 全新 40 步 preflow 与快照

运行目录：

`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\runs\current_solver_preflow40_active_plane_normal_v38u_20260718`

结果：`status=completed`，40/40 preflow history 连续；压力失败 0，CG 未收敛 0，最大 CG 相对残差 `9.821242343045949e-7`，最大投影后 no-slip 残差 `1.4901264222544341e-8 m/s`。求解器耗时 `1633.9934091 s`。

快照前缀：

`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\preflow_snapshots\current_solver_fine_preflow40_active_plane_normal_v38u_20260718\preflow_state`

快照哈希：

- JSON SHA256 `214AE567D0C755EB86CEBCA27B25F9A03547B9F350D4482565734AA38B0EA3CE`
- generation NPZ SHA256 `4F64B8614D5A7A46961984BBD3692170DF92466CC0F05F0D89A8344E53494C66`

另有独立 snapshot-load check，成功报告 `preflow_snapshot_loaded=true`。

### 10.3 从该快照连续完成 50 步 FSI

运行目录：

`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\runs\current_solver_fsi50_from_preflow40_active_plane_normal_v38v_20260718`

结果：

- `status=completed`，requested/completed=`50/50`；
- `step_artifact_validation=passed`，50 个连续 field 和 50 个连续 history；
- 50 个 NPZ 中所有数值数组有限；
- 压力失败 0，CG 未收敛 0，最大 CG 相对残差 `9.490970915774357e-7`；
- 最大投影后 no-slip 残差 `8.099999831756577e-5 m/s < 1e-4 m/s`；
- 最大位移 `0.0003772251767444901 m`，出现在第 9 步；
- 保存法向的最大 `|n_x|=3.579669864848256e-4`，是旧 `1e-5` 门的约 35.8 倍，但最小活动平面法向模仍为 `0.9999998629486639`。

最后一项是关键的通用性证据：新流程不是把阈值稍微调大，而是按搜索维度使用正确的几何不变量。当前证据支持本 50 步生产流程连续运行，不等于已经证明任意 5000 步都不会遇到新的独立物理或数值失效。

### 10.4 最新 Fluent 离线对比和 GIF

离线验证层还修复了两处陈旧契约：列表中的有限浮点数现在逐项使用约 1 ULP 的严格比较，整数身份明确拒绝 bool；canonical Fluent 身份使用规范化后的绝对路径精确匹配，且 CLI 默认路径统一锁定本文件第 208 行的 fresh50，带 `-copy/-spoof` 后缀的近似路径会 fail closed。实际 locked 配置为 `flow_hibm_sharp_interpolate_velocity_rows=false`。报告元数据也改为如实记录只绘制 deformed solid、隐藏 HIBM markers 和 rest positions。相关两个验证文件共 65 个测试通过。

新输出目录：

`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\postprocess\current_solver_fsi50_active_plane_normal_v38v_strict_contract_20260718`

产物校验：

- `comparison_report.json`：`status=diagnostic_complete`，`parity_claimed=false`；
- 50 张连续 PNG，GIF 50 帧、`1260x532`、每帧 `120 ms`、循环播放、固定 `0..31 m/s`；
- 58 行 `CHECKSUMS.sha256` 全部复算通过；
- GIF SHA256 `D986C2A4CAC387EA0C6846131EAAE5490D35A61AF6108065D02DA0497CCF1203`；
- 目视检查第 1、9、50 帧：solid/非流体区域为黑色，未见 HIBM marker 圆环、rest-position 轮廓或白/青 halo。

对比结果必须诚实保留：入口质量流量相对误差 `0.0004295`、出口 `0.0043644` 通过 5% 诊断门；但压力降相对误差 `0.36777`、中心线速度 NRMSE `0.12150`、喉部速度 NRMSE `0.12030`、下游速度 NRMSE `0.20311` 均未通过 5% 门。tip 位移曲线相位一致且峰值都在第 9 步，但当前 solver 峰值 `0.00037103 m`，Fluent 峰值 `0.00052417 m`，峰值幅值偏低约 29%。因此下一阶段问题是流场/压力和结构响应幅值的物理一致性，而不是本次已修复的 inactive-axis 搜索中断。

### 10.5 后续长运行原则

若后续扩展到 5000 步，应继续使用同一套流程：先把每个已知故障压缩为维度/拓扑/守恒等通用不变量的 RED 测试，再跑连续 soak；生产运行只在真实不变量破坏时 fail closed，并保留最后健康 step、失败 step 和完整诊断。不要为某一步、某个 marker 或某个算例增加跳过逻辑。当前下一项独立工作应围绕 Fluent 差异做有证据的物理诊断，不应回退本节的活动平面几何修复。

## 11. 2026-07-19 流场、压力和结构幅值续接

### 11.1 修改前的可复现 SST 基线

本轮以 v39c/v39d 为修改前基线，不以短探针代替完整运行：

- 40 步 preflow：`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\runs\current_solver_preflow40_sst2003_generic_v39c_20260718`
- 50 步 FSI：`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\runs\current_solver_fsi50_from_preflow40_sst2003_generic_v39d_20260718`
- strict comparison：`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\postprocess\current_solver_fsi50_sst2003_generic_v39d_strict_contract_20260718`

v39d 完成 50/50，耗时 `1250.8291268 s`。solver/Fluent tip 峰值分别为 `0.0003985296345/0.0005241735376 m`，均在第 9 步，solver/Fluent 比值 `0.760301`，即幅值偏低 `23.97%`。第 9 步 solver 流体反力为 `0.0092554430 N`，相对 Fluent `0.014623 N` 约为 `0.633`。solver 最终压力极值为 `322.205/-32.229 Pa`，pressure-drop/range relative error 分别为 `0.3762096/0.3776133`。centerline/throat/downstream NRMSE 分别为 `0.1240745/0.1172809/0.2044112`，near-wall backflow fraction absolute error 为 `0.0635903`；质量流量入口/出口相对误差 `0.0005172/0.0045489` 仍通过。`parity_claimed=false` 必须保留。

### 11.2 失败链与通用修复

| 运行 | 真实结果 | 暴露的问题 |
|---|---|---|
| v40a/v40b | 各完成 1 个细网格步，约 1185 s | 只能证明短探针，不证明长程可用 |
| v40c/v40d | MUSCL stage 非有限；v40d 第 6/40 步速度达到约 `2.27e17 m/s` | 缺少 stage rollback/最终提交门 |
| v40e | 第 17/40 步要求超过 4096 个 SST 显式扩散子步 | 显式 `D/h^2` 算法瓶颈，不能靠增大上限掩盖 |
| v41a | 第 1 步 520 个 SST 显式候选非正 | 候选提交前缺少状态快照、回滚和物理时间片减半 |
| v41b | 第 6/40 步失败；前一健康步速度 `61.7656 m/s`、`k=243.168`，final MUSCL rate 非有限 | `div(uq)` 把有限 projection/HIBM `div(u)` 残差变成伪压缩源 |

核心通用修改位于 `simulation_core/fluids/solver.py`：

- 使用 backward-MAC 物理面质量通量的 MUSCL/TVD、平滑极值 MC limiter、SSP-RK2、自动 CFL 分片与 stage rollback/commit gate；
- SST 标量采用显式对流/源项 + 冻结变系数三轴 LOD backward-Euler 扩散；
- SST 动量的 coercive 变系数 Laplacian 采用三轴 LOD backward-Euler，保留完整对称 Boussinesq transpose/cross stress；
- wall omega 使用统一物理目标并在数值上限前 fail closed，不静默裁剪；
- SST 显式候选和隐式结果都在提交前检查正性，失败后恢复 base、减半 trial `dt`，被拒 trial 不推进物理时间；
- 动量和 SST 对流使用 `div(uq)-q div(u)` 的一致 advective-form 修正，使有限投影残差不能放大常量状态；
- SST 连续性修正只使用与标量通量相同的 relative-through-flow 面集：fluid/open 面使用物理速度，moving obstacle/no-slip 面为零，避免壁面运动伪造 `k/omega` 穿透通量。

### 11.3 RED/GREEN 与长程证据

- projection residual 下常量横向速度：旧实现最大误差约 `0.0164511`，修复后 GREEN；
- projection residual 下均匀 SST：旧实现产生 `k` 峰谷差 `0.0317147`，修复后 GREEN；
- moving obstacle 两种 backward-MAC 存储方向：旧实现均产生 `0.0317146` 的伪 `k` 差异，relative-flux 面集修复后 GREEN；
- MUSCL 完整文件：`7/7 passed`，`633.412 s`；
- SST 完整文件（含 moving obstacle）：`24/24 passed`，`2231.369 s`；
- 最终源码的 5000 步小网格 soak：`passed`，`183.662 s`，包含第 2500 步 checkpoint replay、有限性、`k/omega>0` 与确定性合同；这只是算法 soak，不等于 fine-grid 5000 步 FSI 已完成；
- `py_compile` 和相关 `git diff --check` 通过；后者仅有既有 LF -> CRLF warning；
- moving-obstacle 修复只读复审：Blocking/High/Medium=`0/0/0`。

### 11.4 v42a 真实 40 步结果与当前运行

continuity-only v42a：

`validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\our_solver\current_solver_preflow40_muscl_tvd_dual_lod_continuity_v42a_20260719`

真实结果为 40/40 completed，耗时 `1609.6674315 s`。40 行 preflow 连续；pressure failure=0，最大 CG relative residual `9.9728839728e-7`，最大 post-projection no-slip residual `1.1921051168e-7 m/s`，SST 非有限/非正单元和 wall-omega guard 均为 0。最大速度 `54.22739 m/s`，最终 `44.41191 m/s`；全程 `k_max=219.80327`、`omega_max=1514834.125`、`mu_t,max=0.002478871 Pa s`。第 1 步拒绝并恢复 6 个 SST trial，后续 39 步零拒绝；动量 stage 拒绝总数为 0。最大隐式 SST 动量 diffusion CFL 为 `365.91186`，没有再次触发显式子步硬停。

v42a 之后补入 moving-obstacle relative-flux 修复，因 snapshot identity 含核心源码 hash，v42a 快照不能绕过校验直接用于 FSI。当前正在用冻结后的最终源码重跑 v42b 40 步：

`validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\our_solver\current_solver_preflow40_muscl_tvd_dual_lod_continuity_movingwall_v42b_20260719`

v42b 完成前不得写成 completed；完成并校验快照后，50 步加载时必须保留 `--preflow-steps 40`，不能设为 0，否则 strict snapshot identity mismatch。后续仍必须完成全新 50/50 FSI、50 个连续 step fields/history、strict Fluent comparison，并以压力、速度剖面、step-9 反力和 tip 峰值比判断幅值是否真正改善；仅运行稳定不等于 parity。

## 12. 2026-07-20 最终冻结：unsplit 动量扩散、73 步稳态预流、FSI50 与严格 Fluent 对比

### 12.1 冻结和清理状态

- 用户要求“分析完成后暂停，不要有下一步动作，清理工作树，再写交接文档”。本节写完后不再启动仿真、A/B、测试或实现。
- 冻结检查时没有本项目的 Python/Taichi 或后处理进程；所有并行只读审查均已结束。
- 本轮只删除了三个非正式短探针/失败目录：`current_solver_preflow1_unsplit_compile_v46a_20260720`、`current_solver_preflow1_unsplit_template_v46a2_20260720`、`current_solver_preflow8_unsplit_template_v46b2_20260720`，以及本轮触达目录中的 `__pycache__`。
- 正式的 73 步预流、schema-v8 快照、50 步 FSI、50 组逐步工件和严格 Fluent 后处理全部保留，路径见 12.4--12.6。
- `.pytest_cache` 创建于 2026-06-17，早于本轮且 ACL 拒绝访问，未强改权限或删除。
- 仓库在本轮之前就有大量 tracked/untracked 用户修改与历史验证工件；因此 `git status` 仍非 clean。本轮没有执行 `git reset`、`git checkout`、`git clean`、stage、commit 或 push，不能为了表面 clean 破坏这些既有工作。

### 12.2 本轮实现的通用数值修复

主要生产修改位于：

- `simulation_core/fluids/solver.py`
- `benchmarks/official/solid_mpm_fsi_runner.py`

新增/更新的合同位于：

- `tests/solvers/test_cartesian_fluid_sst_transport.py`
- `tests/solvers/test_sst_momentum_helmholtz_contracts.py`
- `tests/benchmarks/test_vertical_flap_sst_runner_contract.py`

实现内容：

1. 把生产 SST 动量黏性步从单次 `x -> y -> z` backward-Euler LOD 物理解，改成体积乘权、共享边系数的 unsplit 七点 Helmholtz 系统 `A = V + dt*K`。
2. 每条 free/free 边只装配一个正 transmissibility，并以相同系数写入两行负非对角；自由子空间因此满足普通欧氏 PCG 所需的对称正定条件。
3. normal MAC 边跨越真实 scalar cell；transverse 边累加两个 component half-patch；分级网格、变量 `nu_t`、外边界、障碍和 canonical A/B owner 都按物理面积/距离进入对角和 Dirichlet RHS。
4. 当前预条件器是固定 Jacobi；求解后用真实 `b-Ax` 残差复核，非有限曲率、非正曲率、无效对角和迭代上限均结构化 fail-closed。
5. 三个速度分量串行复用压力求解 scratch，压力装配随后完整重写；没有 CPU/GPU 场往返。
6. 动量隐式失败使用整场事务：先恢复 advection transaction base，再只用更小的 unsplit IMEX 子步重试；禁止退回旧 xyz-LOD。失败分量不得泄露部分 scatter。
7. runner 新增 `flow_sst_momentum_helmholtz_{converged,iterations_last,iterations_total_last,relative_residual_last,rejected_trial_count_last}` 和 integrator/substep/CFL 诊断。
8. 同时修复旧 LOD helper 中三个独立几何/边界错误：normal variable-viscosity 取真实 crossed scalar cell，normal xmax 使用完整 MAC-cell 距离，所有方向 sweep 上 exact owner 使用权威 orientation-owner target 而不是 stale velocity。

当前 integrator 身份必须保持：

`unsplit_volume_symmetric_pcg_jacobi_frozen_coefficients`

### 12.3 RED/GREEN 和聚焦验证

- 新 Helmholtz 合同最初 RED 为 `3 tests / 8 failures`：normal variable-`nu` 系数、A/B owner 在 transverse sweep 中衰减、xmax normal 距离错误。
- 修复后 5 个装配合同通过；再加二维 manufactured unsplit Neumann mode 和高 CFL `predict()` 路由，共 `7/7 passed`，约 `276.207 s`。
- 6 个既有边界/owner/分级网格合同通过，约 `271.188 s`。
- 新增“失败 PCG 污染全场后必须完整回滚且旧 LOD 零调用”和“moving A/B owner 精确消元”合同：`2 passed + 2 subtests passed`，约 `101.34 s`。
- runner 新诊断字段合同：`1/1 passed`。
- 相关 `py_compile` 通过；`git diff --check` 通过，只有工作树既有 LF -> CRLF warning。
- 另有最终源码的 5000 步小网格 canonical soak（第 2500 步 checkpoint replay）在此前阶段通过；它证明核心算法 soak，不等于 fine-grid 5000 步 FSI 已完成。

### 12.4 正式 73 步 windowed-stationary 预流

运行目录：

`validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\our_solver\current_solver_preflow200_windowed_sst_unsplit_v46c2_20260720`

快照前缀：

`validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\preflow_snapshots\current_solver_fine_preflow200_windowed_sst_unsplit_v46c2_20260720\preflow_state`

结果：

- requested/completed=`200/73`，`status=stop_reason=windowed_stationary`，不是人工停止或超时；耗时 `1681.8510669 s`。
- 最后连续 3 个 10 步窗口通过；最后窗口最大指标为 `mu_t,max relative span=0.00691738 < 0.01`，压力范围 relative span=`0.000125457`，速度峰 relative span=`7.48003e-5`。
- 73 步全部为单一动量隐式子步、零拒绝、Helmholtz/CG 全收敛；Helmholtz 最大分量迭代 `189`、最大三分量总迭代 `549`、最坏真实相对残差 `9.99667e-8`。
- 最大压力 CG 相对残差 `9.00021e-7`；最大投影后 no-slip 残差 `4.76837e-7 m/s`。
- 最后压力极值 `297.3002/-47.2167 Pa`，范围 `344.5169 Pa`。
- 这与旧 v45h 稳态压力范围 `344.3114 Pa` 几乎相同；因此旧 xyz-LOD 的 `dt^2` 交叉项虽然离散上不正确，但不是稳态压力幅值偏低的主因。

快照完整性：

- schema=`8`，field count=`31`，grid=`4x256x320`。
- manifest 文件 SHA256：`54e93dff84c46c2b225eb00d6b0bd449b988824b0b914442c27f68020a4668ec`。
- generation NPZ SHA256：`bdd2d2bf6b703741af27d2c78008437ab3dfcf68c87cf09920482fe0405e123f`，manifest 声明值与实际复算一致。
- identity：config=`6a730c7b41f6a8f0eb55cd96ebb6e744b90c6da5082e571ae0a721f39a755fba`，geometry=`7e42230a98e855ef0a6d24eef16ee4d1870b56c93d62b076bdc82bd3f5ef1b4d`，source=`ef0c655d113272048f6dd889cf3cdc43625a4123459e8bd99590a612b91ea098`。

### 12.5 正式 FSI50

运行目录：

`validation_runs\ansys_vertical_flap_fsi\our_solver_fine_vs_fluent_2026-07-02\our_solver\current_solver_fsi50_from_preflow_stationary_unsplit_v46e2_20260720`

结果：

- `status=completed`，requested/completed=`50/50`，物理时间 `0.025 s`，solver summary 耗时 `1859.6972592 s`。
- `step_artifact_validation=passed`；50 个连续 `step_history/step_NNNN.json` 和 50 个连续 `step_fields/step_NNNN.npz`。
- 50 步全部使用上述 unsplit integrator、单子步、零 rejected trial、Helmholtz/压力 CG 全收敛、pressure failure=0。
- Helmholtz 最大分量迭代 `205`、最大三分量总迭代 `565`、最坏真实相对残差 `9.99197e-8`；最大 CG 相对残差 `9.31078e-7`；最大投影后 no-slip 残差 `1.25088e-6 m/s`。

关键峰值：

| step | solver pressure range (Pa) | solver reaction (N) | solver tip norm (mm) | Fluent tip norm (mm) | tip ratio |
|---:|---:|---:|---:|---:|---:|
| 9 | 337.570236 | 0.008840057 | 0.387013 | 0.524174 | 0.738330 |
| 27 | 339.608788 | 0.008871879 | 0.386991 | 0.505973 | 0.764845 |
| 45 | 339.486144 | 0.008870472 | 0.381544 | 0.484841 | 0.786947 |

step 9 的 Fluent 压力范围和 3-D 等效反力分别为 `551.418874 Pa`、`0.0146225883 N`；solver/Fluent 比值为 `0.612185` 和 `0.604548`。v46 的三峰衰减顺序现在与 Fluent 相同，但幅值差异没有解决。

### 12.6 锁定 Fluent fresh50 严格后处理

输出目录：

`validation_runs\ansys_vertical_flap_fsi\our_solver_vs_native_fluent_fine_2026-07-10\postprocess\current_solver_fsi50_unsplit_v46e2_strict_contract_20260720`

锁定 Fluent 输入：

`validation_runs\ansys_vertical_flap_fsi\official_fluent_fine_fsi_valid_2026-07-10\runs\fresh50_20260713_104843\postprocess_compare31_strict_pressure_20260719_142808_r2`

产物与身份：

- `comparison_report.json`：`status=diagnostic_complete`，`parity_claimed=false`；不能把脚本完成写成 parity。
- final run identity、50-step history、压力语义 `static_gauge_pressure_pa/outlet_0_pa` 均通过。
- `CHECKSUMS.sha256` 共 58 行，全部复算通过。
- Fluent final pressure extrema=`483.8569/-85.2279 Pa`；solver=`293.5264/-63.7279 Pa`。
- Fluent global speed max=`30.4267 m/s`；solver global=`41.4703 m/s`，在 Fluent sample points 上 solver max=`34.4783 m/s`。

v45i -> v46e2 的严格指标：

| metric | v45i | v46e2 | 结论 |
|---|---:|---:|---|
| pressure-drop relative error | 0.453705 | 0.449636 | 仅轻微变化，仍失败 |
| pressure-range relative error | 0.386616 | 0.383665 | 仅轻微变化，仍远超 5% |
| speed-max relative error | 0.131029 | 0.133160 | 略变差 |
| speed-mean relative error | 0.073893 | 0.074139 | 基本不变 |
| centerline/throat/downstream `u` NRMSE | 0.146064/0.124738/0.219174 | 0.145707/0.123967/0.218205 | 基本不变 |
| near-wall `u` NRMSE | 0.315163 | 0.314963 | 基本不变 |
| tip NRMSE/reference peak | 0.143198 | 0.143182 | 基本不变 |

通过的流场门只有 `u_max`、入口质量流量和出口质量流量；pressure drop、speed max/mean、centerline、throat、downstream 仍失败。最重要结论是：**unsplit Helmholtz 修复应保留，因为它恢复了正确离散与事务语义，但“LOD 过度扩散是流场/压力/结构幅值主因”的假设已被正式长程结果证伪。**

### 12.7 当前根因边界和下一线程优先级

已经排除或降级：

- 不是入口少送流量：step 9 出口 `0.0006000599 m^3/s`，理论入口 `10*0.003*0.02=0.0006000000 m^3/s`。
- 不是半域尺寸/薄片尺寸不一致：Fluent 与 solver 都是 `0.1 x 0.02 m` 半域和 `0.003 x 0.01 m` 薄片。
- 不是压力 CG 不收敛：step 9 和完整 50 步的 exact CG residual、physical gate 均通过。
- 不是结构过硬：此前单位载荷 compliance 比例约为 solver/Fluent=`1.1618`；载荷只有 Fluent 的约 60%，结构响应偏低主要跟随载荷。
- 不是单纯牵引面积缩放：solver marker 总面积、3-D extrusion span 和单位载荷检查已对齐；全局压力范围本身就偏低。
- 不是旧 xyz-LOD 主因：v45/v46 长程压力、载荷、流场和位移几乎重合。

下一线程按以下顺序做可证伪诊断，不要直接调参：

1. **最高优先：高 CFL velocity predictor 与单次终端 pressure projection 的耦合。** v46 step 9 的动量对流初始 CFL=`247.67`，内部约 310 个显式对流子步、最大子步 CFL=`0.454`，但一个全局步只报告一次终端压力投影。先写 RED/控制实验，验证无压力子步是否积累非散度自由预测速度；受控 A/B 必须在每个 outer predictor segment 后做完整 projection，不能只是把同一次终端 projection 前的 predictor 切片数改大。
2. **SST 状态和壁面处理身份。** Fluent 只锁定 `kw-sst` 名称，尚未导出实际 inlet `k/omega/mu_t`、wall `y+` 和自动壁面处理；solver 明确用 intensity 5%、viscosity ratio 10 和自有 wall-omega guard。“同名 SST”目前不等于逐点物理状态一致。先导出/比较，禁止直接调 `mu_t` 拟合压力。
3. **控制体动量闭合与 HIBM 压力接口。** 对固定刚性薄片建立入口/出口动量通量 + 壁面压力/剪切力账本。若全局闭合而薄片力偏低，再查 traction sampling；若全局不闭合，查 interface matrix、516 个 pressure-unreached cells、1596 条 active interface rows 和 384 个 marker-nullspace constraints。
4. **低优先复核 out-of-plane 边界。** y 半域已对齐；仍需确认 x 两侧严格等价于 Fluent 2-D 无 x 梯度，而不只依赖当前默认行为。
5. Fluent operating pressure 是 `1013250 Pa`，但比较的是固定密度下的 gauge pressure 差，不能用它解释当前倍率差，也不要把压力加常数当修复。

### 12.8 5000 步通用运行仍缺的基础设施

当前 fine-grid 50 步可连续完成，但不能据此声称 5000 步具备中断恢复能力：

- preflow 没有逐步 observer；运行期间只写初始 config/manifest，完整 `preflow_history` 和终态 snapshot 要等整段成功后才落盘。
- FSI 有逐步 `progress.json`、history 和 fields，但没有可从 `completed_step+1` 精确恢复的完整 checkpoint。
- 现有 preflow snapshot 是固定固体终态证书，只含 fluid 状态；不能伪装成 FSI checkpoint。

下一线程若先做长程基础设施，建议：

1. preflow 每个健康提交步原子写 `preflow_progress.json` 和 `preflow_history/step_NNNN.json`。
2. 新建通用 core 层 FSI checkpoint，复用 `simulation_core/fluids/preflow_snapshot.py` 的 canonical config/source/geometry hash、generation NPZ、manifest 原子替换和先全量验证后恢复思想，但不要放松 preflow certificate。
3. checkpoint 写入缝必须在一个完整耦合步的 fluid -> solid -> marker -> dynamic obstacle/HIBM rows -> pressure anchors -> history 全部提交之后；权威 metadata 保存 `completed_step` 和 `stage=post_solid_boundary_rebuilt`，恢复循环从 `range(completed_step, requested_steps)` 开始。
4. 至少持久化：完整 fluid persistent fields；solid `x/position_increment_residual_m/v/C/F`；marker 动态几何、velocity、normal、area、probe origins、projection-only tip vertices、anchor metadata；完整 history 与严格 identity。`progress.json`、混合阶段 step frame 或 `fluid.save_state()` 都不能当跨进程 restart。
5. 最小真实验收是同一源码下 `4 步连续` 对 `2 步 checkpoint + 新进程续到 4 步`，比较完整持久状态、history、source global index；不能只比较位移图。

### 12.9 性能后续项（不应先于物理根因）

- v46e2 每步 Helmholtz 最大分量迭代约 187--205，已接近性能预警线。若下一线程优化，首选固定主导轴的 SPD line-block Jacobi；仍不足再考虑 `.5,.5,1,.5,.5` 对称回文 line 预条件器。
- 绝不能把单向 `xyz` LOD 直接作为普通 PCG 预条件器；变量 `nu`、分级网格和障碍使其非自伴随，会破坏 PCG 理论并重新引入已删除的交叉项。
- 当前重型 Taichi 特化约为 3 个 initializer + 9 个 component/axis assembly。可在独立分支将 component 改 runtime、axis 保持 static，把 12 个重型特化降到 4 个；必须先做 CUDA lowering 和九种 component/axis 路由测试。该 JIT 优化不能与幅值根因 A/B 混在一起。

### 12.10 下一线程开场检查

1. 先读本节、v46c2 preflow report、v46e2 50 个 step history 和 strict `comparison_report.json`。
2. 确认没有 Python/Taichi 进程，并确认正式工件目录完整；短探针目录已删除，不要尝试续跑。
3. 读取当前 dirty diff，保留所有既有用户修改；不要用 reset/clean 追求表面 clean。
4. 只选择 12.7 的一个可证伪物理假设或 12.8 的 checkpoint 基础设施作为单独任务；先 RED，再最小通用实现，再完整证据。
5. 不得修改 E、密度、marker area、压力/速度/位移缩放或 hardcode Fluent 结果来制造通过。
