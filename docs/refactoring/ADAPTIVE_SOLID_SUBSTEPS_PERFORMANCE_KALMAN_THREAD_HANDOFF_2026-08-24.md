# 自适应固体子步、性能优化与 Kalman 对比实验交接

日期：2026-08-24
原始状态（2026-08-24）：已按用户要求暂停；以下第 2 至 6 节保留当时的事实快照。
2026-08-25 续写状态：统一物理时间审计和自适应固体子步实现已完成；Stage 3 A/B 工具链已获 tooling-only 接受，r01 dry-run 通过，但 production preflow 在第 12 步被新时间审计的固定 32-ULP 假阳性中止；根因修复已完成聚焦 RED/GREEN，三条只读复审均接受进入 fresh source-matched preflow。r01 没有 snapshot，下一轮必须使用 r02。
权威工作树：/home/zhuohengli/worktrees/HIBM-MPM-v46e2-main-integration
分支：codex/perf-r114-hotpath

## 0. 2026-08-25 续写证据边界

- 第一阶段已把每个 FSI 宏步改成流体事务，并为 SST transport、MUSCL momentum、
  Helmholtz 及 predictor segments 增加 accepted physical-time 审计；失败 trial 不计时，
  流体和固体都必须完整消费 `dt_s`，否则 fail closed。
- 第二阶段已把 production 默认改为 `solid_substeps=None`，并在每个宏步从 restored
  accepted 固体状态、弹性波速、最大粒子速度、最小网格间距和 CFL target 选择 `N`。
  固体 trial 失败时恢复 accepted state、重建外力后提高 `N` 重试；只有 accepted trial
  更新 feedback、history 和 checkpoint。显式 `--solid-substeps 1600` 仅保留为同一路径
  的 fixed1600 A/B reference override。
- Stage 1/2 最新只读 reviewer 已对其范围给出 `ship`；这不等于整个项目可发布。
  流体真实聚焦时间门槛为 `6 passed in 1273.64s`。
- Stage 3 已加入完整固体宏步同步计时、flow/HIBM/snapshot/export 分桶、实际
  selector/packed/kernel 计数、严格 runtime identity、正式 manifest/summary 时间边界，
  以及 fail-closed fixed1600/adaptive 比较器。首轮 fresh reviewer 给出 `fix-first`：
  step/final NPZ 伪证据、动态 mask 误拒绝、config/CSV/summary 跨产物未闭合及计时
  evidence 不完整共五项 P1。新增反例先得到
  `6 failed, 1 passed, 16 deselected in 2.90s`；修复后比较器为
  `28 passed in 16.02s`，当时的组合 host/tooling 门槛为
  `177 passed in 23.78s`。
- 第二轮 fresh reviewer 仍给出 `fix-first`，没有 P0，但发现五项新的 P1：preflow
  producer/当前源码/FSI provenance 未闭合；compact JSON 排序导致真实 CSV header
  false reject；dual-face 每面 `64` 实际 `128` marker 的契约错误；把 residual floor
  误作绝对 hard cap；以及 exact NPZ dtype/stage/schema/last-step-final closure 缺口。
  这些问题已按 TDD 修复：
  - 真实 CSV fixture 先得到 `1 failed in 8.61s`，最小修复后继续暴露 marker 契约，
    得到 `1 failed in 7.29s`；两项修复后 baseline 为 `1 passed in 6.55s`；
  - current-source、preflow producer、四种 dtype mutation、last-step/final closure 和
    residual floor 的代表性反例先得到 `8 failed, 38 deselected in 11.43s`，修复后为
    `8 passed, 38 deselected in 12.57s`；
  - 首次全比较器门槛为 `45 passed, 1 failed in 37.53s`，唯一失败是诊断 wrapper 没有
    暴露全网格 finite 的底层原因；修复后该单测 `1 passed in 8.41s`，路径 provenance
    组 `4 passed, 46 deselected in 9.94s`，最终比较器全门槛
    `50 passed in 38.21s`；
  - launcher source surface 中加入比较器本身的测试先 RED
    `1 failed in 7.19s`，实现后 GREEN `1 passed in 6.32s`；launcher 聚焦回归
    `40 passed in 8.59s`，Ruff 和 `git diff --check` 通过。
- 修复后的比较器会现场重算当前 source map，并闭合 sibling `__preflow` producer 的
  manifest/config/compact/summary、runner 定义的 FSI-only config 投影、runtime、
  snapshot manifest/generation/hash、run/output/final paths、dual-face `64 x 2 = 128`
  marker、CSV 内容、exact NPZ schema/dtype/stage 和 last-step/final 13-field parity。
  修复后的五文件组合 host/tooling 门槛为 `200 passed in 42.44s`；四个 PowerShell
  代码块和内嵌 snapshot validator 的 Python AST 均解析通过；四个相关 Python 文件的
  Ruff 与 `git diff --check` 也通过。新的 fresh reviewer 结论仍待本段之后记录；在
  reviewer 明确接受前不得运行 r01 dry-run。
- 第三轮 fresh reviewer 给出 `fix-first`，无 P0/P2，唯一 P1 是真实 runner history
  同时包含 `_n`/`_N` marker/scatter force-unit aliases：compact JSON 已通过
  `_json_safe` 折叠，但 CSV 先从 raw keys 建 header，导致正式 pair 必然在数值比较前
  `history CSV schema mismatch`。production-like alias fixture 和 writer 单测先得到
  `2 failed in 8.29s`；`_write_history_csv` 改为先复制并 canonicalize 每个 row 后，同两
  项为 `2 passed in 8.81s`。修复后五文件 host/tooling 门槛为
  `201 passed in 44.02s`，Ruff 与 `git diff --check` 通过。该修改只统一正式 artifact
  schema，不改变求解器状态、控制方程或数值路径。必须再取得新的 fresh reviewer
  明确接受，才能启动 r01 dry-run。
- 第三轮修复后的 fresh reviewer 最终给出
  `ACCEPT_FOR_R01_DRY_RUN_TOOLING_ONLY`，P0/P1/P2 为 `0/0/0`。随后 r01 dry-run
  exit 0；manifest/config/runtime/source 审计通过，source surface 为 96 files，且
  dry-run 没有生成 snapshot、preflow 或 FSI target。
- r01 production preflow 是唯一 CUDA 作业，但在第 `12/200` 步失败，完成 11 步，
  `elapsed_s=2330.9217`。错误为 SST accepted-time ledger closure：requested
  `5e-4 s`、报告 reconstruction tail `3.79471e-18 s`、固定 32-ULP tolerance
  `3.46945e-18 s`。失败目录只包含 manifest/config/progress/failure，目标 snapshot
  root 完全不存在；必须保留 r01 failure/cache，禁止以 r01 继续或从 reduced
  `step_fields/*.npz` 重启。
- 调查确认这不是少推进物理时间：固定 32 ULP 是当前未提交 Stage 1 审计新增值，
  不是历史求解器门槛。最小 live-arithmetic 重现为 257 accepted slices，内部
  remaining 精确为 0，而 `math.fsum` reconstruction 留下 33 ULP；删除一个真实
  accepted slice 则远超动态容差。由于 failure artifact 用 `:g`，r01 的约 35 ULP
  是从打印尾差与 binary64 ULP 推得，不能说成 artifact 直接序列化的 full precision。
- 第一版 operation-count tolerance 被两位 reviewer 拒绝：SST/MUSCL 首 trial 尚可
  低于 floor 后触达 physical kernel，loop 也可能在正的 `remaining <= floor` 时退出；
  runner 还用默认 `accepted_substep_count=1`、`get(..., 1)`、`getattr(..., 1)` 和
  `max(1, count)` 掩盖缺失/零计数。
- 修复按 TDD 收紧为：shared minimum physical slice；SST/MUSCL 首 trial 和 retry
  均在物理 kernel 前拒绝 `dt <= floor`；loop-tracked remaining 必须先精确为 0；
  之后 operation-count ULP tolerance 只处理 `fsum` ledger reconstruction；runner
  count 必填并从真实 report/attribute 传递，缺失、零和负数 fail closed。
  - host RED：`3 failed, 2 passed, 8 subtests passed in 8.49s`；GREEN：
    `5 passed, 13 subtests passed in 7.23s`；
  - 有效 CUDA RED：`2 failed in 37.86s`，两个禁止桩分别证明 SST/MUSCL sub-floor
    trial 已触达 physical kernel；GREEN：`2 passed in 53.61s`；
  - 直接相关 host 扩大门槛：`82 passed, 1 skipped, 7 subtests passed in 7.61s`；
  - SST positivity retry、MUSCL CFL retry、Helmholtz transactional retry：
    `3 passed in 377.17s`；
  - 五文件 `git diff --check` 通过；排除 integration 文件既有的动态-import E402 后
    Ruff 通过。以上仍不是 fresh preflow、FSI A/B、strict50 或发布证据。
- 数学、集成和全新独立 reviewer 均接受本轮修复进入 fresh source-matched preflow，
  P0/P1/P2 均为 `0/0/0`。三者核对了 shared floor、exact-zero remaining、operation-count
  ULP bound、accepted-only ledger、runner strict count propagation 和外层 rollback；
  结论不覆盖 strict preflow、FSI1/2/8/50 或发布。
- 最新小型 strict-CUDA 单宏步为 `1 passed in 11.95s`，现场断言 actual CUDA、
  `f32`、seed `0`、strict arch、adaptive `N=2`、fixed override `N=4`、完整 `dt_s`、
  宏步 damping 等价、无 retry/OOB/clamp。它不是 fixed1600/adaptive FSI 数值 A/B。
- 预先声明的数值/稳定性/性能门槛已冻结在
  `docs/validation/ANSYS_VERTICAL_FLAP_SOLID_SUBSTEP_AB_GATES_2026-08-25.md`；
  r01 已消耗失败，文档中的下一轮命令和全目标存在性拒绝已递增为 r02；r02 必须
  重新执行 dry-run 身份审计、fresh preflow 和 snapshot manifest/generation/hash
  闭合，不能复用 r01 source hashes 或 cache identity 作为 source-matched 证据。
- 尚未完成 r02 dry-run、fresh source-matched preflow、fixed1600/adaptive strict
  FSI1/2/8 A/B、strict FSI50、
  性能优化、Kalman 五组实验、最终 review 或发布。

## 1. 用户当前要求和正确顺序

用户已经明确：

1. 先把固体 MPM 从默认固定 1600 子步改成真正的自适应子步。
2. 自适应方案必须覆盖完整的 FSI 宏观物理时间步，不能靠少推进物理时间“加速”。
3. 再做不改变物理模型的代码和仿真性能优化，并量化耗时改善。
4. 最后比较 Kalman 的四种放置：interface-only、fluid-only、solid-only、global。
5. 四种方案均需和无 Kalman baseline 比较，先过短门槛，再让合格方案跑 strict 50 步。

不要再把“所有测试做完”理解为“先跑仓库全部旧回归”。当前只运行支撑正在实施功能的最小测试；自适应完成后，再按风险逐步扩大验证。

## 2. 暂停时的 Git 与进程状态

- HEAD：e8a5cf740b7898c315bd7a63f2bca9b1c559b580
- 本地 main：e8a5cf740b7898c315bd7a63f2bca9b1c559b580
- 本地记录的 origin/main：e8a5cf740b7898c315bd7a63f2bca9b1c559b580
- 当前修改尚未 commit、push 或合并 main。
- 写本文档前 git diff --check 通过。
- 工作树有意保持 dirty。禁止 reset --hard、clean、批量删除或覆盖现有修改。
- 写本文档前 tracked diff 为 22 files changed, 2726 insertions, 290 deletions，另有一个未跟踪测试。
- 已终止暂停前运行超过两小时的 solver 回归。
- 已中止刚启动的自适应子步只读分析代理。
- 没有需要继续等待的后台仿真。

主要修改文件：

- benchmarks/official/solid_mpm_fsi_runner.py
- cases/ansys_vertical_flap_fsi.py
- simulation_core/coupling/hibm_mpm/core.py
- simulation_core/fluids/solver.py
- 对应 benchmark、case、integration、solver、validation 测试
- validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02 下的说明和运行脚本

未跟踪文件：

- tests/benchmarks/test_fixed_solid_preflow_progress_behavior.py

被删除但未提交：

- tests/integration/test_ansys_vertical_flap_feedback_conditioned_projection_runtime.py

## 3. 已经做了什么，以及为什么

### 3.1 性能热路径和求解器维护

当前 dirty diff 中已经完成的主要工作：

- SST transport 合并可安全合并的最大扩散率、输入准备、复制和正值保护工作。
- MUSCL transport 复用最终通量，减少重复计算。
- profiling 改为显式 opt-in，避免普通生产运行承担无用同步。
- 删除或绕开若干不必要的 host/device 同步和全场下载。
- runner 补充 pressure-gradient refresh。
- MPM 多子步使用 batched out-of-bounds guard，整批只做一次 fail-closed host report read。
- fixed-solid preflow 增加更清晰的进度行为。
- HIBM component-face/cap-face 完成多轮正确性修复，包括平移不变 straight-C0 判定、canonical component-face 归属和 tip-cap 报告。

目的：减少 kernel launch、重复全网格计算和 host/device 往返，同时保留方程、边界、rollback 和 fail-closed 语义。

目前不能声称生产 50 步仿真加速了多少。已有 96.6% 改善仅属于 integration 测试套件墙钟时间，不是仿真加速。

### 3.2 无效 runtime 测试清理

原 runtime 测试使用 flow_projection_iterations=64、solid_substeps=1。

调查结果：

- 当前分支和同 HEAD 的未修改基线都会在 64 次压力迭代失败。
- 提高到 256 后，两边都会继续前进，并在相同 marker constraint PCG breakdown 处失败。
- 这证明旧配置已经失效，不是当前 production diff 引入的回归。

处理：

- 删除该 92 行 runtime 测试。
- 强化 tests/integration/test_ansys_vertical_flap_feedback_conditioned_projection.py：
  - marker_count=8 时检查 feedback False -> True 和 consumed count；
  - marker_count=0 时保证永不消费 feedback；
  - 检查 gate、投影、solid feedback、row refresh、history 的源码顺序；
  - 检查循环内不能重新清零 feedback；
  - 检查 consumed 计数只在正确条件下增加。

强化后的目标测试：12 passed, 2 subtests passed in 5.55s。

独立只读 reviewer 仍给出 DO_NOT_SHIP：host/source contract 不能替代 source-matched strict-CUDA 两步真实 solver gate。

## 4. 已完成的验证结果

有明确终态的验证：

- Full integration：270 tests in 105.625s，OK，skipped=2，expected failures=2。
  - 旧组合为 3143.943s 且有 2 errors。
  - 约减少 96.6%，但仅是测试套件加速。
- Tools：72 tests in 37.186s，OK。
- Squid config/package：166 tests in 405.111s，OK。
- FSI architecture gate：97 passed, 89 subtests passed in 11.68s。
- Changed benchmark tests：59 passed, 152 subtests passed in 7.84s。
- ANSYS case/validation changed tests：119 passed, 1 xfailed in 2988.59s。
  - xfail 是旧 50 步 web physical target，不是正式 strict 证据。
- HIBM/cap-face 聚焦门槛：
  - C0 original：3 passed + 20 subtests；
  - translated short shared vertex：1 passed + 4 subtests；
  - VF48I：1 passed + 2 subtests；
  - VF48C：3 passed；
  - cap-face：11 passed + 2 subtests；
  - raw-four：3 passed + 5 subtests；
  - cohort/cardinality：6 passed。

暂停前的 6-file solver 回归运行超过两小时，只持续输出通过点号但未自然结束。它被中止，没有最终 pytest summary，不能记为通过。

## 5. 当前关键缺口

### 5.1 自适应固体子步尚未实现

必须明确：暂停时 production 仍不是用户要求的动态方案。

当前代码：

- cases/ansys_vertical_flap_fsi.py 中 VerticalFlapFsiConfig.solid_substeps 默认仍为 1600。
- solid_substep_cfl_report(config) 使用：

~~~text
cfl_minimum =
    ceil(wave_speed * macro_dt
         / (solid_cfl_target * min_grid_spacing))

selected_substeps =
    max(requested_substeps, cfl_minimum)
~~~

- runner 在进入 FSI 循环前只调用一次 selector。
- requested_substeps=1600，所以 CFL 逻辑只能增加到 1600 以上，不能降低。
- 所有 FSI 宏步一直使用同一个计数，不会逐步变化。

历史根因：

- 固定 1600 在最初引入 case 的 commit 4258a10 中就存在，未找到误差或收敛证明。
- CFL guard 后来在 commit 10bdb7c 中加入，但设计为 max(requested, cfl_minimum)，没有解除 1600 下限。
- 前一会话的执行错误，是把大范围旧回归错误地排在自适应实现之前。

### 5.2 物理时间约束

当前 MPM 子步是显式时间积分，不是同一物理时刻的非线性迭代。每个子步都会推进时间。

正确约束：

- 每个 FSI 宏步必须完整推进 dt_s。
- 若选择 N 个子步，则 dt_sub = dt_s / N，全部子步时间和必须等于 dt_s。
- velocity damping 继续使用 damping ** (1/N)，保持宏步总 damping 不变。
- 不能“收敛后提前停止”并少推进剩余物理时间，那会改变物理解。

### 5.3 缺最新 source-matched strict FSI 证据

最新源码还没有完成 fresh strict preflow snapshot、strict FSI1、FSI2、FSI8、FSI50。

reviewer 要求的两步 gate 至少证明：

- feedback consumed 序列 [False, True]，总数 1；
- 第二步 marker count > 0；
- mode 为 hibm_sharp_reconstructed_rows；
- hibm_observer_topology_refreshed=True；
- valid marker > 0、invalid marker = 0；
- projected/no-slip residual 有限；
- 两步 accepted，无 pressure/PCG breakdown。

旧 strict r01 不是当前源码匹配证据。旧结果：

- 50/50 accepted；
- wall time 1056.897s；
- local peak 46.953 m/s；
- exported max 39.4305 m/s；
- displacement 8.53286e-5 m；
- final CG residual 5.077e-7；
- no-slip 8.928e-7；
- tip closure 5.289e-7。

它没有通过 Fluent 速度一致性：

- Fluent max 30.4267 m/s；
- solver exported max 39.4305 m/s；
- speed NRMSE 0.1328。

不要声称 Fluent parity。用户特别提醒实验 Fluent 最大速度约为 30 多 m/s。

## 6. 下一会话先实施自适应固体子步

先写失败测试，再做最小实现。

1. 把 production 默认从固定 1600 改为明确 auto 语义。
   - 可用 None 或独立 auto 字段。
   - 1600 不能继续作为隐含最小值。
   - 固定计数只能作为同一求解路径的 A/B reference override，不能复制第二套 solver。
2. selector 从循环前一次计算移到每个 FSI 宏步计算。
3. 稳定性至少考虑弹性波速、accepted 固体最大粒子速度、最小 MPM 网格间距、CFL target。
4. 待验证的最小候选公式：

~~~text
N = ceil((elastic_wave_speed + max_particle_speed)
         * macro_dt
         / (cfl_target * min_grid_spacing))
~~~

这只是候选，必须在当前 3D case 做 A/B，不能仅凭公式合并。

5. 如果远低于 1600 时轨迹误差不合格，应加误差控制或 fail-closed retry，不要重新写死 1600。
   - 宏步开始保存 accepted solid state；
   - finite、OOB、deformation 或误差门槛失败时恢复并提高 N；
   - retry 仍推进同一个完整 dt_s；
   - 失败 trial 绝不能 commit。
6. 每宏步报告 selected substeps、substep dt、CFL、max speed、retry count、solid wall time；汇总 min/max/mean/total substeps。

最小 TDD：

- 默认配置不再得到固定 1600 下限；
- 低速状态可选择小于 1600 的稳定计数；
- 更高粒子速度单调增加计数；
- N * dt_sub == dt_s；
- 动态 N 保持宏步总 damping；
- 不同 accepted 速度能在相邻宏步选择不同 N；
- NaN、Inf、负速度、非正 spacing/CFL target fail closed；
- report/checkpoint 统计与实际执行次数一致；
- rollback 后只能读取 restored accepted state，不能读取失败 trial。

验证顺序：

1. selector 单元测试，RED -> GREEN。
2. mock solid runner contract，验证逐步重算、完整时间、damping。
3. 小型 CPU/CUDA MPM 单宏步固定1600 vs auto。
4. source-matched strict FSI1。
5. strict FSI2 feedback gate。
6. strict FSI8 A/B。
7. 短门槛合格后才跑 strict FSI50 A/B。

A/B 同时报：

- 实际总固体子步；
- MPM 和总 wall time、加速比；
- tip displacement 时间序列误差；
- solid max speed/velocity 时间序列误差；
- interface force/traction 误差；
- fluid local/exported peak；
- pressure、divergence、no-slip、PCG/CG residual；
- OOB、deformation clamp、retry；
- 相对 Fluent 的位移和速度误差。

不能只比较最后一个标量，也不能把三维误差“正常”当作自动通过。先写容差，再判断。

## 7. 自适应之后的性能优化

1. 分开 profile flow、pressure/projection、HIBM row assembly、traction、MPM、report/export。
2. 量化每个宏步 kernel launch 和 host/device transfer。
3. 优先检查：
   - MPM kernel launch 能否在保留 fail-closed 语义时进一步批处理；
   - 每宏步是否还有多余 to_numpy/from_numpy；
   - report reduction 是否能留在 device，只下载聚合量；
   - pressure/HIBM 是否重复全网格重建；
   - 不需要的 observer/export 是否保持关闭。
4. before/after 必须使用相同 source、backend、snapshot 和 accepted steps。
5. 只能报告实际仿真加速，不能用测试套件时间代替。

## 8. Kalman 后续实验

实验矩阵：

1. no-Kalman baseline；
2. interface-only；
3. fluid-only；
4. solid-only；
5. global。

物理保护边界：

- Kalman 只用于下一 trial 的初始猜测，不直接替换 accepted 物理状态。
- 只在 accepted step 后 commit filter state。
- retry、失败 trial、rollback 必须恢复 filter state。
- 保持相同物理时间窗、dt_s、容差、最大迭代和边界条件。
- 比较实际 pressure/PCG/FSI work 和 wall time，不只看 predictor RMSE。

已有 CPU-only shadow pilot 的 Q/R 使 mean guess RMSE 约恶化 2.88 倍。它不是 production 实现，也不能证明四种放置都无效，但当前 Q/R 不能直接上线。

筛选顺序：shadow/FSI1/2 -> accepted-state 合约 -> FSI8 -> 合格方案 strict FSI50。最终按耗时、工作量、相对 baseline/Fluent 误差和失败率排序。

## 9. 发布要求

当前 reviewer 为 DO_NOT_SHIP，原因是缺 source-matched strict-CUDA 多步 gate。

发布前：

1. 完成自适应和短门槛。
2. 完成最新源码 strict FSI2 reviewer gate。
3. 根据结果完成 FSI8/50 A/B。
4. git diff --check。
5. 审查完整 diff，无 secret、无关文件和误删证据。
6. 让只读 reviewer 基于最新 diff 和正式证据复审。
7. conventional commit。
8. fetch origin/main 并确认可 fast-forward。
9. 推 feature branch，再 fast-forward/push main。
10. git ls-remote 验证远端 commit。

用户此前允许完成验证后推送并合并 main。新会话仍应在发布前明确报告最终 diff、证据和 commit。

## 10. 环境约束

- Windows checkout 只是入口，不是 source of truth。
- 所有编辑、Git、测试和仿真只能在：
  /home/zhuohengli/worktrees/HIBM-MPM-v46e2-main-integration
- 通过 wsl.exe -d Ubuntu-22.04 -- ... 执行。
- 可信 Python：D:\working\taichi\env\python.exe
- 不要从 reduced step_fields/*.npz 重启正式验证。
- 源码改变后必须生成 source-matched fresh preflow snapshot。
- 同时只跑一个昂贵 Taichi/GPU 作业。
- 编辑使用 apply_patch。
- 保留 dirty worktree 中所有无关修改和未跟踪证据。

## 11. 可直接复制给新会话的提示词

~~~text
请继续 HIBM-MPM ANSYS 三维竖直薄板 FSI 项目。先完整阅读：
/home/zhuohengli/worktrees/HIBM-MPM-v46e2-main-integration/docs/refactoring/ADAPTIVE_SOLID_SUBSTEPS_PERFORMANCE_KALMAN_THREAD_HANDOFF_2026-08-24.md

权威工作树只能使用：
/home/zhuohengli/worktrees/HIBM-MPM-v46e2-main-integration
Windows checkout 不是 source of truth。开始时先检查 WSL branch、HEAD 和 git status，保留现有 dirty worktree，禁止 reset、clean 或覆盖他人修改。

最高优先级不是继续跑旧的全量回归，而是立即完成固体 MPM 自适应子步：
1. production 默认不能再以 1600 为下限；
2. 每个 FSI 宏步根据 accepted 固体状态、弹性波速、最大粒子速度、最小 MPM 网格间距和 CFL target 动态选择子步；
3. 每个宏步仍必须完整推进 dt_s，不能用“收敛提前停止”少推进物理时间；
4. 保持宏步总 velocity damping、边界条件、外力、rollback 和 fail-closed 语义；
5. 先写会失败的 selector/runner contract 测试，再做最小实现；
6. 不复制第二套求解器。固定1600只作为 A/B reference override，不再作为 production 默认；
7. 先 selector、mock runner、单宏步和 strict FSI1/2，再做 fixed1600 vs adaptive FSI8；合格后跑 FSI50；
8. 报告实际子步、MPM/总耗时、加速比，以及位移、固体速度、界面力、流体速度、压力、残差、OOB/deformation/retry 误差。不能把测试套件加速冒充仿真加速。

自适应完成并验证后，再 profiling 并继续不改变物理模型的性能优化。之后才做 Kalman 五组：无 Kalman baseline、interface-only、fluid-only、solid-only、global。Kalman 只影响 trial 初值，filter state 只在 accepted step 后 commit，rollback 必须恢复；保持相同物理时间窗和容差。先短门槛，只有合格方案跑 strict 50 步。

当前 reviewer 因缺 source-matched strict-CUDA 两步真实 feedback gate 给出 DO_NOT_SHIP。发布前补齐 consumed [False, True]、count=1、sharp rows、topology refreshed、valid>0、invalid=0、有限 residual、accepted steps、无 pressure/PCG breakdown 的正式证据。

请先用一句话确认当前固定1600根因和准备修改的最小文件范围，然后直接实施，不要再先跑无关长回归。完成审查和验证后，提交、推送 feature branch，并 fast-forward 合并/推送 main；最后用远端 ref 验证。
~~~
