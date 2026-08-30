# ANSYS 竖直薄板 segment aggregation r29 工作日志

日期：2026-08-28
当前状态：用户已授权先备份再重构；完整备份和实际恢复核验已完成。固定材料
参考映射/伴随载荷、cap 一致运动学、owner-aware MAC 候选及恢复身份检查已
落地，正在完成独立审查和回归。本轮重构源码尚无真实 FSI 步数或 Fluent 精度
通过记录；不得用旧 r37 的连续50/恢复50代替。200/5,000 physical-step、全仓库
80% coverage 和稳定生产求解器均未验收。当前详细状态见文末“已授权材料参考重构”。

本文件开头的 r29/r30 参数、mode 与 SHA256 都是保留的历史记录；当前合同、冻结和
运行状态以文末最新专节为准，不能将早期 core/test SHA 视为当前 source identity。
其中“待用户确认”“源码仍为r37”等陈述均为其所在历史阶段的状态，不是本轮授权。

本轮承接
[`ANSYS_VERTICAL_FLAP_MARKER_CLOSURE_R28_WORKLOG_2026-08-27.md`](ANSYS_VERTICAL_FLAP_MARKER_CLOSURE_R28_WORKLOG_2026-08-27.md)
的 step 42 `prepare_author_cardinality` 阻塞。目标是证明并实现三 raw authors
到两个已注册 finite segments 的聚合，而不是放宽 target tolerance 或把未完成
的运行称为 50-step PASS。

## 源码与固定边界

- 权威工作树：`/home/zhuohengli/worktrees/HIBM-MPM-r21-validation`。
- branch：`codex/closure-diagnostic-r23`；基线 HEAD：
  `f61758e0ef09045a0b995067df0d263a118bab61`。
- 当前改动尚未提交或推送。
- `core.py` SHA256：
  `da10764b930c54bd5f5066b6b9d0fd73a67b4891b48de82d1692dd245e24f720`。
- 聚合合同测试 SHA256：
  `1f23168d1aa339811fc3a6c7fb3ccd694d323aedc9a6ab0bf75d1762c60b621b`。
- `dt_s=5e-4 s`、closure tolerance `1.1e-6 m/s`、原有 target-conflict
  tolerance、adaptive solid substeps、A0 carry-forward/IQN reuse off 均不改变。
- accepted macro step 中 fluid 和 solid 各自必须消费完整 `dt_s`；权威
  remaining time 必须为零。代数迭代收敛不能提前结束物理时间推进。

## 聚合合同与实现

实际 r28 witness 为 A direct `(0,9,31)`、B owned relocation-shadow
`(0,8,31)`、C direct `(0,9,32)`，共同落到 axis-2 MAC face `(0,9,32)`。
A 属于 segment `(5,6)`，B/C 属于 `(4,5)`；其 raw targets 不相同。
一次性 probe 还确认：该非插值路径的四个 pair caches 都为零，selector offsets
未提供有限几何 authority。因此不能通过放宽缓存或目标相等判断修复。

实现采用两个阶段：

1. prepare 只在精确 lower-owned `Dlower/Slower/Dupper` 或 upper-owned
   `Dlower/Dupper/Supper` scan 顺序、raw count 3、actual count 1、两个 direct
   source slots 和无第四作者均成立时登记私有 pending mode `128`，暂存
   owner-first/opposite keys。preliminary 排除 owner/shadow 属于同一 ordered
   segment，使既有同段多作者重建
   继续沿原路径处理，不被两段聚合规则错误截获。
2. 独立 Taichi validator 在 prepare observer 之前验证完整 registry、相邻 segment
   和 shared endpoint、projection vertex bounds、author weights/targets、boundary
   anchors、region、三组 normals、search envelope 及 owned shadow transaction。
   search envelope 复用原有严格 helper，分别检查 owner/shadow 和 owner/opposite。
   通过才统一为物理 lower/upper keys 和精确 mode `132`；失败只报告一次
   cardinality，并清除可提交状态。下侧 witness 为 `(lower,shadow),(upper,-1)`，
   上侧为 `(lower,upper),(shadow,-1)`，保留实际 raw scan 顺序。

reconstruction 只消费验证后的两个 segment representatives，通过 MAC face 与
piecewise-linear geometry 求唯一 target；B 仅是 transport witness，不是第三个
独立物理 segment。mode132 不得进入旧的近似相等目标平均分支。

## 合同测试范围

新增的精确测试覆盖：

- r28 形状的三 raw authors、四个 pair caches 为零、A/C representatives；
- marker 顺序反转和 shared-low-endpoint 编号，canonical target 不变；
- strict interior owner，且另一 segment 位于 shared endpoint 的外延；
- targets 差小于旧 conflict tolerance 但几何不唯一时仍拒绝，不得平均；
- 第三 registered segment、重复 registry、malformed indices/weights、region drift、
  stale anchor、unowned/wrong-storage shadow、pairwise normal divergence；
- search envelope 缩到 `1e-6 m` 时必须 cardinality fail-closed；测试恢复搜索半径，
  生产 canonical ledger 保持逐字节不变，transaction transient state 清零；
- A/B/C 属于同一 ordered segment 时保留既有重建结果，覆盖正反 marker 编号；
- A/B 合法而只有 C 越过 search envelope 时仍须原子拒绝。

三组新 fixture 的 B nominal probe 已改为沿 `(0,0.6,-0.8)` 的射线；observer
不再手工发布 shadow，而是验证真实 arbitrate/materialize 已将 B 发布到 A，
并核对 source/base/valid 及 source-keyed actual sample point/velocity。
成功例显式断言 blocked count 为零；no-average 负例也要求 prepare 时无背景
blocked 错误。same-segment reverse fixture 的 nearest marker 为 A/B/C=`0/1/1`；
负例恢复 search metadata，以及临时变动的 marker/projection-vertex counts。

`py_compile` 与 `git diff --check` 已通过。早期“独立静态审查未留 P0/P1”的结论
已被新审查发现的两项 P1 推翻；修复后复审确认这两项源码缺陷已关闭。
早期五组 lowopt 合同通过后又进行了纯 helper 去重；CUDA 单独编译 wrapper
试验因编译器访问违规已撤回。当时默认编译配置尚未 GREEN、FSI50 尚未开始；
最终冻结版的聚焦及数值结果以本文后续专节为准。

## 冷编译诊断边界

- 完整证明曾直接加入巨型 prepare kernel；一次冷编译 40 分钟未完成后取消，
  没有得到测试 PASS/FAIL。
- 当前证明已拆成独立 `@ti.kernel`，旧死块已物理删除。
- 默认编译配置下的三个精确节点在 20 分钟上限后取消，仍未给出首个节点结果。
  `py-spy` 只读采样确认停留在 prepare 的 `prog.compile_kernel(...)`，而非数值推进。
- 以拆分后的源码、CUDA/f32、`opt_level=1`、advanced optimization 开启，
  仅 `cfg_optimization=False` 做早期单项诊断；这些诊断本身不构成数值验收。
- 首次诊断辅助脚本读取未导出的 `external_optimization_level` 属性，在测试方法前
  失败；该打印已删除。此 harness failure 与 solver failure 分开记录。
- 第二次诊断实际完成编译，首个正例在 `453.59 s` 后失败，原因是
  `canonical obstacle relocation has blocked component faces: count=2`，不是
  target/cardinality 冲突。报告为 `pytest_mode132_cfg0_diagnostic_v2.xml`。
- 同源 warm probe 在 `68.42 s` 重现；观测到 raw count `3`、mode `132`、keys
  `(5,6)`、重建 target `1.0`、target conflict `0`。新合成 fixture 把纯 +z nominal
  probe 当作 accepted sample，导致 x/y 面的 progress 都为 `-0.5`，生产 blocked
  检查正确拒绝。报告为 `pytest_mode132_cfg0_diagnostic_v3.xml`。
- 后续编译收缩仅删除两项冗余依赖：`mask == 3` 已证明两个 direct slot 被接受；
  pending conflict 只在最终 count 2 时可观察，因此无需保存循环中的“最后非零值”。
  完整证明、cardinality 检查和默认 CFG 设置均保留，耗时改善尚待实测。
- `core.py=0e08cabf...`、真实 relocation fixture 测试版本 `e08f90bb...` 的默认
  编译运行约 `570 s` 后仍处于 prepare compile；因新审查发现两项 P1 而取消。
  采样 CPU 约 `553 s`、内存约 `11.24 GB`，无测试结果，也无物理时间推进证据。
- 最新去重只把两处参数完全相同、纯读取的 side/cap seam helper 合并计算一次。
  保留插值/非插值的所有原判据，不增加 kernel specialization，不更改默认优化。
  `seam_dedup_v1` 的默认 17 节点尝试在 `1504.7 s`（约 25 分钟）诊断上限
  取消，exit 1；未返回首个节点结果，不是测试断言失败。该次 core 为
  `74f0d3ef47086cfa64ac94a0ef39f604bd98dd2a4fa3d64bfe1a393143db56bc`。
  CPU 采样 `1479.7 s`；内存采样峰值约 `10.61 GiB`，结束前约 `2.19 GiB`。
  `py-spy` 的 Python 栈仍指向 prepare 的 `prog.compile_kernel(...)`。
  不能由内存下降或纯 helper 去重推断编译已经完成，也不能宣称加速。
- 曾试验 `_compiled_projection_only_moving_seam`，仅 CUDA prepare 通过
  `ti.static(ti.cfg.arch == ti.cuda)` 调用 `@ti.real_func` wrapper；其他后端与
  reconstruction 保留原 inline 路径。wrapper 只调用原纯 helper，不复制证明。
  runtime 参数仅显式化原有 i32/ivec3；字段保持 template，不引入浮点精度转换。
  依据为 Taichi 1.7.0 的[官方 real-function 说明](https://github.com/taichi-dev/taichi/releases/tag/v1.7.0)
  与 v1.7.4 的 [CPU/CUDA 字段和向量测试](https://github.com/taichi-dev/taichi/blob/v1.7.4/tests/python/test_function.py)。
  [官方离线缓存测试](https://github.com/taichi-dev/taichi/blob/v1.7.4/tests/python/test_offline_cache.py#L497)
  覆盖真实函数体依赖；不禁用缓存，不改 CUDA stack limit 或全局优化。
- `real_seam_v1` 以 17 节点、独立新缓存目录、CFG=true、opt_level=1、
  advanced=true 和实际 CUDA 运行。core 为 `06feaa98...`，tests 为 `bb3063a4...`。
  stage observer 在 `95.66 s` 到达 `prepare_before`；随后 Windows fatal
  `access violation` 出现在 `kernel_impl.py:969` 的 `prog.compile_kernel`。
  exit 1，没有首项测试结果或数值推进，不能计作编译加速或合同 GREEN。
  该 wrapper 已通过逆向精确 patch 完全撤回；当时 core 恢复为单侧 `74f0d3ef...`。
  已核查没有残留 Python/CUDA 作业。官方支持 typed real functions 不足以证明
  此大型 real-to-inline 动态控制流组合在本机可用；具体编译器原因仍未确定。

报告目录：
`validation_runs/solver_soaks/ansys_vf__aggregation_kernel_split__20260828__focused/`。
冷编译耗时不得按步数线性外推。r27 的 36/36 和 r28 的 41/50 分钟级实测仍以
r28 历史日志为准。

## 独立审查与 RED/GREEN 证据

新审查发现两项 P1：preliminary 错误接管合法同段三作者；validator 漏查 C 的
search envelope。两条新 RED 合同实际复现了错误拒绝和错误接受。

- `pytest_mode132_lowopt_review_red_v1.xml`：pytest `136.39 s`、辅助脚本
  `136.66 s`。JUnit 为 4 failures（正反同段 subtest 各一、失败后的次生
  `IndexError` 一、C-only 输入未抛异常一），不是四个独立 solver 缺陷。
- 修复后 `pytest_mode132_lowopt_review_green_v1.xml`：pytest `148.39 s`，
  `5 passed, 17 subtests passed, 1 failed`。唯一失败为第三段负例的 finally：
  临时 marker count=4 尚未恢复就重建两段 topology。求解器已抛出预期冲突。
  修复测试清理时保留拒绝、ledger 字节不变和 transient 清零断言。
- `pytest_mode132_lowopt_review_green_v2.xml`：exit 0，pytest `123.63 s`、
  辅助脚本 `123.88 s`；`5 passed, 18 subtests passed`。
  该次 core SHA256 为 `7cf0a7aef5baecb9f55b310e2754b880dab9528ad5e7aa3d0148c5a1ca23076f`，
  tests SHA256 当时为 `bb3063a41c7b8d67007cb1e3ca4066ef5fad6311230c550833d235fcdde9fb2e`；随后纯 helper 去重形成单侧 `74f0d3ef...`。

以上 lowopt 均为 CUDA/f32、CFG=false、opt_level=0、advanced=false 的逻辑诊断。
这些进程级临时设置不写入生产配置，不替代默认编译设置或正式数值门禁。

## 配置预检与后续数值门禁

r29 dry-run exit 0，耗时 `8.095 s`。与 r28 配置逐字段比较，仅 `step_count`、
`preflow_snapshot_input_path`、`preflow_snapshot_output_path` 不同，没有数值配置
差异。该预检早于最新源码修复，不构成当前 source identity 验收；它未运行求解器。
该预检当时尚未生成本轮新 preflow，也未运行新源码的 FSI 仿真。

源码稳定后必须重新生成 source-matched preflow；禁止重用旧 r26 snapshot，禁止从
reduced `step_fields` 重启。按 focused tests → fresh preflow → FSI1 → FSI2 → FSI8
→ exact FSI50 的顺序执行，任一步失败先定位，不跳过门禁。

计划产物放在 `validation_runs/solver_soaks/` 下的
`ansys_vf__{preflow,fsi01,fsi02,fsi08,fsi50}__segment_agg__20260828__r30`，
snapshot prefix 为 producer 内的 `state`。dry-run 使用独立目录。

验收必须包括实际 strict CUDA runtime identity、producer/consumer source/config/
geometry identity、完整物理时间账、pressure/closure/FSI health，以及精确 50 个
`step_fields` 和 50 个 `step_history`、completed summary 与 final time `0.025 s`。
派生 SST 求和尾数按现有 roundoff 规则审核，不能误当作未推进的物理子步。

即使本门禁通过，也只证明这一配置和新增合同范围；不构成 Fluent parity、FSI5000
或所有复杂几何的普遍稳健性保证。

## 实际编译配置与快照身份补强

后续门禁选用显式进程配置 `TI_CFG_OPTIMIZATION=0`、`TI_OPT_LEVEL=1`、
`TI_ADVANCED_OPTIMIZATION=1`，保留 CUDA/f32、seed 0、fast_math=true、debug=false。
这没有修改生产默认值、物理配置或 target/closure tolerance；默认 CFG=true 的
冷编译仍未通过。本轮不再重复已撤回的 `@ti.real_func` wrapper 试验。

`pytest_mode132_cfg_off_opt1_advanced1_v1.xml` 冷运行中，当前 core
`74f0d3ef...` 的五组新增合同全部通过（18 个 subtests）。首组 call 为
`460.55 s`；prepare 后首次观测为 launch 后 `366.44 s`。整体运行在旧 extrusion
合同处因三个 subtests 的 z-cache 断言失败而停止：`11 passed, 31 subtests passed,
3 failed in 1248.12s`，辅助脚本 `1248.49 s`。后六个节点当时尚未执行；不能把
这个部分结果称为完整聚焦回归 PASS，更不能称为正式 FSI50。

与默认编译试验的多 GiB 内存增长不同，此次采样约为 0.68--1.0 GiB。该数值只描述
本机实测的编译过程，不是跨机器性能保证，也不能把冷编译耗时线性外推成每步耗时。

运行时报告现在从 `ti.cfg` 读取实际 `default_ip`、`cfg_optimization`、`opt_level`、
`advanced_optimization`、`fast_math`、`debug`，连同 Taichi version 存入
`compiler_configuration`；不会根据环境字符串猜测实际值。
preflow config identity 包含 actual arch/default fp/seed/compiler configuration。
不同编译配置或缺少该身份的旧 snapshot 必须在任何 `from_numpy` 前被拒绝；
cache 路径、requested/strict 标记不混入数值身份，因此相同配置换 cache 目录仍可
严格恢复。源文件和几何身份检查继续保留。

该补强遵循 tests-first：

- 生产修改前：`pytest_compiler_identity_red.xml`，`12 failed, 1 passed,
  167 deselected in 7.77s`。
- 修改后两文件完整测试：`pytest_compiler_identity_green.xml`，
  `180 passed in 8.35s`。
- 将 mismatch 断言收紧为精确 `PreflowSnapshotMismatchError` 后：
  `pytest_compiler_identity_exact_green.xml`，`180 passed in 8.34s`。
- 该四文件补强经独立只读审查无 P0/P1；不等于整套 solver 或 FSI50 已验收。
- relocation 静态合同与 closure 诊断：`pytest_static_and_closure_regression.xml`，
  `12 passed, 27 subtests passed in 7.77s`。

## 旧等距面合同的归因

旧 `test_inactive_axis_extrusion_cohort_uses_one_face_ray` 中 z 射线固定为
`0.625`，两个候选面为 `0.5/0.75`；progress 同为 `2/3`，切向距离平方都为
`0.015625`。selector 对普通单作者保留下侧，但 exact tie 不授予 pair authority。
因此 direct/shadow common pair offset 为 `-1`，两侧 pair cache 都应无权限；
下侧普通单作者仍提交零速度，上侧不激活。

该 selector、classifier、precompute、fixture 和旧测试完整方法均与 HEAD
逐字节相同。历史提交 `6bec377` 同时引入 exact-tie 拒绝和下侧单作者期望，
却保留了下侧 pair cache 有权的断言。隔离重跑已复现同样三个 subtest 失败，
排除了前序测试污染；这里不应修改生产 selector 或放宽几何规则来迎合旧断言。
修正合同将同时断言 offset/key/kind 均无权限，并保留 prepare、commit 和原子性
检查。新增小型 selector probe 覆盖 exact tie 及向两侧偏移的唯一候选，计划在
CFG=false/true 两种编译设置分别验证；实际结果另行追加。

## 单侧实现已完成的聚焦门禁

- `pytest_mode132_cfg_off_isolated_legacy_tail_v1.xml`：旧合同尚未校正时，
  `7 passed, 20 subtests passed, 3 failed in 978.04s`；三个失败仍只属于上节的
  z-cache 旧断言，其余六个兼容性节点通过。
- `pytest_storage_tie_cfg0.xml` / `pytest_storage_tie_cfg1.xml`：分别
  `1 passed in 8.79s` / `1 passed in 8.87s`；每个测试含两处 x 位置、exact tie
  和两侧唯一候选共六个 probe，实际 CFG 配置均打印核实。
- 随后仅加强旧 extrusion 合同，不修改 selector：显式验证 z pair offset/key/kind
  无权限、admission/full-valid 为零，保留原有单作者 prepare/commit 和原子性。
- `pytest_mode132_cfg_off_opt1_advanced1_warm_v1.xml`：exit 0，
  `17 passed, 41 subtests passed in 278.84s`，辅助脚本 `279.24 s`。
  对应 core `74f0d3ef...`、合同测试
  `1d8bc77353c834adf888245351b9982d4dcc55047ae94666513377adbb83e16b`，配置 CFG=false、opt_level=1、advanced=true。
  覆盖新增聚合正反例、原子拒绝、同段兼容、inactive-axis extrusion、端盖接缝、
  插值/non-interpolated 路由以及 target/alpha 冲突旧合同。
- 本轮没有运行全仓测试或测量覆盖率；不能声称 full-suite/80% coverage PASS。

2026-08-28 06:01:13 +09:00 启动冻结源码后的 preflow → FSI1/2/8/50 串行门禁。
入口、检查脚本已独立只读审查；检查器还只读验证了 r27 全部 36 个和 r28 全部
41 个 accepted histories，未发现字段或 roundoff 误拒绝。这是验收脚本预检，
不是新源码数值 PASS；本轮数值结果完成后在下节记录。

## 双侧 ownership 范围补强

最终静态审查确认：上述 `reverse/shared_low_endpoint` 合同只反转 marker 编号，
不是把 relocation ownership 关于 component face 做物理镜像。
当时单侧版本 `74f0d3ef...` 只接纳 lower-owned `Dlower → Slower → Dupper`；
对应的 upper-owned `Dlower → Dupper → Supper` 仍会 cardinality fail-closed。
这不是该补丁新增的回归，但与本轮通用稳健性目标有关，因此继续补充该自然对称
场景的 RED、negative 和同段 legacy 控制后再完成数值门禁。

06:08 +09:00 主动停止 r29 preflow 的已核验 PID 25320，保留该目录的原始
manifest/config/progress 和缓存。停止时仍在冷编译，没有 preflow 完成或数值
FAIL 结论；外层脚本因进程被停止而 exit 1，后续 FSI1/2/8/50 没有启动。
原始 progress 可能仍标记 running，不得据此推断有活动进程或已有数值 PASS。
下一轮使用新的 r30 产物目录，不覆盖这次取消记录。

最小双侧方案保持同一个 validator：pending128 暂存 owner-first/opposite keys，
完整证明仍包括两个 search envelopes、两个已注册相邻段、shared endpoint、
weights/anchors/regions/normals 和 owned transport；通过后统一 canonical
lower/upper representatives 和精确132。失败诊断保留真实 raw scan 顺序；
count2 恢复原 pair conflict，合法同段路径不接管，不新增插值或目标平均权限。

### 双侧镜像 RED 实测

在生产 core 仍为 `74f0d3ef...`、测试为 `668e63af...` 时执行
`pytest_mode132_cfg_off_mirror_red_v1.xml`。2026-08-28 06:23:09 +09:00
启动，exit 1；pytest `455.58 s`，辅助脚本 `455.90 s`：
`1 failed, 3 passed, 12 subtests passed`。

- 唯一失败为合法 upper-owned 正例：`prepare_author_cardinality`、face
  `(0,1,2)`、axis 2、claim count 3、raw keys `(5,6,2)`，witness
  `(-399,-197)`。其 source rows 按实际 scan 顺序为 lower direct `(0,1,1)`、
  upper direct `(0,1,2)`、upper-owned shadow `(0,0,2)`。
- lower-owned 合法正例通过；上下两侧同段 legacy 控制均通过。
- 9 个 upper-owned corruption/atomicity 负例全部通过，包括 opposite-only
  search-envelope 越界。它们目前可被旧 cardinality 守卫拒绝，修复后必须重跑，
  才能证明新增聚合入口没有吞掉非法输入。
- fixture 使用真实 relocation materializer；物理镜像反射两侧 cell 和速度，
  但共享 MAC face 保持 `(0,1,2)` 不变。

确认 RED 进程结束后才授权修改 core；当时尚未取得双侧实现的 CUDA GREEN。

### 镜像负例证据修正与双侧源码冻结

独立审查发现原 upper-only envelope 负例存在两个掩盖因素：support metadata
在 materialize observer 内才设置，晚于 assembly 的 scalar 捕获；而 `5e-5`
和 `2.6e-5` 的 z-anchor 偏移超出了既有 2-ULP anchor tolerance。因此上述
RED 的合法 upper positive 仍有效，但不能把其中 envelope 子例当作独立的
opposite-only 包络覆盖，原始 JUnit 不改写。

最终测试在 assembly 前设置 `(0.5,0.3,0.125)`、anisotropic=true；observer
只把三个 anchor 的 z 统一移动一个 f32 ULP。用实际 f32 center/anchor/weights
断言原有 anchor 容差内、owner/shadow 严格在内、opposite 严格在外。
原子拒绝核验后 restore 同一原始 fixture，以宽 `(0.5,0.5,0.5)` 和相同 observer
偏移作成功对照，显式要求 prepare `(raw3,132,5,6)`、target `-1`、零冲突/blocked
和 transient neutral；不改变生产容差，也不手工构造 shadow。

双侧实现还在运行 CUDA 前修正了静态审查发现的三类问题：

- validator 全程使用已在外层声明的 owner/shadow/opposite，避免 Taichi
  if-local 变量越域，也不把物理 target 重绑定为 opposite；
- potential-shadow 字段读取有显式 bounds block，inactive-axis 动态索引有显式
  非负/非 component-axis block，不依赖逻辑 `and` 的短路行为；
- upper 非插值 direct 的 `actual_geometry` 必为 0，不能误用 lower shadow 的
  `!=0` 条件；最终 raw3/actual1 和完整 owned-transport 证明仍保留。

当前冻结源码/测试为本日志顶部 SHA256。`py_compile`、AST 和 `git diff --check`
通过；两文件快照/实际编译配置回归
`pytest_compiler_identity_symmetric_green.xml` 为 `180 passed in 7.90s`。
这些轻量测试与静态审查不替代最终 20 节点 CUDA 回归或完整 FSI50。

### 冻结版完整 CUDA GREEN 与 r30 启动

独立只读审查接受顶部 `da10764b...` / `1f23168d...`，无剩余 P0/P1 或
fix-first 项。之后保持源码和测试冻结，单个 CUDA 进程运行完整 20 节点：

- `pytest_mode132_cfg_off_symmetric_final_v1.xml`：exit 0；
  `20 passed, 54 subtests passed in 1327.39s`，辅助脚本 `1327.82 s`。
- JUnit suite 的 tests 计数为 74，实际 20 个 testcase，0 failures/errors/skipped；所有打印的
  core/test source identities 各自只有一个值，精确等于顶部冻结 SHA256。
- 实际 CUDA/f32、i32、Taichi 1.7.4、CFG=false、opt_level=1、advanced=true、
  fast_math=true、debug=false，未冒充默认 CFG=true 的通过结果。
- 包含上下 ownership 正例、同段控制、全部 corruption/atomicity 负例及修正后的
  narrow/wide search-envelope 对照；17 个原聚焦节点也全部保留并通过。
- 首次 prepare 的 observer 从 launch 后 `49.09 s` 到 `302.01 s`；整个 22 分钟
  包含多个不同模板 kernel 的编译与合同检查，不是物理仿真的每步耗时。
- 最终源码的静态 relocation / closure 诊断回归：
  `pytest_static_and_closure_symmetric_green.xml`，
  `12 passed, 27 subtests passed in 7.68s`。

2026-08-28 07:15:31.7637664 +09:00 启动全新 r30 preflow → FSI1 → FSI2 →
FSI8 → FSI50 串行门禁。入口脚本核验 core pin、无其他 Python 作业、所有目标
目录不存在；每档结束运行独立产物检查，失败就不启动后续档。
该启动记录本身不构成任何数值 PASS，后续实际结果另行记录。

运行使用已核验的 Windows Python `D:\working\taichi\env\python.exe`，cwd 为
权威 WSL 源码的 UNC 路径；native WSL 直接执行该 Windows interpreter 返回
`Exec format error`，因此没有把 Windows 镜像当作源码或把改动复制回镜像。
Git、源码编辑与源码核对仍在上述权威 WSL 工作树执行。

### r30 fresh preflow 实测结果

`validation_runs/solver_soaks/ansys_vf__preflow__segment_agg__20260828__r30`
已完成，runner `elapsed_s=1028.7380632000059`，含冷编译；外层串行阶段
（包括退出、产物核验）`1052.25 s`。不得把这个初始化阶段的耗时外推为暖态每步
耗时，也不得用它替代后续完整 50 步的实测耗时。

- preflow requested/completed 均为 1，FSI requested/completed 均为 0；
  `state.json` 和其 manifest 指定的 NPZ 已生成，status=completed。
- 独立检查器通过：101 个源码文件的 live/manifest/snapshot identity 相符，
  snapshot manifest、域与状态文件的哈希匹配；未读取旧 r26 或 reduced step fields。
- 完成报告中的实际运行身份为 strict CUDA/f32，Taichi 1.7.4、i32、CFG=false、
  opt_level=1、advanced=true、fast_math=true、debug=false、seed=0。
  progress 中的 requested-before-init 记录不作为 actual runtime 证据。
- 只读采样把较长等待定位到 `prog.compile_kernel(...)` 的 SST momentum
  Helmholtz component kernel；之后日志继续推进到压力 kernel 并正常完成。
- FSI1 于 `2026-08-28T07:33:04.2017181+09:00` 启动，使用该新快照；
  后续 FSI1/2/8/50 仍须逐档通过，尚不声明 FSI50 PASS。

### r30 FSI1 / FSI2 / FSI8 实测结果

三档均从上述同一 fresh snapshot 重新开始，不从上一档的 reduced frames 重启。
三档 runner 与原独立检查器均 exit 0：

| 档位 | accepted/requested | runner elapsed_s | 外层阶段 elapsed_s | final time_s |
| --- | --- | --- | --- | --- |
| FSI1 | 1/1 | 356.020915299996 | 366.22 | 0.0005 |
| FSI2 | 2/2 | 226.56172849999712 | 235.63 | 0.001 |
| FSI8 | 8/8 | 261.89715249999426 | 270.80 | 0.004 |

- 每档 source identity 均为冻结的 101 文件，strict CUDA 与实际编译配置一致，
  snapshot 确认严格恢复；逐步 fluid/solid accepted time 均走满各自 `dt_s`，
  authoritative remaining 均为零，压力/闭合/FSI 与两套 ledger 守卫通过。
- feedback sequence 分别为 `[false]` 和 `[false,true]`，FSI2 覆盖首次消费固体
  feedback 的门禁；FSI8 为一个 false 和七个 true。NPZ/history 序列分别精确为
  1、2、8 帧，runner artifact gate 通过。
- FSI1 的 pressure relative residual 为 `6.943433437352414e-9`；canonical 和
  observer closure residual 分别为 `9.238719940185547e-7` /
  `9.548084562993608e-7 m/s`，均低于未改动的 `1.1e-6 m/s`，FSI iterations=3。
- FSI8 于 `2026-08-28T07:43:06.3951467+09:00` 启动；此时 FSI50 尚未启动。

独立证据审计还要求补强最终离线验收：现检查器会核验逐步文件名、历史和 runner
自报 artifact gate，但未独立重开所有 NPZ 检查字段 shape/finite/阶段；FSI 最终
残差也只与报告自身的 effective tolerance 比较。另做只读 CPU 检查器，重新打开
每帧有效物理字段，并用配置及记录的 candidate RMS 复算阈值及末尾迭代关系。
不改变冻结 solver、正在运行的入口脚本、数值阈值或原始产物；这不是已发现数值
失败，但最终有效字段/收敛声明必须等追加检查通过。其范围与实际结果另行记录。

部分未启用的诊断（例如 centroid）在原始报告中为 NaN；不能据此宣称实际活动
物理场非有限，也不能把关键守卫通过夸大成“报告中所有值均有限”。原始 JSON
保留不改写。

### r30 FSI50 启动与暖态计时边界

正式 50 步于 `2026-08-28T07:47:37.3511592+09:00` 启动，仍从同一 fresh preflow
恢复。启动不构成 completed/PASS；需完整 50 帧、50 个 accepted histories 和
两层独立产物审计通过后才能给出最终结论。

已完成 FSI8 的 step-history 文件时间戳显示：launch 到首份历史为 `223.663 s`，
后七个相邻输出间隔为 `5.534--6.694 s`，平均 `5.847 s`。这些间隔包含步推进和
产物写入，不是纯 kernel benchmark；不线性外推为完整 FSI50 实测或正式加速。

### r30 FSI50 的真实结果：45/50 后 step46 fail-closed

原 step42 已正常接受，随后 step43--45 也正常输出。`failure.json` 实测
`elapsed_s=583.1667435999989`（约 9.72 分钟），status=failed、RuntimeError；
外层串行脚本 exit 1。现场精确 45 个 step NPZ 与 45 个 histories，最后 accepted
time `0.0225 s`，没有 completed summary，也没有第46步产物。此结果不是
exact50 PASS，更不能把后五步估算为已经完成；失败目录原样保留。

新冲突仍为 axis-2 face `(0,9,32)`、`prepare_author_cardinality`、raw count 3：

| raw author | source row | registered segment | weights |
| --- | --- | --- | --- |
| A lower direct | `(0,9,31)` | `(4,5)` | `(0.2427421212,0.7572578788)` |
| B owned relocation-shadow | `(0,8,31)` | `(4,5)` | `(0.8042451143,0.1957549155)` |
| C upper direct | `(0,9,32)` | `(3,4)` | `(0,1)` |

r28/r30 旧正例覆盖的是 A 不同、B=C 同段；这次是 A=B 同段、C 不同。
`da10764b...` 的 preliminary 排除了 owner/shadow 同段，validator 也显式要求
owner/shadow 不同且 shadow/opposite 同段，因此新的两段分组尚未被接纳。
不能只删除该判断就放行：必须延迟至第三作者后证明精确的两个不同 direct
representatives，shadow 属于其中一段，全部注册/ownership/normal/envelope
约束不变，并继续保留 all-three-same-segment 的既有路径与 count2 冲突语义。
A=C 而 B 不同的排列不在当前证明范围，不因“只有两段”自动授权。

重建路径另已核实：mode132 明确跳过 normal-ray face-first 分支，调用既有
`_canonical_component_face_segment_projection_target`，对有限段作 Euclidean
nearest-point、endpoint-support 和距离/tie 判定，不是纯 component-axis 射线。
按 failure 中 F32 坐标的只读复算：face yz 约 `(.005937499925,.050000000745)`；
segment `(4,5)` 的投影参数 `0.3377980949` 为严格内点，距离平方
`1.2830035817e-6 m²`、投影 z target `-0.1502428908 m/s`；segment `(3,4)`
参数 `1.5411479403`，有限端点距离平方 `1.3871730751e-6 m²`。
这项几何诊断不替代完整 GPU RED/GREEN 或生产重跑，也没有修改 endpoint support。

### 追加只读字段/FSI 标量验收与 45-step prefix 边界

staging `verify_segment_aggregation_fields.py` SHA256：
`11c456a31af648af3e1b8f0bd0feb8f8e3025784d76db12b681e2a8f4782c222`；
测试 SHA256：`ecb1541ba5771bd2657b6c1a69ca59d7c7af9664e92e94adb7ed34fb1ce0242f`。
独立审查接受冻结版；6 个 CPU unittest 方法通过（14 个正/负情景）。新增语义
RED 曾暴露真实 ledger NaN 和小 RMS 下自报 tolerance 放大的误接纳，修正后通过。
验收器独立重开 NPZ（allow_pickle=False），将 shape 绑定 config.grid_nodes，
核验有效流域字段、真实 ledger 有限/范围、严格 bool 阶段与准确 pressure 语义；
FSI 最终接受使用从 config 与记录 candidate RMS 复算的阈值，不放大物理 tolerance。
未保存原始 IQN trial vectors，故只能证明标量 histories 与公式一致，不声称重算
原始向量 residual，也不声称所有报告/CSV 的每个字段逐项一致或全部有限。

root 在三个已完成 consumer 上追加运行，FSI1/2/8 均通过。对失败 FSI50 另用
只读 prefix 审计组合原 step_gate 与字段/FSI 检查：101 live source 文件匹配，
45 帧/历史精确序列、feedback `[false,true×44]`，每步 fluid/solid 完整 dt 与
remaining=0，所有已接受步的健康检查通过。最大 pressure residual
`8.0985754821e-7`、两套 closure 最大 residual `1.0008557183e-6 m/s`，
FSI 最大 4 次迭代。审计输出明确保持 `run_status=failed`、
`accepted_prefix_audit=passed`、`exact50_passed=false`，没有改写原产物。

### step46 分组偏置的新增 RED：首次测试与测试自身纠错

源码保持 `da10764b...`，新增合同首个冻结 SHA256 为
`0339ebb35d5bd27c58172e2cf36513d554fbefb592e8c45f77caf9abe3168c7c`。
实际 CUDA/CFG-off 运行 5 个 test methods、10 个 subcases，耗时 `74.44 s`；
JUnit：`pytest_mode132_cfg_off_owner_shadow_same_segment_red_v1.xml`。
两个 A=B、A!=C 正例在 lower/upper 均以预期的
`prepare_author_cardinality`、raw3 失败；错误 opposite anchor、A=C/B-different、
合法 all-same 和既有 B=C 双侧控制共 6 个 subcases 通过。

另外两个 all-same 坏 target 负例没有抛异常，不能把该结果报成负例 PASS。
独立源码审查定位为测试注入错轴：fixture 的 component_axis=2、inactive_axis=0，
测试却改动 target[0]；生产的 segment-author provenance 合同按当前 MAC component
检查 `velocity[component_axis]`，不授权用其他分量的有限扰动拒绝当前 z-face。
修正测试为扰动 target[component_axis]，不为此扩大生产 authority；同时覆盖
first/middle raw author，后者检查早期坏 pair 不被后续合法 pair 覆盖。

独立审查还要求新增 raw2 控制：保留 C 的不同段搜索数据但取消其 active author，
使新 lower lookahead 具备资格、最终却只有 A/B 两名作者。合法情景必须保持旧
same-segment 重建，坏 target 必须恢复 `prepare_pair_arbitration` 并原子拒绝。
以上新增/修正合同须在同一未修改 core 上重新 RED，之后才实施 OR-prelim 与
两 direct representatives 的统一证明；当前尚没有新的 source-matched FSI50 PASS。

## 2026-08-28 续跑/几何实现快照（非数值验收）

本节为追加状态，不改写上文 r28--r30 的历史证据。原 `mode132` 和“三作者排列”
专用补丁路线已经放弃；上文有关它的失败、诊断和 RED 均仅保留作可复核历史，
不代表当前生产 dispatch。

当前生产 dispatch 已切换为 geometry-owned 的全 source A/B/C 组装：先保留并验证
所有来源，再以几何 owner 决定面目标。该 dispatch 已进入生产实现，但其完整 source
验证与数值验证仍为 WIP，不能把实现状态表述为 FSI50 成功。

续跑层的新 `checkpoint_store` schema 2 具有 O(1) 的 head、增量且带 checksum 的
accepted-history journal、两份 numeric NPZ generation 的有界保留，以及与 accepted
state 一同持久化的 observer outbox。restart 严格绑定 source/config/geometry identity，
明确拒绝把普通字段导出 NPZ 当作 restart；写入者只允许一个。临时文件后 replace
提供的是进程中断下的原子发布，不是未经文件系统专项验证的断电耐久性承诺。

backend restore 的 I/O 失败会中止刚构造、尚未对外暴露的新 runtime，并保留先前
durable generation；这不是对构造新 runtime 期间每一次分配或 I/O 的完全事务回滚。

当前已测 gate（均不等于正式数值 run）如下：

- combined checkpoint host：109 passed、2 skipped，21.62 s（runner 26、CLI 10、
  wrapper 29、store 30〔2 skipped〕、codec 14）。
- controller host：85 passed，6.94 s（较早一次）。
- CUDA capture：4 passed，73.83 s；未执行真实的独立进程 CUDA checkpoint/resume。
- segment audit：5 RED、8 passed，43.25 s；corner/cap：2 RED、4 passed，16.80 s。
  随后的 full registered-segment audit 在 strict CUDA 下 19 passed，31.75 s
  （session 4431，exit 0）。

仍在运行的 GREEN 结果在本次记录时为 UNKNOWN，且没有 formal run。最新真实数值
结果仍是 r30 的 45/50 accepted prefix，随后 step46 fail-closed；exact FSI50 尚未通过，
也尚无真实 CUDA checkpoint-resume 数值执行证据。

## 2026-08-28 r31 fresh-preflow rejection and locality correction

本节追加 r31 的 source-locked 诊断，不改写上文历史结果。r31 source lock 为
`55a86822e0f1a7316fc9b730c88b03f04039306a09aa99b4205ccdc0c0fc7933`（63 files）。
fresh preflow `ansys_vf__preflow__continuous__20260828__r31` 在进入 FSI 前
`100.61054 s` fail-closed：accepted FSI 为 0，未产生可用 preflow，因而不能把
其 NPZ/JSON 诊断说成全字段 restart 或数值续跑结果。

现场有 96 个 rejection：captured 260、raw 212、valid 36、support 12、owner 260，
而 owner 侧为 212 valid、20 support、28 corner。root 保存的只读诊断位于
`C:/.../registered_preflow_rejection_r31.*`。首次 cap face 为
`(.000375, .009999999776, .047656249255)`，owner projection 等于该 face；
inactive axis 为 0，anisotropic box radius 为 `(.0012, .003125, .00234375)`。
失败原因是 blanket 的“两段端点都必须 local”规则拒绝 cap `26..27`，尽管实际
anchor/owner projection 在 support 内，而远端未使用 endpoint 的
`|dz| / rz = 1.00000063578`。

生产 locality 规则因此收紧为：owned anchor、recomputed owner projection，以及每个
actual/alias connector 都必须严格位于原有 convex support 内；不再把遥远且未使用的
segment endpoint 当作 connector。nearest owner 仍先于 permission，未扩大 support
或 tolerance，也未提升任何候选。针对 long-segment、mirror、isotropic/anisotropic
和 corner 正例，以及 outside-connector/tangent 负例，实际 RED 为 6 failed、4 passed，
耗时 25.44 s。该 RED 不是数值成功。

此前 54 个 CUDA 测试通过，host combined 为 257 passed、2 skipped（21.19 s）。
full 64 CUDA GREEN 现已通过（六个 geometry 文件，220.54 s），其中包括 10 个新的
clipped-locality 测试。新的 r32 source lock 为
`6c3b5c3de58713fa59e0b0e7f5ebfede6563f065ebdf376b25edfcf29fd952e6`（63 files）；
fresh r32 preflow 正在运行（sole PID 36136，session 21618）。没有新的 formal FSI50
成功或数值续跑成功声明。

## 2026-08-28 r33 source-freeze：r32 结果与 checkpoint 发布缺口

以下追加项保持 r30/r31/r32 旧结果原样。r33 source freeze 为
`69f5d760278d258d58a4c06ece4340fe0518d254f964b13db8d9f1115bb72b3d`
（63 files）；sole GPU preflow（PID 46236，session 41016）运行期间不修改 source。

r32 fresh preflow 在 source strict gate（107 files）下成功：runner
`398.9819721 s`、wrapper `414.0400877 s`。但 r32 FSI01 的物理 step1 虽已 accepted，
首次 checkpoint publication 因生产 history row 缺少 `time_s` 而失败；
`memory_accepted=1`、`durable_accepted=0`，wrapper `366.6365083 s`、progress
`357.1897965 s`。该尝试没有 step export、checkpoint 或完整成功，不能改报为续跑成功。

窄修复在真实 history-row producer 写入 canonical physical timestamp。host 证据只
从该 producer 的 AST 执行所选 `step`/`time_s` 表达式，再以 complete state fixture
通过 accepted-checkpoint wrapper 写入/加载；它不覆盖完整 history row schema，也不覆盖
完整 runner commit 路径。可选 raw IQN trial-vector observer payload 与 accepted checkpoint
的 numeric-only codec 不兼容，现于 runtime build 前 fail-fast；普通 checkpoint observer
与不使用 checkpoint 的 raw-IQN export 均保持支持，且没有静默删除 metadata。

lazy geometry allocation 的 review P2 已关闭：旧路径会额外分配 612 bytes/cell，当前格
为 `4.78125 MiB`、`128^3` 为 `1.1953125 GiB`，现已避免。实测为 genuine 6 RED、1 pass
（`7.14 s`），12 host GREEN（`7.16 s`），以及 71 个 geometry/lazy-regression CUDA
测试（`188.67 s`）。新增 metadata 12 tests passed（`7.05 s`），覆盖 cap SST-gradient
与 11 个 invalid-history atomic 情景。stdlib trace 的最终 combined host 为 271 passed、
2 skipped（`51.04 s`）；本地 line coverage 为 wrapper 86%、codec 91%、store 88%、
initial controller 86%、active controller 88%，不代表 global 或 GPU coverage。test-only
actual-entry `iqn_ils` correction 为 1 passed、27 deselected（`6.58 s`）。

所有已审 review P2 均关闭；真正的 CUDA checkpoint commit/resume 与完整 FSI50 仍待 r33，
本节不声明数值成功。

## 2026-08-28 r33 checkpoint 续跑证据与 r34 边界

本节只追加后来证据，不改写上文 r32 的 `time_s` 缺失失败、r30 的 45/50 前缀或历史 RED/GREEN 结论。

- r33 fresh preflow 已通过 strict source gate：runner `377.7516787 s`、wrapper `391.7570719 s`。随后 FSI01 的物理 step 1 已 accepted，但 ordinary 0D Unicode snapshot-stage tag 被 codec 拒绝，故 `memory_accepted=1`、`durable_accepted=0`；progress `370.7574285 s`、wrapper `380.3095066 s`。没有成功 checkpoint、restart、续跑或数值 PASS。

- Unicode scalar 采用显式 opt-in JSON metadata；physical arrays 仍须 finite 且 numeric。真实 full-runner commit、disk 和 observer replay 检查已在下述 31-test 子集中通过，随后移除了旧 IQN compatibility guard。

- host-only synthetic checkpoint-store 5,000 accepted-record soak：总 `417.0396941 s`，10 个 500-record block 为 `39.7719`--`41.2474 s`；最终为 2 个 NPZ generation、5,000 个 journal、`3,307,049 bytes`。detached launcher 的 exit code 未捕获；metrics 仅在全部最终断言完成后写出，另一次独立 strict-load verification 的 exit code 为 0，Taichi runtime 的 `prog` 保持 `None`。这不是 5,000 步物理、CUDA 或 FSI 保证。

- F 个 accepted steps 的 journal 只追加当步 report delta；每步仍序列化完整 state 与固定 preflow payload P，工作量 O(F*P)。当前 P=1，并非随 F 增长的 preflow prefix；restart 有意 O(F) 读取/materialize journal/report/RAM。这只定义有限资源边界，不能承诺真实 5,000-step 数值表现。

- 未改 geometry/dispatch/lazy suite 在 strict-CUDA profile 下为 71 passed、`188.67 s`，其中包括 host tests，不能写成“71 CUDA tests”。完成的 ten-file host gate 为 301 passed、2 skipped、`54.18 s`；local line coverage：wrapper 86%、codec 91%、store 88%、initial controller 86%、active controller 88%、predictor 81%。Unicode codec 共 27 cases（含 2 个 encoder-negative）。real-producer/full-runner-commit/disk/replay/IQN/physical-negative worker gate 为 31 passed、`7.55 s`，是 301 的子集，不得相加。

r34 source 已冻结为 `bf0262db073f7a426b5aa9fd9bea9009dd63e4d4f528cea12365afcf82cc0b20`（63 files）；当时运行中的 strict-CUDA preflow 已随后完成，短门禁结果见下文，不隐含 FSI50 结论。

## 2026-08-28 r34 短续跑实测与未豁免 parity

本节补记前节“running”之后完成的短门禁，保留 r31--r33 全部失败记录，也不豁免任何 comparison 结果。

- fresh preflow 通过 107-source gate：solver `235.8071233 s`、wrapper `245.5991915 s`。FSI01 为 1/1 passed：solver `210.8294641 s`、wrapper `221.1777761 s`。continuous FSI02 为 2/2 passed：solver `237.5784015 s`、wrapper `247.2869525 s`；warm step 2 为 `8.0053065 s`。
- 从同一 FSI01 output separate-process resume 到 step 2 的数值与 full-state audit 通过，wrapper `227.8522954 s`。116 个 arrays 和 5 个 U0D Unicode scalars 均 readonly，journals complete，正常保持 2-NPZ retention。

- 独立的 continuous-FSI02 与 FSI01-to-resume-step2 比较**不是 bitwise parity**。step 1 在 resume 前已经有约 `1e-19` 的 time 差异。`continuous_r34_resume02_comparison.json` 状态为 `differences_measured_not_waived`，有 38/116 个 array differences、2/61 个 control-scalar differences。此证据不推断 CUDA cause，也不豁免 parity。

r34 continuous FSI50 已随后失败；以下补记持久化边界与 r35 后续门禁。

## 2026-08-28 r34 continuous-FSI50 持久化失败与 r35 后续门禁

该 50-step run 有 7 个 durable frames，physical step 8 已 accepted，随后 manifest 的 `os.replace` 以 `WinError 5` 失败；progress elapsed 为 `348.5026734 s`，完整 `failure.json` 已保留。它不是 FSI50 success，也没有已捕获的 source-matched 8-step prefix。outer driver 随后打印 decoded stderr 时发生 GBK `UnicodeEncodeError`，故 exit 为 1、没有 `result.json`，也没有可信 wrapper elapsed；staging driver 已改为 UTF-8/log-before-print。

- WSL UNC 上 ordinary Windows native Python read handle 已实测使 destination replace 返回 `WinError 5`。生产共享 `atomic_file.py` 只对 Windows 5/32/33 做同一 replace operation 最多 8 次、合计不超过 `0.95 s` backoff retry，覆盖 checkpoint-store 的 3 个 replace 与 CLI JSON/NPZ publishers。manifest swap 前 old head 保持 published，绝不重试 physical work；persistent lock 仍失败，可能留下 unreferenced orphan。这符合 [Microsoft `FILE_SHARE_DELETE` 语义](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)，delete access 允许 rename。

- 证据为 6 个 manifest-contention RED、3 个 helper-boundary RED、2 个 CLI RED，并保留 old JSON positive；focused 为 60 passed、2 skipped、`18.17 s`。实际 WSL-UNC contention 为 20 passed、`17.56 s`。完成的 11-file host trace 为 321 passed、2 skipped、`75.62 s`；该次 missing-coverage mode 打印的 100% 不构成 coverage claim。reviewer 已接受 helper 与 readonly restore harness。

r35 source 已冻结为 `e02c5e1150995ed2471c4b880d2765758d138739c52d90102aac3fa332725ab2`（64 source files，含 `atomic_file`）；新的 strict fresh preflow 正在运行（PID 37272，session 81914）。kernel/source geometry 与 physical tolerances 未改。r35 必须先 fresh source-matched preflow 后才可 FSI50；尚无结果结论。

## 2026-08-28 r35 实际 preflow、31-step failure 与诊断分支

本节只更新上文 r35 的“running”状态，不改写 r30--r34 的失败、host-only 或 parity
边界。

- r35 fresh preflow 已通过 strict 108-source gate：solver
  `235.48416759999236 s`，outer wrapper `245.7792247000034 s`。这是
  source-matched preflow 成功，不是 FSI50 成功。
- 后续 continuous FSI50 在 31 个 accepted/durable steps 后失败（`31/50`），失败于
  尝试 physical step 32：runner last-step elapsed `564.8447479 s`，wrapper
  `581.146270000012 s`。strict full-source geometry audit 给出 physical
  generation 331、`owner_reason=2`（strict-support）、owner markers 14/15、source
  markers 15/16、face `(0,4,28)`、component axis 1，count 8（包括 faces 与
  sources）。这是 fail-closed geometry rejection，不能表述为 50-step success。
- durable head 为 K31，generation `7c7d4403d6ad4862b5a9709efa5fad7d`。任何
  replay 之前，完整 failed output 已 recoverably copy 到 staging
  `continuous_r35_failed31_before_replay`：100 files、10,345,146 bytes，全部
  SHA-256 已验证；原 failure artifacts 保留为历史证据。
- staging K31→K32 diagnostic/recovery 正在运行；post-restore exact-state audit
  与 geometry dump 尚无结果，故此处不声明 restored-state、resumed-step 或数值 parity。
- 5,000-record checkpoint 仍只是 synthetic host persistence soak，不能推导为
  5,000 physical FSI steps 或 50-step 成功。

`checkpoint_host_r35_line_coverage.json` 记录 host gate 为 321 passed、2 skipped，
`75.62 s`。selected-host executable-line measurements：accepted-state 86.5574%、
active-Kalman 88.5417%、initial-guess controller 86.6388%、atomic-file 100%、
checkpoint codec 91.8919%、checkpoint store 88.4837%。方法是 stdlib trace hits
与 compiled executable lines 的交集；它明确不是 branch、GPU 或 repository-wide
coverage。

## 2026-08-28 K31 精确恢复、step32 现场复现与方法修正边界

`reproduce32c` 的 post-restore audit 为 `passed=true, mismatches=[]`，仍是
K31/head `7c7d4403d6ad4862b5a9709efa5fad7d`。随后同一 source/face/owner 的
严格支持域冲突再次发生，53 个现场数组在 rollback 前直接读取并逐项验证 dtype、
shape 与 SHA-256。wrapper `282.50864770000044 s`、exit 1；没有 accepted step32。
scratch transaction generation=3（原连续运行=331），不是 physical step/time。

前两次诊断也保留：第一次被 staging audit 的错误模块引用阻止；第二次错误地要求
重建缓存保留历史 revision=685。已核实 macro snapshot 后保存的 live revision=686，
恢复重新绑定应精确使用 686。修正后的审计保留全部物理数组、控制器、IQN、哈希与
配对内容的严格一致性，只按明确重建契约校验四个版本标签，并独立检查 installed
anchor revision/hash；6 个 host-only 正反例通过。前两次都没有推进 step32。

独立 F64 复算证明：global owner 距 face `2.3775320834 mm`，合法原始 anchor
（vertex15）距 face `2.3821148871 mm`。后者在源/face box 内，z 余量约
`2.207607 um`；前者更近，却在两种 box 外约 `15.881563 um`。因此不能靠换中心或
放宽 target tolerance 修复。当前 reviewed 下一步是分类/来源仍用原 open box，
聚合几何连接改用 `conv(open D_face union {P_global})` 的最小凸包，并严格限制
P 的 active-plane 对角线欧氏距离。全局 nearest、所有 raw audit、拓扑/法向、
sole commit 与 `1.1e-6` marker closure tolerance 不变。该 connector-domain 语义
改变必须有对称 RED、严格 raw-boundary/外部 hull-facet negative、旧回归与新的
source-matched preflow/FSI50；本条仅记录设计与诊断证据，尚未宣告实现或 FSI50。

## 2026-08-28 r36 投影闭合域实现与已通过的测试

- 已将 reviewed 的有界 `conv(open D_face union {P_global})` 实现于
  `component_face_segment_audit.py`；raw 三个开盒、global nearest、角点/alias/
  normal-fan、物理时间与误差门槛均保留，不按作者数量或排列分支放行。
- 精确 F32 14/15/16 现场夹具先在 strict CUDA 得到 `owner_failure=2` RED：
  `projection_support_r36_capture_red_b.xml`，22.92 s。更早的测试导入错误仅是
  collection failure，不算算法 RED。实现后新 66 项 strict-CUDA 测试全部通过，
  23.41 s；包括轴与存储顺序对称、额外插入的域外合法 connector、严格 hull
  facet/owner-bound/非有限值，以及三个原 raw-box boundary negative。
- 新旧 geometry regression 合计 137 passed，172.95 s；其中有 host 与 device
  测试，不能称为全部 CUDA 或全仓库覆盖率。CLI 整模块 14 passed，13.79 s；
  两个 observer 统一清除旧 running-progress failure 字段，保留本次 event 的诊断。
- 独立实际 diff 审查无阻塞。r36 的 64-source lock 为
  `5b7134212538026988aed62a788b153d6ea20a966f1f83e4a1ba91792a383779`；
  新 strict fresh preflow 正在运行。此条尚不构成 FSI50 或 5,000-step 成功。

## 2026-08-28 r36 48-step 边界、r37 统一 disk 域与当前门禁

本节只追加 r36 的实际 fail-closed 结果和 r37 的当前实现/测试状态；不改写此前
r30--r35 记录，也不声明新的 preflow、checkpoint-resume 或 FSI50 成功。

- r36 fresh preflow 通过 strict 108-source gate，outer wrapper 为
  `215.04502249999496 s`。后续 continuous FSI50 exit 1：48/50 steps 已
  accepted/durable，step 49 失败；wrapper `673.7183566000022 s`，最后 accepted
  step elapsed `656.4502538000088 s`。因此这不是完整 FSI50 成功。
- 失败现场的 53 个数组均已逐一 SHA-256 核验。strict audit 有 4 个 raw
  `reason=7`，随后 finalizer 另记 4 个 `reason=9`（count=8）；owner
  `reason=0`。唯一短路径为 `14 -> 13 -> 12`，normal fan 为
  `0.5640962694098663 rad`，没有 alias 或 branch。其关键 connector vertex13
  到 face 的距离为 `2.548860933 mm`，位于 active-plane 外接半径
  `R=3.90625 mm` 内，却在旧 P-hull 外；这说明 P-hull 对合法曲边 connector
  仍过窄，而非 global owner、fan 或 raw provenance 失败。
- r37 将 aggregation locality 统一为原 anisotropic face domain 的最小开
  Euclidean disk：`R^2 = sum(active radius^2)`；scalar support 继续使用原来的
  `r.x^2`。raw 的三个 open box、B 的 global-nearest owner、normal fan、alias、
  corner、已有 tolerance 与 physical macro time 都不变。connector 的聚合语义
  因此明确变宽；旧 P-hull 实现已删除。该变化不豁免 raw source/anchor/route
  检查，也不把任何失败 step 变成 accepted step。
- TDD 证据：实际 capture 为 6 RED，`16.58 s`；curved aggregation-locality
  的 3 axes × 2 storage × 5 matrix 为 12 failed、19 passed，`27.24 s`。
  新的 focused strict-CUDA gate 为 107 passed，`70.57 s`，覆盖 strict
  boundaries 与 actual captures。旧 71-regression 已为 71 passed，`184.80 s`；
  独立 Sol 实际 diff 审查为 SHIP、无阻塞。它们仍不能将 107 个 focused tests
  表述为完整数值成功。
- r37 的 64-source 已冻结为
  `65af075de01d90e593bf03ed2e5e6f57a9aac10e54e5e998b5191e294a0c1e3a`；fresh
  preflow 正在运行（PID 24040，session 32934），尚无结果。没有新的 checkpoint
  continuation 或 FSI50 成功；本节也不把此前 host-only synthetic 5,000-record
  persistence soak 表述为 5,000 physical steps。

- r37 fresh preflow 已通过 strict 108-source gate：solver
  `245.96080210000218 s`、outer wrapper `256.79527399998915 s`。随后
  continuous FSI50 已启动，sole GPU 为 PID 40976、session 84828；本记录时它仍在
  运行，尚无 accepted-step 完整结果、checkpoint continuation 结果或 FSI50 成功。
  该状态取代上条“preflow 正在运行”的瞬时描述，不改变其余历史证据。

- r37 strict-108-source campaign 的 continuous FSI50 现已 50/50、exit 0：solver
  `769.536465600002 s`、CLI `771.245453999989 s`、outer wrapper
  `782.3919382000022 s`。first step 为 `261.5633334000013 s`；后续 49 个 warm
  steps 的 min/mean/median/max 分别为 `8.218361999999615 s`、
  `10.362905155102027 s`、`10.311846800002968 s`、`15.661034800010384 s`。
  50 个 step fields、FSI scalar、physical time、pressure 与 closure 均通过；raw
  IQN vectors 未导出，故本条不声明 vector audit。
- 独立 full-checkpoint audit 通过：50 records、116 arrays、5 个 U0D scalars、
  physical time `0.025`、正常 two-NPZ retention 且无 orphan；retained bytes 为
  `1878415`，state-array uncompressed bytes 为 `3052090`。r37 K8 generation 为
  `3ac89cb7bc5e4923bcae8632bb5a2458`，K9 generation 为
  `6b3f08a320b64c58aad104ed36963809`，均为同一 trajectory 的完整 prefix。
  64-source lock 仍为
  `65af075de01d90e593bf03ed2e5e6f57a9aac10e54e5e998b5191e294a0c1e3a`；checkpoint
  identity 的 source 为
  `ab8ee48c5275abcb0f8db200797aec5792a0e75dac36caef371d69f7c5ff8cf7`。
- paired recovery 尚未成功：root 计划将完整原 output recoverably 移至同 parent 的
  `continuous50_reference` r37，原路径安装 K8，再执行
  `paired_interrupt09 -> paired_resume50`；reference 将保留且不删除数据。当前也
  没有 5,000 physical-step 成功或 Fluent parity 声明。

## 2026-08-28 r37 成对恢复与本轮只读根因分析

本节更新此前“计划移动 reference／paired recovery 进行中”的瞬时状态，不改写
已独立核验的连续 50/50 成功。按用户“继续分析”，本阶段没有改动求解器源码、
容差或物理配置，也没有启动新的仿真。64-file source lock 仍精确匹配 r37。

- 完整 continuous50 已 recoverably 移至同目录的
  `ansys_vf__continuous50_reference__continuous__20260828__r37`；原路径安装了其
  同轨迹 K8 前缀。K8 post-restore 审计通过，`mismatches=[]`。随后在 durable K9
  发布后，故障注入 wrapper 按计划终止自己的 child；child exit 1 是主动终止，
  不应写成正常数值退出。wrapper 完成耗时 `271.0909398000076 s`。
- 重放所得 K9 generation 为 `5d978254ae0944b3812fc53fade03f4d`，已独立保存完整
  前缀。与原 continuous K9 比较：90 个 array entries 中 33 个有差异（含 state/
  outbox 重复镜像），61 个 control scalars 中 2 个有差异，全部物理时间账本及
  source/config/geometry identity 精确相同。状态为
  `differences_measured_not_waived`，不声明位级或数值轨迹等价。
- 第 8 步导出 NPZ 的 whole-file SHA 改变已确认仅为封装：41 个数组及解压 NPY
  字节、dtype/shape、顺序、CRC 和压缩大小均相同；28 个 ZIP members 的 DOS
  timestamps 变化。没有改动 full-file integrity gate；不是物理状态漂移。
- K1--K8 的 journal 标量逐项精确。K9 两路均 3 trials，各 trial 的 CG、平流、
  SST、固体子步计数相同。首次 fixed-point evaluation 就出现 residual RMS
  `0.033168228009225635` 对 `0.03316800686826735 m/s` 的差异；candidate RMS 为
  `0.018113972084486195` 对 `0.018113629923403658 m/s`。现有记录不能定位
  首个不同的 kernel。两条 initial-guess 统计差异由接受结果计算，是结果而非
  初猜配置不同的证据。
- 源码链核验：两路 trial 0 都先恢复 captured base；marker restore 从 physical
  rows 和 static binding 重建 projection-only cap rows，固体每个 substep 先清
  网格，压力 warmstart 每 trial 失效，SST work arrays 从 live k/omega 填充后使用。
  尚未找到闭合的“漏存字段／旧缓存先读”因果证据。恢复审计并未直接比较全部
  cap/scratch 首次输入字节，所以也不能声称所有 kernel inputs 已相等。
  MPM P2G 的 f32 并行 atomic accumulation 是候选机制，不是已确认根因；marker
  force scatter 已明确串行，不能混为一谈。
- 后续 K9-to-50 续跑已通过 K9 restore audit，但重连时无 Python 进程，最后
  durable step 为 21；无 result.json、failure.json 或 geometry error dump。
  原因及 exit code 未知，不能称 solver fail-closed 或 completed recovery。
  K21 generation `87508e21481a4659bd7d82d79421e087` 的 68-file immutable prefix
  已保存。独立身份推导的 strict reload 通过：21 条连续 journal、精确物理时间
  `0.0105 s`、90 个 readonly arrays（numeric values 有限）、正常两代 NPZ、无
  orphan。这证明落盘前缀完整，不证明 paired50 完成或恢复轨迹等价。

本轮分析报告位于
`C:/Users/lizhu/.codex/visualizations/2026/08/27/01a0434a-786c-7bc2-a8c6-1a381dda007a`：

- `continuous_r37_k8_step8_npz_comparison.json`
- `continuous_r37_k9_trial_diagnostics_comparison.json`
- `continuous_r37_unfinished21_prefix_audit_v2.json`（v1 的 taichi_imported 标签不准确，
  保留原件；v2 明确 imported=true、runtime_initialized=false）

下一项因果控制是同一 immutable K8-to-K9 的固定输入重放，并在 trial 0 的
restore+guess、边界组装、SST/predictor/projection、MPM 前后取直接只读指纹。
重复路径本身有差异也不自动证明 GPU 原因。运行健康、完整状态持久化、轨迹
可复现性是三项独立验收；本轮没有以跨运行比较放宽物理 closure tolerance。
200-step 扩展暂停，5,000 physical steps 与 Fluent parity 均未验证。完整逻辑及
官方并行执行说明见现有
[持续执行设计](../refactoring/ANSYS_VERTICAL_FLAP_CONTINUOUS_EXECUTION_DESIGN_2026-08-28.md#2026-08-28-r37-read-only-restart-discrepancy-analysis)。

## 2026-08-28 r37：恢复 50 完成、K51 新拒绝与系统性运动学 RED

本节取代前文的即时运行状态。生产求解器仍为冻结 r37；没有调整 target、closure、
pressure/FSI 容差，没有将未推进的物理时间记为完成，也没有提交或推送。

### 同输入阶段重放与完整恢复

- 两条同一 immutable K8→K9 重放均正常完成。22 个阶段指纹齐全，restore 到首个
  boundary-before 的 535 个字段精确；boundary-after 首现队列/append 顺序、
  component compact labels 与部分 anchor winner 差异。232 个 pressure rows 的
  canonical multiset 与 component membership 精确。首次 solid 子步入口的直接
  solid fields 精确，子步后出现 f32 级差异；不能将此提升为所有隐含状态均相同。
- 从 B9 接续至 50，child exit 0，solver `503.9677220000003 s`，wrapper
  `516.1597354000005 s`。这些时间只覆盖恢复后的运行，不是全新冷启动 50 步。
  最终接受 `50/50`、`t=0.025 s`，K50 generation
  `ae9ad6a7d95f486c82200713a63d20cf`。strict 108-source、50 fields/history、
  完整 physical time 与 checkpoint 审计均通过。仍没有导出 raw IQN trial vectors，
  因此 scalar history 检查不能冒充 raw-vector 审计。
- 恢复结果与原 uninterrupted50：116 数组中 44 个、61 controls 中 2 个不同。
  两路各自完整时间门禁通过，但 fluid accepted-time 累加值有最高
  `3.2526065174565133e-19 s` 的跨运行差异，不能宣称所有账本字节相等。
  50 个流体 masks 精确；最终 pressure max-abs difference `0.6130584479704808 Pa`、
  NRMSE/ref-max `0.00029665437599709526`；speed max-abs difference
  `0.029831301220154982 m/s`、NRMSE `8.415865736878026e-5`；tip 最大分量差
  `1.107342541217804e-6 m`。这是恢复轨迹间的测量，不是 Fluent 误差或新增容差。
- 完整 161-file 恢复参考已逐文件 SHA 核验保存在
  `validation_runs/solver_soaks/ansys_vf__recovered50_reference__continuous__20260828__r37`。
  原 uninterrupted50 与旧 K21 未明中断证据也保留；本次成功不能解释旧中断原因。

### K50→200：只接受 50 步，K51 fail-closed

- extension child exit 1，wrapper `205.38547879999987 s`。通过 K50 restore 后，
  post-solid observer 的 boundary assembly 被拒绝；没有接受第 51 步。
- 错误为 registered component-face source audit rejection count 52：28 raw
  rejection 加 24 face rejection；首 source `(0,8,31,1)`、target `(0,9,31)`、
  source primitive `3--4`、global owner `4--5`，owner reason 3（side/corner）。
  首 face 最近 owner 唯一且在原 strict support 内，不是半径不足。
- 对 53 个 dump 数组独立核验 SHA/shape/dtype 后：20 个 face 属于 `4--5`，
  几何弦法向被 endpoint material normal 的逐边规则翻转；另 4 个属于 `7--8`，
  未翻转但真实 signed distance 约 `-8.46266 um`，超过约 `0.02235 um` 的
  坐标 roundoff 界。仅修首个 flip 不会消除后 4 个冲突。
- 25 条真实边无非共享端点交叉，front/back 均 y 单调，但 front marker 5 有
  `99.806 deg` 折角，front arc 为 `16.704 mm`。简单曲线不能证明材料几何正确。
  dump 不含 obstacle mask 或同步 solid.x/F，不能从 raw_kind=0 推断两侧 mask。
- 失败后与已保存 recovered K50 完整比较：**116 数组、61 controls、全部物理时间
  账本 exact**；两代 NPZ、无 orphan。该 exact 结论仅比较同一 K50 在失败前后，
  不应混同前述两个不同运行轨迹。

### 与 Fluent 的数值证据边界

- 只复制历史 Fluent 数据，不复制 Windows 镜像代码。60 个 postprocess 文件、
  59 个 checksum entries 及 100 个原生 case/data 文件（585,608,259 bytes）均
  SHA 核验。canonical fresh50 input manifest SHA 为
  `19f57f2c9d6a09b2935335ad4c53af3946f68d4e3f26a2479863936f8acf125f`。
- 基准通过 50 steps、231000 residual rows、350 step/equation snapshots、7 类
  equation 与 strict `static_gauge_pressure_pa` / `outlet_0_pa` 语义；没有启动 Fluent。
- 使用原有 measurement helpers 对 recovered 小网格作 **不具 fine50 资格** 的
  同物理时刻诊断：最终 speed NRMSE/ref-max `0.07325381093220852`；raw pressure
  NRMSE/ref-max `0.4617041960686483`，RMSE `222.17314 Pa`。原始压力均值差
  `163.16335 Pa` 未被扣除或拟合。
- 最大 solid displacement：本求解器末步 `6.834697 mm`，Fluent 末步
  `0.216674 mm`；50 点位移 NRMSE/ref-peak `3.7910604001268005`。streamwise
  force 波形 NRMSE `0.151492867`，transverse 为 `7.45543944`；单位为 N/m，
  我方 3D 力除以固定 span `0.003 m`。网格/预流/数值身份不一致，不能把这些值
  声称为正式同条件精度，但也不能据此声称当前数值可信。
- case summary 的 `max_displacement_relative_error` 对照 CASE_SPEC 静态值，
  **不是 native Fluent 本次轨迹误差**。当前细网格 IQN/adaptive 50 步尚未启动。

### 不依赖失败步编号的 RED

- 实际 feedback kernel 只以宏观末态速度推进 surface 位置，而 solid 已累积所有
  子步位移。CPU 真 kernel 测试（仅隔离 runtime 初始化，无数值逻辑 mock）结果：
  6 signed/3-axis 恒速对照通过；6 两子步往返与 3 宏步分割不变性失败，
  `9 failed, 6 passed in 12.18 s`。二进制精确 fixture 的误差为 `dt=1/32` 的
  位移，不是舍入噪声。这证明该 API 不具材料位移一致性，不证明 K51 唯一成因。
- cap 的位置采用 `1.5*x_tip-0.5*x_prev`，速度只复制 tip；对称 derivative RED 为
  `12 failed, 12 passed in 9.67 s`。本次 cap pressure=false，不能把它说成当前
  cap 载荷错误；但它确实参与 projection geometry 与 wall velocity。
- K50 的 marker 宏观 Euler 更新 residual 最大 `1.836586180850317e-9 m`。
  最近参考粒子加局部 F-offset 的诊断候选与 marker 最大差 `0.641013 mm`；
  该候选并非已验证非仿射材料真值。solid particle 最大位移本身为 `6.834697 mm`，
  故 Fluent 大差异不只是可视化表面漂移。
- 新方法需要固定材料映射及其转置载荷传递，同时核验力、力矩、功率、根部反力、
  面积语义、IQN trial 与 accepted 恢复边界。至少 7 个生产文件，另有 MAC-face
  构造义务；已请求用户确认架构影响范围，确认前保持求解器冻结。详见持续执行设计。

### 比较器工具状态与本地证据

current IQN/adaptive 比较分支已经过独立只读审查并精确安装到权威 WSL 代码。
真实 comparator API 的 staging CPU integration 曾 `23 passed in 52.67 s`：包括 T1、
Picard/zero-rank 合法路径、128 physical markers 独立 cross-check，及 CG、MG
fallback、invalid row、49/51 frames、strict pressure、错误 raw vectors/布局、
旧 partial12 无最终认证和原有 5% 失败门槛。合成 Fluent fixture 仅把 canonical
目录重定向到临时封存数据，其余七类 residual、hash/count、metric/render 代码实际运行。
这不是正式 Fluent 精度结果。
旧比较器三个测试文件也已在 staging import overlay 上 `130 passed in 16.20 s`；
仅把资源根指回权威 WSL 树，未修改 canonical 相对路径或正式数值门槛。第一次
`129 passed, 1 failed` 是覆盖层自身文件位置导致锁定资源根不同，保留其原始 XML。

落地后不使用 overlay，直接对权威 WSL 源码运行上述新旧四个测试文件：
**153 passed in 67.34 s**。独立审查逐字节核对五个实际目标与已审 staging，
`5/5 cmp` 及 SHA-256 均一致；未使用旧 patch 或旧测试。新 profile 的标准库
trace 可执行行覆盖为 `199/200 = 99.5%`（对应 23 项测试，`102.61 s`）；
这是该单一新模块的行覆盖，不是分支、GPU 或仓库整体覆盖率。

CLI 显式选择 `--comparison-profile current_iqn_adaptive`；默认仍为
`legacy_final`。两者均保留严格 50 步、canonical Fluent 来源、压力语义与原有
5% diagnostic gate。新分支额外检查完整物理时间和原始 IQN trial vectors，
但合成测试通过不能代替一次真实 fine50/Fluent 验收。比较器落地后，r37 的
64-source numerical lock 仍精确为
`65af075de01d90e593bf03ed2e5e6f57a9aac10e54e5e998b5191e294a0c1e3a`。
求解器材料/载荷/MAC 架构变更仍待用户确认，不在本次比较器落地范围内。

本节机器证据位于前述 host staging 目录，关键文件：

- `continuous_r37_trace_ab_comparison.json`、`continuous_r37_trace_ab_boundary_semantics.json`
- `continuous_r37_trace_b_resume50_checkpoint_audit.json`、`continuous_r37_trace_b_resume50_trajectory.json`
- `continuous_r37_recovered50_reference_copy.json`、`continuous_r37_extend200.result.json`
- `continuous_r37_extend200_geometry.npz` / `.json` / `_support_audit.json`
- `continuous_r37_failed51_accepted50_comparison.json`
- `fluent_canonical_baseline_preflight_r37.json`、`continuous_r37_small_vs_fluent_nonqualifying.json`
- `material_surface_motion_red_r37.xml`、`material_cap_velocity_red_r37.xml`
- `continuous_r37_material_surface_k50.json`、`fluent_profile_integration_r37_v2.xml`
- `fluent_profile_installed_regressions_r37.xml`、`fluent_profile_executable_line_coverage_r37.json`

后续必须先通过材料/载荷/几何及独立固体回归，再 fresh source-matched preflow、
同条件 fine50/Fluent，之后实际 200→5000 physical steps；任何 synthetic 持久化或
几何循环均不得冒充物理长程仿真。当前状态仍不满足稳定且准确的生产求解器验收。

## 2026-08-28 已授权材料参考重构：先备份，再实现，再验证

用户要求“先对现在的代码进行备份，然后去进行重构”。本节取代前文的待确认与
冻结 r37 状态，但不改写旧运行的成功/失败证据，也不提交或推送代码。

### 可恢复备份

备份路径：
`/home/zhuohengli/backups/HIBM-MPM-r21-validation/pre_material_refactor_20260828T112045Z`。
`working-tree.tar.gz` 包含修改、未跟踪、ignored/cache 和验证产物；根 `.git`
链接单独记录，Git refs/history 保存在 `history.bundle`。实际解包到
`restore-probe/` 并恢复 bare Git repository 后，3,407 regular files、3,718 paths、
SHA256、metadata 和 HEAD 全部核验，备份前后树一致。备份在首次重构编辑前完成；
这是同机备份，不是异地灾备。

### 实现与未改变的数值边界

- 固定参考 Cartesian W 同时定义 `x_gamma=W*x`、接受态 `v_gamma=W*v` 与
  `f_particle=W.T*f_gamma`。保留有符号半格外推，不裁权、不扩大真实几何域。
  unity 使用无量纲运算界；端点比较同时包含粒子及壁面自身 f32 量化误差。
- 实际 f32 粒子载荷先暂存，审计 force/torque/material virtual power 后才提交；
  cap 载荷和速度使用同一位置映射的转置/导数。它不是全耦合能量守恒证明。
- 表面法向与压力探针跟随同一已验证几何；固定参考面积政策保留。默认物理面
  marker offset 为0，probe offset 独立为 `physical_face_offset=0.51`。
- IQN trial guess 不被通用 restore 改成 Wv；只有 post-solid candidate/accepted
  state 检查材料一致性。绑定 SHA、cap/拓扑身份进入 layout/checkpoint；完整
  恢复先检查再写入，坏 cap 不得污染 macro state。旧 r37 快照不可续跑新源码。
- MAC 候选在 progress ranking 前经同一 global-nearest-owner 几何资格筛选；
  不选更远方便的 owner、不删 raw author、不豁免最终 source audit/sole commit。
- Neo 初始化坐标 f64 计算、单次 f32 存储；support/damping 账本区分末子步力与
  接受宏步累计冲量。`pure_fixed_mass` 在未夹紧网格节点上审计被固定粒子丢弃的
  PIC/APIC 动量及角动量份额；已夹紧节点不重复记账。这不是包含自由粒子
  PIC/FLIP 差异的全局传递动量闭合证明。
- 不调整 target、closure、CG 或 FSI 容差。fluid/solid 接受时间仍各自消费完整
  `dt_s`；代数迭代收敛不能截短物理推进。

新 Fluent profile 为 `current_iqn_adaptive_material_reference`。保留 strict
pressure、canonical fresh50、精确50 frames/history、raw IQN、完整物理时间及旧
5% diagnostic gate；新增绑定身份与每步实际 adjoint audit、五个有限3-vector
support/damping字段、summary 与最后 accepted history 的精确一致性。实际 CLI
摘要已转存材料字段，profile 消费既有 JSON canonical `_N` 力单位键。

### 本轮完成的 focused 证据

以下 JUnit/JSON 均位于
`C:/Users/lizhu/.codex/visualizations/2026/08/27/01a0434a-786c-7bc2-a8c6-1a381dda007a`：

- `material_wall_input_roundoff_red.xml`：真实平移/镜像物理壁面6失败，12个超出
  总量化预算的负例通过；`material_wall_input_roundoff_green.xml`：68通过，
  pytest `5.78 s`。不使用物理 target tolerance 处理量化问题。
- `material_integrated_cpu_regressions.xml`：236通过，`81.15 s`；覆盖材料实际
  kernel、cap、绑定、runner 和完整恢复。这不是独立236个物理仿真。
- `material_transfer_strict_cuda.xml`：88通过，`81.84 s`；对应 runtime JSON 的
  87条观测均为CUDA、fallback=false、cfgopt=false、opt1、advanced=true、
  offline-cache=false；另1项是host-only元数据兼容。8个scope源码/测试SHA前后精确。
- `material_profile_final_evidence_red.xml`：41失败/1通过，缺失反力/最终摘要及
  summary-history 不一致原先被接受；`material_profile_cli_export_red.xml`：
  实际摘要缺少材料字段，1失败；`material_profile_cli_canonical_json_red.xml`：
  经真实原子写出的 history 力键是 `_N` 而新profile读 `_n`，1失败。
- `material_profile_cli_canonical_json_green.xml`：63通过，`112.34 s`，包含真实
  summary及50个history原子写出→完整comparator；fixture仍是合成Fluent数据。
  较早的100项 `material_profile_cli_consistency_green.xml` 未覆盖canonicalizer，
  不应代替这个最新门禁。
- `material_native_fluent_legacy_regressions.xml`：旧比较器三个文件130通过，
  `16.65 s`；不修改历史profile或Fluent阈值。
- `material_mac_existing_strict_cuda.xml`：既有 MAC 组装、owner、全 raw capture、
  path audit 等71项通过，`143.08 s`。其中有 host-only 测试；运行时严格CUDA，
  14个限定源码/测试SHA前后一致，不能把测试数量当成71个物理算例。
- `material_support_final_cpu_green.xml`：最终带零贡献原子操作 guard 的 support
  和 runner 诊断13项通过，`215.51 s`。APIC角冲量的有效RED证据为
  `support_policy_impl/red-affine-angular-corrected-oracle.xml`；较早轴权重写错的
  oracle 不是有效算法RED，保留但不作为正确性证据。
- `material_solid_initialization_strict_cuda.xml`：16项通过、36项未选，`674.35 s`。
  覆盖原有固体动力学/夹持/完整恢复和实际材料算例初始化；runtime严格CUDA，
  10个scope SHA前后不变。旧悬臂测试共20,000个固体子步，不是耦合FSI长程、
  网格/时间收敛或Fluent精度证明。
- `material_host_line_coverage.json`：绑定模块227/239行（94.979%），新材料
  profile 138/138行（100%）；对应131项测试、`214.65 s`。这是标准库trace的
  两个纯host模块可执行行覆盖，不是分支、GPU kernel或全仓库覆盖率。

测试有重叠，不能相加冒充独立验收规模。以上不是全仓库80%覆盖率、真实FSI50、
数值Fluent精度或5000步稳定证明。固体/MAC回归已完成，独立只读review确认本次
审查范围无剩余阻断项；其结论仅为可以进入新数值验证，归档于
`material_reference_r38_review.md`，不是生产或精度ship。

67个数值源码文件已冻结，source-set SHA为
`a5e73ba968a04a207bd0d330ea5e6e26b263162ad07da152cd8b5adb9a70d78a`；
逐文件清单在 `material_reference_r38_source_lock.json`。首次全新小网格预流于
2026-08-28 23:05（Asia/Seoul）启动，独立目录为
`validation_runs/solver_soaks/ansys_vf__preflow__material_small__20260828__r38`。
后续严格按 fresh preflow、真实连续/恢复短程、同条件fine50/Fluent 顺序推进；
只有前置门禁通过后才允许更长soak。运行中状态不能计作成功证据。

### r38 首次全新预流与短程审计

上述 small preflow 已实际完成，1个固定固体预流步、**0个FSI接受步**；
runner `elapsed_s=629.0527135`、solver `627.5299085 s`、外层进程总计
`636.0029586 s`。严格CUDA/CFG-off/opt1/cache-off与67文件source lock均保持。
`material_r38_small_preflow_gate.json` 确认完整运行manifest的113个相关源码文件
精确匹配、snapshot manifest/NPZ/hash一致；这113个文件的范围大于数值锁的67个。

小网格仍是4×32×64、48粒子、24物理marker，预流模式是明确的
`single_step_legacy`，不是fine配置的windowed stationary。预流压力相对残差为
`3.973641327924933e-08`，原门槛为`1e-6`。其traction readiness为
`not_evaluated`，stress valid/invalid为0/24，和旧r37单步预流一致；不能把此
快照的结构/来源核验说成流场稳态或载荷有效性证明。载荷有效性由真实FSI步另验。

新预流单步计时`574.3463474 s`，其中flow advance `563.3676835 s`、momentum
predictor `244.4945596 s`、SST transport `101.5737284 s`（包含首次编译）。
旧r37外层记录为`256.7952740 s`、预流步`203.8014076 s`；源码和界面方法不同，
这不是隔离单因素性能A/B。两次只读调用栈采样从pressure kernel compile推进到
stress AST materialize，不能将全段都称为同一内核卡死，也不能将cold时间换算成
warm每步时间。r38后续逐步成本仍待真实耦合运行实测。

短程审计脚本 `audit_material_reference_small_run.py` 重用既有压力、时间、完整
artifact gate，以及生产raw-IQN/material审计。独立review发现并修复“只比
producer/consumer相同而未锁方法”的误通过边界：显式small几何/预流差异之外，
固定fine profile的数值方法政策，并交叉检查summary Kalman off/modified=false。
`material_small_audit_identity_red.xml` 为16失败/4通过；GREEN为20通过
（pytest `0.96 s`）。这是host审计测试，不是20个物理仿真。

实际2步任务已从新预流启动，目录
`validation_runs/solver_soaks/ansys_vf__fsi02__material_small__20260828__r38`。
它的启动期调用栈已保存为 `material_r38_small_fsi02_startup_stack.txt`；采样时
已进入coupling trial的MUSCL内核编译，虽然progress文件仍显示旧initialization
标签。只有接受步/完整checkpoint发布后才按实际结果更新证据。

### r38 两步实跑、首次恢复拒绝及证据保全

上述 fresh FSI02 已完成2/2接受步（总物理时间0.001 s），每步3次IQN trial。
runner `712.5911805 s`、solver `711.1130757 s`、外层 `719.6532201 s`；
首次接受步累计 `702.9374291 s`，第二步累计 `710.9674985 s`，第二步增量约
`8.03 s`。不能用冷启动时间推算warm逐步成本，也不能以这个小网格结果推算fine50。
第2步24个stress marker均有效，CG相对残差`4.830794212906835e-09`；
marker closure `9.751001925906166e-07`，原门槛`1.1e-6`不变。

`material_r38_small_fsi02_physical_audit.json` 通过2步独立raw-IQN、完整fluid/solid
物理时间、材料身份、实际adjoint账本和summary/history一致性检查。第2步力残差
`2.3093584519672147e-10 N`，运算界`3.014601002048844e-08 N`；力矩与材料
虚功残差也分别小于各自运算界，五个反力/阻尼向量均有限。
`material_r38_small_fsi02_checkpoint_audit.json` 通过完整checkpoint检查：124个
数组、6个unicode scalar、2条journal、保留2个NPZ且无孤儿；这是2步证据，
不是50步或Fluent精度证明。

首次同目录resume04在恢复K2后、执行第3步前被host诊断拒绝，外层
`68.8839375 s`。唯一mismatch是诊断仍构造旧5键marker geometry，漏读新live
`material_surface_binding_identity`；不是已经推进到第3步的数值失败。
`postrestore-identity-red.xml` 为1失败/3通过，修正诊断的live读取后GREEN为
4通过（pytest `0.38 s`）。缺失或错误live身份仍严格拒绝，不从checkpoint补值；
不新增capture/refresh或任何数值写入。独立只读review同意重新运行恢复审计，
但该结论不代替真实恢复通过。host helper SHA为
`f3c964927b13a2ab228fb09928fb98acabe596805d9b4ca066b3cd2861668dd8`。

失败尝试与K2全目录已复制至独立
`validation_runs/solver_soaks/ansys_vf__rejected_resume04__material_small__20260828__r38`，
18个文件逐一验证源before/after及副本SHA，记录为
`material_r38_small_resume04_rejected_copy.json`。原运行入口不会自动清理旧终态
`failure.json`；重新审计前将这一已核验文件原地重命名为
`failure.resume04.rejected.json`，并保留独立副本的原文件名，没有删除失败证据。
文件SHA仍为`a365c8c3382bea91819d0a0dcd96102746ca81ec2d7e3e39db771b99a8dc8d98`；
K2 manifest SHA仍为`b2bd29cd52cf0836889c7b0903bd3b424a095aa59534aaa58e72f1839b7827f1`，
generation仍为`8f352ab0890742d998bd8659e8bae9e9`。这是明确的人工证据归档步骤，
不是生产入口已经具备自动attempt生命周期管理的证明；严格门禁仍拒绝当前失败。

独立continuous04于23:36（Asia/Seoul）从相同新预流启动，目录
`validation_runs/solver_soaks/ansys_vf__continuous04__material_small__20260828__r38`。
它不加载resume-only helper，数值源码67文件保持冻结；须完成后再串行执行
`resume04_reaudit`，禁止并行重任务。连续/恢复4步、fine50和Fluent比较仍待结果。

### 2026-08-29：连续4步与完整恢复4步完成，长程另列阶段

continuous04已完成4/4，物理时间0.002 s；runner `730.0717757 s`，外层
`737.3446579 s`。首次接受步累计`704.4198292 s`，其后3步增量分别为
`7.9084880`、`7.9803185`、`8.1690129 s`。physical audit和完整checkpoint audit
均通过；后者用显式禁止优化模式的新审计入口再次核验也通过。

resume04_reaudit从原K2完成到K4；runner `711.8829910 s`、外层
`719.8281721 s`，第4步warm增量`7.8763316 s`。恢复瞬间的
`material_r38_small_resume04_reaudit_postrestore_audit.json` 为passed=true，
0 mismatch：直接读取恢复后的macro state、边界、控制器、完整账本及counter，
并检查材料身份、压力镜像及warmstart无效化。projection-only cap和runtime scratch
按其明确重建政策另验，不冒充保存态的逐字节字段。
`material_r38_small_resume04_reaudit_physical_audit.json` 与
`material_r38_small_resume04_reaudit_checkpoint_audit.json` 均通过：每步完整fluid/
solid物理时间、raw IQN、材料/反力账本和最终摘要一致，124数组/6 unicode scalar、
4条journal、2个live NPZ，无孤儿。两端最新K4均独立匹配当前source/config/geometry。

独立review发现旧比较脚本只对数组和62个selected control标量作exact判定，漏掉
部分anchor/observer及journal元数据；不能据此称完整checkpoint一致。新候选
递归比较整个state和全部records，包括空结构、严格类型、scalar和数组。Python
诊断NaN按checkpoint codec的单一sentinel标签比较，numpy scalar仍按dtype/bytes；
物理数组有限性门禁不变。behavior RED为
`checkpoint_comparison_identity_corrected_oracle_red.xml`（10失败/16通过）；
较早`identity_red.xml`有3项测试字段名写错，不作为那3项有效RED。另有numpy
float64 NaN payload负例1失败/26通过；最终`checkpoint_comparison_final_green.xml`
27通过（`0.68 s`）。真实K4经main自比较exit0、0完整差异，证据为
`material_r38_small_continuous04_self_comparison.json`。审计脚本的`-O/-OO`及
`PYTHONOPTIMIZE`负例RED4失败→GREEN4通过（`26.49 s`），防止断言检查被优化删除。
此前实际审计使用优化级别0，不受该缺口影响。

实际两轨迹K4比较不是exact：`material_r38_small_continuous04_vs_resumed04.json`
记录41/124数组不同、2个selected初始预测诊断不同，3个fluid accepted-time值
差`1.0842e-19`至`2.1684e-19 s`；总macro time同为0.002 s。全state/records
共2243处差异，其中包含计时、输出路径、几何hash和浮点诊断等，不能把它们称为
2243次物理失败或静默删掉。主要字段实测最大绝对差为：solid x
`3.725290298461914e-09 m`、fluid velocity `4.186853766441345e-05 m/s`、pressure
`9.242795342458976e-04 Pa`；其相对L2差分别为`4.2239e-08`、`5.4009e-07`、
`2.3299e-06`。完整报告保留全部差量，状态为`differences_measured_not_waived`。

这两个运行是分别从相同预流启动，K2之前就已有差异（例如step2速度峰值
`42.3414192199707`与`42.34143829345703 m/s`），故不能把K4差异全归因于restart。
恢复瞬间live-vs-saved审计通过与两条后续轨迹逐位一致是不同结论；本轮未证明后者。
source lock的67文件在两次实际运行前后都保持不变，未放宽任何数值门槛。

遵照用户2026-08-29追加的顺序要求，先完成当前重构、运行attempt生命周期、回归/
review及文档收尾，再开展完整50步、Fluent比较与分级长程稳定性验证。当前没有
启动长程仿真，也不能声称fine50、5000步或Fluent精度已经通过。

完整比较器最终独立review为ship（仅验证器范围），SHA为
`0d930d2276169eb7d4d12f9cbc4dd6791c7986c2ed8c48670e09667a45447c88`。
已提升为staging正式审计脚本，旧版本保存在
`checkpoint_comparison_impl/legacy_compare_accepted_checkpoints.py`。最终版本重新生成
`material_r38_small_continuous04_self_comparison_final.json`（exit0、0差异）和
`material_r38_small_continuous04_vs_resumed04_final.json`（exit2、差量与前版相同）。
完整差异保存在JSON，stdout只显示数量摘要；r38的67文件source lock再次核对一致。

### 2026-08-29：运行attempt生命周期完成，重构阶段关闭

`simulation_core/diagnostics/run_attempt.py` 与实际validation CLI已完成独立review并
应用到权威WSL。恢复预检先用生产 `canonical_source_sha256` 核对当前源码，再校验
loaded state与同一head的accepted step/generation；只有全部预检通过，才在
`attempts/<唯一id>/metadata.json` 原子发布旧终态文件的SHA、size、checkpoint和
source身份，然后移动原始failure/interruption bytes。新失败仍写为新的active终态。
完成态/oracle消费同时要求summary/progress completed及没有active终态。
这不重试物理步、不豁免checkpoint source，也不放宽任何数值容差。

真实Windows/WSL诊断先复现了 `Path.exists()/is_symlink()` 漏报Linux链接；对应
`lstat` 仍提供reparse属性。因此helper以目录条目和reparse属性拒绝异常路径。
`run_attempt_unc_symlink_measurement.json`保留原候选漏报证据；
`run_attempt_symlink_probe.json`最终通过Linux与native Windows UNC各4类共8项
（failure、interruption、dangling、archive-directory），均不移动原条目并拒绝
伪完成态。只使用自动清理的一次性目录，没有改动真实运行记录。
此实现只承诺单写者、普通进程异常时bytes留在active或archive；未做目录fsync，
不宣称掉电耐久。Python属性语义参见
[Python 3.10 stat documentation](https://docs.python.org/3.10/library/stat.html)。

正式回归已放在 `tests/diagnostics/test_run_attempt.py`、
`tests/integration/test_ansys_vertical_flap_run_attempt.py` 和既有checkpoint CLI测试，
不依赖本机STAGING路径。合法K2到K3旧CLI behavior RED（3条history）保存在
`run_attempt_impl/run_attempt_behavior_red_k2_to_k3.xml`；其中preflight被mock，
所以只证明旧main遗留active failure的生命周期缺口，不冒充完整checkpoint解码。
正式测试覆盖成功归档/失败后原档保留、新active failure、同一head复用、source/
generation负例、metadata写入/首及次rename异常、noop、目录及reparse拒绝。

扩大host回归首轮为272通过/3失败/3跳过，证据
`run_attempt_host_line_coverage_tests.xml`。三失败是旧测试接口失配：仍patch旧
`Path.replace`、期待20次固定退避，以及调用私有fresh-output函数时缺少已必需的
`resume=False`。只更新两份测试，改为实际 `os.replace` 加显式 `winerror=5`，
严格检查同一temporary/target、8次精确退避总0.95 s、原bytes和临时文件清理；
fresh输出仍拒绝非空目录。没有因此改变生产实现或减少物理验证要求，独立review
确认不是削弱验收。

同一278项集合重跑：`run_attempt_host_final_tests.xml` 为275通过、3跳过，
`38.24 s`，没有失败。跳过项都是native Windows符号链接创建权限：两项旧
checkpoint retention测试、一项新helper测试；新helper已有上述真实双平台测试，
不将两项旧skip伪称为通过。`run_attempt_host_final.json`记录新helper executable
lines `139/153 = 90.8496732%`，不等于branch、GPU或全库80% coverage。首轮失败
证据没有覆盖。helper SHA为
`c211114b3d139cf74e505e3efcc1e071bf686c2a8202066b49b0d1a634b1a69a`，
CLI SHA为`cf6f3deb9606de6bfcaee9c9a3481288d72bdfff75482797e863df65f1e4d2a8`。

另以原r38 rejected-resume的18个真实文件生成一次性完整副本，复用记录的CLI参数
仅替换output/checkpoint路径，执行真实main、head读取和source预检（没有mock
这些预检）。`run_attempt_real_old_source_rejection.json`通过：旧K2 source
`28519e07857a2bc416335b635d6312b37cb192af385bab29e554350bd8accbe4`
在任何archive/progress/manifest写入前拒绝；18文件SHA、历史原目录均不变，
Taichi未初始化，runtime configuration及solver禁止桩未触发。不是物理仿真结果。

重构前backup的tar、Git bundle和CHECKPOINT.md再次SHA核对全部匹配原清单，
备份仍位于 `/home/zhuohengli/backups/HIBM-MPM-r21-validation/pre_material_refactor_20260828T112045Z`。
最终源码重新冻结为 `material_reference_r39_source_lock.json`，68文件集合SHA：
`fdb5919fbb897c061ef507a03a923d9a6dd5ddd3a63e5b7f3975b18c9425e4dc`；
生产checkpoint source身份为
`cda30f7b85eec55276e5942830491fa43df698e93c8d0a5187fe95dad89f5b1d`。
旧r38的材料/固体/MAC数值代码没有在此次host收尾中改变，但严格source身份已变，
因此旧预流/accepted checkpoint不能在r39复用。r38真实短程结果仍只作历史证据。

到此用户授权的备份、重构实现、聚焦CPU/CUDA、短程连续/恢复、host收尾及独立
复核阶段完成。下一阶段才用新r39源码生成严格预流、完整fine50和锁定Fluent比较，
然后逐级验证200/5000步。不能把本阶段完成称作数值精度或长期稳健性已经通过。

### 2026-08-29：r39 fine 预流完成，独立数值资格阶段开始

最终源码的 fine dry-run 已通过52项配置身份核对，外层 `7.03055 s`；dry-run
不计为任何物理步。实际 fine 预流于00:59:44（Asia/Seoul）串行启动，目录
`validation_runs/solver_soaks/ansys_vf__preflow__material_fine__20260829__r39`。
这是4×256×320流场、5120固体粒子、128个物理marker，requested preflow上限200，
没有复用r38旧snapshot。运行期间没有修改生产Python源码或启动第二个GPU任务。

生产预流在第78步满足windowed_stationary后正常停止：`preflow_converged=true`，
mode/status/stop_reason均为`windowed_stationary`；runner `1628.1685360 s`，
外层 `1635.4559099 s`。最后一步 `13.1230015 s`，其中flow advance
`13.1067949 s`。78步全是固定固体预流，不是78步FSI；summary的FSI
requested/completed仍为0，final_time_s仍为0。

`material_r39_fine_preflow.command.json`、`.stdout.log`、`.result.json`保留完整
启动、输出与退出证据。实际runtime为Taichi1.7.4 strict CUDA/f32，CFG关闭、
opt_level=1、advanced开启、cache关闭；68源码集合在运行前后SHA保持一致。
raw traction readiness为`flow_only/not_evaluated`，不能称为traction-ready。
本条只记录生产端完成；独立snapshot完整性、当前source/config、全部窗口历史与
物理健康审计仍需另验，通过后才允许启动fine50。

Fluent参考数据也以当前WSL的生产校验代码、native Windows Python重新只读预检：
`fluent_canonical_baseline_preflight_r39.json`通过50对原始case/data身份、所需
postprocess文件checksum、231000条残差/350条snapshot摘要及50条force history
时间网格。canonical input manifest SHA仍为
`19f57f2c9d6a09b2935335ad4c53af3946f68d4e3f26a2479863936f8acf125f`；
force history当前SHA为
`a1eb635aed919ceb1445358c8f78947dca4e773fa0a077b025edd10dde5f3c8a`。
后者记录当前内容身份，不是独立预期force digest；正式比较仍会重新校验数据。
预检没有初始化Taichi，也没有计算r39对Fluent的数值误差。114项完整source映射
在这次只读预检前后保持一致。fine50、Fluent误差门槛和长程运行均尚未通过。

#### r39 预流独立审计通过；fine50 已启动但未验收

`material_r39_fine_preflow_gate.json`已通过：生产strict snapshot loader重新验证
完整NPZ/字段、canonical边界权限、独立当前config/source身份；实际runtime和
114项源码映射一致，78条历史及生产重算的3个稳态窗口通过，Taichi未初始化。
geometry digest在此仅验证snapshot内部完整性，后续真实恢复会独立重建live几何。
flow_only/not_evaluated保持原样，traction_qualified=false。
窗口union最接近1%门槛的是涡黏性峰值相对span `0.009980517682825126`；门槛未改。

审计器先有一次真实history等值误拒，不是生产求解失败。
`material_r39_fine_preflow_history_difference.json`记录156个缺键差异，全部为每步
两个legacy `marker_action_reaction_residual_n`/`scatter_action_reaction_residual_n`。
production CLI原有export规则将它们无损合并为对应大写N，snapshot保留双别名。
独立review逐一核对78步156组snapshot_n/snapshot_N/compact_N，有限float的IEEE64
bytes完全一致（包括零符号），无其他数值或结构差异。只在staging审计history层
复用这两个明确别名的语义，冲突仍拒绝，不采用通用casefold或浮点容差。
旧B886审计器的拒绝证据与诊断保留，未修改生产源或原运行数据。

预流审计器最终SHA为
`d6f0f56514f5fa50e1d095f18f7fd38e88f5420f8e0701ab335a6a2721ee9da6`。
它还校验raw readiness/mode再与生产重算结果对比，防止normalizer覆盖错误声明；
flow_only允许生产实际支持的evaluated/not_evaluated，而非硬锁后者。68冻结和
114完整源码映射均在审计前后核对。真实normalizer的RED及别名RED分别保存在
`fine_gate_impl/fine_preflow_raw_red.xml`和`fine_preflow_history_alias_red.xml`；
root独立最终 `fine_preflow_alias_root_final_tests.xml` 为18通过（`8.37 s`）。
Sol只读复核与真实78步审计均通过，才启动正式fine50。

后置consumer admission额外锁定config/report的输入prefix为已审producer的
实际output prefix，且比较全部保留的preflow history与完成步数；三项config/
source/geometry identity本身不等于具体snapshot状态身份。对production原有的
9项可选projection NaN哨兵，复用 `_mutable_preflow_report_value` 再严格JSON
比较。独立实际78步witness证明702处差异全为这9项NaN到null，无其他差异；
Inf、未授权/错误路径NaN、有限值、类型及signed-zero变化均继续拒绝。没有改动
生产编码逻辑或物理字段有限性门槛，也没有把该witness称为fine50已完成。
consumer审计器SHA为
`98f4cc759235f1d7eae97511cff57030e6c9521f8bc7c0d80da58d3ad0e51862`；
两套审计器的root独立最终 `fine_gates_root_final_tests.xml` 为34通过（`11.68 s`），
Sol限定范围复核ship。这些仅是host验收工具测试，不是34个FSI物理步。

正式 `fine formal50` 于01:54:13（Asia/Seoul，runner PID30024）启动，目录
`validation_runs/solver_soaks/ansys_vf__formal50__material_fine__20260829__r39`。
复用本次已审78步snapshot，不重算预流；实际Taichi日志确认arch=cuda。
保持唯一昂贵GPU任务，34项离线host测试未初始化第二个GPU。启动证据为
`material_r39_fine_formal50.command.json`及`.stdout.log`；完整50步、完整checkpoint
审计、正式Fluent误差与200/5000长程都仍需真实结果。

#### r39 fine50 真实失败：25步接受，第26步IQN未收敛

本次没有完成50步。进程exit1，生产failure elapsed为`2349.9980146 s`
（39.1666分钟），外层launcher为`2357.4426474 s`。step26在16次迭代后报
`FsiCouplingConvergenceError`：relative residual `2.605355e-3`，absolute
residual `5.889273e-6 m/s`，原relative门槛仍为`1e-3`、absolute仍为0。
progress为failed，accepted step25、time0.0125 s；没有step26已接受帧或completed
summary。68源码集合在运行前后不变。不得把此次算作fine50、Fluent比较通过，
也不启动200/5000长程。

冷启动到首个accepted-step stdout为`730.4347251 s`；step2..25主要约54–69秒/步。
第26步明显变慢后达到16次上限。前9步只读计时检查显示step2..9所有trial的flow
平均42.256 s、solid平均2.611 s；HIBM计时部分嵌套flow，不能再直接相加。
accepted trial的flow平均13.085 s，其中momentum约6.014 s、SST约2.335 s，
余量包含压力/边界/诊断，不能命名为纯PCG。observer的export时间在写盘后才回填，
step JSON中的0不代表真实零开销；这些timer也不覆盖checkpoint。旧r28原配置的
6–12秒/步不作为本次fine配置的测量。

`material_r39_failed_fine50_prefix_audit.json`为独立只读检查：82个运行文件在审计
前后SHA完全不变；25对step_fields/history的原始IQN、完整流固macro dt、材料
伴随审计通过。生产checkpoint loader验证独立current source/config、完整state
及25条journal；98个state数组只读且所有物理数值有限，共`99,249,965`未压缩bytes。
只保留K25/current和previous两份NPZ，没有额外generation；失败step26没有提交。
该host审计没有初始化Taichi，完整114文件source映射不变。其runtime期望来自已审
同源预流，后续实际replay仍必须重建并核对live runtime和几何，不能省略恢复审计。

本轮不能从最后一对residual判定发散、secant病态或噪声floor。step24/25均3次
收敛；最终absolute分别约`8.382e-6`/`6.246e-6 m/s`，但step26的candidate RMS
由末值反推约`2.260e-3 m/s`，远低于step25的`2.619e-2 m/s`。低速度尺度下的
相对收敛停滞只是待证假设，不授权放宽门槛。

已确认的诊断缺口是：generic solver把完整16轮scalar/向量轨迹放在
`FsiCouplingConvergenceError.report`，CLI却只保存`.diagnostics`；因此本次
failure.json没有完整失败轨迹，不能事后捏造。计划先备份当前全树与失败现场，
在不改任何生产源码的exception-only staging wrapper下做一次同源K25单步诊断
恢复。改CLI也会改变严格source身份，不能先改CLI再偷偷复用r39 checkpoint。
该replay无论结果如何都不是新的fresh50或长期稳健性证明。

早期Fluent比较仍限定1..9相同时间层的导出标量：第9步最大位移ours
`0.361443 mm`、Fluent`0.569496 mm`，约`-36.53%`；依既有比较器反转streamwise
并除以span0.003 m后，流向力低约36%–45%，第1步已存在。独立核查原始Fluent
首步case/data可重建Fx=`4.827743836088798 N/m`：两侧压力贡献
`4.8293587673212395 N/m`，水平顶盖压力的Fx严格0，顶盖剪切仅减去
`0.0016149312324411462 N/m`。因此缺顶盖/该黏性项不能解释ours约2.17 N/m的
首步流向力缺口；这不等于已定位其上游压力误差根因。

两边rho_f/mu/rho_s/E/nu及域/瓣片尺寸相同，Fluent原始case-config证明为2D
瞬态线性结构、平面应力支持、Rayleigh关闭；ours有0.995宏步速度阻尼，且当前
流体报告明确finite3D slab、x面未配置strict periodic/slip，不能称严格同一2D
模型。预流的全局pressure/velocity extrema也已不同，但尚未统一采样掩膜，不把
这些extrema当作正式pressure误差或唯一归因。完整Fluent CLI因缺50步不运行。

5000步容量只读审查：正常accepted写入没有发现按历史长度重复扫描的O(N²)
路径；但history/report及一次完整resume仍为O(N)。按本次1..15文件尺寸，
5000步的step_fields（已经含IQN trial arrays）+step_history线性外推约10.32 GiB，
不含journal、两份完整checkpoint及最终报告，也不是实跑验证。checkpoint旧代
删除遇OSError只warning，反复失败会积累旧NPZ；长程验收需包含此运维风险。

#### r39 失败现场第二份备份、单步诊断与保留性审计

修改任何生产源码前，已将重构后的全树及原始fresh50失败现场再备份到
`/home/zhuohengli/backups/HIBM-MPM-r21-validation/material_r39_failed_fine50_20260829`。
3620个regular文件、3947个路径的前后清单、解压恢复SHA、tar内容比较、gzip测试
和Git bundle恢复验证均通过。tar SHA为
`94b55bd43d3d8a1d9e17d622c498b2bd14ede4242855b156ccec505c4aa2c374`，
bundle为`8a889443aed85ce4e70aa79e6d9aba3c5a4dc96000d66a2451a9576c16e78d55`，
CHECKPOINT.md为`a05d77d003d87ff01ab23fc4231b0f2bab1c557240107bf120198082d535bfc0`。
这是同机可恢复备份，不是异地灾备。原fresh50的82文件仍在该备份的
`restore-probe/validation_runs/solver_soaks/ansys_vf__formal50__material_fine__20260829__r39`。

之后仅以staging exception-only wrapper执行一次同源K25到26诊断恢复。生产
源码、物理参数、IQN16次/1e-3/absolute0及压力门槛均未改变；wrapper的8项host
测试和Sol审查先通过。`material_r39_fine_iqn_replay26_postrestore_audit.json`
证明真实live恢复的声明物理字段及控制状态匹配K25，mismatches为空；明确排除
可重建projection cap和runtime scratch，不能据此声称所有内部状态已逐字节核对。

诊断恢复仍失败：step26的16次IQN后absolute residual为
`4.6567788166375205e-6 m/s`，relative为`0.0020602542066067574`；candidate
RMS为`0.002260293318030518 m/s`，对应原门槛`2.260293318030518e-6 m/s`。
runner `917.7150784 s`，外层`938.6112583 s`。没有接受第26步，仍为25步、
0.0125 s。完整16×128×3 guess/candidate/residual现保存在
`material_r39_fine_iqn_replay26_trace.json/.npz`。它是另一次轨迹，不能冒充
第一次fresh失败时未保存的16轮数据。

`material_r39_replay26_preservation_audit.json`通过物理payload与checkpoint
保留性审计：K25 head、两代NPZ、25条journal及所有数组dtype/shape/raw bytes
均保持；78个accepted文件中77个文件SHA不变。末帧`step_0025.npz`因正常resume
outbox重新调用`zipfile.writestr`而改变ZIP DOS时间戳，全部成员数组bytes不变；
仅清零local/central headers的四字节DOS时间字段后，两个ZIP完整bytes相同。
因此不能声称78文件原始bytes全不变，也不能把容器时间戳变化称为物理状态改写。
负例详情保留在`material_r39_replay26_preservation_difference.json`。
原fresh failure按既有生命周期归档到
`attempts/5b58fe04f3a04619977497f89a16249a/failure.json`，SHA仍为
`04e0694653a874c6ced91041654f5d84040eaaaf43f3ed500af13ecd1e2b32ee`。

独立只读复算完整16轮的原始数组SHA、7类标量指标和15个IQN next-guess，均与
生产算法逐bytes一致；首次guess等于备份K25最后candidate。rank最终保持8，
没有fallback/limit/reuse，系数范数最大约1.688，不支持“错误IQN实现/系数爆炸”
的归因。尾部absolute约4.657e-6到6.666e-6，candidate相对均值的RMS散布约
4.769e-6。相邻第13/14轮guess RMS差`4.22445e-7`，candidate差`7.45590e-6`
（增益约17.65）；尚未用相同guess重复计算，不能仅据此定性为随机噪声。
下一步为同K25、同runtime原生begin/evaluate/rollback的A/B/A，不接受任何物理步。

#### MPM 精度根因：独立算术实验与实际求解器RED

K25实际1284个固体子步，每步f32 dt为`3.894080862210103e-7 s`。纯host审计
`material_r39_solid_precision_host_audit.json`发现10200个非零对角dt*C增量中
1140个在f32加单位阵时被舍入掉。这只能定位算术风险，不能单独证明实际核行为。

随后`diagnose_material_r39_solid_cuda_arithmetic.py`用与r39相同CUDA编译设置
运行独立核，`15.203 s`：恒定正负0.01/s梯度累计1284次，f32 F应变均为0，
显式f64递推分别为`5.000012283096211e-6`和`-4.999987159859387e-6`。
对K25全部合法F（det约0.98379到1.01554，奇异值约0.97581到1.02424），无需
限幅却仍做f32 SVD重构，F最大扰动约1.073e-6；以同一个host-f64平面应力算子
评价重构前后输入，最大应力扰动约1.7634 Pa、RMS约0.2431 Pa。
这项应力数字隔离的是SVD输入扰动，不冒充完整生产载荷误差。Taichi1.7.4的SVD
默认使用runtime.default_fp，因此仅将F field改f64而不明确计算dtype并不足够。

真正生产MPM公共step的3项CUDA行为测试已先RED（`solid_precision_red_cuda.xml`，
`144.66 s`）：正负0.01/s的8粒子APIC affine patch完整1284子步，C均维持1%
以内、OOB和clamp均0，但所有Fyy-1为0，目标为正负`4.99999983e-6`；另对
线性小应变模型的`F=I+skew_yz(0.0123)`，sym(F-I)恰为0且初始v=C=0，
一步后产生最大`5.92003971e-7 m/s`虚假速度，clamp仍0。
后者是线性模型的零小应变测试，不是把有限旋转误当线性模型零应力。
这三个生产RED已经证明通用精度缺陷；仍需A/B/A确认其与step26停滞的因果链。

截至此记录，68/114生产Python源码仍与r39冻结一致，没有实施F64/SVD修复，
没有放宽target/pressure/IQN容差或增加IQN次数来通过仿真。后续修复必须覆盖
持久F精度、save/restore及严格checkpoint dtype，不允许恢复时隐式降精度。
必须重新生成同源预流和完整fine50，再进行锁定Fluent验收；200/5000仍未启动。

#### r39 同一接受态 A/B/A：重复性损失定位到固体阶段

同源诊断 `material_r39_iqn_aba` 已完成，外层 `750.8178024 s`。原生 runtime
begin 一次、evaluate A/B/A_repeat 三次、rollback 一次，未调用 accept；A/B
来自前次失败轨迹的 guess 索引 12/13，第三次与 A 使用相同 guess。wrapper
先经过 17 项 host 测试与只读审查，实际三次均执行 **1283** 固体子步、
子步 dt `3.897116134060795e-7 s`，各推进完整宏步 `0.0005 s`，retry、clamp、
OOB 均为 0。这是 K26 诊断，不能套用 K25 报告中的 1284 子步。

`material_r39_iqn_aba_audit.json` 通过；补齐审计器的完整 37-key owner schema
和 source-trace/launcher dependency SHA 前后检查后，6 个 main 级负例先真实
RED、16 项 host GREEN。第二名只读审查者从三个原 NPZ 逐一核对 111 数组的
dtype/shape/finite/SHA，并重算全部 74 个字段差量，均与审计 JSON 相符。
原始 capture：metadata SHA
`bcf3245e96ec4c61c1e3d111264d85bf873c8969b7937fd5f709bf05c2dfb367`；
A NPZ `07e8292adfdf7a5ac97baadc301bdc84f7a3bf4711821734f0a00d645339d8d9`；
B NPZ `35ed9ec33d9b8e1caaf7a6b80a4c6208507b8c24c09837c3ab7227542b213598`；
A_repeat NPZ `a123ff5635d2dc78186901a07489e6ccb3115317bac1812bd96da522a3770ed2`。

A 与 A_repeat 的记录输入中，7 个 flow_enter 字段、5 个 solid_enter 状态字段
以及 **5120 粒子的 external_force_n 均逐字节相同**。输出则为：

- marker candidate 速度每向量 RMS 差 `7.915648345685995e-6 m/s`，最大分量差
  `2.0475854398682714e-5 m/s`，256/384 分量不同；原 IQN 门槛约 `2.26e-6 m/s`。
- 固体粒子速度每向量 RMS 差 `5.366408043475175e-6 m/s`，F 最大差
  `9.655952453613281e-6`。
- 流体 pressure 最大差仅 `3.2374600777984597e-9 Pa`，marker F_gamma 最大差
  `1.51704278479603e-15 N`；这些差量在实际 f32 粒子外力中完全舍入消失。
- A/B 的 guess RMS 差 `4.22445251974235e-7 m/s`，candidate RMS 差
  `6.895540809473421e-6 m/s`，小于同输入 A/A_repeat 的差量。

证据支持将主要重复性问题定位到 solid_enter 之后，包含 MPM scratch、并行
原子累加与算术精度。它还不能单独证明 SVD、F 累积或任一特定 kernel 是唯一
原因，也不能证明拟议 F64 修复足够。下一门槛是固定同一 reference/static/W/
粒子状态/外力的 **固体独立新旧源消融**；新源只显式扩大诊断 F 输入的存储
精度，不把旧 checkpoint 冒充可恢复的新源接受态。

实际 restore audit passed、mismatches 为空；仍停留 K25、0.0125 s，原 checkpoint
head/两代数据/25 journal 未变。仅末帧 ZIP 时间戳再次重编码，物理数组 bytes
保留；前次单步诊断 failure 归档到
`attempts/674030e91fde46ffbb89c57d945e27da/failure.json`，原 fresh failure 归档
不变。当前 failure 类型为刻意中止的 `DiagnosticTrialProbeComplete`，不能将
诊断 exit0 当作第26步或 fresh50 数值成功。

#### Fluent 首步压力力缺口：前后表面拆分

独立读取 canonical fresh50 的原始 step_0001 case/data 并积分，而非借助图像
或错时刻采样：Fluent 前表面压力力 `4.544001881549617 N/m`，后表面
`0.28535688577162344 N/m`，压力合计 `4.82935876732124 N/m`。ours 使用当前
物理 marker traction*A、反转流向并除以 span=0.003 m 后，前表面
`2.417897277561854 N/m`，后表面 `0.237421222382516 N/m`，合计
`2.655318499944369 N/m`。压力总缺口为 **45.017162%**，其中 **97.7951%**
来自前表面。所有128物理 marker 的面积/单位/力分解与行动反作用核对正常。

Fluent 前表面大部分高度的压力约467–474 Pa；按2 mm等高段积分，前表面平均
压力依次为467.862、467.608、471.808、470.152、394.571 Pa，后表面约
-27.48到-34.26 Pa。ours 只能据现有分区力换算前表面平均241.790 Pa、后表面
-23.742 Pa，不能伪造对应的逐高度压力曲线。

现有 ours K1 导出中的二维 pressure 是 pre_solid_projection 的 span mean，
marker x/n 与 history pressure-pair map 则是 post-solid/accepted-anchor-refresh；
二者不在同一几何时刻。用该 pair map 索引 span-mean mask 有64个前表面位置落到
非流体显示 fallback，因此不能作为合格的同掩膜压力比较。后续需保存同一次
traction 采样的实际 x/n/A、probe cells/weights、side/reference p、t/F 与 geometry
revision，再判断边界模型、流场还是采样误差。没有因此修改压力门槛、符号、
参考值或 Fluent 比较器，也没有宣布 Fluent 验收通过。

#### Neo MPM f64 deformation repair: focused evidence and non-claims

已落地的精度合同只扩大持久形变状态：`F`/`saved_F` 为 f64；`C`、`v`、grid
velocity 与既有 P2G/APIC 布局仍为 f32。step 内的单位阵、`dt*C` 递推、本构关键
局部量、stress map 与 P 显式为 f64；合法原始 `F` 不再经过 SVD 重构，只有真实
奇异值越界或负 Jacobian 才投影。checkpoint 对 solid `F` 严格要求 f64，旧 f32
输入在任何 owner 写入前拒绝；保存、恢复和宏 rollback 保留合法 low bits。没有
修改 pressure、IQN、target、形变限幅或其它数值容差。

先前实际 CUDA RED `solid_precision_red_cuda.xml` 为 3/3 failed（144.66 s）：
1284 子步的正/负恒定应变率 f32 `F` 累积为零，且线性 `I+skew` 产生虚假速度。
修复后的 strict-CUDA 选定集合为 12 项，583.24 s 中 11 passed、1 failed；唯一失败是
nominal lower-bound 的 **3 ULP 测试断言**。把该测试专用断言扩至 8 ULP 后，
`solid_precision_nominal_bound_cuda.xml` 的 1 项在 52.24 s 通过；这不是生产形变限幅或任何
求解器容差放宽，也不冒充一次调用中12项全通过。host checkpoint gate 为 59 个
测试加 13 个通过的 subtest（9.50 s），另有 2 个 nested x64 restore test（67.22 s）
及 9 个 selected Neo case（265.30 s，另37项未选中）。

同一固定 reference/static/W/粒子状态/外力的 solid-only 三重复消融没有接受
FSI 步：旧最大 candidate RMS repeatability 为
`8.241885584584458e-6 m/s`，新为 `8.335008754803797e-9 m/s`，改善约
`988.827x`；旧到新 mean candidate RMS shift 为
`0.0002059366128032301 m/s`，为旧 mean-vector RMS 的 9.11046%，并非只去噪而平均
响应不变。新旧各3次 solid-only trial 均实际执行 1283 substeps、完整宏步
`0.0005 s`，retry=0、clamp=0、OOB=0。新旧源只变化于 Neo、accepted checkpoint
和 squid checkpoint 三个生产文件；诊断输入只将旧 f32 F 精确提升为 f64，其余
状态、外力、质量、reference 和 W 均逐 bytes 相同。审计逐一验证6个NPZ、所有
数组SHA、GPU W 的独立 host 重算及上述指标，并获独立只读审查。
fixture NPZ SHA 为 `3659d67f910c017750e1e1ae6a2316a4813cd292fba06c125c2a567d30afe02f`；
candidate result SHA 为 `086cb2435c7015d9ad298bf9c933e591367a73ffcf0ceb6d55b5d1a983e89a62`；
comparison SHA 为 `de404d882c6327cec172782c9e399157c7ff2dc3c18b2a7484ef7f594ce5426a`。
这些不是一次 accepted FSI step、fresh50 或 Fluent 结果。

新修复尚未生成完整 source freeze，也没有重新做 fresh preflow、50 accepted
steps 或 Fluent 验收；r39 仍是 25/50 accepted 后在第26步失败的历史记录，不能被
本节覆盖。Fluent 高度口径仍为 `0.02 m` 且匹配；K0 已有 pressure 差异，并且
同一已核验导出中的 `pressure`/`fsi_pressure` 原始 bytes 相同。这既不证明几何
错配，也不定位压力/载荷根因；根因仍未定。

#### FSI convergence failure artifact: real CLI RED/GREEN

为避免 r39 失败后丢失 16 轮 IQN 轨迹，CLI 现在只对精确
`FsiCouplingConvergenceError` 写独立 `fsi_coupling_diagnostics`：完整 context、
dataclass report 及原始 guess/candidate/residual 向量。旧
`pressure_solve_diagnostics` 的语义保持不变；failure export 保留已经接受的
step/time，非 FSI RuntimeError 不会被标成 FSI 或继承旧 FSI 字段，新的 `running`
进度会清除陈旧字段。诊断序列化或写入失败只被记录为 reporting error，不替代
primary error。

这是实际 WSL CLI 集成证据，而非早期 staging stub：test-only 旧 CLI RED
`fsi_failure_export_actual_red.xml` 为 4 failed / 2 passed，5.50 s（缺完整字段、
陈旧字段清理、helper 与恢复清理）。应用候选 CLI 后，`fsi_failure_export_actual_green.xml` 为 105
passed，27.79 s，覆盖 6 个新 failure 测试及 run-attempt、checkpoint CLI、
output contract 和 runner；CLI SHA-256 为
`f4adbfe692187c2e64876c51c2a3c54ad9cc27a95aca3f326da1f015d605e0b7`。这只补齐
失败诊断保留，不能作为 accepted FSI step、fresh preflow、fine50 或 Fluent 的成功
证据。

随后用 stdlib line events 与可执行行交集测量新 CLI diff：33/33 新增可执行行被
9个相关 host 测试覆盖（7.52 s）；报告 `fsi_failure_export_added_line_coverage.json`。
这是新增行覆盖率，不是 branch、GPU kernel 或全库80%覆盖率。新鲜只读审查核对了
实际源码、RED/GREEN XML、primary exception 的 bare raise、原子写盘和接受进度边界。

#### 外部物理面 Q 合同审计：已证实差异，修复尚待 RED/GREEN

同一旧预流 NPZ 中，x 外部 normal masks 均为0；projection 两端净/绝对 Q 均为0，
但 MUSCL primal-Q 在 xmax 读取 `velocity[nx-1].x`（最后内部 MAC 面）。按生产
公式重建的 xmax 净 Q 为 `4.50473272e-9 m3/s`、绝对 Q 为 `2.06603459e-5 m3/s`；
入口 Q 为 `5.999999918e-4 m3/s`，比例分别约7.51 ppm和3.44339%。这个 ledger 由
动量和 SST 共享，不能把 cell-centered vx 或账面 Q 冒充真实外部边界流量。

重建的流向动量表面项约 `-2.33697e-5 N` 也不是实际净动量损失：更新含
`div(Q*u)-u*div(Q)`，外面零斜率的相同状态在精确算术下抵消。因此目前只证实
边界账本/CFL语义不一致，未证实它解释45%低载荷。既有 `['ymax']` 也不构成严格
二维约束；不能通过改变几何、对称标签或比较门槛冒充修复。

下一步为三轴双侧正反向、exact normal/tangential-only、显式出口/入口、graded
面积和逐格散度一致性的真实测试。共享外面规则必须每次收到与投影相同的边界
模式，不能从SST的默认出口推断流体边界，不能增加未保存的活动模式缓存；内部
HIBM wall-relative Q 保持独立语义。完成实现、回归与审查之前，不启动新长程仿真。

#### 外部物理面 Q：旧生产源码的扩展 RED 已完成

`physical_face_flux_extended_red.xml` 对当前 WSL 生产实现实际执行，控制台结果为
13 failed、5 passed、6 subtests passed，163.21 s；XML 没有 collection error。
其中5个失败直接观察到错误数值：默认 xmin 法向为11而非0、tangential-only
法向为7/17而非0、graded-grid散度37/64格不一致（最大差4.5），以及空间局部
normal mask 仍让13/16个未声明面读入13的内部速度。另7个失败项是新的显式
拓扑参数尚不存在（含2个公开入口冲突 subtest），runner 的1项失败为未透传
`pressure_outlet_zmin`。不能把 API 缺失或13个失败项称为13个独立数值根因。

测试时 solver SHA 为
`b25e7ac7ecd854cac8055219c689649fc6c5eaa660be33d01437f2e85f4fed29`，runner SHA 为
`f08a386f3c3cd16061c2601849b6498f103ffaec7826b59cf3e33d20392471ee`；XML SHA 为
`882ce2a9944edec4f82752a0867ac42a2d2e6756bfc29b1f0c4ed7b5b14c002f`。
独立补充的 `test_physical_face_flux_partial_topology.py` SHA 为
`67c8a607374fce15921686a078988f832657d2b0b2d6f10a8dc07e7bd3979fc3`，覆盖空间
局部掩码、显式零法向、障碍相邻外面以及公开 predict/SST 拒绝时18个物理数组
和SST壁面配置不变。此时边界生产代码尚未修改，尚无该重构 GREEN 结果。

F64修复另完成 `solid_precision_constitutive_x64.xml`：7 passed、39 deselected，
385.36 s，覆盖线弹性纯剪切、平面应力、Saint-Venant–Kirchhoff 非小应变、外力
动量与FLIP基线。XML SHA 为
`30f3baf55939ff4b62185c6adc3f4ef31f2bb8a172519c83be1e0cf7d518c494`。这是CPU
x64的本构回归，不是CUDA全套、更不是耦合50步或长程验收。

#### 共享外边界规则已应用，新增合同 GREEN；旧数值回归进行中

共享 `_physical_exterior_normal_velocity` 已替换现有3个 primal-Q kernel 和投影
散度中的外面分支，没有保留并行旧实现；内部 HIBM 相对壁面 Q 与投影绝对壁速
仍分开。predict、SST及每个SSP源/重试显式接收与投影相同的拓扑，SST在壁距
准备之前拒绝冲突，动量通量入口也在同步源状态之前拒绝冲突。没有新增未保存
的活动模式缓存，没有修改步长、压力/target/IQN门槛或Fluent比较门槛。

独立审查额外发现通用 sharp HIBM 入口未向predict透传原有投影参数；新增
`test_sharp_physical_face_flux_routing.py` 在真实旧入口上四种bool组合均RED，
4 failed、0 errors、4.94 s。失败是实际predict调用只收到dt和advection_scheme，
不是测试支架异常。随后仅向该predict补上与project相同的两个bool参数；必要的
现有spy签名和源代码调用计数同步，原有数值断言保持。只读审查最后给出候选
可进入应用/GREEN的结论，不等于数值验收通过。

应用前8个源码/测试文件均核对实际dirty基线；应用后3个生产SHA为：

- solver：`2d385ea7bdcf423e67a1988dd3e754b9ffe93358fdf47793d711e0209c15f69e`
- official runner：`a644c3e39ef6105d79554893d3a19966f8eba4b4c70340536b02f2f178a5f463`
- generic HIBM core：`419e4eee2646fbae0b2f400a2a18a894d74eb829757529d00283e06926d909a1`

新增generic RED XML SHA为
`e4d5502a41371503e930d97ce833b322af640503f3fac6b703538cb8ca88c826`；此前在旧边界
生产源码上的3项throughflow基线通过（429.24 s，58 deselected），包含动量
二阶精度、SST幅值保持和低CFL入口面通量，XML为
`physical_face_flux_legacy_throughflow_baseline.xml`，SHA
`07e6b3cf06cad1061965ad6199cd7411f3b2a27cf3fa7c94bfab04eb4146db58`。新源码必须
再次通过这些门槛；不能用旧基线通过冒充新实现结果。

r40暂存启动器已准备好独占dryrun/preflow/formal50，实际完整114源文件、启动器、
旧常量来源、inventory函数、两验收gate及diagnostic依赖均受锁约束。dryrun包含
与formal50相同的field/IQN/checkpoint选项，并用生产52-key配置身份检查。原r39
启动器bytes保留，两个gate仅增加显式policy注入，原件已备份。主任务独立重跑
45个host测试于20.21 s通过，XML
`material_precision_validation_campaign_root_host_green.xml`，SHA
`dc168bc7ad3d4fdb7049dc580075a0694f8b96cd000297ef83600c805916455a`。尚未创建
真实r40锁、预流或FSI输出；先完成当前新源码GREEN和回归，再重新冻结。

新增四个测试文件在实际新 WSL 源码上完成 `physical_face_flux_contract_green.xml`：
20 passed、8 subtests passed，168.66 s，SHA
`9e3e15788bc2258a26494ddd6264a8d71254010d3d3c528b4d8a7c3b146f8669`。相关既有
host集成为34 passed、4 subtests passed、1 skipped，7.99 s，XML
`physical_face_flux_host_integration_green.xml`，SHA
`2037493406cad5e0c7d2a4a0a636fcf92f2191ddad300e24c80baf0c5f269a4d`。跳过的是
旧 `official_fluent_reference_input` 目录缺失的导入测试；不是canonical fresh50
原始资料缺失，也不是Fluent比较通过。独立只读审查再次核对实际3份生产SHA、
4份测试及两份XML，对本次重构给出ship：官方/generic入口、SSP各源和重试路由
一致，Euler/RK2动量仍走原内核，未发现必须修正项。旧MUSCL、SST和移动壁面
数值回归仍须单独完成；本结论不认证fresh50或长期运行。

#### 原始 K0 出口动量审计：已独立复现，仅作诊断

暂存 `audit_preflow_outlet_momentum.py`（SHA
`e950e0272303be0595dd8c9613b8409698b5e848e7702baf2f274eecf014c443`）只读canonical
Fluent `fsi_setup.cas.h5/dat.h5` 与旧r39完整preflow NPZ，不恢复checkpoint、不运行
求解器。Fluent面按全局一基inclusive IDs与实际fluid c0/c1邻接重建，压力力用
外法向边长矢量；`SV_WALL_SHEAR` 已是积分力，不再乘边长。ours用实际zmin
backward-MAC行和zmax精确外面法向，按存储dx/dy、rho与span换算。旧快照源码
身份保留为历史事实，不伪装成当前源码。当前共享外面helper保留这两个显式port规则。

首次sandbox收集失败不算RED；重新在权威目录运行时为预期缺新模块的collection
RED。实现后18个纯host测试通过，0.49 s（全局面IDs/重叠、面积和厚度、正负流向、
无写入、非法输入），`preflow_outlet_momentum_host_green.xml` SHA
`5b2365f724321816177545b3770f4b92d694fdad78938000486f42f547ab7c57`。
实际输入独立运行输出 `preflow_raw_outlet_momentum_r39.json`，SHA
`dd710ceb44a8777a6ba571f1385547e151b354f4f6f9ac7d0b8aad8c8351b6a6`：

- Fluent出口流向动量 `6.641229869957881 N/m`，全域净流出 `4.2412298699578805 N/m`。
- ours出口 `4.216296317359675 N/m`、入口 `-2.3999999672174446 N/m`，净流出
  `1.8162963501422307 N/m`；出口比值 `0.6348667942412924`。
- Fluent出口U范围约 `[-5.03748,29.22355] m/s`，ours约 `[-0.91681,21.77700] m/s`。

这证实旧K0尾流/射流分布已经不同，不能仅调整力单位解释差异；不证明某一个
代码缺陷就是原因。Fluent已取得压力和壁面剪切减动量流出的余项约
`0.05868575696 N/m`，没有完整端口法向应力和离散插值项，不能叫作精确离散守恒
误差或通过标准。该历史诊断不能替代新r40完整50步、严格压力语义与Fluent验收。

#### 共享Q后的完整状态合同与generic账本生命周期（2026-08-29续）

上一批旧MUSCL/HIBM回归实际结束为25 passed、9 subtests passed、1 failed，
1896.35 s；`physical_face_flux_muscl_hibm_regression.xml` SHA为
`fa4a022c00ebfc9c6cf1b4bcda281e8e97d5d53a3ec4476f321957d31761757b`。
失败发生在generic predictor之前：band sweep返回0仍会失效当前generation，
调用方却先break而跳过重建，后续reachability reader正确拒绝未封存账本。
源码对照证实这是原有排序缺口，不是新增predict参数造成的新物理错误。

修复复用现有assembly/seal helper：4个band循环和post-solid首次band均在退出或
读取前重建；两处air conversion即使返回0也重建；pre/post overflow与tiny正转换
在第一次flood前封存。审查补上pre-overflow helper的`nonlocal velocity_report`，
保证局部报告回写；没有改失效器、sealed guard、投影次数或物理时间。实际core SHA为
`08421027436f96e91c8edbd7c3d93383645cdb47f07e8da97909ce8cd95db1d3`。

- 原真实CUDA节点 `test_sharp_fluid_solve_runs_predictor_before_projection`：
  1 passed，359.21 s；原两次0.00025 s predictor、5次projection和调用顺序断言保留。
  XML `generic_ledger_lifecycle_cuda_green.xml` SHA：
  `f12e8541783fc01a8a4502c17685e9b2993653ebbc1ed74afb6cb43ce1dfa88e`。
- 新host测试执行生产AST statement blocks与nested helpers，使用generation-aware
  double检查首次reader和外层报告；不是替代真实canonical/CUDA检查。旧core
  `419e4eee...` 上18个控制流子项失败、0 errors；新core上4 passed、22 subtests
  passed，1.51 s。XML `generic_ledger_lifecycle_host_green.xml` SHA：
  `c52d57c5b19d5c71348310f775030d08f055da65464223e1467772fc05ad364a`。
  按真实编译文件名的`sys.settrace`核验11/11新增可执行语句；不是全仓或分支覆盖率。
  标准`trace`对动态exec命名空间计数为0的尝试未作为覆盖证据。

另外两份真实CUDA RED证明只统一Q还不够：在Q不变时，仅改变最低侧raw normal，
动量通量仍变化；SST center/normal strain和隐式normal行仍把未声明外面当外推。
`physical_mac_state_consistency_red.xml` 为12个数值失败子项、13个通过子项，
62.16 s，SHA `253b4d3170e173e6dbb22e294be33cdf87d94c701be4ca14553fb4a701586977`；
`physical_mac_normal_operators_red.xml` 为18个数值失败子项，86.30 s，SHA
`29ff61b14254e7f0d7b7d1070d9fe49cf046903ef4bf2c7d4d8fd733cb30e0a0`。
两者无collection/调用参数错误；不能把子项数当独立根因数。

共享helper现返回`[prescribed, absolute normal]`：默认closed/exact-zero固定，
显式zmin出口或zmax True无exact的零值仍为自由外推。最低面同步、最高面ghost、
SST三个source路径/应变/transpose及normal Helmholtz矩阵统一消费该合同。
没有把最高侧最后内部MAC行清零，没有把normal closure变成切向无滑移或额外
wall-correlation摩擦。原no-slip/目标/压力/IQN/Fluent门槛不变，未加活动模式缓存。
应用前完整diff通过独立静态审查；实际solver SHA为
`569c20c78ab855ddf5c472af46d7c336307e943d7bb6d4b99ef1bed93d1137c1`。

首轮新数值GREEN：`physical_mac_state_operators_green.xml`，7 passed、51 subtests
passed，364.72 s，SHA
`6423faa873833077eeb1897c0d449357a3dbe9a5488638a77ad2c7dcb9782a61`。
包括对称closed/exact、raw残留不影响Q/动量、非零nu的normal helper、空间局部
normal/tangential mask，以及6种拓扑下current/previous来源。该次stage测试SHA为
`07e1f70ba27674806d863d90640369e67c3e3f1db177585e4213692417f866f3`；其dt=0的
transpose只核center/cached gradient，nu=0的solve只核source和最终状态。审查要求
补真实transpose divergence oracle、已组装row-kind和非零nu最大面diagonal。
初版7+51门禁本身不包含这些增强证明；增强结果另见下文，不能追溯性地并入初版GREEN。

旧私有kernel调用仅作参数迁移，原断言/容差保留。SST阶段AST与生命周期合跑
14 passed、40 subtests passed，7.87 s；`physical_mac_stage_lifecycle_host_green.xml`
SHA `a1af6665999927bb68d9dfa5e6b6f1ae17e88e10b78bcccc6821fdceb43ddd76`。
该旧MUSCL/HIBM/Helmholtz/Q批次已完成：46 passed、31 subtests passed、5 failed，
1983.16 s。`physical_mac_muscl_hibm_helmholtz_regression.xml` SHA为
`cabc4db95b6ede0b9091eaa629bd0d3ba821599968b285990021d2f97675470f`；XML的82项
包含31个subtests，不能当成82个主测试。五个失败均来自旧MUSCL测试文件，其他
HIBM/Helmholtz、物理Q及路由检查没有失败：

- pulse原raw sum预期48，实际34.73882293701172；极值断言通过。48包含应封闭的
  xmin行上12个非零值。合法初态应先使该行为0，但也不能简单把恒定预期改成36：
  自由行更新有`u*div(Q)`连续性修正，还需计入固定/自由双控制体界面的动量通量。
- 正弦剪切二阶测试测得阶数0.2937035448651879，原门槛1.8；其x法向初值非零但
  没有登记解析外边界，与默认closed物理合同矛盾。
- 常量横向速度2的断言有48/96元素不符；graded dual-Q局部常量5得到4.6875。
  两个fixture同样未声明承载非零x法向的物理面。
- 冻结SST系数的mock仍只接收2个位置参数，真实调用已传4个；这是测试签名错误，
  不是SST数值发散。

第一版MUSCL fixture迁移已应用，测试文件SHA为
`63c723876adbf4f9a5a8ac9f7d355ac44e7c436ad0cb5365ddee9af3c147eb7f`。两个常量
fixture明确登记x两端normal exact；mock机械更新签名。CUDA复跑实际为1 passed、
2 subtests passed、1 failed，210.75 s；XML
`physical_mac_fixture_migration_regression.xml` SHA为
`b6dfed5d6ef36e9fd9f9defc5764a7a420ad0f5a831aa0801f391c1568aa4a3c`。唯一失败不是
生产求解器数值错误，而是测试支架错误限定一个宏步只能有1个自适应子步：zmin的
半控制体使名义CFL至少为0.5，高于初始目标0.45，生产代码合法自动分为2片。

替代observer按真实accepted片序列检查每一对stage，要求0 retry并用现有ULP规则
核验完整requested dt；正弦fixture仍调用真实public predict和SSP核，只在每个实际
stage0前把external解析边界移到该片`t+h`。不写自由MAC行、不mock守卫、不改变
生产CFL/时间步或构成新的生产时变边界API。原[1,1,:,0]列、z窗口、1.8阶/0.95幅值
及pulse动量容差均保留。新多片候选SHA `f5864edf...` 已通过独立静态复核并精确应用
到权威WSL测试文件，哈希、Python语法和diff空白检查通过。完整MUSCL CUDA数值复跑
随后为13 passed、6 subtests passed，1235.08 s；JUnit含19项、0
failures/errors/skips，XML SHA为
`0e44f629303888ec56399d173463cd407bf90d4c56c303345e2a71f5e68d9e57`。不能再把
single-slice写成预期。

pulse保留原极值限制，以全部`i>=1`自由ux双控制体核验
`delta M = rho*h/2*(R0+R1)`，每个accepted片和四个宏步合计分别检查。R使用真实两阶段
通量的六个外表面直接求和及每个自由控制体的`u*div(Q)`；host以f64独立归约，
不能漏掉x面1的固定/自由交界。独立几何V=1/256逐值核对生产ledger；原容差按rho*V
等价换为`2.8125e-7 kg m/s`，没有放宽。这只检查真实flux/SSP的离散全自由区预算，
不单独证明flux构造物理正确，前面的face/CFL与精度回归仍保留。

增强stage测试已应用（SHA
`7f919c24f65c451024a63f340d8dfc8425de376148497aea047299425d373703`），独立审查核对
其transpose divergence及真实row-kind/非零nu最大面diagonal预期。它与两个旧SST
制造解夹具的CUDA批次实际为2 failed、1 passed、6 subtests passed，353.39 s；XML
`sst_boundary_fixture_red_and_stage_strengthening.xml` SHA为
`84fbe9ea5f9e4f53edabac6b4aa566f106eee1b3d472d610455fd90e79296659`。增强真实stage
本身GREEN；两个失败来自旧制造解未声明所需物理边界，因此不能把这两个RED说成
增强矩阵失败，也不能把增强结果追溯性地并入初版7+51门禁。

两个SST夹具声明及moving-obstacle mock的机械签名迁移已应用，测试SHA为
`d12a6b42c3dfa06e20b5b49d56798e65a89ccfc48bad1bd37e55d79ae8b7b8b7`；correlation
transpose probe的机械签名迁移SHA为
`f8ac1ab362304275121fdb8161da24acbfcb8a44e2613f02a3d204e4aff916a6`。修正后的SST、
viscosity、transpose及真实generic-HIBM批次完成为32 passed、26 subtests passed、
30 deselected，3334.81 s；JUnit含58项、0 failures/errors/skips，XML SHA为
`4f013e41fa550d1c98dbe59739e1f721af19ceeb18b75cd52661ed75fc9f8620`。尚未冻结真实
r40，也未新跑fresh preflow、formal50、Fluent比较或长程仿真；必须先完成当前重构
和回归，再按此顺序验证长时间稳健运行。

最终独立只读审查完整核对fluid solver的582行差异、sharp core的17行差异及变更调用面
和测试，未发现生产阻断项，给出仅限创建r40 source freeze的`ship`。审查确认physical-
face状态、current/previous路由、implicit/transpose配对、retry回滚及ledger重封存一致；
这不是fresh preflow、accepted FSI step、fine50、Fluent或长程结果。

## 2026-08-30 r47：K200完成、速度云图与dual-root精确50步比较

r47在新源码匹配的preflow及短fresh/resume门禁后，以同一进程从K50继续到K200。
resume attempt最终exit code为0，接受`200/200`步，物理时间为`0.1 s`；新增150步
耗时`8933.812750 s`。每个接受步均为3次IQN迭代，并记录2次同一物理时刻的被拒
代数trial；被拒trial不推进物理时间。checkpoint journal连续到K200，没有缺步或
断链。

marker operator、pressure-nullspace operator和solver scratch资源分别恒定为
`18,309,056`、`41,902,024`和`23,592,968` bytes，没有随步数增长。全段最坏
pressure exact relative residual为`9.919589959059803e-7`，仍低于未修改的`1e-6`
门槛，但裕量很小，不能写成宽裕通过，也不能据此放宽压力门槛。K200是本轮实测的
长程边界，不是K5000可以完成的证明或保证。

速度云图保存在被Git忽略的本地目录
`validation_runs/solver_soaks/ansys_vf__k200_velocity_viz__material_fine__20260830__r47`。
K50/K100/K150/K200共用`0..45 m/s`色标，另有K200单帧图。可复核哈希为：

- `figures/velocity_magnitude_step_0200.png`：
  `a67836c10238dcc310c6247278dbe0f6b61a5f0f5cfe84db43ed60feb2b50582`；
- `figures/velocity_magnitude_steps_0050_0100_0150_0200.png`：
  `a4758b320d27afd31c2d1558e145edfabace1066333d92b2d033bb6ac73f9c31`；
- `velocity_render_manifest.json`：
  `e97ebf6cb4c46f98611b52a05fbd3f5b9efd9a9bf5ddcb5a6f01a24dcff95de0`。

锁定比较采用dual-root：完成的K50 resume attempt提供聚合history和terminal
control-plane记录；canonical artifact root的checkpoint head可以已经到K200，
比较器只读取并严格验证其K1--K50前缀。门禁核对attempt-v2 provenance、源码/
配置/几何身份、canonical journal和精确50份field/history，不能用canonical中
保留的旧K50 summary冒充当前K200 head，也不能静默混合两个root。最终强化后的
主调用路径在真实r47上通过50/50步，每步逐项绑定journal、per-step JSON和聚合CSV的
460个公共history字段；只允许两个已声明的journal-only残差别名、CSV边界的严格布尔/
空值表示以及9个受零count/flux约束的空集合NaN坐标。任一非核心字段变化同样fail-
closed，且不会靠数值tolerance掩盖差异。

比较结果为diagnostic complete，不是parity或Fluent真值证明。主要归一化差异为：

- speed：`13.33%`；
- gauge pressure：`19.07%`；
- tip displacement waveform：`20.92%`；
- maximum solid displacement waveform：`23.23%`；
- streamwise force waveform：`37.52%`；
- transverse force waveform：`57.25%`。

其中v速度为`4.27%`，out-of-plane force leakage为`0.0`，这两个单项通过5%检查；
主要指标既未通过现有5% diagnostic gate，也未通过另行解释的10%高一致性目标。
Fluent只是锁定参考，不一定是准确真值。10%只有在几何、边界、时间窗、observable
定义和网格/时间收敛对齐后才是高一致性目标；3D/extruded与2D差异可能贡献偏差，
但不能自动豁免当前较大的压力、位移和力差异。跨求解器比较口径不修改内部pressure、
closure、conservation、no-slip或accepted physical-time门槛。

全字段绑定和Windows create-only发布重试接入主路径后，锁定比较重新生成到
`validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/material_reference_r47_fresh50_20260830_r3`。
r3的dual-root合同对K200 canonical head通过，5% diagnostic gate仍按预期失败；
主要指标相对r2的差值逐项为0，WSL `sha256sum -c`核验全部59个文件通过。最终
有界回归为dual-root/Fluent `169 passed, 3 skipped, 4 subtests passed`，checkpoint/
runtime/resume/lifecycle `530 passed, 4 skipped, 24 subtests passed`。新的只读
Sol/Ultra审查未发现P0/P1/P2并给出`ship`；该结论不扩大K200、Fluent或checkpoint
未直接密码学绑定step NPZ的证据边界。
