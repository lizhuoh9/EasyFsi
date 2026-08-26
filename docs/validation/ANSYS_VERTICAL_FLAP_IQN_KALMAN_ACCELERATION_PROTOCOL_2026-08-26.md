# ANSYS 竖直薄板 IQN--Kalman 初值加速验证协议

日期：2026-08-26
状态：r06 Q0--Q3 source-matched strict-CUDA FSI50 快速学术实验已完成；结论仅适用于 `dt_s=1e-4 s`、终止物理时间 `5 ms`
适用分支：`codex/perf-r114-hotpath`

## 1. 研究问题与边界

Kalman 只预测下一物理步的第一个 marker-velocity guess；同一物理步内的
收敛由共享 `solve_fsi_runtime` 和 IQN-ILS 完成。滤波后状态不得写回 Fluid、
Solid、Marker 或 accepted interface state。旧
`ANSYS_VERTICAL_FLAP_KALMAN_FSI50_RESULTS_2026-08-26.md` 是独立的
modified-physics 写回实验，不能作为本实验的加速证据。

第一轮不实现 Aitken。IQN 第一次固定 Picard 松弛及病态回退必须保留。

| 组 | Coupling | 第一次 guess | 用途 |
|---|---|---|---|
| E0 | direct explicit | carry-forward | 历史背景，不作 Kalman 加速分母 |
| Q0 | IQN-ILS | carry-forward | 主 baseline |
| Q1 | IQN-ILS | linear extrapolation | 简单预测基线 |
| Q2 | IQN-ILS | Kalman | 研究方法 |
| Q3 | IQN-ILS | accepted-state oracle replay | 只测理论上限 |

核心比较为 Q0 对 Q2，同时必须报告 Q1；Q3 不得称为可部署算法。

## 2. 冻结配置与 source identity

Q0--Q3 必须使用相同源码映射、fresh strict-CUDA preflow snapshot、backend、
Taichi dtype/seed/cache identity、`dt_s`、网格、粒子、marker、材料、边界、
adaptive solid controller、Fluid controller、输出设置和收敛预算。每个 run
保存逐文件 source SHA256，并与 producer 和比较器现场重算值完全一致。禁止从
`step_fields/*.npz` 重启。

冻结参数：`max_iterations=16`、`relative_tolerance=1e-3`、
`absolute_tolerance_mps=0`、`history_limit=8`、
`initial_picard_relaxation=0.5`、`svd_relative_cutoff=1e-10`、
`kalman_writeback_mode=off`。除 `initial_guess_mode` 外，Q0--Q3 不得改变
其他参数。算法改变后必须更换 run identity 并重做 fresh source-matched preflow。

正式 runner 通过 `--initial-guess-mode` 选择 `carry_forward`、
`linear_extrapolation`、`kalman` 或 `oracle_replay`。Q2 还必须显式给出
`--initial-guess-kalman-q`、`--initial-guess-kalman-r` 和预锁定的 warmup；
Q3 必须给出 `--initial-guess-oracle-path`。Oracle producer 在进入 JIT/数值求解
前校验 completed 状态、Q0 配置、源码 SHA、逐帧 SHA、步数和 layout，并一次性加载
到内存；正式运行期间不再读 producer 文件。

## 3. 硬正确性门槛

每个 trial 都从相同 accepted `t_n` 恢复，并让 Fluid 和 Solid 各完整推进一个
`dt_s`。rejected trial 对 accepted physical time、feedback、filter、history、
checkpoint 和 accepted work ledger 的贡献为零；每个物理步只 commit 一次。

所有晋级运行还必须满足：

- actual backend 为 strict CUDA，所有状态和报告 finite；
- `fluid_accepted_time_s == solid_accepted_time_s == requested_macro_dt_s`，
  remaining unadvanced time 为零；
- 无 pressure/CG/PCG breakdown、OOB、deformation clamp 或未恢复 retry；
- Fluid、MPM、HIBM trial count 与 coupling iterations 一致；
- predictor 每物理步最多调用一次，只在 accepted step 后 update/commit；
- marker count/order/topology layout identity 不变，否则 fail closed；
- Q0、Q1、Q2 每步收敛且 history 长度等于请求步数。

FSI2 还必须满足 feedback consumed 为 `[False, True]`、count 为 `1`；第二步
marker count 大于零，mode 为 `hibm_sharp_reconstructed_rows`，observer topology
refreshed，valid marker 大于零、invalid marker 为零。

## 4. 预锁定的物理解一致性

Q1/Q2 相对 Q0 在每个 FSI8/FSI50 accepted 时刻必须同时满足：

- area-weighted marker velocity 和 marker position NRMSE `<=0.005`；
- tip/maximum displacement NRMSE `<=0.005`，近零绝对 floor `5e-8 m`；
- interface/pressure force NRMSE `<=0.01`，近零绝对 floor `2e-5 N`；
- pressure extrema、Fluid velocity peak、MPM max speed、solid kinetic/strain
  energy NRMSE `<=0.01`；
- no-slip 和 divergence 不恶化超过 Q0 的 `1.01` 倍并保持 finite；
- accepted/rejected/retry/OOB/clamp 分类与 Q0 相同。

正式论文结论还要用 `dt_s/2` matched run 估计时间离散误差；最终差异必须小于
`max(上述门槛, dt_s 对 dt_s/2 的离散误差包络)`。短门槛不等于 Fluent parity。

## 5. 晋级顺序与停止条件

1. Gate A：host-only IQN、transaction、predict-once/commit-once。
2. Gate B：真实 Taichi CPU fields，两次 trial 从同一 base 得到同一 candidate。
3. Gate C：source-matched strict-CUDA Q0 FSI1。
4. Gate D：Q0 FSI2 feedback gate。
5. Gate E：Q0 FSI8；随后 Q3 oracle，再决定是否运行 Q1/Q2。
6. Gate F：只有合格组运行 matched FSI50。

Q0 不收敛、时间/rollback 失败、平均 iterations 接近 1、Q3 相对 Q0 的 total
Fluid/MPM trials 降幅小于 `10%`、重放不确定或 Q2 不优于 Q1 时，停止在线
Kalman 加速主张。`10%` trial 降幅和 `5%` warm wall-time 降幅是研究价值目标，
未达到时必须报告未观察到有意义加速。

## 6. 迭代、工作量与计时

逐步报告 coupling iterations、first/final relative/absolute residual、完整 residual
和 update-mode history、IQN rank/condition/fallback、Fluid/MPM/HIBM trial count、
CG calls/iterations、MPM attempted/accepted substeps、snapshot/restore、IQN、
predictor、Fluid、Solid、HIBM 和 output wall time，以及完整时间/feedback/健康证据。

汇总给出 iterations 的逐步序列、total、mean、median、P95、maximum，不能只给
平均值。每组同时报告原始端到端耗时和排除首次 CUDA/JIT/cache 的 warm estimate：
FSI8 使用 steps 2--8，FSI50 使用 steps 2--50，再按请求步数线性外推。warm 值
必须明确标为外推，不能伪装成 raw total，也不能用 pytest 耗时代替仿真耗时。

实现中的 trial-work ledger 对 rejected 和 accepted trial 一视同仁，分别累计
Fluid solve、Solid macro solve、CG iterations、Fluid transport substeps、MPM
substeps 以及启用 profile 时的 Fluid/HIBM/Solid wall time；accepted feedback
step count 与 feedback-consuming trial count 是两个独立字段。正式
`our_solver_summary.json` 直接暴露上述累计值、初始猜测模式/控制器报告，以及
iterations 的 total/min/max/mean/median/P95。

## 7. 快速门槛记录（非正式 fine-grid 结论）

| Run | 结果 | Coupling work | 其他迭代/子步 | 物理时间 |
|---|---|---|---|---|
| r03 E0 direct FSI1 | 完成 | 1 次 explicit pass；Fluid solve 1；Solid solve 1 | pressure CG 272；MPM 1280；retry 0 | Fluid=Solid=`5e-4 s`，remaining=0 |
| r03 Q0 IQN FSI1（修复前） | freshness gate 失败 | 完成 IQN iteration 0；accepted step 0 | Solid solve 0；无法从失败产物恢复 CG work | accepted time 0 |
| r04 小网格 Q0 smoke（修复后） | 人工中止于首次 preflow/JIT | FSI iteration 0 | 未形成数值结果 | accepted time 0 |
| r05 小网格 Q0 FSI1 | 完成 1/1 | iterations=`[3]`，rejected trial=2，Fluid/Solid solve=`3/3` | residual=`[4.3154e-2, 2.1433e-2, 3.861e-6]`；MPM attempted substeps=444；fallback=0 | Fluid=Solid=`5e-4 s`，remaining=0 |
| r06 小网格 Q0 FSI2 | 完成 2/2 | iterations=`[3,3]`，rejected trial=4，Fluid/Solid solve=`6/6` | step residual final=`[5.365e-6, 5.525e-6]`；MPM attempted substeps=891；feedback=`[False,True]` | 每步 Fluid=Solid=`5e-4 s`，remaining=0 |

r03 的 direct 与 Q0 使用同一 source-matched snapshot；因此 direct 成功只排除了
snapshot 本身不可用，不能替代 Q0 Gate C。静态根因是 IQN trial restore 会推进
marker geometry revision，从而使 pressure-pair anchors 失效，而旧代码直到 commit
才刷新 anchors。当前单点修复在每次 IQN guess 应用后、traction sampling 前刷新
anchors，并且不再把 same-time trial guess 标记为 accepted prior-step feedback。

对应 host-only TDD 契约先 RED、后 GREEN。r04 因首次 CUDA kernel 编译超过快速
smoke 的等待预算而被人工停止；它既不是 PASS，也不是数值 FAIL。r05/r06 证明修复
后的 Q0 小网格路径、完整 physical time 和两步 feedback 顺序可行，但它们的原始
elapsed 包含 inline preflow/JIT，不能作为性能结果。此后加入了 Q1/Q2/Q3 控制器、
严格 Oracle preflight、predict-once/accepted-only commit 以及全 trial work ledger；
相关非 CUDA 聚焦测试已通过。随后完成了 source-matched r06 Q0--Q3
strict-CUDA FSI50 快速学术实验；四组均完成 50/50 accepted steps：

| 组 | 第一次 guess | Coupling trials | 平均 trials/步 | 排除首次 CUDA/JIT/cache 后的结论 |
|---|---|---:|---:|---|
| Q0 | carry-forward | 150 | 3.00 | baseline，`1.000x` |
| Q1 | linear extrapolation | 150 | 3.00 | 相对 Q0 无迭代减少，也未观察到加速 |
| Q2 | Kalman | 150 | 3.00 | 相对 Q0/Q1 无迭代减少，warm 耗时反而更长 |
| Q3 | accepted-state oracle replay | 52 | 1.04 | 相对 Q0 的 warm speedup 为 `2.306x`，仅代表理论上限 |

因此，Q3 证明“更准确的下一步首猜”在当前 IQN-ILS 路径中存在显著的理论
加速空间；但当前 Q2 Kalman 配置没有改善迭代次数，也没有带来实际 warm-time
收益。按照第 5 节预锁定的停止条件，本轮结果不支持宣称 Kalman 已实现加速。

上述结果是终止时间仅 `5 ms` 的快速学术 pilot，不是原计划终止时间 `25 ms`
的正式 50 步比较，不构成 Fluent parity，也没有完成 adaptive solid 与
fixed1600 的 source-matched A/B。因此不得据此声称完整物理模型验证、Fluent
一致性或 adaptive-solid 性能收益。
