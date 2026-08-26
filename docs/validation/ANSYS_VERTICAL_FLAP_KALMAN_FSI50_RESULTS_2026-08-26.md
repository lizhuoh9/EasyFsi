# ANSYS 竖直薄板 Kalman modified-physics FSI50 结果（2026-08-26）

## 结论边界

本轮包含 no-Kalman、interface、fluid、solid 和 global 五组。四个 Kalman 模式
均允许滤波后状态写回，属于 **modified-physics 实验**，不能解释为相同物理
模型下的加速比较，也不能据此宣称 Fluent parity。

五组以及排除首次 CUDA/JIT/cache 成本的 warm no-Kalman 基线均完成 50/50 个
FSI 宏步。它们的 97-file source map 与当前工作树逐文件匹配，strict feedback、
物理时间、有限性和健康检查通过，没有 pressure/PCG breakdown、MPM retry 或
rejected macro step。production 可用 `kalman_mode=off` 回到无写回基线。

## 迭代与子步统计

| 模式 | CG 调用 | CG 总迭代 | 平均/次 | 范围 | HIBM cycles | Helmholtz | SST | Momentum | MPM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline/off | 50 | 11,952 | 239.04 | 208--256 | 50 | 26,444 | 13,579 | 15,712 | 64,195 |
| Interface | 50 | 11,952 | 239.04 | 208--256 | 50 | 26,447 | 13,624 | 15,720 | 64,195 |
| Fluid | 50 | 11,936 | 238.72 | 208--240 | 50 | 26,541 | 13,538 | 15,709 | 64,195 |
| Solid | 50 | 11,952 | 239.04 | 208--256 | 50 | 26,473 | 13,579 | 15,711 | 64,195 |
| Global | 50 | 11,936 | 238.72 | 208--240 | 50 | 26,516 | 13,538 | 15,709 | 64,195 |

HIBM 每个宏步 1 个 cycle（预算 2）。MPM 每个宏步 1,280--1,286 个自适应
子步。五组均无 restart、retry 或 breakdown。

## Strict feedback 与物理时间

六个 source-matched 50 步运行的前两步 feedback consumed 均为
`[False, True]`，count=1；第二步 marker=128，mode 为
`hibm_sharp_reconstructed_rows`，topology refreshed=true，valid/invalid
marker=128/0，CG breakdown count=0，相关残差有限。

50 步最大时间闭合误差：macro fluid
`6.505213034913027e-19 s`、SST reconstruction
`3.686287386450715e-18 s`、momentum advection
`6.505213034913027e-19 s`；reported fluid/solid remaining 均为 0。

## Warm no-Kalman 基线耗时

目录：
`validation_runs/ansys_vertical_flap_fsi/our_solver_vs_native_fluent_fine_2026-07-10/runs/ansys_vf__kalman_mp__off_warm__fsi50__20260826__r02`

第 2--50 步稳态样本外推到 50 步，以排除首次 CUDA/JIT/cache 成本：

| 指标 | 50 步外推耗时 |
|---|---:|
| End-to-end | 836.733 s |
| Flow | 720.241 s |
| Solid | 44.759 s |
| Flow + solid | 765.000 s |
| HIBM（flow 子集） | 75.294 s |

这些数值不是包含首次初始化成本的原始总计，也不是 pytest 套件耗时。

## 云图数值范围

五模式统一色标：速度最大值 39.435536 m/s；压力范围
-49.604482--289.957325 Pa；固体位移最大值 82.655504 micrometre。
warm baseline 第 50 步速度最大值 39.434994 m/s，压力范围
-49.604193--289.906430 Pa。渲染 PNG 为本地分析产物，不纳入 Git 历史。

## 提交前聚焦门槛

只覆盖本次自适应固体子步、accepted physical-time 记账和 Kalman 写回合约，
不代表仓库全量回归：

```text
72 passed, 3 subtests passed in 6.29s
```

测试文件：

- `tests/benchmarks/test_adaptive_solid_substeps_contract.py`
- `tests/coupling/test_active_kalman_writeback.py`
- `tests/benchmarks/test_modified_physics_kalman_contract.py`
- `tests/solvers/test_time_stepping.py`

## 尚未满足的 main 发布门槛

当前 fixed1600/adaptive 正式 A/B 只有与当前 source map 不匹配的 FSI1；尚无
当前源码匹配的 FSI2、FSI8、FSI50 和 comparison report。因此结果支持
feature-branch WIP 备份和 Kalman 实验复核，不支持宣称 adaptive A/B 已完成，
也不支持在该门槛补齐前合并 main。

