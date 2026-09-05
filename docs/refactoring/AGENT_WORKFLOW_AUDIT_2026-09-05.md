# Agent 工作流审计与环境恢复 — 2026-09-05

本次解决的是反复中断的操作与指令问题：明确当前 worktree 和持久解释器，
消除旧技能中的模型及流程冲突，修正 PowerShell/WSL 执行方法，并把历史错误
转成下次可以执行和核查的规则。FSI1/2/3 的数值通过状态没有因此改变。

## 范围与依据

用户要求先根据 [GPT-6 Astra 官方指南](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)
更新 AGENTS/skills，扫描本项目聊天历史，并明确将 reviewer 改成 Astra / max。
指南建议迁移时复查指令冲突、任务持续执行方式和验证成本；这里的环境路径、
暂停边界及数值门槛来自当前用户要求、实际文件和运行证据。

历史扫描按 `cwd` 属于本项目筛选，快照时间覆盖 2026-06-13 至 2026-09-05：

- 索引中有 3,479 个项目任务，3,479 份对应日志全部存在。
- 内容筛选读取覆盖全部 48 个用户主任务，以及 557 个相关子任务，共 605 个。
- 其余子任务只做元数据清点；没有逐行语义审读全部约 25.24 GiB 历史。
- 35,333 个关键词候选包含继承上下文和重复输出，不能当作故障次数。
- 原始日志和脱敏筛选结果保留在本机，不复制到项目仓库。

审计过程也发生过读取过大的任务输出、猜错 helper/相对引用路径，以及重复使用
不合适解释器的诊断错误；这些是 agent 的操作错误，已经纳入下面的规则。

## 历史问题与防复发措施

| 已观察到的问题 | 本次落实的规则或修复 | 证据定位 |
|---|---|---|
| 旧默认 worktree 与当前 handoff 分支不同 | 以当前 handoff 明确选择的 WSL root 为准；先查 branch、HEAD、dirty paths | H3:241,279 |
| `/tmp` 下正式解释器消失 | 按冻结 requirements 重建到持久 venv；运行前重核身份，不用同名 CPU venv 替代 | H1:49411,52964；本次恢复记录 |
| CRLF shell 脚本和 Windows Python 在 WSL 执行失败 | 检查 shell/解释器；多行逻辑写成 UTF-8 无 BOM、LF 文件，再传直接参数 | H3:241,440 |
| PowerShell/Bash 嵌套引号、命令替换、管道被提前解释 | 禁止把复杂 Bash 插进多层字符串；数据用参数或 JSON 文件传递 | H1:32065,52498,52783 |
| WSL E_ACCESSDENIED 与 IPC 解码失败被混为一谈 | 区分权限、传输、解析、依赖和数值错误；有授权的 WSL 操作用 scoped escalation | H2:46,51；H4:25,52 |
| 进程搜索命中诊断命令自己，PID 文件也可能陈旧 | 匹配真实 executable 与父子关系，排除诊断进程；核查 OS 进程和终止产物 | H2:84–87；H5:6292；H6:5710–5722 |
| 长时间无 stdout 被理解成暂停；真实失败又报告过迟 | 始终轮询同一 session/PID；结合活动与写盘周期判断；观测到终止立即更新状态 | H1:42592–42790 |
| 自动 goal continuation 险些越过明确暂停 | 暂停优先；先停止自己的进程并交接，等待后来直接用户恢复指令 | H1:52977–53086 |
| 空的中断目录被误当可复用名称 | 输出标签只创建一次；即使空目录也占用，不删除后重用 | H1:53037,53058,53068 |
| 猜测 checkpoint 安装目录、helper 名称 | 按当前 catalog 的实际路径及源码符号查找；`checkpoint` 位于 `gstack-checkpoint` | H1:52998；H7:360–361 |
| formal 配置遗漏绝对残差阈值，短 preflight 未进入相关分支 | 串查 spec→case→solver→diagnostics→acceptance，覆盖近零幅值的相对/绝对判据 | H1:52463–52801 |
| L2 最优解与最终最大 f32 全行残差门不等价 | 检查优化目标与最终审计目标是否一致，保留原门槛与 fail-closed | 当前 handoff 的 `f2320f5` 修复记录 |
| component/formal 身份混用导致无效证据或重复长跑 | 分别核对显式 manifest；只变 formal runner 不使未变的 15 文件 component 证据失效 | H1:52731,52801,52874 |
| 源码改变后旧 strict snapshot 不能继续使用 | 保留 source-matched restart 门；缩减字段/历史数组不是完整 checkpoint | H8:26360,37899,49878 |
| 旧 ECC 要求所有修改 80% coverage、所有状态不可变，模型建议仍全部列为 GPT-5.4 | 全局规则改为与变更对应的验证；数值状态允许必要原地更新，保留所有权和回滚 | 实际全局 AGENTS 与 ECC 两个生成源 |
| Sol Advisor 强制主任务 Sol/Ultra、先声明路由才能读文件 | 尊重用户配置的主模型；必要委派前说明理由，保留精确 reviewer 的真实性要求 | 实际维护源及安装/cache 副本 |
| gstack 自动前置流程与任务范围冲突；不存在的 ECC 技能目录仍被宣称完整安装 | `proactive: false`；仅显式调用 gstack；删除失实目录声明，按实际 catalog 发现技能 | 实际 `.gstack/config.yaml` 与空 `.agents/skills` |
| 本次旧配置草稿覆盖了准备期间变化的 `service_tier`，安装后语义核对才发现 | 已按安装前备份恢复；草稿创建时保存原内容/hash，写入前核对同一个 hash，变化时重新合并；安装后检查无关字段 | 本次 `windows-correction-result.json` 与语义验证结果 |

这套规则降低已知操作错误的复发风险，不能保证未运行的数值轨迹一定通过。
遇到新数值失败仍必须保存 artifact、定位根因并核对同类契约。

## 修改落点与维护方式

- 全局 `C:/Users/lizhu/.codex/AGENTS.md`，以及生成它的
  `C:/Users/lizhu/.codex/ECC/AGENTS.md`、`ECC/.codex/AGENTS.md` 一起维护。
  旧的重复 ECC catalog 声明不再保留，后续本地同步不会自动带回原文本。
- Windows 项目入口与当前 WSL worktree 的 `AGENTS.md` 使用相同导航规则；
  不改其他历史 worktree、GUI 项目或求解器文件。
- 新增个人 `windows-wsl-execution` 和 `hibm-fsi-validation` skills；
  针对性更新 `karpathy-guidelines` 和 `taichi-docs`，不重写无关技能。
- Sol Advisor 维护源位于 `C:/Users/lizhu/.codex/marketplaces/sol-advisor`。
  先更新源，再同步 `plugins/sol-advisor` 和 `plugins/cache/sol-advisor/sol-advisor/0.6.0`。
  这是 0.6.0 的本地定制，不是宣称新的上游发布；上游更新前要重新合并本地差异。
- 通用 reviewer 和 Sol Advisor 专用 reviewer 均配置为
  `gpt-6-astra` / `max` / `sandbox_mode = "read-only"`。
  兼容角色 ID `sol_advisor_sol_reviewer` 保留；Luna/max、Terra/high 保留。
  主模型原本已是 Astra/max，本次只更新 reviewer 和相关描述。
- 安装前核对目标文件 SHA，保存逐文件备份及安装清单，避免覆盖并发修改。
  备份位于本次 Codex visualization 工作目录内，处于技能发现目录之外。

## R26A 持久环境恢复

2026-09-05 核查的当前源树为：

```text
/home/zhuohengli/worktrees/HIBM-MPM-r25b-live
branch: codex/turek-hron-fsi123-validation-r26a
HEAD: 70a2a1d1022d8240ed994882b1f953e6cd5787ff
python: /home/zhuohengli/.venvs/hibm-mpm-r26a-py310/bin/python
```

旧 `/tmp/hibm-mpm-r26a-py310/bin/python` 已不存在；无法从现有证据确定删除原因。
新环境由 `/usr/bin/python3` 3.10.12 创建，按该 HEAD 的 `requirements.txt`
安装直接冻结依赖，包括 NumPy 2.1.2、SciPy 1.15.3、Taichi 1.7.4。
`pip check` 通过。没有用环境检查启动 CUDA 或正式算例。

在本次项目文档修改前，clean tree 下执行 runner 的真实 `capture_provenance`
得到以下与暂停记录一致的指纹：

```text
formal source: f02c7efb41ad0802737fccab198284ef7fe7395e0b95b43712f43e6dfb7b3e24
formal config: f0d755926db9be429dae65af730da63f27032d478a8e1957200b6984b99bc03d
host: db88ab4094ab1be43ad58b7c18cc59c756a45e1ac4f44978bc6238a4738c6a47
component source (15 files, recomputed): 478eb7707fd20761654443b334782a252ffee90dc29ba2c9b9707ba747b67c89
```

host schema 1 绑定 CPython/NumPy/SciPy 身份；该匹配不证明所有间接依赖、
驱动、CUDA 执行或后续数值轨迹与历史完全相同。正式运行还要执行既有门禁。
恢复命令使用以下环境前缀，并在选定 worktree 内执行实际 runner 参数：

```sh
env -u PYTHONPATH -u PYTHONHOME \
  LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  SIMULATION_TAICHI_OFFLINE_CACHE=1 PYTHONUNBUFFERED=1 \
  /home/zhuohengli/.venvs/hibm-mpm-r26a-py310/bin/python
```

这只是解释器与环境前缀，不是完整启动命令。正式标签必须重新确认未占用，
完整参数取自当前 campaign 合同；不盲目复用 handoff 里的短 SHA 或旧标签。

## 验证与当前边界

实际完成的检查：

- Codex 官方 `quick_validate.py` 对四个个人技能及 Sol Advisor orchestration
  共 5 个技能通过。验证使用 WSL 系统 Python 中已有的 PyYAML；Windows Taichi
  Python 没有该依赖，不能把它当作通用文档验证环境。
- 维护源的完整 `verify.sh` 通过；真实 Windows agents 目录上的
  `install-agents.sh --check --target-dir /mnt/c/Users/lizhu/.codex/agents` 通过。
  缺失的 WSL `jq` 及其 `libjq1`、`libonig5` 依赖已安装；没有进行系统升级。
  verifier 修复 Python 3.10 的 tomli 回退，并将缺失 rg 时的静默漏检改为
  grep 的“命中/未命中/执行错误”三态处理。
- 安装清单中 46 个目标的最终 SHA 全部匹配，37 个 Windows 文件实际变化，
  包括维护源和重复安装副本；这不是 37 个不同技能。主模型和其他配置字段
  与安装前备份保持一致，两个 reviewer 都是 Astra/max/read-only。
- 独立 Astra/max 审查修正了 campaign 适用范围、辅助 lane 停止边界、
  无条件重复验证和旧 verifier 文案断言，最终结论 `ship`；五个历史情景的
  行为复核覆盖静默计算、用户暂停、component/formal 分域、跨 shell 失败、
  checkpoint 实际路径。
- 15 文件 component source SHA 重新计算后完全未变，无需重跑 nx4/nx8。

当前任务通过显式模型参数运行了 Astra/max 独立 reviewer；配置和安装文件
检查并不证明本任务启动时已经加载的旧命名角色发生热更新。下次创建实际
命名 reviewer 时仍须核对公开运行时 metadata，不能只凭 TOML 声称已加载。
以上验证范围是指令、插件迁移和环境身份，不算 FSI1-S0 或 FSI2/3 通过。

项目 AGENTS、文档索引和本报告会产生可审查的未提交变更。
formal runner 的 clean-source gate 仍然有效：不忽略 dirty files、不修改
provenance 判据，也不为了启动数值任务偷偷提交。本次没有启动 FSI 长任务，
没有重跑 component chain，没有修改数值容差或求解器，没有 commit/push。

`missing field code_mode_host_duration_ns` 已确认是工具传输层症状，
现有证据没有证明其根因，也不能声称 AGENTS 修改修复了底层 IPC。
恢复后先核查实际操作结果，再决定是否重试，避免未知状态下重复启动。

## 本机历史证据键

下列 ID 对应 `C:/Users/lizhu/.codex/sessions/YYYY/MM/DD/` 内的原始 JSONL。
表中定位为文件行号；完整筛选报告留在本次 visualization 目录的
`history_scan/FSI_HISTORY_EVIDENCE_AUDIT.md`，不纳入仓库。

- H1：2026/09/02，`01a0615e-3b99-7220-ab0b-c92fcb3292a6`。
- H2：2026/09/05，`01a06efe-4eaa-7611-92af-4279c3ac17c4`。
- H3：2026/08/27，`01a0434a-786c-7bc2-a8c6-1a381dda007a`。
- H4：2026/09/05，`01a06f3f-5608-7132-bed7-3d529c0ec427`。
- H5：2026/06/13，`019ec124-7b7a-7ba1-90cc-a6d0afcd7ec0`。
- H6：2026/06/15，`019eca17-b453-77a3-9da5-221bb1ac2fa1`。
- H7：2026/09/05，`01a06f3c-1b43-7841-ac96-ca57e7365e03`。
- H8：2026/08/24，`01a0342f-230b-72d0-a124-2e489284ec43`。


## 后续稳健性工作确认的执行教训

后续数值代码任务复现了 WSL 普通 -- 对参数的二次 shell 解释。带正则 |
的 PowerShell 引号不能阻止该解释；直接 --exec 能保留原参数。所有使用相对
路径、pytest 缓存和输出目录的命令同时显式传 --cd 到当前 WSL worktree。
绝对 Python 脚本路径和 git -C 都不能替代进程 cwd。个人
windows-wsl-execution 技能已补入这两条有实际复现的规则。

代码编排工具的 JavaScript 模板字符串与 PowerShell here-string 是不同的
解释层。在 JavaScript 原始模板字符串中直接嵌入 Markdown 反引号会提前结束
模板，失败发生在工具调用之前；使用不含该分隔符的文本或独立文件，不能把
String.raw 当作通用转义。测试夹具也必须复用实际 import/report API，并先确认
是否被更早的重复标记校验拦截；只有到达目标求解路径的失败才算有效回归。

本节是前述指令审计之后的新工作记录。求解器变更、回归和下一次 component
重认证状态以 docs/TUREK_HRON_VALIDATION.md 的稳健性更新为准，不沿用本报告
此前“仅更新指令、component 身份未变”的历史结论。
