# CDEQ+ 向 CLLM/Jacobi 场景扩展的完整方案

> 文档性质：研究路线与实现交接文档
>
> 更新时间：2026-07-14
>
> 目标：将已发表的 CDEQ 扩展为 TPAMI 稿件。主要新增方法是 initializer
> 和 continuous-time schedule；LLM/Jacobi 解码作为一个新增应用场景，
> 用于验证 CDEQ+ 的跨领域泛化能力。
>
> 边界：本文档不修改、替代或中断当前并行进行的 CDEQ+ 实现和实验。

## 1. 最终决策

LLM 部分应定位为 **CDEQ+ 的跨领域应用验证**，而不是一篇新的并行解码方法。

TPAMI 扩展稿的贡献层次应当固定为：

1. 已发表的 CDEQ 是整篇工作的基础。
2. CDEQ+ 提出两项主要方法改进：
   - learned initializer；
   - continuous-time schedule。
3. 在原 CDEQ 的数据集、模型和任务上，系统验证两项改进。
4. 增加一个 LLM 章节，将同一套 CDEQ/CDEQ+ 实例化到 Jacobi 轨迹上，
   证明其并不局限于传统 DEQ。

LLM 章节应明确复用 CLLM 的 Jacobi 轨迹构造，不声称 Jacobi 轨迹、
固定点一致性或并行 token refinement 是本文首次提出。

整条路线的一句话表述是：

> CDEQ+ 是一个 solver-trajectory distillation 框架。其 initializer 和
> continuous-time schedule 不仅适用于传统 DEQ 轨迹，也能够在只做状态接口
> 适配的情况下迁移到自回归 LLM 的 Jacobi 固定点轨迹。

## 2. PCCoT-DEQ 实验带来的判断

此前 PCCoT Picard 实验暴露了数值收敛与语义性能之间的不一致：

- 随着迭代次数增加，DEQ 相对残差持续下降；
- quick-start 样例的答案在很早的迭代步便稳定；
- 验证集 exact match 随迭代步数呈非单调变化；
- 在前 64 个验证样本上，EM 在第 7 步达到 `31/64 = 0.484375`，
  第 20 步下降为 `27/64 = 0.421875`；
- 更多迭代同时带来更高推理耗时。

因此，必须区分：

```text
fixed-point residual 下降 != 任务性能提升
```

这并不否定 CDEQ，而是说明：如果教师固定点本身较弱，或者与任务目标不对齐，
一致性学生仅仅更准确地回归这个固定点，并不能自然超过教师性能。

因此，PCCoT-DEQ 的固定点不适合作为 TPAMI 扩展的主要证据。LLM 场景更适合使用
Jacobi 轨迹，因为在 greedy decoding 设定下，Jacobi 固定点被定义为与目标模型的
AR 输出一致。此时教师上限来自选定的 AR LLM，而不是一个可能在收敛过程中损害
语义性能的 PCCoT latent equilibrium。

## 3. TPAMI 扩展稿的定位

### 3.1 支撑扩展稿的新增内容

期刊扩展不能只依靠一个 LLM 章节。完整新增内容应包括：

- learned initializer；
- continuous-time schedule；
- 四组严格消融：CDEQ、CDEQ+Init、CDEQ+CT、CDEQ+Init+CT；
- 原数据集上的扩大实验；
- solver、step budget、初始化和轨迹分析；
- 新增 LLM/Jacobi 应用；
- 更完整的误差分析和 limitations discussion。

IEEE Computer Society 的相关材料通常把约 30% 的实质性新增作为 conference
extension 的参考标准。最终 cover letter 和 Introduction 应逐项说明相对已发表
CDEQ 论文新增了什么，而不能只依靠页数变长。

### 3.2 LLM 章节只回答一个问题

LLM 章节的核心研究问题应当是：

> 同一套 CDEQ+ 设计能否蒸馏另一类固定点求解过程，即自回归语言模型诱导的
> Jacobi iteration？

该章节不负责提出新的 serving system、speculative decoder、acceptance algorithm
或 LLM 训练范式，也不需要覆盖并行解码领域的所有问题。

### 3.3 推荐论文表述

推荐英文定位：

> We adopt Jacobi trajectories following CLLMs to instantiate CDEQ+ for
> autoregressive language models. Our goal is not to introduce another Jacobi
> decoder, but to examine whether the same solver-trajectory distillation
> framework and its two extensions generalize beyond conventional DEQs.

推荐与 CLLM 的关系表述：

> CLLMs provide the task-specific Jacobi trajectory construction and a direct
> parallel-decoding baseline. CDEQ+ provides the previously defined
> time-conditioned consistency module, learned initialization, and
> continuous-time training mechanism whose cross-domain transfer is evaluated
> in this work.

需要避免的表述：

- “本文首次将 consistency distillation 应用于 Jacobi decoding”；
- “本文首次提出 LLM 的 Jacobi trajectory”；
- “本文提出一种新的 CLLM 架构”，除非后续确实形成独立架构贡献；
- “数值残差越低，LLM 推理准确率越高”；
- 在没有匹配训练目标和验证的情况下，把 few-step 当作可随意调整的运行参数。

## 4. 与现有 LLM 方法的关系

### 4.1 CLLM

CLLM 从目标自回归模型收集 Jacobi 轨迹，并通过 consistency objective 与
AR-preservation objective，使轨迹中间状态更快收敛到 AR 固定点。

在本项目中，CLLM 应被视为：

- LLM 轨迹构造协议的来源；
- 最直接的相关工作和实验 baseline；
- 可复用模型、轨迹数据和评测脚本的来源；
- 必须明确承认的 prior art。

CDEQ+ 期刊稿的新增贡献不是使用 Jacobi 轨迹，而是证明已经定义好的两项 CDEQ+
改进可以迁移到这类新的 solver trajectory。

### 4.2 Jacobi Forcing 与 noisy-training 方法

后续工作已经覆盖 progressive noise schedule、student-generated trajectory、
large-block training、noisy SFT、retrieval/recycling 和 multi-block decoding。

这些工作应在 Related Work 中讨论，但第一版 LLM 实现不应主动吸收以下内容：

- rejection recycling；
- tree verification；
- 新的 multi-block scheduler；
- online trajectory refresh；
- 新的 serving engine。

### 4.3 参数高效是辅助优势，不是唯一创新

冻结 backbone、只训练较小的一致性模块是有价值的，但不能成为唯一差异：

- CLLM 官方实现可以使用 QLoRA；
- Medusa 等方法也可以在 frozen LLM 上训练轻量模块。

因此，trainable parameter count、memory 和 modularity 应作为实验优势报告，
而不是单独承担方法创新。主贡献仍然是 CDEQ+ 两项改进及其跨 solver/domain 泛化。

## 5. LLM 场景的最小数学定义

### 5.1 Jacobi 教师轨迹

给定 prompt `x` 和冻结的自回归语言模型 `p_theta`，greedy AR generation 满足：

```text
y_i = argmax_y p_theta(y | x, y_<i).
```

对于长度为 `n` 的未来 token block，Jacobi decoding 从一个初始化序列
`y^(0)` 出发，并行更新所有位置：

```text
y_i^(k+1) = argmax_y p_theta(y | x, y_<i^(k)).
```

教师轨迹为：

```text
J = {y^(0), y^(1), ..., y^(K)}.
```

在 greedy decoding 条件一致时，收敛终点 `y^(K)` 应与目标 LLM 的 AR 输出一致。

### 5.2 连续状态接口

CDEQ+ 不能直接对整数 token ID 做连续插值。token ID 是离散符号，线性插值没有
有效的语言模型含义。

应将每个 Jacobi token 状态转换为连续状态：

```text
s_k = R_theta(x, y^(k)),
```

其中 `R_theta` 来自冻结的目标 LLM。候选状态包括：

1. final-layer hidden states，建议作为第一版默认选择；
2. pre-softmax logits；
3. token probability distributions。

优先使用 final-layer hidden states，原因是它的维度是 model hidden size，而不是
vocabulary size，并且可以通过冻结的 LM head 解码。

### 5.3 大模型维度匹配

对于 7B LLM，如果直接在完整 hidden size 上应用原始 `H -> 3H -> H` MLP，
辅助模块可能过大。必要时可以增加 bottleneck projection：

```text
h       -> down_projection -> u
[u; t]  -> CDEQ+ updater   -> delta_u
delta_u -> up_projection   -> delta_h
```

该 projection 是适配模型维度的工程接口，不应被包装成新的主要贡献。论文中必须
报告 projection rank、参数量和实际 latency，并加入 parameter-matched baseline。

### 5.4 Jacobi 轨迹上的 CDEQ baseline

基线 CDEQ 应尽量复用已发表定义：

- time-conditioned consistency updater；
- adjacent-state/EMA consistency；
- 对 Jacobi endpoint representation `s_K` 的回归；
- 相同的 boundary interpolation 原则；
- 除非单独设计 few-step objective，否则只支持已训练的一步推理。

当前参考 loss 可以在概念上保留：

```text
loss = 0.2 * adjacent_ema_consistency
     + 0.8 * endpoint_regression.
```

hidden state 场景可以考虑 normalized MSE 或 cosine distance；logit 场景可以考虑
distributional KL。表示与距离函数属于实现选择，应通过消融确定，而不是作为第三项
主要方法贡献。

### 5.5 CDEQ+ initializer

LLM initializer 应保持现有 CDEQ+ 的语义：

- 学习指向教师 endpoint representation 的 residual；
- 直接回归 `s_K`；
- 如果原 CDEQ+ 将 initializer 和 updater 分开优化，则在 consistency objective
  前 detach initializer output；
- 分别报告 initializer-only 和 initializer+CT 的效果。

概念形式为：

```text
s_init_hat = s_0 + I_psi(s_0, prompt_context)
L_init     = distance(s_init_hat, s_K).
```

initializer 不是 draft LLM，不应描述为 speculative decoding。

### 5.6 Continuous-time schedule

复用 CDEQ+ 的 rho-shaped time grid：

```text
t_j = (
    epsilon^(1/rho)
    + j/(N-1) * (T^(1/rho) - epsilon^(1/rho))
)^rho.
```

当前默认值为：

```text
N = 38
epsilon = 0.002
T = 5
rho = 7
```

在 LLM 轨迹中：

- 将 Jacobi 轨迹起点映射到高 time/noise；
- 将收敛终点映射到低 time/noise；
- 使用现有 progressive rule 采样连续 `r < t`；
- 只在连续 hidden/logit states 之间进行插值；
- 保持原 CDEQ+ boundary-condition design。

continuous-time mechanism 应被描述为同一个 CDEQ+ 组件在新场景中的复用，
而不是新的 LLM diffusion model。

### 5.7 输出解码

对于预测得到的 endpoint hidden state `s_hat`，复用冻结的 LM head：

```text
logits_hat = LMHead_theta(s_hat)
y_hat      = argmax(logits_hat).
```

第一版实现使用 greedy decoding，以匹配 vanilla Jacobi 和 CLLM 的主要理论假设。
除非 reviewer 或后续研究明确需要，否则 sampling support 不在当前范围内。

## 6. 第一版 LLM 实现的范围边界

### 6.1 在范围内

- 收集或复用 Jacobi trajectories；
- 提取 hidden/logit trajectories；
- 在模型维度需要时增加 input/output projection；
- 复用现有 CDEQ consistency updater；
- 复用现有 initializer；
- 复用现有 continuous-time schedule；
- 执行已经被训练目标支持的一步 CDEQ/CDEQ+ inference；
- 与 CLLM 做正式比较。

### 6.2 不在第一版范围内

- 把 CDEQ 重写成通用 Jacobi flow operator；
- 修改 LLM backbone architecture；
- 从头预训练或 instruction-tune 一个 LLM；
- 新的 speculative decoding 或 token verification；
- adaptive halting；
- stochastic decoding；
- online/on-policy distillation；
- production serving engine；
- 在没有匹配目标的情况下重复调用 CDEQ updater。

这些范围边界保证 LLM 章节与其在 TPAMI 扩展稿中的作用相匹配。

## 7. 模型与数据集选择

### 7.1 Feasibility setting

第一步优先选择一个官方 CLLM 已支持的 setting，以便复用模型、trajectory 和评测
协议。推荐首先使用 GSM8K，原因是：

- 它直接衡量 reasoning accuracy；
- 与现有 PCCoT/GSM8K 经验一致；
- CLLM 提供已发表的 7B math setting 和公开 artifacts；
- exact-match 评测简单明确。

接口调试可以先使用较小模型，但最终 TPAMI 结果至少应包含一个可信的 billion-scale
LLM，而不能只有 GPT-2 Small。

### 7.2 Submission-level setting

建议的最低正式配置：

- 一个 math/reasoning task，例如 GSM8K；
- 一个结构明显不同的任务，例如 Spider、HumanEval、MBPP 或代码生成任务；
- 如果资源允许，headline result 至少包含一个 7B 模型。

资源有限时按以下顺序执行：

1. 先在 GSM8K 上完成全部消融；
2. 第二个任务优先选择 CLLM 官方已有的 code/structured setting；
3. 复用官方 checkpoint 和 trajectory，不重新训练目标 LLM；
4. 不追求大量 benchmark 但缺少直接 baseline 的表面规模。

一个任务可以作为 feasibility gate，但不足以支撑最终稿中“LLM 泛化性”的完整结论。

## 8. 必须包含的 baseline 和 ablation

### 8.1 核心比较矩阵

| Method | 作用 |
| --- | --- |
| AR decoding | 原始模型质量和 latency 参照 |
| Vanilla Jacobi | 未蒸馏 teacher solver 的行为 |
| CLLM | 最直接的 LLM consistency-distillation baseline |
| CDEQ-Jacobi | 已发表 CDEQ 在 Jacobi states 上的实例化 |
| CDEQ-Jacobi + Init | 单独验证 initializer |
| CDEQ-Jacobi + CT | 单独验证 continuous-time schedule |
| CDEQ+-Jacobi | 同时启用两项改进 |

如果现有预算允许复现，可在正式稿中加入 Jacobi Forcing 作为更强近期参考。它不是
第一轮 feasibility experiment 的前置条件，但必须在 Related Work 中讨论。

### 8.2 公平性要求

- 所有方法使用相同 target LLM 和 task split；
- 使用相同 Jacobi block length 和 greedy decoding 假设；
- latency 在相同 GPU、软件栈、batch size、prompt length 和 generation limit 下测量；
- 报告完整 wall-clock latency，不只报告 solver steps 或 NFE；
- 将辅助模块 overhead 纳入时间和显存；
- 报告精确 trainable parameter count；
- 一步 CDEQ 与可变步 CLLM 比较时，必须呈现完整 quality/latency Pareto；
- 只有在硬件和评测协议完全一致时才能直接引用官方 CLLM 数字，否则需要重跑。

### 8.3 指标

任务质量：

- GSM8K exact match/test@1；
- code task 的 pass@1；
- Spider execution accuracy；
- 与 target AR greedy output 的 agreement。

推理效率：

- end-to-end latency per example；
- tokens per second；
- relative-to-AR speedup；
- LLM forward evaluations/NFE；
- peak inference memory；
- auxiliary-module latency。

训练成本：

- trainable parameter count 及其相对 backbone 的比例；
- peak training memory；
- training duration 和 GPU 数量；
- trajectory 收集数量与存储成本。

轨迹诊断：

- 每个 Jacobi step 到 endpoint 的 hidden/logit distance；
- 每一步与 endpoint 的 token agreement；
- CDEQ/CDEQ+ endpoint regression error；
- initializer 训练前后的 endpoint error；
- representation error 与任务正确性的相关性。

## 9. 实验顺序和决策门槛

### Phase 0：保护当前并行 CDEQ+ 工作

开始 LLM 实现之前：

- 完成或 snapshot 当前正在运行的 CDEQ+ 工作；
- 记录精确 commit、checkpoint 和正式 metrics；
- LLM 扩展放入独立 branch 或隔离模块；
- 不为了兼容 LLM 重写当前 GPT-2 CDEQ/CDEQ+ 路径；
- 只有在现有行为有测试保护后，才抽取 shared utilities。

成功信号：

```text
开始 LLM 工作后，原 CDEQ+ 实验仍然可以独立复现。
```

### Phase 1：只验证 trajectory

任务：

1. 加载选定的 frozen LLM。
2. 复现 greedy AR 输出。
3. 复现 vanilla Jacobi 收敛到相同 greedy endpoint。
4. 收集 token trajectory 和 continuous hidden/logit states。
5. 验证 shape、endpoint identity、storage cost 和 determinism。

成功信号：

```text
在被检查的样本上，Jacobi endpoint token 与 greedy AR token 完全一致。
```

停止条件：

- 如果在相同 greedy setting 下无法复现 endpoint equivalence，不开始 CDEQ+ 训练；
- 先解决 causal mask、KV cache、padding、EOS 和 block boundary 语义。

### Phase 2：CDEQ-Jacobi baseline

任务：

1. 选择 hidden states 或 logits 作为连续表示。
2. 只增加 target LLM 所需的最小维度 adapter。
3. 在不启用 Init/CT 的情况下训练已发表 CDEQ objective。
4. 测量一步质量和 latency。

成功信号：

```text
在 target LLM 保持冻结时，CDEQ-Jacobi 能够学习非退化的 endpoint mapping，
并显著降低 endpoint representation/token error。
```

停止条件：

- 如果 baseline 不能降低 endpoint error，暂不加入 Init 或 CT；
- 先检查 representation choice、normalization 和 projection capacity。

### Phase 3：四组 CDEQ+ ablation

在完全一致的数据和 compute 下训练：

```text
Init=0, CT=0
Init=1, CT=0
Init=0, CT=1
Init=1, CT=1
```

成功信号：

- 每个组件产生可解释影响；
- 组合模型改善 quality/latency 或 quality/training-cost Pareto；
- 增益在多个 seed 或 task subset 上稳定。

### Phase 4：正式比较

任务：

- 在同一协议下运行 AR、Jacobi、CLLM、CDEQ 和全部 CDEQ+ 消融；
- 加入第二个任务；
- profile wall-clock 和 memory；
- 生成论文表格与 trajectory figure；
- 在扩大范围之前先写清 limitations。

成功信号：

```text
即使 CDEQ+ 没有在每一个 speed metric 上超过 CLLM，LLM 章节仍能够以受控实验
支持 CDEQ+ 的跨领域迁移结论。
```

可接受结果按强度排序：

1. quality/speed 接近或超过 CLLM，同时显著降低 trainable parameters 或训练显存；
2. 在固定低 step budget 下取得更好的质量；
3. Init 和 CT 在传统 DEQ 与 LLM 场景均稳定有效；
4. 如果结果为负，能够严谨说明 continuous trajectory distillation 在离散生成中的
   失败边界。

## 10. 后续代码隔离建议

当前 PCCoT checkout 含有并行进行中的未提交 CDEQ+ 工作。在该工作稳定之前，
不应把 LLM 改动混入现有核心文件。

建议未来组织方式：

```text
llm_cdeq/
  jacobi.py              # teacher trajectory collection
  representations.py     # hidden/logit extraction
  consistency.py         # LLM adapter around existing CDEQ+ modules
  train.py
  evaluate.py

configs/llm_cdeq/
  gsm8k.yaml
  second_task.yaml

scripts/llm_cdeq/
  collect_trajectories.sh
  train_ablation.sh
  evaluate_matrix.sh
```

以上只是后续建议布局，本文档阶段不创建这些文件。第一版应尽可能复用已验证的
consistency module，同时把 LLM-specific mask、EOS、block construction 和 state
extraction 隔离在独立代码中。

每个 checkpoint/trajectory 必须保存：

- target model identifier 和 revision；
- tokenizer identifier 和 revision；
- dataset、split 和 sample IDs；
- Jacobi block length 与 initialization rule；
- maximum Jacobi steps 与 convergence rule；
- state representation 与 layer index；
- projection rank/dimension；
- Init/CT switches 与 time schedule；
- trainable parameter names/counts；
- software/GPU environment；
- random seed。

## 11. 主要风险与处理方式

| 风险 | 后果 | 处理方式 |
| --- | --- | --- |
| 对 token ID 插值 | continuous-time 解释无效 | 在 hidden states 或 logits 上操作 |
| CDEQ baseline 学不会 endpoint | Init/CT 结果无法解释 | 先通过 baseline gate 再做消融 |
| 辅助 MLP 过大 | 参数高效优势消失 | 使用并报告 bottleneck projection |
| hidden distance 与 token 不对齐 | MSE 低但生成质量差 | 同时跟踪 token agreement 和任务指标 |
| one-step collapse | 重复、非法或退化输出 | 保留 endpoint/AR 监督并检查 EOS/mask |
| CLLM 使用不同评测协议 | 比较无效 | 在相同模型、数据、硬件和 decoding 下重跑 |
| LLM 改动污染当前 CDEQ+ | 并行工作不稳定 | 模块隔离并等待受保护 baseline |
| 范围扩大到 serving research | TPAMI 扩展失焦 | 不加入新的系统级解码方法 |
| LLM 是唯一大幅新增 | conference extension 不充分 | 让 Init/CT 和原领域扩展承担主贡献 |

## 12. 推荐论文结构

```text
1. Introduction
2. Related Work
   2.1 Deep Equilibrium Models
   2.2 Consistency Distillation
   2.3 Parallel and Jacobi LLM Decoding
3. Background: Published CDEQ
4. CDEQ+
   4.1 Learned Initializer
   4.2 Continuous-Time Schedule
   4.3 Training and Inference
5. Analysis
   5.1 Initialization Error
   5.2 Trajectory Coverage
   5.3 Approximation and Task Error
6. Experiments on Original Domains
7. Extension to Autoregressive Language Models
   7.1 Jacobi Trajectory Construction
   7.2 Continuous State Interface
   7.3 Experimental Setup
   7.4 Results and Discussion
8. Limitations
9. Conclusion
```

Section 7 建议篇幅：

- 约 0.5 页：Jacobi formulation 及与 CLLM 的关系；
- 约 0.5 页：hidden/logit state interface 与 CDEQ+ 复用；
- 约 1 页：实验表、trajectory figure 和讨论；
- 详细实现与附加消融放入 supplementary material。

## 13. Conference-to-journal difference ledger

写作期间持续维护下表，并在 cover letter 中给出精简版本。

| Journal addition | 类型 | 所需证据 |
| --- | --- | --- |
| Learned initializer | Method | 公式、训练细节、单独消融 |
| Continuous-time schedule | Method | schedule、边界条件、单独消融 |
| 扩大原领域数据集 | Experiment | matched baselines 和重复实验 |
| Solver/step analysis | Analysis | residual、quality、runtime 曲线 |
| LLM/Jacobi extension | New application | CLLM baseline 与跨领域结果 |
| Extended limitations | Analysis | teacher ceiling 与 semantic mismatch |

最终稿必须明确引用已发表会议论文，并在 Introduction/cover letter 中解释新增内容，
不能只以篇幅增加作为扩展依据。

## 14. 工作清单

### 14.1 Research design

- [ ] 固定 LLM 只是 application section 的角色。
- [ ] 选择第一个 CLLM-compatible model/task。
- [ ] 决定使用 hidden states 还是 logits。
- [ ] 确定 module dimension 或 bottleneck rank。
- [ ] 固定一步训练和一步推理的精确定义。

### 14.2 Implementation

- [ ] 用 Git/checkpoint metadata 保护当前 CDEQ+ baseline。
- [ ] 复现 AR/Jacobi endpoint equivalence。
- [ ] 保存 deterministic Jacobi trajectories。
- [ ] 训练 CDEQ-Jacobi baseline。
- [ ] 训练全部四组 Init/CT ablation。
- [ ] 在相同协议下运行 CLLM。
- [ ] 加入第二个任务。

### 14.3 Evaluation

- [ ] 报告 task quality 与 AR agreement。
- [ ] 报告 wall-clock latency 和 TPS。
- [ ] 报告 NFE/Jacobi iterations。
- [ ] 报告 trainable parameters 和 training memory。
- [ ] 绘制 trajectory distance 与 token agreement。
- [ ] 确认性能差异不是来自不同 decoding budget。

### 14.4 Writing

- [ ] 明确 Jacobi trajectories 遵循 CLLM。
- [ ] 不声称 Jacobi consistency distillation 是本文首次提出。
- [ ] 将 Init 和 CT 作为 CDEQ+ 的主要贡献。
- [ ] 将 LLM 结果写成 generalization evidence。
- [ ] 提供 conference-to-journal difference statement。
- [ ] 将 PCCoT residual/EM mismatch 写入 teacher-alignment limitation。

## 15. 参考资料与起点

- CLLMs: Consistency Large Language Models：
  <https://proceedings.mlr.press/v235/kou24a.html>
- CLLM 官方实现、模型与轨迹：
  <https://github.com/hao-ai-lab/Consistency_LLM>
- Jacobi Forcing：
  <https://arxiv.org/abs/2512.14681>
- Make Some Noise / TR-Jacobi：
  <https://aclanthology.org/2024.emnlp-main.718/>
- Medusa frozen-backbone baseline：
  <https://arxiv.org/abs/2401.10774>
- TPAMI information and calls for papers：
  <https://www.computer.org/digital-library/journals/tp/cfp-ieee-pattern-analysis-machine-intelligence>

## 16. 最终工作原则

不要为了单独提高 LLM 章节的新颖性而重新设计整套方法。当前应优化的是一个清晰、
可辩护、工作量受控的 TPAMI 扩展：

```text
published CDEQ
    + learned initializer
    + continuous-time schedule
    + broader original-domain evaluation
    + one carefully controlled Jacobi/LLM application
    = CDEQ+ journal extension
```

LLM 实验的成功标准是：在最小方法改动下，证明 CDEQ+ 可以迁移到新的 solver
trajectory，并且与 CLLM 做透明、公平的比较。它不需要被扩展成一篇独立的 CLLM
替代方法论文。
