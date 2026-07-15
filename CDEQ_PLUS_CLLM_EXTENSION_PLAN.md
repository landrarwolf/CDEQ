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

## 0. 2026-07-14 执行补充与固定协议

本轮可行性验证已经固定为 GSM8K、greedy decoding、Jacobi block size 16；先复现
官方 CLLM checkpoint，不将 7B CLLM 全量重训作为前置条件。固定上游版本和远端
目录记录在 `CLLM-src/UPSTREAM.md` 与 `CLLM-src/configs/llm_cdeq/gsm8k.yaml`。

连续时间方向统一为：`y0 → epsilon=0.002`，`yK → T=5`，`rho=7`；`t=T`
为严格 identity boundary，训练域为 `[epsilon,T]`，一步推理使用既有 CDEQ 的
`t=0` boundary 调用。离散 loss 固定为 `0.1 × adjacent EMA MSE + 0.9 × endpoint
SmoothL1`，CT 使用 `q=1.1,d=100,k=8,b=1` 的 progressive rule。

工程实现隔离在 `CLLM-src/llm_cdeq/`，四个稳定入口分别为 `prepare_states`、
`train`、`evaluate` 和 `profile`。模型、原始轨迹、hidden cache、checkpoint 与实验
输出只保存在远端，不进入 Git。本轮首先证明 CDEQ-Jacobi 学到非退化的
endpoint mapping，再验证轨迹进度感知的自适应少步推理：允许 endpoint-targeted
updater 在真实推理中调用少量多次，但平均与 P95 调用次数必须显著小于常规 CLLM。
Init 和 CT 仍需分别具有正向作用，Init+CT 最佳或并列最佳；暂不要求一步超过 CLLM。

官方 `augTrue` 数据经实物检查后还需一个确定性的规范化步骤：其
`answer_trajectory_ids` 含有重复、交错的 augmented candidates，并非可直接按 JSON
下标解释的时间序列。缓存构造因此先用冻结 Abel 的一次 token transition 建图，
再保留到达 self-mapping endpoint 的最长严格对齐链。只有链上满足
`LMHead(R(x,y^k))=y^(k+1)` 的状态进入连续 hidden trajectory；不会按原始 JSON
顺序插值，也不会对 token ID 插值。

## 0.1 2026-07-15 实际执行结果与当前结论

官方工件已在远端完成逐文件校验。Abel-7B-001、CLLM 7B Math 和 GSM8K
Jacobi 数据均匹配固定 revision；环境为 Python 3.10、PyTorch 2.1.2、
Transformers 4.36.2、Accelerate 0.25.0、Datasets 2.15.0。FlashAttention
2.4.1 在当前环境不可用，因此本轮质量复现和 cache 构造统一使用 SDPA，速度结果
只作为同 backend 的服务器内比较。

已通过的官方复现和工程 gate：

- 官方 GSM8K demo 正确输出 `500`，生成 384 tokens，约 103.15 tokens/s；
- 100 个 Abel block 上，使用同一 no-cache causal forward primitive 的 greedy AR
  与 vanilla Jacobi endpoint 为 `100/100` 一致；
- 64-block 四组过拟合均可下降，其中扩展到 160 steps 后 baseline endpoint error
  下降 18.34%，CT 下降 48.51%；
- 远端完整测试为 `28 passed`；
- target backbone 参数量为 6,738,677,760，rank-512 baseline adapter 可训练参数为
  5,770,752，约占 backbone 的 0.0856%。

cache 构造曾发现一个重要数据问题：按原始 record 数做 grouped split 时，两个超大
augmented `data_id` 会垄断 2k train cache。实现已改为对 `data_id` 做确定性
group-level hash split，并在每个 split 内跨问题轮转取样。重建后的 cache 为：

- train 2,000 blocks / 2,000 unique `data_id`；
- validation 512 blocks / 512 unique `data_id`；
- train/validation `data_id` overlap 为 0；
- 6 个 hidden-token 未对齐候选被拒绝，所有入库状态的冻结 LM-head 对齐为
  `432,803/432,803`；
- states 为 BF16 `[N,17,16,4096]`，全部有效 rho time grid 从 0.002 单调到 5.0；
- 有效 trajectory length 覆盖 2--17，train/validation 平均分别为 10.764/10.854。

2k/512 feasibility baseline 的显式 identity 指标为 endpoint relative error
`1.011137`、token agreement `43.1885%`。默认配置 20 epochs 达到 `0.876771`
和 `45.0562%`。Token CE 权重 0.05/0.1/0.2 均提高少量 token agreement，但会明显
恶化 hidden error，因此固定 `token_ce_weight=0`。

随后完成 rank `{256,512,1024}`、LR `{1e-4,3e-4,1e-3}`、local/global
`{0.1/0.9,0.2/0.8}` 的 successive halving。最终最优为 rank 512、LR `1e-3`、
local/global `0.2/0.8`：validation error `0.860156`，token agreement `45.4224%`，
相对 identity 分别改善 14.94% 和 2.23 个百分点；另一 local=0.1 配置为
`0.862611` 和 `46.0205%`。两者均未达到本计划规定的 20% / 5pp baseline gate。

训练集诊断也只从 identity `0.998723/43.9875%` 改善到
`0.838668/47.1469%`，说明主瓶颈不是 train/validation 泄漏或典型过拟合，而是当前
逐 token bottleneck updater 的 hidden approximation/优化能力。由于 teacher
trajectory 长度覆盖健康且冻结 LM-head 对齐为 100%，token alignment 和 CT coverage
不是第一嫌疑。按预注册停止规则，本轮不扩大到 10k/1k，也不启动四组正式消融；
应先重新审视 representation normalization、跨 token coupling 或 projection/updater
capacity，并将本轮作为可复现的负 feasibility 结果保留。

官方完整 GSM8K accuracy 和 500-sample speed profiler 仍在远端独立后台运行；其最终
数字写入复现报告后，才判断官方 checkpoint 的 56.4±1.0 与同机 2x speedup gate。

## 0.2 新增决策：轨迹进度感知的自适应少步推理

完整研究假设、算法定义、证伪条件和分阶段实验见
[`ADAPTIVE_CDEQ_PLUS_RESEARCH_IDEA.md`](ADAPTIVE_CDEQ_PLUS_RESEARCH_IDEA.md)。

一步到达 endpoint 仍是训练时的监督目标，但不再假设一次近似调用在真实推理时必须
精确到达 endpoint。若第一次输出只落在教师轨迹某个中间邻域，则估计该输出对应的
归一化轨迹进度 `p in [0,1]`，将其映射到 rho time，并再次调用同一个 updater：

```text
z_0 = Init(s_0)                         # initializer 只执行一次
z_1 = F(z_0, 0)
p_m = H(z_m)
t_m = rho_time(p_m)
z_(m+1) = F(z_m, t_m)                   # 少量重复，仍以 endpoint 为目标
```

对于含 `K+1` 个状态的教师轨迹，`p_k=k/K`，并使用

```text
t(p) = (epsilon^(1/rho) + p * (T^(1/rho)-epsilon^(1/rho)))^rho.
```

这里 `H` 是轻量 progress estimator，而不是 endpoint predictor。线上推理不能通过
生成完整教师轨迹再做最近邻来判断 point，否则会抵消加速；同一样本教师轨迹上的
masked hidden nearest-point 只用于 oracle 可行性实验、progress 标注和诊断。

推理时间必须单调但保守：`p_(m+1) >= p_m`，未验证稳定前不得直接把 `t` 设为 `T`，
因为 `t=T` 是 identity boundary，错误的过早估计会使迭代卡死。停止条件联合使用
progress 置信度、连续 greedy token 稳定性和 masked hidden update norm，并始终设置
`max_calls`。initializer 只在第一轮执行。

训练按最小改动原则分三层推进：

1. 保持当前 `all teacher states -> endpoint` 主损失不变，先在 held-out cache 上使用
   oracle nearest-time 测量 1/2/3/4-call 曲线；如果 error/agreement 不随调用改善，
   先否决当前 recurrence 假设，不训练 progress head。
2. oracle 有效后，使用缓存已有的 `(s_k,p_k,t_k)` 训练轻量 `H`，比较
   oracle-time、learned-time 和 fixed schedule；这会增加辅助进度监督，但不改变
   CDEQ endpoint、Init 或 CT 的主目标。
3. 如果 learned-time 有效而 student rollout 出现 off-trajectory drift，再加入小比例
   stop-gradient rollout augmentation：对学生自产状态继续监督到同一 endpoint，
   不把目标改成 next point，也不跨离散 argmax 反传整条 rollout。

默认优先验证只重复轻量 adapter 的 latent recurrence，以最大化相对 CLLM 的 NFE
优势；同时将每轮 greedy token 经冻结 Abel 重新编码回 canonical hidden manifold
作为稳健性 fallback/消融。后者每轮会增加一次 backbone forward，必须单独报告
target-backbone NFE，只有在平均调用数仍远低于 CLLM 时才可作为最终方案。

## 0.3 Adaptive Stage A 实际结果

连续轨迹 oracle projector 和 1/2/3/4-call latent recurrence 已实现，并在当前 2k
tuned checkpoint 与严格匹配的 512-block validation cache 上完成验证。结果未通过
oracle gate：endpoint hidden error 从 call 1 的 `0.859063` 逐步恶化到 call 4 的
`0.870933`，token agreement 维持在约 `45.25%`，没有实质提升。

第一次输出的平均连续轨迹投影进度仅为 `0.027146`，projection distance 为
`0.500858`；重复调用后进度仍停留在约 `0.03`，而 distance 增至 `0.563718`。训练
split 上得到同方向结果，因此当前失败不是 learned-time 估计误差，而是 updater 输出
off-trajectory 且不具备稳定 self-composition。按 gate 规则，暂不训练 progress head，
不启用 rollout loss，也不升级 checkpoint；完整诊断见
[`ADAPTIVE_CDEQ_PLUS_RESEARCH_IDEA.md`](ADAPTIVE_CDEQ_PLUS_RESEARCH_IDEA.md)。

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
- 主目标仍训练每个 teacher state 一步指向 endpoint；重复调用先作为被严格验证的
  inference recurrence，必要时只增加 progress supervision 与 detached rollout
  exposure，不将监督目标改成 next point。

当前参考 loss 可以在概念上保留：

```text
loss = 0.1 * adjacent_ema_consistency
     + 0.9 * endpoint_regression.
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
N = 实际 Jacobi 轨迹长度
epsilon = 0.002
T = 5
rho = 7
```

在 LLM 轨迹中：

- 将 Jacobi 轨迹起点 `y0` 映射到 `epsilon=0.002`；
- 将收敛终点 `yK` 映射到 `T=5`；
- 使用现有 progressive rule 采样连续 `r < t`；
- 只在连续 hidden/logit states 之间进行插值；
- 保持 `t=T` 严格返回输入、一步推理从 `t=0` 调用的 boundary design。

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

### 5.8 Adaptive few-step refinement

单步输出不再被直接视为最终成功，而被视为一次 endpoint projection 的近似结果。
模型通过轻量 progress estimator 判断输出位于轨迹的等效进度，再用对应 `t` 做下一次
endpoint-targeted refinement。第一版限制 `max_calls in {1,2,3,4}`，并同时报告：

- output-to-teacher-trajectory distance，验证“输出落在某个中间 point 附近”的前提；
- oracle-time 与 learned-time 的差距；
- 每次调用后的 endpoint hidden error、token agreement 和 GSM8K accuracy；
- 平均、median、P95 adapter calls，以及 target-backbone NFE；
- stable、max-calls、cycle、EOS 等停止原因。

若 oracle-time 下多次调用仍不改善，则问题不在 time estimator，而在 updater 的
self-composition/off-manifold behavior；此时不得仅通过增加推理次数掩盖失败。

## 6. 第一版 LLM 实现的范围边界

### 6.1 在范围内

- 收集或复用 Jacobi trajectories；
- 提取 hidden/logit trajectories；
- 在模型维度需要时增加 input/output projection；
- 复用现有 CDEQ consistency updater；
- 复用现有 initializer；
- 复用现有 continuous-time schedule；
- 执行一步 endpoint mapping，并验证轨迹进度感知的 1--4 次自适应少步 refinement；
- 训练或校准轻量 progress estimator；
- 必要时加入 detached rollout exposure；
- 与 CLLM 做正式比较。

### 6.2 不在第一版范围内

- 把 CDEQ 重写成通用 Jacobi flow operator；
- 修改 LLM backbone architecture；
- 从头预训练或 instruction-tune 一个 LLM；
- 新的 speculative decoding 或 token verification；
- stochastic decoding；
- 无上限或无停止验证的重复 updater；
- online/on-policy distillation（离线 detached rollout augmentation 除外）；
- production serving engine；
- 在没有 progress 条件、停止准则或多步曲线验证的情况下重复调用 CDEQ updater。

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

### Phase 3.5：adaptive few-step feasibility

1. 不重训 updater，先在 held-out cache 上运行 oracle nearest-time 的 1/2/3/4-call
   latent recurrence；要求 endpoint error/token agreement 随调用数单调或近单调改善。
2. oracle 有效后训练 progress estimator，比较 oracle-time、learned-time 与固定 time
   schedule，并校准 normalized progress error 与置信度。
3. learned-time 出现 rollout gap 时，再加入 10%--20% detached student rollout
   augmentation；主监督仍为所有输入状态到 endpoint。
4. 比较 adapter-only recurrence 与 Abel canonical re-encoding；后者按每轮一次 target
   forward 计入 NFE 和 wall-clock。
5. 固定 `max_calls` 后评测 500 样本与完整 GSM8K，报告调用数分布、停止原因以及
   quality/latency Pareto。

成功信号：相对 one-shot 明显改善 endpoint/token 指标，learned-time 接近 oracle-time，
且平均与 P95 target-equivalent calls 显著小于常规 CLLM。若 oracle 多步不改善，停止
该方向并诊断 output-to-trajectory distance，不直接扩大训练或调用预算。

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

第一版已按该原则落地到 `CLLM-src/llm_cdeq/`，并把 mask、EOS、block
construction、state extraction、cache schema 和 checkpoint schema 隔离在官方
CLLM 源码之外。现有 DEQ/CDEQ 入口未被改写。

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
| student rollout 偏离 teacher manifold | 多次调用反而漂移或过冲 | 先做 oracle-time 诊断，再加入 detached rollout exposure |
| progress 过估计 | 过早到 `t=T` 后被 identity boundary 卡死 | 单调保守更新、置信下界、`T-delta` cap 与 max-calls |
| token 稳定但落在错误平台 | 错误早停 | 联合 progress 置信度与 hidden update norm，不单独依赖 token unchanged |
| canonical re-encoding 过贵 | 相对 CLLM 的加速消失 | 同时报 adapter-only 与 re-encoding NFE/latency |
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

- [x] 固定 LLM 只是 application section 的角色。
- [x] 选择第一个 CLLM-compatible model/task：GSM8K / Abel-7B / block 16。
- [x] 决定使用 final-layer shifted hidden states。
- [x] 确定 `4096→512→4096` bottleneck。
- [x] 固定 endpoint-targeted 一步训练定义。
- [x] 固定轨迹进度感知的 adaptive few-step 推理假设与验证顺序。

### 14.2 Implementation

- [x] 用 Git 分支与快照 commit 保护当前 CDEQ+ baseline。
- [ ] 复现 AR/Jacobi endpoint equivalence。
- [ ] 保存 deterministic Jacobi trajectories。
- [ ] 训练 CDEQ-Jacobi baseline。
- [ ] 训练全部四组 Init/CT ablation。
- [x] 运行 oracle-time 1/2/3/4-call recurrence gate（未通过）。
- [ ] 训练并校准 progress estimator（因 oracle gate 失败而暂停）。
- [ ] 仅在 rollout gap 出现时加入 detached rollout augmentation（当前不启用）。
- [ ] 在相同协议下运行 CLLM。
- [ ] 加入第二个任务。

### 14.3 Evaluation

- [ ] 报告 task quality 与 AR agreement。
- [ ] 报告 wall-clock latency 和 TPS。
- [ ] 报告 NFE/Jacobi iterations。
- [ ] 报告平均/median/P95 calls、停止原因和 learned-time/oracle-time gap。
- [x] 报告每轮 output-to-trajectory distance 与 endpoint 变化曲线（Stage A）。
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
