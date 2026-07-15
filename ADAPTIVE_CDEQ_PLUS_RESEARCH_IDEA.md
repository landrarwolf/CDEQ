# 轨迹进度感知的自适应 CDEQ+：研究设想与验证计划

> 文档性质：待验证的研究 idea，不代表当前 CDEQ+ 已经成功。
>
> 当前基线：官方 CLLM 已跑通；现有 CDEQ+-Jacobi feasibility 未通过预设门槛。
>
> 目标：在保持“所有轨迹节点一步指向 endpoint”训练语义的同时，允许真实推理以
> 少量自适应调用逐步逼近 endpoint，并使调用次数显著少于常规 CLLM。

## 1. Idea 名称与一句话定义

建议名称：

**轨迹进度感知的自适应少步一致性细化**

英文工作名：

**Trajectory-Progress-Aware Adaptive Few-Step Consistency Refinement**

一句话定义：

> CDEQ+ 训练时仍要求每个教师轨迹节点一步预测最终 endpoint；推理时不强制一次近似
> 调用必须精确到达 endpoint，而是估计当前输出在教师轨迹上的等效进度，用相应时间
> 条件再次执行少量 endpoint-targeted refinement，直到稳定或达到调用预算。

这不是将 updater 改成 next-point predictor，也不是恢复原始 Jacobi 的逐点迭代。
每一次 updater 调用的目标始终是最终 endpoint。

## 2. 直观例子

假设教师轨迹有 30 个状态：

```text
point 0, point 1, ..., point 29
```

训练仍要求：

```text
F(point k, t_k) -> point 29,  for every k in [0,29].
```

真实推理从 point 0 开始。第一次调用虽然以 point 29 为目标，但受模型容量和优化误差
限制，输出可能只落在 point 15 附近。此时不直接接受结果，而是：

1. 估计当前输出的归一化进度为 `p ~= 15/29`；
2. 使用 rho schedule 将 `p` 转为对应的时间 `t`；
3. 以该 `t` 再调用同一个 endpoint predictor；
4. 重复少量几次，直到足够稳定或达到 `max_calls`。

当前时间约定为 `epsilon=0.002, T=5, rho=7`：

```text
t(p) = (
    epsilon^(1/rho)
    + p * (T^(1/rho) - epsilon^(1/rho))
)^rho.
```

在该例中，`p=15/29` 对应 `t ~= 0.3196`。由于 rho grid 是非线性的，point 15 的
时间不是 `2.5`。当前 GSM8K cache 最多包含 17 个状态；30 个状态只用于解释一般思想。

## 3. 数学形式

设教师轨迹为：

```text
s_0, s_1, ..., s_K,
t_0=epsilon < ... < t_K=T,
```

其中 `s_K` 是 endpoint representation。现有 CDEQ+ updater 为：

```text
F_theta(s, t) -> endpoint approximation.
```

训练主目标保持为：

```text
L_endpoint = d(F_theta(s_k, t_k), s_K).
```

自适应推理定义为：

```text
z_0 = Init(s_0)                   # initializer 仅执行一次
z_1 = F_theta(z_0, 0)

for m = 1, ..., M-1:
    p_hat_m = H_phi(z_m)
    t_hat_m = rho_time(p_hat_m)
    z_(m+1) = F_theta(z_m, t_hat_m)
```

`H_phi` 是轻量 progress estimator，预测 `[0,1]` 上的等效轨迹进度，而不是预测
endpoint。最终通过冻结 LM head 做 greedy token decoding。

## 4. 核心假设与可证伪条件

该 idea 成立需要以下假设：

1. 单次 updater 的不完美输出仍落在教师轨迹或其邻域，而不是任意 off-manifold 区域。
2. 使用输出对应的等效 `t` 再次调用 updater，可以继续降低 endpoint error。
3. 少量重复调用不会系统性过冲、震荡或退化。
4. progress estimator 能以远低于 backbone/updater 的开销近似 oracle time。
5. 达到目标质量所需的平均和 P95 调用次数显著少于常规 CLLM。

最关键的证伪条件是：即使使用真实教师轨迹提供 oracle time，2/3/4 次调用仍不能
持续改善 endpoint hidden error 或 token agreement。若出现这种情况，问题不是进度
估计器，而是 updater 不具备可重复组合的动力学性质；应停止该方向，而不是增加调用数。

## 5. 当前训练是否需要修改

### 5.1 不需要改变的部分

以下部分保持不变：

- 所有 teacher states 均直接监督到 endpoint；
- `0.1 x adjacent EMA consistency + 0.9 x endpoint regression` 主损失；
- initializer 直接监督到 endpoint，并在进入 updater 前 detach；
- CT 仍对教师 hidden trajectory 做连续插值；
- `t=T` 仍为严格 identity boundary；
- target LLM 与 LM head 全部冻结。

因此，不应把训练目标改成：

```text
F(point k, t_k) -> point k+1.
```

否则会退回逐步模拟教师 solver，偏离 CDEQ 的 endpoint consistency 语义。

### 5.2 最小新增训练

可部署版本至少需要训练或校准 progress estimator：

```text
H_phi(s_k) -> p_k = k/K.
```

优先预测 normalized progress `p`，再通过固定 rho mapping 转换为 `t`，以适应不同
轨迹长度。可以比较 regression、ordinal bins 以及 regression+ordinal 的组合。

progress head 可复用 `down(s)` 后的表示，做 masked pooling，再输出标量或有序区间。
其参数量和 latency 必须单独报告。

### 5.3 条件式新增训练

现有 updater 只见过教师状态和教师状态之间的连续插值。第二次推理的输入是学生自己
产生的 `z_1`，可能存在 exposure bias。只有在 oracle recurrence 有效、learned-time
rollout 仍出现明显退化时，才加入 detached rollout augmentation：

```text
z_student = stopgrad(F_theta(s_k, t_k))
p_oracle  = nearest_teacher_progress(z_student)
L_rollout = d(F_theta(z_student, rho_time(p_oracle)), s_K).
```

要求：

- 学生自产状态仍监督到相同 endpoint；
- 不跨离散 argmax 反传整条 rollout；
- rollout 样本先限制在训练 batch 的 10%--20%；
- progress head 可先训练并冻结，避免 updater 与 progress head 相互欺骗；
- 单独消融 rollout augmentation 的收益与开销。

## 6. 如何判断当前位于哪个 point

### 6.1 Oracle nearest-time：只用于研究诊断

在有完整教师轨迹的 held-out cache 上，计算学生输出与各教师状态的 masked normalized
hidden distance，选择最近 point 对应的 `p/t`。

优点是直接回答“第一次输出是否真的落在 point 15 附近”。缺点是线上若生成完整教师
轨迹再做最近邻，会完全抵消加速，因此 oracle 不得作为正式推理器。

### 6.2 Learned progress head：正式候选

用缓存中的 `(s_k,p_k)` 训练轻量 `H_phi`。正式评测至少报告：

- normalized progress MAE；
- ordinal/bin accuracy；
- calibration error 与置信度；
- learned-time 相对 oracle-time 的 endpoint 指标差距；
- head 参数量与推理 latency。

由于一个 Jacobi block 内不同 token 可能异步收敛，第一版先预测 block-level progress；
若不稳定，再评估 per-token progress，并以慢 token 的保守分位数决定全局 `t`。

### 6.3 Token stability：适合停止，不足以单独估计时间

连续两轮 token agreement、stable prefix、logit margin 和 EOS 状态可以作为 progress
head 特征或停止信号，但不能单独证明已经接近正确 endpoint。错误输出也可能形成稳定
平台，因此 token unchanged 不能作为唯一停止条件。

### 6.4 Fixed schedule：必要对照组

固定的 2/3/4-call time schedule 不需要 progress head，应作为简单 baseline。只有
learned-time 稳定优于 fixed schedule，才能说明轨迹进度感知本身产生了价值。

## 7. 两种推理路径

### 7.1 Adapter-only latent recurrence

直接将 `z_m` 送回轻量 updater：

```text
z_(m+1) = F_theta(z_m, t_hat_m).
```

优点：

- 不增加 target LLM forward；
- 最符合“大幅降低相对 CLLM 推理调用数”的目标；
- adapter 额外调用通常远低于 7B backbone 成本。

风险：

- `z_m` 可能偏离冻结 LLM 的 canonical hidden manifold；
- hidden error 可能下降但 token 质量不改善；
- 重复调用可能过冲、震荡或 collapse。

该路径应作为第一候选，但必须通过 output-to-trajectory distance 和多步单调性验证。

### 7.2 Canonical re-encoding fallback

每轮把预测 hidden 经 LM head 解码为 greedy tokens，再由冻结 Abel 重新编码成
canonical shifted hidden，之后调用 updater。

优点：每一轮都重新投回目标 LLM 的有效 representation manifold。缺点是每轮增加一次
backbone forward。必须将 target-backbone NFE、adapter NFE 和 wall-clock 分开报告。

只有当所需轮数仍显著少于常规 CLLM，并且质量收益明显时，该路径才可作为最终方案；
否则仅作为诊断和消融。

## 8. 时间更新和停止规则

时间更新必须满足：

```text
p_(m+1) >= p_m.
```

但不应无条件强制大幅前进。progress 过估计会提前到 `t=T`，而 `t=T` 是 identity
boundary，可能让错误状态永久卡住。推荐：

- 使用 progress 置信下界或保守分位数；
- 限制单次最大进度增量；
- 未验证稳定前将时间 cap 在 `T-delta`；
- 检测二周期或多周期 token/hidden oscillation；
- 设置硬上限 `max_calls in {1,2,3,4}`。

建议停止需要联合满足多个条件：

1. progress/end-bin 置信度足够高；
2. 连续 greedy token 稳定；
3. masked relative hidden update norm 足够小。

达到 `max_calls` 时输出当前最好结果，同时记录为 budget stop，不应伪装为收敛成功。

## 9. 分阶段实验

### Stage A：无需重训的 oracle gate

- 使用现有 one-step checkpoint；
- 在 held-out cache 上测试 1/2/3/4-call latent recurrence；
- 每轮用教师轨迹 nearest point 提供 oracle `p/t`；
- 记录 endpoint error、token agreement、output-to-trajectory distance 和 cycle/collapse；
- 同时运行 fixed schedule。

通过条件：相对 one-shot，至少一个少步预算显著改善 endpoint/token 指标，且总体趋势
单调或近单调。若 oracle 不通过，停止后续 progress-head 工作。

### Stage B：learned progress

- 在缓存 teacher states 上训练 progress head；
- 比较 learned-time、oracle-time 与 fixed schedule；
- 校准置信度和时间单调更新；
- 测量 progress head latency。

通过条件：learned-time 接近 oracle-time，并稳定优于或不劣于 fixed schedule。

### Stage C：rollout exposure

仅在 Stage A 通过、Stage B 出现 student rollout gap 时执行：

- 生成 1--3 round detached student states；
- 使用训练期 oracle 或冻结 progress head 赋予时间；
- 仍以 endpoint 作为唯一全局目标；
- 搜索 rollout ratio `{0.1,0.2}`；
- 比较是否恢复多步单调性。

### Stage D：真实 GSM8K 与 CLLM 对比

- `max_calls={1,2,3,4}` quality/latency Pareto；
- adapter-only 与 canonical re-encoding；
- 500 样本 speed profile；
- 完整 GSM8K accuracy；
- 相同硬件、attention backend、block size 和 greedy decoding；
- 与官方 CLLM 的实际调用数分布直接比较。

## 10. 评价指标与成功标准

表示和 token 指标：

- endpoint relative hidden error；
- endpoint token agreement；
- output-to-teacher-trajectory distance；
- 每轮指标改善量；
- AR greedy endpoint agreement；
- GSM8K exact match。

进度估计指标：

- normalized progress MAE；
- ordinal accuracy；
- calibration error；
- learned-time/oracle-time gap。

推理效率指标：

- mean/median/P95 adapter calls；
- mean/median/P95 target-backbone NFE；
- end-to-end latency 和 tokens/s；
- progress head latency；
- stop-reason distribution；
- cycle、repetition 和 EOS collapse rate。

最终成功条件应同时包含：

1. adaptive few-step 明显优于 one-shot CDEQ+；
2. learned-time 的方向与 oracle-time 一致；
3. Init 和 CT 的主要结论不被多步机制反转或掩盖；
4. 平均与 P95 target-equivalent calls 显著小于常规 CLLM；
5. wall-clock 确实形成更优 quality/latency Pareto，而不只是在口径上减少 steps。

## 11. 建议实现位置

在 oracle gate 通过前，不修改 checkpoint schema 或正式训练入口。第一阶段优先新增
独立诊断脚本，复用现有 cache 和 checkpoint：

```text
CLLM-src/llm_cdeq/adaptive.py       # recurrence、oracle time、stop state
CLLM-src/llm_cdeq/evaluate.py       # 1/2/3/4-call metrics
CLLM-src/llm_cdeq/analyze.py        # per-call curves
CLLM-src/tests/llm_cdeq/            # monotonic time、T cap、stop、Init once
```

Stage A 通过后再加入：

```text
CLLM-src/llm_cdeq/progress.py       # progress head 与 loss
CLLM-src/llm_cdeq/train.py          # progress supervision；可选 rollout exposure
CLLM-src/llm_cdeq/model.py          # progress module/checkpoint package
CLLM-src/llm_cdeq/profile.py        # calls/NFE/latency Pareto
```

新增测试至少覆盖：

- initializer 只执行一次；
- progress/time 单调且不越界；
- 未验证 endpoint 时不会直接进入 `t=T` identity trap；
- stable、budget、cycle 和 EOS 停止原因；
- oracle nearest-time mask/EOS 正确；
- rollout state detach；
- backbone checksum 不变；
- checkpoint round-trip 和旧 checkpoint 兼容。

## 12. 需要进一步讨论的研究问题

1. progress 应预测 block-level、per-token，还是两者结合？
2. hidden nearest distance 应使用 normalized L2、cosine，还是 LM-head-aware metric？
3. 纯 latent recurrence 是否足够保持在 teacher manifold 附近？
4. 是否需要显式的 idempotence、contraction 或 monotonic-progress regularization？
5. progress head 应独立训练、冻结后训练 updater，还是联合训练？
6. fixed schedule 是否已经足够，learned progress 是否真正必要？
7. 如何定义“显著少于 CLLM”：平均调用数、P95、wall-clock，还是三者同时约束？
8. adaptive refinement 是否会改变 Init/CT 原本的消融解释？

## 13. 当前结论

现阶段最合理的判断是：

- CDEQ 的 all-states-to-endpoint 主训练目标不需要改变；
- “判断当前到了哪个 point”不是一个免费的 if 判断，而是需要 oracle 诊断后再训练或
  校准的 progress estimation 问题；
- 重复 updater 是否有效必须先用 oracle-time 实验证明；
- rollout augmentation 是条件式补救，不应在核心假设尚未成立前加入；
- 研究目标从“严格一步到 endpoint”调整为“用显著少于 CLLM 的少量自适应调用达到
  更好的 quality/latency Pareto”。

因此下一步不是立即重训，而是先实现 Stage A 的 oracle 1/2/3/4-call gate。
