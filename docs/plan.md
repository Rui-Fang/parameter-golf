# 基于 Universal/Recursive Transformer 的量化友好循环模型设计计划（面向 parameter-golf）

## 1. 目标定义

这份计划的目标不是单纯做一个“更纯”的 Universal Transformer，而是设计一条**在 parameter-golf 约束下真正可提交、可训练、可压缩、可扩展**的循环模型路线。这里的核心约束是：

* **10 分钟 / 600s 训练窗口**
* **16MB artifact 上限**
* **shared-weight recurrence 带来的参数效率**
* **低比特压缩下跨 loop 误差累积与分布漂移**

目前 repo 中与 loop / recurrence 相关的经验基本已经说明：**循环模型的主矛盾不再只是 activation 是否低比特，而是同一组低比特 shared core 被重复调用时，cross-loop 的量化形状误差与误差传播会迅速放大。** 这与 `LoopQ` draft 中总结的两类瓶颈一致：一类是不同 loop 的激活统计和几何结构不一致，另一类是误差会沿 recurrent trajectory 递推放大。

---

## 2. 问题重述：真正要压的是什么误差

我建议把问题从“QAT 能不能让 shared weights 适应量化”提升为：

> **能否主动设计一组极小的 loop-wise 模块，让不同 loop 进入 shared low-bit core 之前，先被搬运到一个更统一、更量化友好的 canonical 形状上？**

这里要压的不是抽象的量化误差，而是更具体的：

### cross-loop quantization shape error

同一个 shared quantized block 在不同 loop 上看到的输入具有：

* 不同的尺度范围
* 不同的 dominant channel 排布
* 不同的 outlier 结构
* 不同的 boundary role

于是同一个量化配置无法同时兼顾所有 loop。`LoopQ` draft 在问题形式化部分已经把这一点说得很清楚：共享的旋转与共享的 clipping/scale 参数，无法同时匹配不同 loop 的统计结构；同时，早期 loop 引入的偏差会通过后续 loop 继续放大。

---

## 3. 总体设计原则

我建议用下面四条原则约束整条方案。

### 原则 A：前后边界 unique，中间 core recurrence

不要让同一组 shared block 同时承担三种角色：

* token → latent 的初始建模
* 中间 iterative refinement
* final readout / pre-logit decoding

这类功能异质性过强，会显著恶化 shared low-bit core 的量化鲁棒性。repo 里的多条相关提交都在不同程度上印证了这一点：#386 的深循环方案需要 `x0 residual mix` 与 skip 才能稳定；#927 依赖跨 loop skip、late ValueEmbedding 和 per-loop scales；#990 则直接采用了“flat encoder + shared crawler + flat decoder”的半共享结构。

### 原则 B：loop-wise 自由度只放在 control plane

不要给每个 loop 一套大 adapter。
应该只允许极小的 loop-wise 参数，例如：

* per-loop norm bias
* per-loop diagonal scale
* per-loop residual mix
* per-loop attn / mlp step scale
* late-loop value reinjection

#1088 的一个重要经验就是：**shared 主干不变，但 norm/bias 在不同深度保持独立，可以起到 depth embedding 的作用。**

### 原则 C：QAT 不是后补，而是 shared core 的训练先验

对于 loop 模型，QAT 不能只理解为“最后适配量化 artifact”；它必须从训练早期就介入 shared core 的动力系统学习。#927 明确指出 recursive model 的量化误差会 through loops compound，因此采用了 **int6 QAT from step 0**；#990 也明确发现 **more looping = worse quantization resilience**。

### 原则 D：优化目标不是“每个 loop 表示一样”，而是“每个 loop 在进入 shared low-bit core 前，对量化器而言更像”

这意味着我们需要一个**loop-wise canonicalization** 模块，而不是简单的 per-loop bias 拼贴。

---

## 4. 建议的主模型：Boundary-Unique Canonicalized Recurrent Transformer

### 4.1 核心结构

建议首版直接使用：

```text
Embedding + lexical helpers
→ 1 unique stem block
→ 3 shared core blocks × 4 loops
→ 2 unique tail blocks
→ final norm + unique lm_head
```

也就是：

* **1 个 unique stem**
* **3 个 shared core blocks**
* **4 次 loops**
* **2 个 unique tail**
* 最后 norm 与 lm_head 继续 unique

有效深度为 **15 blocks**，但物理存储的 block 数远少于常规 15L。这个配置兼顾了三件事：

1. 保留足够明显的 recurrence 红利
2. 不把所有功能都压给 pure shared UT
3. 在 600s 训练预算下，不至于因为循环过深导致 step 数掉得太厉害

这与 #857 的结论一致：更深的虚拟深度在同 step 下通常有利，但 wallclock 下未必能覆盖吞吐损失，因此 recurrence 深度需要克制。

---

## 5. 本方案的核心创新：Loop-wise Canonicalization under QAT

### 5.1 基本思想

shared core 应该始终在一个**canonical quantization frame** 中工作。
因此每个 loop 在进入 shared core 之前，先做一次极小的 loop-wise 搬运；shared core 输出后，再映回 recurrent latent space。

数学上可以写成：

[
\tilde h_t = T_t(\mathrm{Norm}(h_t)) + b_t
]

[
u_t = F_{\text{shared}}^{\text{QAT}}(\tilde h_t)
]

[
h_{t+1} = \alpha_t h_t + \beta_t h_0 + \gamma_t T_t^{-1}(u_t) + \delta_t v_t
]

其中：

* (F_{\text{shared}}^{\text{QAT}})：共享低比特 core
* (T_t)：第 (t) 个 loop 的小型 canonicalizer
* (h_0)：stem 之后的 base state
* (v_t)：可选的 late-loop token/value reinjection
* (\alpha_t,\beta_t,\gamma_t,\delta_t)：每个 loop 的小尺度系数

### 5.2 为什么这比普通 per-loop bias 更强

普通的 per-loop bias / scale 只能缓解幅值问题，无法系统修正：

* channel 重要性排序
* outlier 位置迁移
* shared quantizer 对不同 loop 的轴不匹配

而 (T_t) 的作用是让不同 loop 的 hidden state 在进入 shared low-bit core 前，被搬运到一个更统一的表示坐标系。这样 shared core 学到的不是“兼容所有 loop 的折中动力学”，而是“canonical frame 中的稳定动力学”。

这和 `LoopQ` 草稿中的 **trajectory-aware range correction** 与 **sparse geometry correction** 在思想上一致，但这里建议把它们前移到 **QAT 训练期**，并做成闭环的前后配对模块，而不是只作为 PTQ 阶段的局部修补。

---

## 6. canonicalization 模块的参数化方式

首版不要复杂，采用三层逐级增强设计。

### 6.1 第一层：Per-loop diagonal canonicalizer

[
T_t^{(0)}(x)=D_t x
]

其中 (D_t) 是 per-loop 的对角缩放向量。

作用：

* 解决主要的 range inconsistency
* 让不同 loop 的 activation RMS / clipping 区间更接近
* 参数极少
* 容易 identity 初始化

这是最值得先做的部分。

### 6.2 第二层：Per-loop norm bias

在 pre-norm 后给每个 loop 一个独立 bias，或少量 gain+bias。
这个设计已经在 #1088 中被证明有价值，作者明确指出 pre-norm bias 实际上发挥了类似 depth embedding 的作用。

作用：

* 给 shared core 注入 loop identity
* 修正轻量几何偏移
* 改善不同 loop 之间的角色分化

### 6.3 第三层：Sparse geometry exceptions

只在少量高敏感模块中，使用若干个共享 support 的 Givens 旋转或 2×2 channel mixers：

[
R_t=\prod_r G(p_r,q_r,\theta_{t,r})
]

注意：

* **不应全模块铺开**
* 只给最敏感的 shared core 入口或晚期 loop 使用
* 应是第二阶段 ablation，而不是首版硬上

这对应 `LoopQ` 中的 sparse geometry correction 思想，但要严格受 budget 约束。

---

## 7. 必须继承的 repo tricks（按优先级排序）

下面列出我认为**应纳入主方案**的 repo trick，并标明来源。

### 7.1 `x0 residual mix`

**来源：#386, #927**
作用：把 stem/base state 持续注入后续 loop，抑制 recurrent drift，防止 shared core 在深循环中偏离初始语义锚点。#386 把它作为深度循环稳定器之一；#927 也保留了 `resid_mixes`。

### 7.2 `per-loop norm bias / depth embedding`

**来源：#1088**
作用：在保持 shared 主干不变的前提下，为不同 loop 提供轻量阶段身份；同时帮助不同深度的 shared block 输入分布更可分。

### 7.3 `per-loop attn scale / mlp scale`

**来源：#927**
作用：把每个 loop 看成不同步长的迭代器，而不是完全同分布的展开层。可显著改善 early / middle / late loops 的功能分工。

### 7.4 `late ValueEmbedding / token reinjection`

**来源：#927**
作用：在深循环后防止表示过度平滑，仅在最后 1–2 个 loops 重新注入 token identity。#927 将其作为 late-loop 关键组件之一。

### 7.5 `deduplicate before quantization`

**来源：#857**
作用：共享块只存一份，再用映射重建；这是 recurrence 模型在 artifact 上获得直接收益的必要手段。#857 明确把它列为关键技巧。

### 7.6 `progressive depth schedule`

**来源：#1088, #895**
作用：先用较浅循环训练，再逐步增加循环深度；既减少早期训练不稳定，也提高 wallclock 利用率。#1088 在 10 分钟 track 中使用了 `NUM_LAYER_SCHEDULE`；#895 则在 4 小时研究中用 2→3→4→5 progressive depth 展示了 recurrence scaling 的不同性质。

### 7.7 `int6 QAT from step 0`（或非常早启动）

**来源：#927**
作用：shared core 在训练初期就适应 recurrent quantization noise，而不是后期被动修补。#927 明确指出 recursive model 中量化误差会沿 loops 累积，因此从 step 0 启用了 int6 QAT。

### 7.8 `BigramHash`

**来源：#857, #927**
作用：极低成本的 lexical shortcut，减轻 shared core 自己学习全部局部模式的压力。对小模型/短训练预算尤其划算。

### 7.9 `SmearGate`

**来源：#857, #927**
作用：低成本 token continuity / local history mixing，对小模型常有正收益。

### 7.10 `LeakyReLU² / ReLU² core MLP`

**来源：#857**
作用：在 shared core 中尽量避免标准 GLU 的 spike amplification。#857 用了 `LeakyReLU(0.5)^2`；而 `LoopQ` draft 也指出 GLU 类模型在 loop/quant 场景下边界更敏感。

---

## 8. 不建议首版直接继承的东西

### 8.1 `pure shared block × 很多 loops`

#386 的纯 shared block × 12 很有研究价值，但首版不建议直接照搬。原因是 pure shared block 必须兼任 embedding formation、latent refinement、late readout 三种角色，量化压力过大。

### 8.2 大规模 loop-specific adapter

这会直接侵蚀参数共享的核心收益，也容易让 artifact 超支。

### 8.3 首版就依赖复杂 eval stack

例如 TTT、Hedge Mixer、复杂 score-first 评估。#927 和 #895 都显示这些东西很强，但首版应先把 base architecture 做稳，再考虑评测侧技巧。

---

## 9. 量化设计

### 9.1 精度分配

建议采用分区量化策略：

* **shared core 大矩阵**：`int6 QAT`
* **unique stem / unique tail / lm_head**：`int8`
* **loop-wise control plane**：`FP32`
* **小 passthrough 张量**：按 artifact 开销选择 `fp16` 或 `fp32`

理由：

* shared core 被重复调用，误差会累计，所以必须在训练中适配低比特
* boundary unique 模块不重复累积，保持 int8 更稳
* control plane 很小，不值得为了几 KB 再压低精度

这与 #927 的核心思路以及 #990 对 crawler 单独提高精度缓解量化脆弱性的经验是一致的。

### 9.2 量化阶段顺序

建议：

1. shared core 从 step 0 或极早期进入 fake quant
2. unique boundary 模块先全精训练，再在后半段轻量感知量化
3. 导出前做 dedup + int6/int8 export + zstd/zlib 压缩

---

## 10. 训练设计

### 10.1 Progressive depth

推荐 schedule：

```text
step 0–1500:   2 loops
step 1500–3500: 3 loops
step 3500+:    4 loops
```

所有 depth 在 warmup/priming 阶段提前 compile，避免切换时 recompilation。#1088 已经证明这个工程技巧很关键。

### 10.2 QAT schedule

建议首版直接：

* shared core：`int6 fake quant from step 0`
* unique tail/head：在训练后 30% 开始轻量 QAT 或直接 PTQ
* control plane：不量化

### 10.3 优化器与状态平均

建议保留：

* matrix params 用 Muon 或类似矩阵优化器
* control / scalar params 用 AdamW/Adam
* EMA 开启
* SWA 只在尾段稀疏收集少量 checkpoint

#895 说明大规模 SWA 很强，但在 600s 设定下不应照搬其重型版本。

---

## 11. 显式正则：让 canonicalization 真正朝“统一量化形状”优化

如果只用 CE loss，loop-wise canonicalizer 也会学，但方向不一定稳定。建议增加两类便宜的辅助目标。

### 11.1 Cross-loop stationarity loss

对进入 shared core 前的 (\tilde h_t) 统计：

* per-channel RMS
* groupwise second moment
* clipping ratio
* top-outlier channel rank / overlap

最小化这些统计在 loop 间的 spread。
目标不是让表示相同，而是让 shared quantizer 看到的输入分布更统一。

### 11.2 Identity / smoothness regularization

约束：

* (D_t) 接近 1
* bias 接近 0
* 相邻 loops 的 canonicalizer 变化平滑
* Givens 角度小而稀疏

这样可避免 loop-wise 模块“演变成小 adapter 网络”。

---

## 12. 最小可行模型（MVP）

如果要快速验证路线，不建议一次做满。MVP 版本建议如下。

### 结构

* 1 unique stem
* 3 shared blocks × 4 loops
* 2 unique tail
* final norm + lm_head unique

### loop-wise control plane 只包含

* per-loop norm bias
* per-loop diagonal scale
* per-loop residual mix with (h_0)
* per-loop attn/mlp scales
* late VE on last 2 loops

### lexical / low-cost helper

* BigramHash
* SmearGate

### 量化

* shared core int6 QAT
* boundary / head int8
* control tensors FP32

这是最小闭环，不应再加大的 loop adapter。

---

## 13. 实验顺序（推荐 ablation 路线）

### 阶段 1：建立 baseline

先做：

* Boundary-Unique + Shared Core
* 不加 canonicalizer
* 只保留 `x0 residual mix + per-loop scale + BigramHash + SmearGate`

目标：确认 semi-shared 架构本身比 pure shared UT 更稳。

### 阶段 2：加入最小 canonicalization

依次 ablate：

1. 无 canonicalizer
2. 只有 per-loop norm bias
3. norm bias + diagonal scale
4. 再加 residual mix coupling
5. 再加 late VE

目标：判断真正的增益来自哪里。
我预测 **norm bias + diagonal scale + residual mix** 会是主增益来源。

### 阶段 3：再尝试 sparse geometry exceptions

仅在最敏感模块试：

* shared core attn input
* shared core mlp input
* 最后 1–2 loops 的 value path

只给极小 budget。若收益有限，应直接砍掉。

### 阶段 4：压缩与 artifact

比较：

* int6+zstd
* int6+zlib
* control tensor fp16 vs fp32 store
* dedup map 方案

---

## 14. 预期收益

如果这条路线成立，我预期它带来的收益不是来自“增加很多表达能力”，而是来自三方面：

1. **shared core 更容易训练**
   因为它始终在更统一的 quantization frame 中工作

2. **cross-loop 量化形状误差更小**
   剪裁区间、离群通道位置、通道重要性排序在不同 loop 间更一致

3. **循环深度上限被抬高**
   也就是在相同 artifact 预算下，允许更多有效深度而不至于量化崩掉

---

## 15. 主要风险与失败模式

### 风险 1：control plane 演化成隐式 adapter

若正则不够，per-loop 模块可能承担过多建模功能，侵蚀 shared core 的意义。
解决：加强 identity / smoothness regularization。

### 风险 2：canonicalizer 只学会调尺度，不学几何

这不一定是坏事。若 diagonal scale 已经覆盖大部分收益，说明 shape mismatch 的主矛盾主要在 range drift，而不是更复杂的旋转不匹配。此时不必强上 geometry exceptions。

### 风险 3：loops 太深导致吞吐掉得太多

这在 #857 已经被明确观察到。解决方法不是盲目压更多 loops，而是保持 3×4 或 2×4/5 这种中等深度。

### 风险 4：QAT 让训练太脆

如果从 step 0 的 int6 QAT 太难训，可以只对 shared core 的最关键矩阵先开 fake quant，其余延后数百步。但原则上仍应尽早引入。

---

## 16. 最终推荐配置（一句话版本）

> **1 unique stem + 3 shared blocks × 4 loops + 2 unique tail**
> **shared core 使用 int6 QAT，从训练早期开始；前后边界保持 int8；loop-wise control plane 仅包含 norm bias、diagonal canonical scale、residual mix、attn/mlp scales 与 late VE；再叠加 BigramHash、SmearGate、dedup-before-quant 与 progressive depth schedule。**

这条路线吸收了 repo 中最有价值的 recurrence 经验：

* #386 的 `x0 residual mix` 与 shared depth 稳定性思路 
* #1088 的 `per-depth norm bias` 与 `depth schedule` 
* #927 的 `int6 QAT from step 0`、`per-loop scales`、`late VE` 
* #857 的 `deduplicate before quantization`、`LeakyReLU²`、`BigramHash`、`SmearGate` 
* #990 的 `boundary unique + middle shared` 结构直觉，以及“更多 loops 更怕后处理量化”的警告 
* `LoopQ` draft 中关于 cross-loop 统计漂移与误差传播的形式化问题分解 

---

## 17. 结论

这条路线最重要的判断不是“再加更多 loop-wise 参数”，而是：

> **QAT 应该与极小的 loop-wise canonicalization 联合设计，让不同 loop 在进入 shared low-bit core 前，被搬运到更统一的量化形状。**

如果这个判断成立，那么 shared recurrence 的下一步就不再是“继续纯共享”或“继续堆 PTQ trick”，而是：

* **边界功能 unique**
* **中间动力学 shared**
* **loop identity 只在 control plane 中表达**
* **shared core 在 canonical quantization frame 中学习稳定迭代**

这是一条兼顾研究价值与 parameter-golf 可提交性的路线。
