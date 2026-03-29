# Plan B: 基于 #927 框架的 Boundary-Unique / Core-Recurrent 重构计划

## 1. 结论先行

Plan B 现在不再是“先冻结 #927 原结构，再把 canonicalization 当作小补丁验证”的保守路线。

新的 Plan B 改成下面这条主线：

1. **继续使用 #927 的训练、量化、导出、TTT 与 sliding eval 框架。**
2. **模型骨架改成“前边界 unique + 中间 core recurrence + 后边界 unique”。**
3. **encoder/decoder split + U-Net skip 不再作为新主架构的一部分。**
4. **canonicalization 和 QAT 从第一版开始联动设计，不再拆成两个独立问题。**
5. **ablation 放到主线跑通之后，不再作为第一阶段阻塞项。**

换句话说，Plan B 的目标不再是“保留 #927 结构，只做 control-plane 增量”，而是：

> **以 #927 为工程框架，直接重写成 boundary-unique / core-recurrent 架构，并让 canonicalization 成为这个新 core 的量化友好接口，而不是事后附着的修饰模块。**

---

## 2. 这次为什么要直接改架构

上一版 Plan B 的出发点是降低变量数，这个判断在“只验证最小 canonicalizer”时是合理的；但如果目标已经明确转向“前后边界 unique，中间 core recurrence”，那么继续把 #927 的 encoder/decoder split 视作必须保留的稳定骨架，反而会妨碍设计落地。

原因有四个。

### 2.1 #927 的 split + skip 是为它自己的递归拓扑服务的

#927 的主体是：

- 4 个 shared blocks
- 7 次 loop
- 前 3 次 loop 视作 encoder
- 后 4 次 loop 视作 decoder
- 中间通过 U-Net skip 把早期激活重新注入晚期 loop

这套设计的目的，是在“所有 block 都共享、而且 loop 单调向前”的前提下，给后半程补一个回看早期特征的通道。

但我们现在要做的不是“保留这个拓扑再局部微调”，而是把职责重新拆成：

- 前边界负责进入 recurrent core 前的表示整形
- 中间 core 负责主要的循环计算
- 后边界负责从 recurrent core 退出并形成读出前表征

一旦边界已经 unique，split + skip 的原始职责就和边界层的职责重叠了。

### 2.2 canonicalization 需要一个更单纯的 core 状态空间

如果保留 encoder/decoder split + U-Net skip，那么 recurrent core 在不同 loop 上看到的输入会同时混入：

- 当前状态
- x0 锚点
- 早期 encoder 激活的回注
- late VE 注入
- per-loop control

这会让“canonicalization 到底在对齐什么分布”变得不清楚。对 canonicalization 来说，最理想的对象不是一个不断被 skip stack 打断的双阶段轨迹，而是一个**单调推进、统计目标更一致的 core hidden state**。

### 2.3 boundary unique 本身就承担了“角色分工”

#990 给出的最重要启发不是某个具体实现细节，而是：

- unique boundary 与 shared core 要分工
- 不要让 shared core 同时承担所有角色

如果我们已经接受这个判断，那么最自然的下一步就是：

- 把“输入准备”和“输出整理”交给 unique boundary
- 把“反复迭代的主计算”留给 shared core

在这个前提下，再保留 encoder/decoder split，实际上是在同一个模型里并列维持两套结构性归纳偏置。

### 2.4 ablation 已经明确后置

这次的要求已经明确：

- QAT 还是要和 canonicalization 结合
- ablation 可以事后做

因此本轮 Plan B 不再要求一开始就把“新架构是否优于旧架构”和“canonicalization 是否有效”完全拆开。

主线优先级改成：

1. 先把 boundary/core 新骨架做对
2. 再把 canonicalization + QAT 接到这个骨架上
3. 最后再做回退式 ablation 去验证哪些部件真的必要

---

## 3. 新的主架构定义

### 3.1 总体原则

新的 Plan B 采用：

- **前边界 unique**
- **中间 core recurrence**
- **后边界 unique**

并保留 #927 已经成熟的外层工程框架：

- 训练循环
- Muon/AdamW 优化器划分
- int6 QAT 路径
- quant/export 路径
- sliding eval
- score-first TTT
- BigramHash / SmearGate / late VE / XSA 这些已验证模块

### 3.2 推荐的第一版参数银行划分

直接沿用 #927 的 4 个参数银行数量，但改变角色分工：

1. `front_boundary_block`：unique
2. `core_block_a`：shared
3. `core_block_b`：shared
4. `back_boundary_block`：unique

也就是把原来 `4 shared blocks` 改成：

- 2 个 unique boundary blocks
- 2 个 shared core blocks

### 3.3 推荐的执行次序

在保持总有效深度仍接近 #927 的前提下，第一版推荐：

- 1 次 front boundary
- 5 次 core loop
- 1 次 back boundary

即总共仍是 7 次主计算，但中间 5 次只在 core 内循环。

更具体地说，可以先采用：

- step 0: front boundary
- step 1..5: `core_block_a` / `core_block_b` 交替 recurrence
- step 6: back boundary

这样做有两个优点：

- 总时延和 #927 同量级，方便沿用 10 分钟预算
- unique 与 shared 的职责切分是清楚的，不需要再靠 encoder/decoder 语义来解释 loop

### 3.4 是否保留 encoder/decoder split + U-Net skip

**Plan B 的答案是：可以去掉，而且本版建议直接去掉。**

理由不是“skip 一定无用”，而是它已经不再是这个新架构的最合适稳定器。

去掉的原因：

1. boundary unique 已经提供了前后职责分工，skip 的结构性价值下降
2. skip 会把早期状态再次注入 core，破坏 core hidden state 的单一统计目标
3. canonicalization 想约束的是 recurrent core 的分布，而不是“core + skip 回注”的混合分布
4. 量化上，skip 回注会增加跨阶段幅度突变，不利于 core 的低比特稳定性

这并不意味着完全不保留任何长程锚点。Plan B 只是不再使用：

- `num_encoder_loops`
- `num_decoder_loops`
- `skip_weights`
- `skips.append / skips.pop` 这条 U-Net 路径

替代稳定器改成：

- `x0` residual anchor
- boundary-to-core 的显式 canonicalization
- per-loop control tensors
- 更保守的 resid mix 初始化

如果后续发现删掉 skip 后训练明显不稳，再单独考虑加一个**轻量 boundary bridge**，而不是恢复整套 encoder/decoder U-Net 语义。

---

## 4. canonicalization 和 QAT 的新定位

### 4.1 canonicalization 不是“小修正”，而是 boundary/core 接口的一部分

在新的 Plan B 里，canonicalization 不再只是“shared block 入口上的一个可选 adapter”。

它应该承担两个职责：

1. **front boundary 输出到 recurrent core 输入的对齐**
2. **core 内不同 loop 之间的状态规范化**

也就是说，canonicalization 是 boundary/core 架构的一部分，而不是后挂在 #927 上的附属品。

### 4.2 推荐拆成两层 canonicalization

第一版推荐把 canonicalization 拆成两类模块：

1. `BoundaryCanonicalizer`
   - 放在 front boundary 之后、进入 first core step 之前
   - 负责把边界输出映射到更适合 recurrent core 循环的状态空间

2. `LoopCanonicalizer`
   - 放在每一次 core loop 进入 shared block 之前
   - 负责让 core 内部跨 loop 的状态分布更稳定

可选地，若需要，也可以加：

3. `ExitCanonicalizer`
   - 放在最后一次 core 输出进入 back boundary 之前
   - 负责把 core latent frame 映射回更适合读出的边界空间

但这第三层应当是次优先级；第一版先保证 boundary-in 和 loop-wise 两层成立。

### 4.3 canonicalization 必须和 QAT 联动

这一轮不接受“先做 canonicalization，后面再看 QAT 怎么接”。

Plan B 的要求是：

- core recurrence 从第一版开始就工作在 QAT 约束下
- canonicalization 的目标就是改善这个 QAT 条件下的 recurrent stability

因此，建议如下：

1. **shared core 的大矩阵继续走 int6 QAT 路径**
2. **boundary blocks 的大矩阵也继续走同一套 fake-quant 机制**
3. **canonicalization 参数、depth identity 参数、control tensors 保持高精度导出**
4. **QAT 不再作为后期开关，而应尽量保持从主训练早期就参与约束**

如果确实需要 warmup，也只允许很短的 warmup；不能再回到“先学一个浮点动态，最后再尝试量化适配”的旧逻辑。

### 4.4 control tensor 的高精度保留要扩大

由于 canonicalization 现在是主架构的一部分，以下参数应明确纳入 control-plane 白名单：

- boundary canonicalizer 的 scale / bias / mix
- loop canonicalizer 的 scale / bias / mix
- 任何 depth identity / loop identity 小张量
- 可能新增的 boundary bridge 小张量

也就是说，`CONTROL_TENSOR_NAME_PATTERNS` 不能只停留在旧的：

- `attn_scales`
- `mlp_scales`
- `resid_mixes`
- `skip_weights`

而要更新成围绕新架构的控制张量集合。

---

## 5. donor 的重新定位

### 5.1 #927：工程框架 donor

#927 现在仍然最重要，但角色已经从“结构主干”变成了：

- 训练与验证主框架
- QAT 与 quant/export 主框架
- BigramHash / SmearGate / VE / XSA 的成熟整合参考
- score-first TTT + sliding 的现成路径

Plan B 不再要求保留 #927 的 loop 拓扑本身。

### 5.2 #990：边界 / core 分工 donor

#990 现在变成结构上最直接的 donor。

它提供的不是可直接抄的代码树，而是明确的设计原则：

- unique boundary 与 shared core 要分工
- more looping 会放大量化脆弱性
- shared core 应该只承担它最擅长的重复计算部分

这恰好支撑新的 Plan B 主架构。

### 5.3 #1088：depth identity donor

#1088 依然非常关键，因为它说明：

- 即便主干共享，per-depth bias 也足以形成有效的 depth identity

这对新的 core recurrence 很重要，因为我们删掉了 encoder/decoder 语义后，loop identity 就更需要靠小型 control plane 显式表达。

### 5.4 #857：激活与模块组合 donor

#857 的主要价值现在集中在：

- LeakyReLU²
- late QAT 经验
- SmearGate / BigramHash / VE 的成熟组合

其中最值得优先考虑的结构增量仍然是 **LeakyReLU²**，因为它有机会改善 recurrent core 在 QAT 下的非线性稳定性。

### 5.5 #386：无 skip 条件下的稳定化提醒

#386 继续提供一个重要提醒：

- recurrent 模型的稳定性锚点不是只能靠 skip
- `x0` residual mix 本身就是一个强稳定器

这对 Plan B 很关键，因为我们正准备主动移除 U-Net skip。

### 5.6 #895：后续训练调度 donor

#895 的 progressive depth 现在不作为首轮实现约束，但保留为第二阶段优化方向。

只有在 boundary/core 主线跑稳后，才讨论：

- core loop 数是否采用 progressive schedule
- 前后边界是否固定，只有 core depth 动态增长

---

## 6. 具体实施计划

### Phase 1：先完成骨架重写

首轮代码修改目标不是做 ablation，而是直接把 [reference/pr-927/records/track_10min_16mb/2026-03-26_RecursiveTransformer_4B7L_VE_QAT_TTT](../reference/pr-927/records/track_10min_16mb/2026-03-26_RecursiveTransformer_4B7L_VE_QAT_TTT) 的 `train_gpt.py` 骨架改成新拓扑。

必须完成的改动：

1. 删除 encoder/decoder split 相关字段
   - `num_encoder_loops`
   - `num_decoder_loops`
   - `num_skips`
   - `skip_weights`

2. 删除 forward / forward_logits 里的 skip 栈逻辑
   - `skips.append(...)`
   - `if skips: x = x + ... * skips.pop()`

3. 新增新的参数银行定义
   - `front_boundary_block`
   - `core_blocks`
   - `back_boundary_block`

4. 新增新的 loop 参数
   - `num_core_loops`
   - 可选 `num_core_banks`

5. 把原本“按 encoder / decoder 阶段分配”的控制张量改成“按 core loop 序号分配”

这一阶段的结果应该是：

- 模型已经是 boundary/core 新架构
- 但不要求第一天就完成所有正则和观测

### Phase 2：把 canonicalization 直接嵌进新骨架

在新骨架稳定编译后，马上接 canonicalization，而不是把它后置成可选 patch。

这一阶段至少做：

1. front boundary -> core 的 `BoundaryCanonicalizer`
2. 每个 core step 前的 `LoopCanonicalizer`
3. canonicalizer 参数并入 control-plane 高精度保留路径
4. 若需要，在 core -> back boundary 前加轻量 exit canonicalizer

要求：

- canonicalization 只改状态分布，不引入大容量 adapter
- 参数规模保持明显小于 block 主体
- 不把 canonicalizer 做成另一个隐式子网络

### Phase 3：同步修正 QAT 与导出路径

QAT 不单独另起一章，而是随着 canonicalization 一起落地。

这里要完成：

1. 检查 fake-quant 是否覆盖新的 boundary/core 大矩阵
2. 检查 canonicalization / depth identity / boundary control 是否走高精度导出
3. 更新导出时的 control tensor 名称白名单
4. 重新验证 artifact 大小是否仍受控

目标不是追求最小文件，而是避免：

- canonicalization 小参数被错误量化
- QAT 只约束了旧的 shared block，却没约束新的 boundary/core 路径

### Phase 4：先跑主线，再补观测

这一阶段的优先级已经不是“先做 off/on ablation”，而是先把主线跑通。

至少需要：

1. 一次 smoke train
2. 一次 pre-quant / post-quant 对照
3. 一次 sliding eval
4. 如果路径还在，跑一次 TTT+sliding

只有在主线能跑通后，再补以下 instrumentation：

- front boundary 输出 RMS / absmax
- core 每一步 canonicalization 前后 RMS / absmax / clip ratio
- core 最终输出到 back boundary 前的 drift
- back boundary 输出到 final norm 前的幅度变化

### Phase 5：ablation 后置

这次 ablation 明确后置，不阻塞主线重写。

建议顺序：

1. 新架构 + canonicalization + QAT 主线先跑通
2. 再做 `with/without skip` 的反证式 ablation
3. 再做 `with/without boundary canonicalizer`
4. 再做 `with/without loop canonicalizer`
5. 再做 LeakyReLU²、progressive depth 等增强项

也就是说，ablation 的职责是**解释已经实现的主线**，而不是阻止主线开始。

---

## 7. 新 Plan B 下哪些保留，哪些移除

### 保留

- #927 的训练框架
- #927 的 quant/export 框架
- int6 QAT 主路径
- BigramHash
- SmearGate
- late ValueEmbedding
- XSA last-n
- score-first TTT + sliding eval
- x0 residual anchor

### 明确移除

- encoder/decoder split 语义
- U-Net skip 栈
- `skip_weights`
- 以 skip 为核心的稳定化解释

### 明确新增

- front unique boundary
- back unique boundary
- recurrent shared core
- boundary canonicalizer
- loop canonicalizer
- 面向新架构的 control tensor 白名单

### 延后考虑

- progressive depth
- LeakyReLU²
- 更复杂的 exit canonicalizer
- 几何型 canonicalization
- 双 core bank 以上的更激进设计

---

## 8. 验收标准

新的 Plan B 验收顺序也调整了。

### 一级验收：新骨架成立

- boundary/core 架构成功替换 #927 原 loop 拓扑
- encoder/decoder split + skip 已完整移除
- 训练、导出、eval 路径都还能跑

### 二级验收：canonicalization 和 QAT 协同成立

- canonicalization 没有被量化路径破坏
- post-quant gap 没有恶化到不可接受
- recurrent core 的状态统计开始趋稳

### 三级验收：再谈收益与 ablation

- sliding 或 TTT+sliding 指标有改善
- 或同等指标下更稳、更易量化
- 事后 ablation 能解释哪些结构是必要的

---

## 9. 最终建议

新的 Plan B 的一句话版本是：

> **把 #927 当成训练/量化/导出框架，而不是必须保留的 loop 拓扑；直接改成“前边界 unique + 中间 core recurrence + 后边界 unique”，同步去掉 encoder/decoder split 与 U-Net skip，并把 canonicalization 做成 boundary/core 接口的一部分，从第一版就和 QAT 一起设计；主线先跑通，ablation 放到后面。**

这版 Plan B 比上一版更激进，但它和当前目标是一致的：

- 架构上真正转向 boundary/core 分工
- 训练上不把 QAT 和 canonicalization 分开
- 工程上继续复用 #927 最成熟的外层框架