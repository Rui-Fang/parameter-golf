# Plan C: 面向 QAT 的 Loopwise Alignment / Boundary-Core Recurrent 代码与实验计划

## 1. 结论先行

Plan C 不再把 canonicalization 当作一个单独追逐的主概念。

新的主线是：

1. 继续以 #927 对应的现有主脚本为工程母体：
   - /home/u0725807/workspace/parameter-golf/records/track_10min_16mb/2026-03-26_RecursiveTransformer_4B7L_VE_QAT_TTT/train_gpt.py
2. 结构上从 encoder/decoder split + U-Net skip 改成：
   - unique front boundary
   - shared recurrent core
   - unique back boundary
3. 在 shared core 前只保留极小的 loopwise alignment 控制面：
   - depth-specific bias
   - bounded affine alignment
   - weak cross-loop carry
4. QAT 从第一版起就围绕 shared core 的统计稳定性设计，而不是最后附加。
5. 实验目标优先级改成：
   - 先降低 post-quant gap
   - 再追 pre-quant bpb
   - 最后再考虑更复杂的 loopwise 模块

一句话概括：

> Plan C 的核心不是“学习一个强 canonicalizer”，而是“用极小的 loopwise 对齐模块，把不同 loop 送回 shared low-bit core 更统一的工作点”。

---

## 2. Plan C 要解决的真问题

当前脚本已经有一版最小 LoopCanonicalizer，但它还是附着在 #927 的 split + skip 主体上，问题有三个。

1. 不同 loop 的角色仍然主要由 encoder/decoder 位置和 skip 流来决定，loop identity 不够直接。
2. shared core 看到的输入分布仍混合了：
   - x0 锚点
   - skip 回注
   - VE late injection
   - per-loop control
   - 当前 canonicalizer 预处理
3. QAT 仍是全局开关，无法表达“shared core 应更早进入低比特约束、boundary 和辅助路径稍晚进入”的优先级。

Plan C 的判断是：

- 如果目标是帮助 QAT，真正应优先做的是 loopwise alignment，而不是更强的 per-loop 变换。
- alignment 的成功标准不是 loop 更可分，而是 loop 进入 shared core 前的统计更接近。
- 一旦 alignment 模块变成每轮一个小 adapter，它就大概率会放大而不是缩小跨 loop 分布差异。

---

## 3. 基线与目标脚本

### 3.1 主改文件

Plan C 的唯一主改文件是：

- train_gpt.py， 目前这一代码直接复制来自 `reference/pr-927/records/track_10min_16mb/2026-03-26_RecursiveTransformer_4B7L_VE_QAT_TTT/train_gpt.py`

### 3.2 现有脚本里必须正视的真实状态

这个脚本已经包含了以下要素，Plan C 不应假装它们不存在：

- `shared_blocks`
- `num_blocks`
- `num_loops`
- `num_encoder_loops` / `num_decoder_loops`
- `skip_weights`
- `attn_scales`
- `mlp_scales`
- `resid_mixes`
- `ValueEmbedding`
- `SmearGate`
- `BigramHashEmbedding`
- `LoopCanonicalizer`
- `CONTROL_TENSOR_NAME_PATTERNS`
- 全局 `_QAT_ENABLED / _QAT_BITS / _QAT_MLP_BITS`

因此 Plan C 不是从零造一个新模型，而是要把当前脚本重构成新的主路径，同时尽量复用：

- 训练循环
- 优化器分组
- 量化导出
- eval / sliding / score-first TTT
- lexical helpers

---

## 4. 新主架构

### 4.1 结构定义

Plan C 的第一版目标结构：

```text
embed + bigram + smear
-> front boundary block(s), unique
-> shared core block bank, recurrent loops
-> back boundary block(s), unique
-> final norm + head
```

推荐第一版参数银行数：

- `num_front_blocks = 1`
- `num_core_blocks = 2`
- `num_core_loops = 5`
- `num_back_blocks = 1`

若 `num_core_loops` 解释为“完整扫过 shared core block bank 的轮数”，则在 `num_core_blocks = 2`、`num_core_loops = 5` 时，core 的 effective depth 是 `10`，总主计算是 `1 + 10 + 1 = 12`。这比最初的 7-step 草案更深，因此后续 wallclock 配置应按 effective depth 重新估算。

- 第 0 步：进入 recurrent state
- 中间 5 步：共享 core 迭代 refinement
- 最后 1 步：离开 recurrent state 并准备读出

### 4.2 为什么直接去掉 encoder/decoder skip

Plan C 直接把以下逻辑移出主线：

- `num_encoder_loops`
- `num_decoder_loops`
- `skip_weights`
- `skips.append(...)`
- `skips.pop()`

不是因为 skip 在所有递归模型里都无用，而是因为在 boundary/core 架构里，它会掩盖两个更应该显式建模的对象：

1. loop identity
2. cross-loop state carry

Plan C 的替代物不是“什么都不补”，而是：

- depth-specific bias
- affine-lite alignment
- weak cross-loop carry

---

## 5. 代码内容设计

## 5.1 Hyperparameters 改造

保留原参数，同时新增或重命名下列字段。

### 结构参数

```python
num_front_blocks = int(os.environ.get("NUM_FRONT_BLOCKS", 1))
num_core_blocks = int(os.environ.get("NUM_CORE_BLOCKS", 2))
num_core_loops = int(os.environ.get("NUM_CORE_LOOPS", 5))
num_back_blocks = int(os.environ.get("NUM_BACK_BLOCKS", 1))
```

第一版建议保留 `num_loops` 作为兼容 alias：

- 如果只设置 `NUM_LOOPS`，则自动推导为 `NUM_CORE_LOOPS`
- 若显式设置 `NUM_CORE_LOOPS`，则以新字段为准
- `NUM_CORE_LOOPS` 表示完整 sweep 次数，不是单个 core step 数；effective core depth 为 `NUM_CORE_BLOCKS * NUM_CORE_LOOPS`

### alignment 参数

```python
align_enabled = bool(int(os.environ.get("ALIGN_ENABLED", "1")))
align_mode = os.environ.get("ALIGN_MODE", "bias")
align_scale_clamp = float(os.environ.get("ALIGN_SCALE_CLAMP", 0.125))
align_mix_init = float(os.environ.get("ALIGN_MIX_INIT", 0.0))
depth_bias_enabled = bool(int(os.environ.get("DEPTH_BIAS_ENABLED", "1")))
carry_enabled = bool(int(os.environ.get("CARRY_ENABLED", "1")))
carry_init = float(os.environ.get("CARRY_INIT", 0.05))
```

约定：

- `ALIGN_MODE=off|bias|affine`
- `bias` 表示只做 per-loop bias
- `affine` 表示做有限幅度的 `(1 + delta) * norm(x) + bias`

### QAT 调度参数

```python
qat_mode = os.environ.get("QAT_MODE", "shared_early")
qat_core_start_frac = float(os.environ.get("QAT_CORE_START_FRAC", 0.05))
qat_boundary_start_frac = float(os.environ.get("QAT_BOUNDARY_START_FRAC", 0.20))
qat_aux_start_frac = float(os.environ.get("QAT_AUX_START_FRAC", 1.00))
```

第一版支持三种策略：

- `step0`: 全部从 step 0 QAT
- `shared_early`: core 先启，boundary 后启，aux 默认不 fake quant
- `late_global`: 兼容当前 late QAT 行为，仅作回退对照

### 观测参数

```python
collect_loop_stats = bool(int(os.environ.get("COLLECT_LOOP_STATS", "0")))
loop_stats_every = int(os.environ.get("LOOP_STATS_EVERY", 200))
```

---

## 5.2 模型存储布局改造

当前脚本里的：

```python
self.shared_blocks = nn.ModuleList([...])
```

重构为：

```python
self.front_blocks = nn.ModuleList([...])
self.core_blocks = nn.ModuleList([...])
self.back_blocks = nn.ModuleList([...])
```

同时删除：

- `self.num_encoder_loops`
- `self.num_decoder_loops`
- `self.num_skips`
- `self.skip_weights`

保留并重解释：

- `attn_scales` -> 只对 core loops 生效
- `mlp_scales` -> 只对 core loops 生效
- `resid_mixes` -> 只对 core loops 生效

边界层不需要每步 control tensor，边界层应该是固定 unique role。

---

## 5.3 新的小模块

### 5.3.1 `LoopDepthBias`

第一版最重要模块。

```python
class LoopDepthBias(nn.Module):
    def __init__(self, num_loops: int, dim: int):
        self.bias = nn.Parameter(torch.zeros(num_loops, dim, dtype=torch.float32))

    def forward(self, x: Tensor, loop_idx: int) -> Tensor:
        return x + self.bias[loop_idx].to(dtype=x.dtype)[None, None, :]
```

作用：

- 给 shared core 明确 loop identity
- 仅改变中心，不大幅改变分布宽度
- 比直接加 loop embedding 更像量化友好的控制面

### 5.3.2 `LoopAligner`

它替代当前 `LoopCanonicalizer` 的概念中心。

建议第一版直接复用字段位，但内部逻辑改为有界对齐，而不是开放式 canonicalization。

```python
class LoopAligner(nn.Module):
    def __init__(self, num_loops: int, dim: int, mode: str, scale_clamp: float):
        self.bias = nn.Parameter(torch.zeros(num_loops, dim, dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.zeros(num_loops, dim, dtype=torch.float32))
        self.mode = mode
        self.scale_clamp = scale_clamp

    def forward(self, x: Tensor, loop_idx: int) -> Tensor:
        xn = F.rms_norm(x, (x.size(-1),))
        if self.mode == "bias":
            return xn + self.bias[loop_idx].to(dtype=x.dtype)[None, None, :]
        delta = torch.tanh(self.log_scale[loop_idx]) * self.scale_clamp
        scale = (1.0 + delta).to(dtype=x.dtype)[None, None, :]
        bias = self.bias[loop_idx].to(dtype=x.dtype)[None, None, :]
        return xn * scale + bias
```

设计原则：

- 先 `rms_norm`
- 只允许小幅度缩放
- 默认从 identity 开始
- 不做内容相关 gating
- 不引入 per-loop MLP

### 5.3.3 `CrossLoopCarry`

这不是大 residual branch，而是一条弱状态旁路。

```python
class CrossLoopCarry(nn.Module):
    def __init__(self, num_loops: int, dim: int, init: float):
        self.logit = nn.Parameter(torch.full((num_loops, dim), logit(init), dtype=torch.float32))

    def forward(self, x: Tensor, prev_state: Tensor | None, loop_idx: int) -> Tensor:
        if prev_state is None:
            return x
        gate = torch.sigmoid(self.logit[loop_idx]).to(dtype=x.dtype)[None, None, :]
        return x + gate * prev_state
```

它的职责不是替代 shared core，而是补回去掉 U-Net skip 后最必要的“跨轮状态连续性”。

---

## 5.4 `Block` 接口策略

Plan C 不建议大改 `Block.forward(...)` 的签名。

当前 `Block` 已经接受：

- `x`
- `x0`
- `attn_scale`
- `mlp_scale`
- `resid_mix`
- `v_embed`

第一版应把 loopwise alignment 尽量放在 `GPT.forward(...)` 外围完成，而不是把更多 loop 参数穿透到 `Block` 内部。

这样做有三个好处：

1. `Block` 仍保持“共享 core 的原子计算单元”角色
2. loopwise 自由度被限制在 control plane
3. diff 最小，便于回退和 ablation

---

## 5.5 `GPT` 前向路径重写

建议用一个统一的 core loop 取代当前 encoder / decoder 两段。

### 5.5.1 新前向伪代码

```python
x = self._embed(input_ids)
x = x + self.bigram(input_ids)
x = F.rms_norm(x, (x.size(-1),))
if self.smear is not None:
    x = self.smear(x)
x0 = x

for block in self.front_blocks:
    x = block(x, x0, front_attn_scale, front_mlp_scale, front_resid_mix)

prev_state = None
for loop_idx in range(self.num_core_loops):
    if self.depth_bias is not None:
        x = self.depth_bias(x, loop_idx)
    if self.loop_aligner is not None:
        x = self.loop_aligner(x, loop_idx)
    if self.cross_loop_carry is not None:
        x = self.cross_loop_carry(x, prev_state, loop_idx)
    ve = self._get_late_ve(input_ids, loop_idx)
    block = self.core_blocks[loop_idx % self.num_core_blocks]
    x = block(
        x,
        x0,
        self.attn_scales[loop_idx],
        self.mlp_scales[loop_idx],
        self.resid_mixes[loop_idx],
        v_embed=ve,
    )
    prev_state = x

for block in self.back_blocks:
    x = block(x, x0, back_attn_scale, back_mlp_scale, back_resid_mix)

x = self.final_norm(x)
logits = self._logits(x)
```

### 5.5.2 边界层控制

第一版边界层不需要独立的每层 `attn_scales/mlp_scales/resid_mixes` 参数表。

边界层可以直接：

- 用 block 内部默认结构
- 复用一组固定的 boundary scalar params
- 或简单采用 `torch.ones / [1,0]` 的常数控制

建议先用常数控制，减少变量数。

### Note: 为什么不保留 core handoff anchor

曾考虑过一种额外设计：

- core 前半段围绕 front boundary 之后的 `core_anchor`
- 在 core 中途记录一次新的 handoff anchor
- core 后半段改为围绕 handoff anchor 工作

这个想法的直觉是把 shared core 再切成“早期 refinement”和“晚期 refinement”两段，进一步减轻单一 anchor 的职责负担。

但在把 `num_core_loops` 纠正为“完整 sweep 次数”之后，这个设计不再适合作为 Plan C 默认主线，原因有两个：

1. 它天然更像 effective-step 控制，而不是 loopwise 控制。若 handoff 发生在一个 sweep 中间，就会重新引入按 step 切分语义，与 Plan C 想强调的 per-loop control plane 冲突。
2. 它会让 shared core 同时承担“跨 loop recurrence”和“中途阶段切换”两种机制，增加解释难度，不利于先把真正的 loop / carry / late VE / late XSA 语义跑通。

因此当前实现选择：

- 保留 `stage_anchors`，即 boundary/core/back 之间的阶段锚点
- 不在 core 内部再加入额外 handoff anchor

如果未来要重新探索 handoff，应明确把它设计成 loop-boundary 上的切换，而不是 sweep 内部的 step-level 切换。

---

## 5.6 XSA 与 VE 的重解释

### 5.6.1 XSA

XSA 不再以总 loop index 的 encoder/decoder 位置为依据，而是以 core loop index 为依据。

第一版建议：

- `xsa_last_n` 只作用于 `num_core_loops` 的最后几轮
- 不作用于 front / back boundary

### 5.6.2 ValueEmbedding

VE 也改成“只注入最后若干个 core loops”，不再依赖 encoder / decoder 划分。

保留：

- `ve_last_n`

但解释改成：

- `ve_last_n` = 最后多少个 core loops 注入 VE

这与 Plan C 的目标一致：

- boundary 负责结构角色
- core 负责迭代 refinement
- VE 只在接近退出前对 value path 做晚期修正

---

## 5.7 QAT 路径重构

当前脚本的 `_QAT_ENABLED` 是单布尔全局开关，这不够表达 Plan C 的优先级。

### 5.7.1 最小可行改法

给 `CastedLinear` 增加一个 `_qat_role` 字段：

```python
self._qat_role = "core" | "boundary" | "aux"
```

初始化时约定：

- `core_blocks.*` 内线性层 -> `core`
- `front_blocks.*` / `back_blocks.*` -> `boundary`
- `bigram.proj`, `ve.proj`, `embed_proj`, `embed_proj_rev`, `lm_head` -> `aux`

全局开关改成：

```python
_QAT_CORE_ENABLED = False
_QAT_BOUNDARY_ENABLED = False
_QAT_AUX_ENABLED = False
```

在 `CastedLinear.forward(...)` 中按 role 决定是否 fake quant。

### 5.7.2 默认调度

Plan C 推荐默认调度：

- 0% - 5%: 全部浮点 warmup
- 5% - 20%: 仅 core 开启 QAT
- 20% 之后: boundary 也开启 QAT
- aux 默认保持浮点导出控制，不进入主 fake quant

理由：

- 先强约束 shared core
- 再约束 boundary
- 避免 lexical helper 和小控制路径过早卷入 fake quant 噪声

### 5.7.3 回退对照

必须保留两个回退模式：

- 完全沿用当前 late global QAT
- 完全 step0 global QAT

否则无法判断 Plan C 的收益来自结构，还是来自 QAT schedule。

---

## 5.8 control tensor 与优化器分组

### 5.8.1 `CONTROL_TENSOR_NAME_PATTERNS`

Plan C 需要把以下对象纳入 control tensor 白名单：

- `attn_scales`
- `mlp_scales`
- `resid_mixes`
- `depth_bias`
- `loop_aligner`
- `cross_loop_carry`
- `q_gain`
- `smear`
- `ve.scales`

不再包含：

- `skip_weights`

### 5.8.2 参数分组

矩阵参数：

- front/core/back block 中的 2D 权重
- `bigram.proj.weight`
- `ve.proj.weight`
- optional head matrices

标量或 control 参数：

- 所有对齐模块参数
- loop control tensors
- norm 邻近 bias
- carry gates
- `q_gain`
- VE scales

原则：

- 所有小控制参数走 scalar lr
- 所有对齐参数保持 FP32 导出

---

## 5.9 观测与日志

Plan C 第一版必须加统计接口，否则无法验证 alignment 是否真的在帮助 QAT。

### 5.9.1 记录位置

每个 core loop 记录两组统计：

1. `pre_align`
2. `pre_block`

也就是：

- 进入 depth bias / aligner 之前
- 完成 depth bias / aligner / carry 之后、进入 shared block 之前

### 5.9.2 记录指标

每个 loop 至少记录：

- RMS
- abs max
- p01 / p50 / p99
- channel-wise mean abs 的均值
- 与 loop 0 的 RMS 差
- 与前一 loop 的 p99 差

### 5.9.3 成功判据

Plan C 不把“统计更像”定义成完全重合。

第一版只要求：

- `pre_block` 的跨 loop RMS spread 小于 `pre_align`
- `pre_block` 的跨 loop p99 spread 也下降
- 同时 post-quant gap 没恶化

---

## 6. 实施顺序

### Phase C0: 当前主干基线重新编号

建立两个真正的对照：

- `C0A`: 当前工作树脚本，`CANONICALIZE_ENABLED=0`
- `C0B`: 当前工作树脚本，`CANONICALIZE_ENABLED=1`

目的：

- 量化当前极小 canonicalizer 是否本身已有正效应
- 给 Plan C 提供真实对照，而不是只和抽象的 #927 对照

### Phase C1: 结构改造，不加 alignment

改成 boundary/core/back 架构，去掉 skip。

配置：

- `ALIGN_ENABLED=0`
- `DEPTH_BIAS_ENABLED=0`
- `CARRY_ENABLED=0`

目标：

- 验证新骨架本身是否可训
- 记录去掉 skip 后吞吐和 pre-quant 损失

### Phase C2: 加 depth-specific bias

只上：

- `LoopDepthBias`

不加 affine，不加 carry。

目标：

- 验证单独 loop identity 是否就能改善统计和 quant gap

### Phase C3: 加 affine-lite aligner

配置：

- `ALIGN_MODE=bias` 先跑
- 再试 `ALIGN_MODE=affine`

约束：

- `ALIGN_SCALE_CLAMP <= 0.125`

目标：

- 测试小幅度 affine 是否真的缩小跨 loop 分布差异
- 避免演化成每轮单独 reparameterization

### Phase C4: 加 weak cross-loop carry

在 C3 最优设置上加：

- `CARRY_ENABLED=1`

目标：

- 测试“去 skip 后最小状态旁路”是否能回补训练稳定性

### Phase C5: 分角色 QAT schedule

在 C4 的最好结构上对比：

- `QAT_MODE=late_global`
- `QAT_MODE=step0`
- `QAT_MODE=shared_early`

目标：

- 确认收益来自对齐模块本身，还是来自更合理的 shared-core QAT 调度

### Phase C6: VE / XSA 重新挂接

在最优 C4/C5 结构上重新评估：

- `VE_LAST_N=0,1,2`
- `XSA_LAST_N=0,2,4`

原则：

- 只对最后若干个 core loops 生效
- 不重新引入 encoder/decoder 语义

---

## 7. 实验矩阵

### 7.1 主实验编号

| 编号 | 结构 | 对齐模块 | carry | QAT | 目的 |
|---|---|---|---|---|---|
| C0A | 现脚本 | off | skip on | 当前 | 真实基线 |
| C0B | 现脚本 | current canonicalizer | skip on | 当前 | 当前最小 canonicalizer 基线 |
| C1 | boundary/core/back | off | off | late_global | 结构可训性 |
| C2 | boundary/core/back | depth bias | off | late_global | loop identity 贡献 |
| C3a | boundary/core/back | bias align | off | late_global | 仅中心对齐 |
| C3b | boundary/core/back | affine-lite | off | late_global | 小幅度尺度对齐 |
| C4 | boundary/core/back | best of C3 | on | late_global | 状态旁路贡献 |
| C5a | best C4 | same | on | step0 | 强 QAT 约束 |
| C5b | best C4 | same | on | shared_early | 推荐主线 |
| C6 | best C5 | same | on | same | VE/XSA 重新挂接 |

### 7.2 每个实验都要记录的指标

- train ms/step
- total steps in 600s
- pre-quant val_loss
- post-quant val_bpb
- quant gap
- artifact size
- `pre_align` 与 `pre_block` 的跨 loop RMS spread
- `pre_align` 与 `pre_block` 的跨 loop p99 spread

### 7.3 成功阈值

Plan C 第一版的成功不要求立刻刷新当前最好成绩。

推荐阈值：

1. C2 / C3 / C4 中至少有一个配置相对 C1：
   - post-quant gap 下降 >= 0.002 bpb
2. 最优配置相对 C0A：
   - 600s steps 不下降超过 12%
3. `pre_block` 的跨 loop p99 spread 至少下降 15%
4. artifact 保持在 16MB 以内

### 7.4 kill criteria

以下情况直接停止该方向：

- affine-lite 明显拉大跨 loop p99 spread
- carry 打开后吞吐下降过大但 quant gap 无改善
- shared_early QAT 明显比 step0 和 late_global 都差
- alignment 模块带来的收益只体现在 pre-quant，不体现在 post-quant

---

## 8. 推荐默认配置

第一版推荐主线配置：

```bash
NUM_FRONT_BLOCKS=1
NUM_CORE_BLOCKS=2
NUM_CORE_LOOPS=5
NUM_BACK_BLOCKS=1
ALIGN_ENABLED=1
ALIGN_MODE=bias
DEPTH_BIAS_ENABLED=1
CARRY_ENABLED=1
CARRY_INIT=0.05
QAT_MODE=shared_early
QAT_CORE_START_FRAC=0.05
QAT_BOUNDARY_START_FRAC=0.20
QAT_AUX_START_FRAC=1.00
VE_LAST_N=2
XSA_LAST_N=2
COLLECT_LOOP_STATS=1
```

这是 Plan C 的推荐原因：

- 先用最小 bias 对齐，而不是一上来就上 affine
- 先给最弱 carry，替代去掉 skip 后最必要的状态通道
- 让 core 先进入 QAT 约束
- 把 VE / XSA 约束在最后几轮

---

## 9. 明确不做的事

Plan C 第一版明确排除下面这些设计。

1. 每个 loop 一个 MLP/adapter/LoRA
2. 内容相关的 loop gate
3. 大幅度 per-loop scale 或可逆变换
4. 重新把 U-Net skip 换个名字加回来
5. 一开始就把 progressive depth 和新结构一起上
6. 一开始就把 GPTQ、复杂 export 改造、对齐模块一起大改

原因很简单：

- 这些改动会让变量数爆炸
- 很难判断收益来自 alignment 还是来自 per-loop 自由度扩张
- 对 QAT 目标不一定友好

---

## 10. Plan C 的最终判断标准

Plan C 是否值得继续，不看它有没有创造一个“更花哨的递归模型”，而看三件事。

1. shared core 入口的跨 loop 统计是否更收敛
2. post-quant gap 是否更小
3. 在 600s wallclock 下，吞吐损失是否还能接受

如果答案是：

- 统计更收敛
- quant gap 更小
- 吞吐只小幅下降

那么 Plan C 就成立。

如果答案是：

- pre-quant 变好
- 但 post-quant 没变好

那说明我们做出来的是更强的每轮适配，而不是更好的量化对齐。

这时就应该收缩自由度，而不是继续给 canonicalization 加表达力。
