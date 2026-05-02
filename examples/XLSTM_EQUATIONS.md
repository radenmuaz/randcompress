# xLSTM: Extended LSTM Equations and Architecture

**Paper**: [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)

## Overview

xLSTM introduces two main improvements to traditional LSTMs:
1. **Exponential gating** with sophisticated numerical stabilization
2. **Modified memory structures**: sLSTM (scalar) and mLSTM (matrix)

---

## 1. Exponential Gating Mechanism

### Problem with Standard Sigmoid Gating
Standard LSTM uses sigmoid gates: $\sigma(x) = \frac{1}{1 + e^{-x}}$

This has limited expressiveness and gradient flow issues at scale.

### xLSTM Solution: Exponential Gates with Normalization

For any gating operation, xLSTM uses:

$$\text{gate} = \min(1, e^{x - m})$$

where $m$ is a **stability constant** (maximum value for log-space computation).

**Soft Capping** (optional): 
$$\text{gate}_{\text{capped}} = \gamma \cdot \tanh\left(\frac{\text{gate}}{\gamma}\right)$$

where $\gamma$ is the soft cap value (typically 15-30).

---

## 2. Scalar LSTM (sLSTM) - Four State System

### State Variables

The sLSTM maintains **four states** instead of the traditional two:

$$\mathbf{s}_t = (\mathbf{y}_t, \mathbf{c}_t, \mathbf{n}_t, \mathbf{m}_t)$$

where:
- $\mathbf{y}_t \in \mathbb{R}^{H}$: Output hidden state
- $\mathbf{c}_t \in \mathbb{R}^{H}$: Cell state (numerator, unnormalized)
- $\mathbf{n}_t \in \mathbb{R}^{H}$: Normalization denominator
- $\mathbf{m}_t \in \mathbb{R}^{H}$: Maximum log value (for numerical stability)

### Forward Pass

**Input Processing:**

$$\mathbf{z} = \mathbf{W}_x \mathbf{x}_t + \mathbf{R} \mathbf{y}_{t-1} + \mathbf{b}$$

Split into four components:
$$[\mathbf{i}_{\text{raw}}, \mathbf{f}_{\text{raw}}, \mathbf{z}_{\text{raw}}, \mathbf{o}_{\text{raw}}]^T = \mathbf{z}$$

**Gate Computation with Exponential Normalization:**

The forget gate combines logsigmoid with the stability constant:
$$\log f_t = \log\sigma(\mathbf{f}_{\text{raw}}) + \mathbf{m}_{t-1}$$

Compute new maximum for stability:
$$\mathbf{m}_t = \begin{cases} 
\mathbf{i}_{\text{raw}} & \text{if } \mathbf{n}_{t-1} = 0 \\
\max(\mathbf{i}_{\text{raw}}, \log f_t) & \text{otherwise}
\end{cases}$$

Gate values with exponential stabilization:
$$\mathbf{i}_t = \min(1, e^{\mathbf{i}_{\text{raw}} - \mathbf{m}_t})$$

$$\mathbf{f}_t = \min(1, e^{\log f_t - \mathbf{m}_t})$$

$$\mathbf{o}_t = \sigma(\mathbf{o}_{\text{raw}})$$

**Cell State Update:**
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tanh(\mathbf{z}_{\text{raw}})$$

**Normalization Tracking:**
$$\mathbf{n}_t = \mathbf{f}_t \odot \mathbf{n}_{t-1} + \mathbf{i}_t$$

**Output:**
$$\mathbf{y}_t = \mathbf{o}_t \odot \frac{\mathbf{c}_t}{\mathbf{n}_t}$$

where $\odot$ denotes element-wise multiplication.

### Numerical Stability Benefits

1. **m tracks the magnitude**: Exponentials are always $\leq 1$ (log-space subtraction)
2. **Gradient friendly**: Bounded outputs, smooth gradients
3. **Parallel formulation**: Can be computed in chunks (not shown here, but possible)

---

## 3. Matrix LSTM (mLSTM) - Parallel Form

### State Variables

The mLSTM maintains matrix states:
$$\mathbf{C}_t \in \mathbb{R}^{H \times H}: \text{Covariance matrix}$$
$$\mathbf{n}_t \in \mathbb{R}^{H}: \text{Normalization vector}$$
$$m_t \in \mathbb{R}: \text{Stability constant (scalar)}$$

### Multihead Architecture

Query, Key, Value projections with multihead structure:
$$\mathbf{Q}_t = \text{Proj}_Q(\mathbf{x}_t) \in \mathbb{R}^{L \times (D/L)}$$
$$\mathbf{K}_t = \text{Proj}_K(\mathbf{x}_t) \in \mathbb{R}^{L \times (D/L)}$$
$$\mathbf{V}_t = \text{Proj}_V(\mathbf{x}_t) \in \mathbb{R}^{L \times (D/L)}$$

where $L$ is number of heads, $D$ is embedding dimension.

### Recurrent (Single-Step) Form

**Input Pre-activations:**
$$i_{\text{raw}} = \text{Linear}_i([\mathbf{x}_t, \mathbf{y}_{t-1}, \text{prev_cell}])$$
$$f_{\text{raw}} = \text{Linear}_f([\mathbf{x}_t, \mathbf{y}_{t-1}, \text{prev_cell}])$$

**Gate Values (exponentially stabilized):**
$$\log f_t = \log\sigma(f_{\text{raw}})$$
$$m_t = \max(\log f_t + m_{t-1}, i_{\text{raw}})$$

$$f_t = \min(1, e^{\log f_t + m_{t-1} - m_t})$$
$$i_t = \min(1, e^{i_{\text{raw}} - m_t})$$

**Scaled Keys:**
$$\tilde{\mathbf{K}}_t = \frac{\mathbf{K}_t}{\sqrt{D_h}}$$

**Covariance Update (Matrix Memory):**
$$\mathbf{C}_t = f_t \odot \mathbf{C}_{t-1} + i_t \odot (\tilde{\mathbf{K}}_t^T \mathbf{V}_t)$$

**Normalization Vector Update:**
$$\mathbf{n}_t = f_t \odot \mathbf{n}_{t-1} + i_t \odot \tilde{\mathbf{K}}_t$$

**Output Computation:**
$$\text{num} = \mathbf{Q}_t^T \mathbf{C}_t$$
$$\text{denom} = \max(|\mathbf{Q}_t^T \mathbf{n}_t|, e^{-m_t})$$
$$\mathbf{h}_t = \frac{\text{num}}{\text{denom} + \epsilon}$$

---

## 4. Parallel (Chunked) Form - mLSTM Over Sequence

For efficient computation over full sequences, mLSTM can be parallelized:

### Forget Gate Matrix (Causal)

For timesteps $i, j$ within a chunk where $j \leq i$:

<!-- $$\text{log\_fg\_matrix}[i, j] = \sum_{k=j}^{i-1} \log\sigma(f_{\text{raw}}_k) + i_{\text{raw}}_i$$ -->
$$\text{log\_fg\_matrix}[i, j] = \sum_{k=j}^{i-1} \log\sigma(f_rawk) + i_rawk_i$$

### Decay Matrix with Stabilization

$$\log D[i, j] = \text{log\_fg\_matrix}[i, j]$$

Subtract maximum for stability (per-row stabilization):
$$\max\_log\_D = \max_j \log D[i, j]$$
$$D[i, j] = e^{\log D[i, j] - \max\_log\_D}$$

### Attention-like Computation

$$\text{QK\_matrix}[i, j] = \mathbf{Q}_i^T \mathbf{K}_j / \sqrt{D_h}$$
$$C\text{\_matrix}[i, j] = \text{QK\_matrix}[i, j] \times D[i, j]$$

### Normalization

$$\text{normalizer}[i] = \max(|\sum_j C\text{\_matrix}[i, j]|, e^{-\max\_log\_D})$$
$$C\text{\_normalized}[i, j] = \frac{C\text{\_matrix}[i, j]}{\text{normalizer}[i] + \epsilon}$$

### Output

$$\mathbf{h}_i = \sum_j C\text{\_normalized}[i, j] \times \mathbf{V}_j$$

---

## 5. Complete xLSTM Block Architecture

### Residual Connection

$$\mathbf{z}_t = \text{LayerNorm}(\mathbf{x}_t)$$

### Split into Parallel Paths

**sLSTM Path:**
$$\mathbf{y}_t^{(s)} = \text{sLSTM}(\mathbf{z}_t, \text{state}_{t-1}^{(s)})$$

**mLSTM Path:**
$$\mathbf{y}_t^{(m)} = \text{mLSTM}(\mathbf{z}_t, \text{state}_{t-1}^{(m)})$$

### Projection & Gating

$$\mathbf{y}_t = \sigma(\text{Linear}_{gate}([\mathbf{y}_t^{(s)}, \mathbf{y}_t^{(m)}])) \odot \text{Linear}_{proj}([\mathbf{y}_t^{(s)}, \mathbf{y}_t^{(m)}])$$

### Feed-Forward Network

$$\mathbf{ff}_t = \text{GELU}(\mathbf{W}_1 \mathbf{y}_t + \mathbf{b}_1) \times \mathbf{W}_2 + \mathbf{b}_2$$

### Final Residual

$$\mathbf{out}_t = \mathbf{x}_t + \mathbf{y}_t + \mathbf{ff}_t$$

---

## 6. Training Objectives

### Causal Language Modeling Loss

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \text{CrossEntropy}(\text{softmax}(\mathbf{W}_{\text{out}} \mathbf{y}_t + \mathbf{b}_{\text{out}}), \mathbf{x}_{t+1})$$

### With Label Smoothing

$$\mathcal{L}_{\text{smoothed}} = (1 - \alpha)\mathcal{L} + \frac{\alpha}{V}\sum_i \log p(i)$$

where $\alpha$ is smoothing factor, $V$ is vocabulary size.

---

## 7. Hyperparameters Summary

| Parameter | Typical Value | Purpose |
|-----------|---------------|---------|
| $H$ (hidden size) | 256-2048 | Model capacity |
| $L$ (num heads) | 8-16 | Multihead degree |
| $D_h$ (head dim) | $H/L$ | Per-head dimension |
| $\gamma$ (soft cap) | 15-30 | Gate saturation control |
| $\epsilon$ | $10^{-6}$ | Numerical stability |
| dropout | 0.1-0.2 | Regularization |
| weight\_decay | $10^{-6}$ | L2 regularization |
| learning\_rate | $10^{-4}$ | Base LR (with warmup) |

---

## 8. Key Implementation Notes

### Initialization
- **Weight matrices**: Normal distribution $\mathcal{N}(0, 1/\sqrt{H})$
- **Recurrent matrices**: Orthogonal initialization
- **Biases**: Zero initialization
- **Initial states**: All zeros

### Numerical Stability Tricks
1. Use log-space for forget gates: $\log\sigma(x)$ directly
2. Subtract max before exponential: $e^{x - \max(x)}$
3. Cap exponentials at 1: $\min(1, e^{x})$
4. Use bfloat16 for mixed precision training
5. Gradient clipping by norm (norm=1.0)

### Computational Efficiency
- **sLSTM**: $O(T \cdot H)$ per layer (recurrent, sequential)
- **mLSTM**: $O(T^2 \cdot H)$ per layer in parallel form, $O(T \cdot H^2)$ in recurrent form
- **xLSTM block**: Combines both, typically $O(T \cdot H^2)$ with FFN
- Comparable to Transformers but with constant memory per step (RNN property)

---

## References

1. Beck et al., 2024. "xLSTM: Extended Long Short-Term Memory". arXiv:2405.04517
2. Hochreiter & Schmidhuber, 1997. "Long Short-Term Memory". Neural Computation
3. Vaswani et al., 2017. "Attention is All You Need". NeurIPS
