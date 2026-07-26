# How microGPT Works
**Number flow, layers, attention, and training motivation**

> **Author:** J Zhao  
> **Topic:** Transformer Architecture Study Notes

---

## Chapter Map

1. What the toy GPT is learning
2. Token IDs, embeddings, and positions
3. Attention: Q, K, V and weighted retrieval
4. Transformer layer interpretation
5. Prediction, loss, and Adam update
6. Motivation: why these strange formulas exist

---

## Chapter 1 — What the Toy GPT Is Learning

### MicroGPT Architecture Overview

The decoder-only Transformer processes tokens in order:

> Embeddings → causal self-attention blocks → output logits → next-token probabilities

### The Model Is a Name Generator

The code trains a character-level autoregressive model.

For a name like `emma`, the sequence is:

```
[BOS, e, m, m, a, BOS]
```

The model learns conditional probabilities with a finite context window `B = block_size`:

$$P_\theta\!\left(x_{t+1}\mid x_{\max(0,t-B+1)},\ldots,x_t\right)$$

Here $T$ is the number of prediction steps for one example. For a name with $n$ characters, $T = n + 1$ because the final BOS/end token is also predicted.

A full name probability is factored as:

$$P_\theta(x_1,\ldots,x_T)=\prod_{t=0}^{T-1}P_\theta\!\left(x_{t+1}\mid x_{\max(0,t-B+1)},\ldots,x_t\right)$$

### The Whole Pipeline

```
text (emma) → token IDs [26,4,12,12,0] → vectors h_t^(0) ∈ ℝ^16 → Transformer (context mixing) → next-token probabilities
```

$$h_t^{(\ell)} = \text{hidden vector at position } t \text{ after layer } \ell$$

$$h_t^{(0)} = E_{\rm tok}[x_t] + E_{\rm pos}[t]$$

$$\text{discrete text} \longrightarrow \text{fixed IDs} \longrightarrow \text{learned vectors} \longrightarrow \text{probabilities}$$

---

## Chapter 2 — Token IDs, Embeddings, and Positions

### Token IDs Are Fixed; Embeddings Are Learned

The tokenizer gives fixed IDs:

$$a \mapsto 0,\quad b \mapsto 1,\quad\ldots,\quad z \mapsto 25,\quad \mathrm{BOS} \mapsto 26$$

The token embedding matrix is learned:

$$E_{\mathrm{tok}} \in \mathbb{R}^{27 \times 16}$$

If `m` has ID $12$, then:

$$E_{\mathrm{tok}}[m] = E_{\mathrm{tok}}[12,:] \in \mathbb{R}^{16}$$

Initialization:

$$E_{\rm tok}[i,j] \sim \mathcal{N}(0,\, 0.08^2)$$

```python
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)]
    for _ in range(nout)
]
state_dict['wte'] = matrix(vocab_size, n_embd)
```

### One-Hot View vs. Embedding Lookup

```
letter a → ID 0 → one-hot e_0 ∈ ℝ^27 --[e_0^T E_tok]--> learned vector E_tok[0] ∈ ℝ^16
```

> **One-hot vector = a standard basis vector $e_i$**  
> Every one-hot vector is a unit vector, but not every unit vector is one-hot. The code directly performs the row lookup.

### Position Embeddings

Define the embedding dimension:

$$\boxed{d := \text{embedding dimension} = \texttt{n\_embd} = 16}$$

The position embedding matrix is also learned:

$$E_{\mathrm{pos}} \in \mathbb{R}^{B \times d}$$

In this toy code, $B = 16$, $d = 16$, so $E_{\mathrm{pos}} \in \mathbb{R}^{16 \times 16}$.

> This square shape is accidental: context length and embedding width both happen to be 16.

### Example: Positions in `emma`

| Position $t$ | Input token | Target token |
|:---:|:---:|:---:|
| 0 | BOS | e |
| 1 | e | m |
| 2 | m | m |
| 3 | m | a |
| 4 | a | BOS |

The first `m` has representation:

$$h_2^{(0)} = E_{\mathrm{tok}}[m] + E_{\mathrm{pos}}[2]$$

The second `m` uses the same token embedding but a different position vector:

$$h_3^{(0)} = E_{\mathrm{tok}}[m] + E_{\mathrm{pos}}[3]$$

### Why Add Token and Position Vectors?

The model needs one $d$-dimensional vector containing both identity and position:

$$h_t^{(0)} = E_{\mathrm{tok}}[x_t] + E_{\mathrm{pos}}[t]$$

> **Addition is not sacred. It is cheap and keeps dimension fixed.**

A more literal alternative would be concatenation:

$$\begin{bmatrix}E_{\mathrm{tok}}[x_t]\\ E_{\mathrm{pos}}[t]\end{bmatrix} \in \mathbb{R}^{2d}$$

But that widens the model or requires another projection.

---

## Chapter 3 — Attention

### From Hidden Vector to Q, K, V

At each position, the vector is first normalized:

$$\tilde h_t = \operatorname{RMSNorm}(h_t)$$

Then three learned linear maps are applied:

$$q_t = W_Q\tilde h_t,\qquad k_t = W_K\tilde h_t,\qquad v_t = W_V\tilde h_t$$

Here $W_Q, W_K, W_V \in \mathbb{R}^{16 \times 16}$.

They are structurally similar, but their roles differ in the attention formula.

### Q, K, V Roles

| Vector | Formula | Role |
|---|---|---|
| $q_t$ | $W_Q\tilde h_t$ | Query — what I seek |
| $k_t$ | $W_K\tilde h_t$ | Key — how I match |
| $v_t$ | $W_V\tilde h_t$ | Value — what I provide |

> **Q, K decide routing. V supplies the routed content.**

### Attention Score and Output

For head dimension $d_h = d/H = 16/4 = 4$, the score from current position $t$ to source position $s$ is:

$$s_{ts}^{(h)} = \frac{(q_t^{(h)})^T k_s^{(h)}}{\sqrt{d_h}}$$

Softmax gives weights:

$$\alpha_{ts}^{(h)} = \frac{e^{s_{ts}^{(h)}}}{\sum_{r\le t} e^{s_{tr}^{(h)}}}$$

The head output is:

$$o_t^{(h)} = \sum_{s\le t} \alpha_{ts}^{(h)} v_s^{(h)}$$

### Attention as Differentiable Retrieval

The current position $t$ with query $q_t$ attends to all past sources $\{0, 1, \ldots, t\}$, computing compatibility scores $q_t^T k_s$, then weighted-sums their values:

$$o_t = \sum_{s \le t} \alpha_{ts} v_s$$

### Matrix Form of Attention

$$Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V$$

$$\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^T}{\sqrt{d_h}}\right)V$$

In the code, vectors are handled one position at a time with a key-value cache. The mathematics is the same causal version: $s \le t$.

---

## Chapter 4 — Transformer Layer Interpretation

### One Transformer Layer

This toy GPT has `n_layer = 1`. That one Transformer layer contains:

1. RMSNorm before attention
2. Multi-head causal attention
3. Output projection $W_O$
4. Residual addition
5. RMSNorm before MLP
6. Two-linear-layer MLP: $W_2\operatorname{ReLU}(W_1 x)$
7. Residual addition

For the MLP, $d_{\rm ff} = rd$ where expansion ratio $r = 4$, independent of the number of attention heads $H$.

### Layer Diagram

```
h_t^(0)
  ↓
RMSNorm
  ↓
multi-head attention (Q, K, V → o_t)
  ↓
W_O o_t
  ↓
h_t^(1) = h_t^(0) + W_O o_t   ←── attention residual
  ↓
RMSNorm
  ↓
W_2 ReLU(W_1 x)
  ↓
h_t^(2) = h_t^(1) + MLP       ←── MLP residual
```

### How Attention Heads Are Concatenated

Each head returns a vector of dimension $d_h = d / H$. In microGPT: $d = 16$, $H = 4$, $d_h = 4$.

The four head outputs are $o_t^{(1)}, o_t^{(2)}, o_t^{(3)}, o_t^{(4)} \in \mathbb{R}^4$.

They are joined by concatenation:

$$o_t = \operatorname{Concat}\!\left(o_t^{(1)}, o_t^{(2)}, o_t^{(3)}, o_t^{(4)}\right) \in \mathbb{R}^{16}$$

Then the output projection mixes the four coordinate blocks:

$$a_t = W_O o_t$$

### $W_O$, $W_1$, and $W_2$

After head concatenation, $o_t \in \mathbb{R}^{16}$. The output projection:

$$a_t = W_O o_t,\qquad W_O \in \mathbb{R}^{16 \times 16}$$

The MLP with expansion ratio $r = 4$:

$$\operatorname{MLP}(x) = W_2\operatorname{ReLU}(W_1 x)$$

$$W_1 \in \mathbb{R}^{64 \times 16},\qquad W_2 \in \mathbb{R}^{16 \times 64}$$

> $r = 4$ is independent of the number of attention heads $H$.

---

## Chapter 5 — Prediction, Loss, and Adam

### Output Prediction

The final hidden vector $h_t^{(2)} \in \mathbb{R}^{16}$ is projected by the output matrix:

$$W_{\mathrm{out}} \in \mathbb{R}^{27 \times 16}$$

to produce logits:

$$\ell_t = W_{\mathrm{out}} h_t^{(2)} \in \mathbb{R}^{27}$$

> **Logits are raw, unnormalized scores before softmax.**

Softmax produces probabilities:

$$p_{t,j} = \frac{e^{\ell_{t,j}}}{\sum_{k=0}^{26} e^{\ell_{t,k}}}$$

### Loss: Negative Log-Likelihood

The target is a discrete token ID:

$$\boxed{y_t = x_{t+1} \in \{0,\ldots,V-1\}}$$

The loss at position $t$:

$$L_t = -\log p_{t,y_t}$$

For the full sequence:

$$L(\theta) = -\frac{1}{T}\sum_{t=0}^{T-1}\log P_\theta\!\left(x_{t+1}\mid x_{\max(0,t-B+1)},\ldots,x_t\right)$$

This is cross-entropy / negative log-likelihood. It is small when the model assigns high probability to the observed next token.

### Parameter Space

$$\theta = \{E_{\rm tok}, E_{\rm pos}, W_Q, W_K, W_V, W_O, W_1, W_2, W_{\rm out}\}$$

$$\dim(\Theta) = \underbrace{2Vd}_{E_{\rm tok}+W_{\rm out}} + \underbrace{Bd}_{E_{\rm pos}} + \underbrace{12Ld^2}_{\text{attention and MLP matrices}}$$

For one Transformer layer:

$$12d^2 = \underbrace{4d^2}_{W_Q, W_K, W_V, W_O} + \underbrace{8d^2}_{W_1, W_2}$$

With $V=27$, $B=16$, $d=16$, $L=1$:

$$2(27)(16) + 16(16) + 12(1)(16^2) = 4192,\quad \Theta = \mathbb{R}^{4192}$$

### Adam Update

For each parameter coordinate, let $g_t = \partial L_t / \partial\theta$. Adam keeps two running estimates:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

Bias correction:

$$\hat m_t = \frac{m_t}{1-\beta_1^t},\qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

Update:

$$\theta_t = \theta_{t-1} - \eta_t \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}$$

---

## Chapter 6 — Motivation

### Why Attention?

The old bottleneck: compress the whole previous sequence into one state:

$$h_t = f(h_{t-1}, x_t)$$

Attention avoids this by keeping all previous position representations and retrieving selectively:

$$o_t = \sum_{s\le t} \alpha_{ts} v_s$$

> **Do not memorize everything in one vector; retrieve what is relevant.**

### Why Query, Key, and Value?

The model must answer two questions:
- Where should I look?
- What information should I take?

So it separates matching from content:

$$q_t = W_Q x_t \quad \text{(request)},\quad k_s = W_K x_s \quad \text{(index)},\quad v_s = W_V x_s \quad \text{(content)}$$

$$\alpha_{ts} = \operatorname{softmax}_s\!\left(\frac{q_t^T k_s}{\sqrt{d_h}}\right),\qquad o_t = \sum_s \alpha_{ts} v_s$$

### Why Dot Product and Softmax?

The dot product is a cheap compatibility score:

$$q_t^T k_s = \|q_t\|\,\|k_s\|\cos\theta$$

In matrix form, all pairwise scores are computed by $QK^T$.

Softmax converts arbitrary scores into differentiable selection weights:

$$\alpha_s = \frac{e^{s_s}}{\sum_r e^{s_r}}$$

It is a smooth version of choosing the most relevant source.

### Why Divide by $\sqrt{d_h}$?

If coordinates of $q$ and $k$ have roughly unit variance, then $q^T k = \sum_{j=1}^{d_h} q_j k_j$ has variance about $d_h$, so typical scale grows like $\sqrt{d_h}$.

Dividing by $\sqrt{d_h}$ keeps score magnitudes stable and prevents softmax from saturating too early.

### Why Multiple Heads and an MLP?

Multiple heads allow several retrieval patterns in parallel:

$$o_t = o_t^{(1)} \Vert o_t^{(2)} \Vert \cdots \Vert o_t^{(H)}$$

Then $W_O$ mixes the head outputs.

- **Attention** communicates across positions: *which previous information matters?*
- **The MLP** performs nonlinear computation at each position: *how should the gathered information be transformed?*

### One-Sentence Summary

> **Embeddings create token-position vectors. Attention retrieves relevant previous information. The MLP transforms it nonlinearly. The output head turns it into next-token probabilities.**

---

## Appendix A — Mathematical Notation vs. Code Notation

$$h_t^{(\ell)} \leftrightarrow \texttt{x},\quad t \leftrightarrow \texttt{pos\_id},\quad \ell \leftrightarrow \texttt{li}$$

| Mathematics | Code |
|---|---|
| $E_{\rm tok}$ | `state_dict['wte']` |
| $E_{\rm pos}$ | `state_dict['wpe']` |
| $W_Q$ | `state_dict[f'layer{li}.attn_wq']` |
| $W_K$ | `state_dict[f'layer{li}.attn_wk']` |
| $W_V$ | `state_dict[f'layer{li}.attn_wv']` |
| $W_O$ | `state_dict[f'layer{li}.attn_wo']` |
| $W_1$ | `state_dict[f'layer{li}.mlp_fc1']` |
| $W_2$ | `state_dict[f'layer{li}.mlp_fc2']` |
| $W_{\rm out}$ | `state_dict['lm_head']` |

The code reuses `x`; the mathematical notation keeps layer and position explicit.

---

## Appendix B — Why the Initialization Scale Is 0.08

$$E_{\rm tok}[i,j] \sim \mathcal{N}(0,\, 0.08^2)$$

The value $0.08$ is a heuristic, not a mathematical constant. It breaks symmetry while keeping early activations, attention scores, and gradients moderate.

- **Too large:** dot products $q^T k$ may be large, making softmax too sharp.
- **Too small:** signals and gradients may be weak.
- **More systematic alternatives:** Xavier and He initialization.

```python
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)]
    for _ in range(nout)
]
```

---

## Appendix C — RMSNorm

For $x \in \mathbb{R}^d$:

$$\operatorname{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \varepsilon}$$

General RMSNorm is:

$$\operatorname{RMSNorm}_g(x) = g \odot \frac{x}{\operatorname{RMS}(x)}$$

Residual additions can cause scale drift. RMSNorm stabilizes attention-score and MLP-input magnitudes. Unlike LayerNorm, it does not subtract the mean.

> **RMSNorm stabilizes the scale of hidden vectors before learned transformations.**

---

## Appendix D — Learned vs. Fixed RMSNorm Gain $g$

Standard RMSNorm usually learns $g \in \mathbb{R}^d$ with $g_i = 1$ at initialization.

- **Fixed $g = \mathbf{1}$:** normalization only
- **Learned $g$:** normalization plus coordinatewise rescaling

This microGPT fixes $g = \mathbf{1}$ implicitly, so $g$ is not a trainable parameter.

```python
def rmsnorm(x):                 # actual microGPT version
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# learned-gain version would use:
# return [gi * xi * scale for xi, gi in zip(x, gain)]
```

Fixing $g$ is valid for a tiny model; learned $g$ usually improves flexibility at negligible cost.

---

## Appendix E — Dimensions of Q, K, and V

For each attention head:

$$q_t, k_t, v_t \in \mathbb{R}^{d_h},\qquad d_h = \frac{d}{H}$$

For microGPT: $d = 16$, $H = 4$, $d_h = 4$, so $q_t, k_t, v_t \in \mathbb{R}^4$.

Across all heads: $H d_h = d = 16$.

> $d$ determines the total Q, K, V dimension, while $d_h = d/H$ is the dimension per head.

---

## Appendix F — Mathematical Form of the MLP

$$\operatorname{MLP}(x) = W_2\operatorname{ReLU}(W_1 x)$$

where $x \in \mathbb{R}^d$, $W_1 \in \mathbb{R}^{d_{\rm ff} \times d}$, $W_2 \in \mathbb{R}^{d \times d_{\rm ff}}$.

For microGPT: $d = 16$, $d_{\rm ff} = 64$.

Let $z = W_1 x \in \mathbb{R}^{64}$, $\operatorname{ReLU}(z)_i = \max(0, z_i)$, then $y = W_2\operatorname{ReLU}(z) \in \mathbb{R}^{16}$.

Coordinate form:

$$y_i = \sum_{j=1}^{64}(W_2)_{ij}\max\!\left(0,\sum_{k=1}^{16}(W_1)_{jk} x_k\right)$$

> **$W_1$ expands $16 \to 64$, ReLU adds nonlinearity, $W_2$ compresses $64 \to 16$.**
