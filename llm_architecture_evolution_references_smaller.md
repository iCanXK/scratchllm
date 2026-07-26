# LLM Architecture Evolution
**From the Initial Transformer to microGPT, GPT-1, GPT-2, and GPT-3**

> **Prepared with ChatGPT**

---

## Roadmap

1. [The Initial Transformer (2017)](#the-initial-transformer-2017)
2. [microGPT](#microgpt)
3. [GPT-1](#gpt-1)
4. [GPT-2](#gpt-2)
5. [GPT-3](#gpt-3)
6. [Evolution Summary](#evolution-summary)
7. [References](#references)

---

## The Initial Transformer (2017)

### Core Idea

The paper *Attention Is All You Need* introduced the **Transformer**: a sequence model built from

$$\text{self-attention} + \text{feed-forward layers} + \text{residual connections} + \text{normalization}$$

Key properties:
- Encoder–decoder architecture for sequence-to-sequence tasks such as translation.
- No recurrence and no convolution in the main sequence-processing path.
- Positional encoding is added because attention alone does not encode order.
- Decoder uses masked self-attention plus cross-attention to the encoder outputs.

### Architecture

**Encoder** (repeated $N$ times):

```
Input Embedding
  + positional encoding
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Feed Forward
  ↓
Add & Norm
```

**Decoder** (repeated $N$ times):

```
Output Embedding (shifted right)
  + positional encoding
  ↓
Masked Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Encoder–Decoder Attention ← (receives encoder output)
  ↓
Add & Norm
  ↓
Feed Forward
  ↓
Add & Norm
  ↓
Linear
  ↓
Softmax → Next-token probabilities
```

---

## microGPT

### What It Is

microGPT is a tiny, dependency-free Python implementation of a decoder-only GPT designed to expose the complete training and inference algorithm rather than maximize efficiency.

- Character-level language model on names.
- One Transformer layer: $L = 1$.
- Model width $d = 16$, number of heads $H = 4$, head size $d_h = 4$.
- Uses **RMSNorm** and a two-layer MLP $W_2\operatorname{ReLU}(W_1 x)$.
- Uses a learned token embedding $E_{\text{tok}}$ and learned positional embedding $E_{\text{pos}}$.

### Architecture

```
Token ID x_t
  ↓
E_tok[x_t] + E_pos[t] = h_t^(0)
  ↓
RMSNorm
  ↓
Masked multi-head attention (Q, K, V → o_t)
  ↓
W_O o_t
  ↓                             ←── attention residual from h_t^(0)
h_t^(1) = h_t^(0) + W_O o_t
  ↓
RMSNorm
  ↓
W_2 ReLU(W_1 x)
  ↓                             ←── MLP residual from h_t^(1)
h_t^(2) = h_t^(1) + MLP
  ↓
LM head / logits
  ↓
Softmax probabilities
```

---

## GPT-1

### Generative Pre-Training

Introduced in *Improving Language Understanding by Generative Pre-Training* (2018).

- Decoder-only Transformer adapted from the original Transformer by **discarding the encoder and cross-attention**.
- Pretraining objective:

$$P(x_1,\ldots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

- Fine-tuning attaches task-specific heads after generative pretraining.

**Typical GPT-1 configuration:**

| Hyperparameter | Value |
|---|---|
| Layers | 12 |
| Heads | 12 |
| Hidden size | 768 |
| Context | 512 |

### Architecture

```
Input token IDs
  ↓
Token embedding + learned position embedding
  ↓
┌─────────────────────────────────────────────┐
│  Transformer block                          │
│  masked self-attention + feed-forward  ×12  │
└─────────────────────────────────────────────┘
  ↓
Final hidden states
  ↓
LM head / logits
  ↓
Softmax → Next-token probabilities
```

> **Key architectural change from the initial Transformer:** Remove the encoder and cross-attention; keep only a stack of **causal decoder blocks**.

---

## GPT-2

### Scaling the Decoder-Only GPT

GPT-2 kept the same basic decoder-only architecture as GPT-1, but scaled model size, dataset size, and context handling enough to demonstrate strong zero-shot and multitask behavior.

- Same core pattern: token embeddings → stack of masked self-attention blocks → LM head.
- Larger context length: **1024 tokens**.
- Largest published model: ~**1.5B parameters**.
- Emphasis shifted from "pretrain then fine-tune" toward **"pretrain and prompt/use directly."**

### Architecture (GPT-2 as an Enlarged GPT-1)

```
BPE token IDs
  ↓
Token embedding + learned position embedding
  ↓
Decoder-only Transformer stack
(same pattern as GPT-1, but much larger)
  ↓
LM head / logits
  ↓
Softmax
```

| Variant | Layers | Heads | Hidden | Context | Params |
|---|---|---|---|---|---|
| GPT-2 small | 12 | 12 | 768 | 1024 | 117M |
| GPT-2 XL | 48 | 25 | 1600 | 1024 | ~1.5B |

---

## GPT-3

### Scaling to Few-Shot Learning

GPT-3 retained the decoder-only Transformer pattern but scaled it drastically. The headline result was strong **in-context learning**: the model can perform new tasks from prompts alone, with zero, one, or a few demonstrations.

- Decoder-only Transformer with causal self-attention.
- Context length: **2048 tokens**.
- Largest model: **175B parameters**.
- Major conceptual shift: **prompting becomes a central interface to behavior**.

### Architecture

```
Prompt tokens
  ↓
Token embedding + position embedding
  ↓
Decoder-only Transformer stack
(deep causal self-attention network)
  ↓
LM head
  ↓
Next-token distribution
```

**GPT-3 highlights:**

| Hyperparameter | Value |
|---|---|
| Layers | 96 |
| Hidden size | 12,288 |
| Attention heads | 96 |
| Context length | 2048 |
| Parameters | 175B |

> **Interpretation:** Architecturally close to GPT-2, but large enough for few-shot and in-context learning to become prominent behaviors.

---

## Evolution Summary

### Architecture Evolution at a Glance

| Model | Core architecture | Objective | Scale | Shift |
|---|---|---|---|---|
| Initial Transformer | Encoder–decoder | Sequence-to-sequence | 6 encoder + 6 decoder layers | Attention replaces recurrence |
| microGPT | Tiny decoder-only GPT | Character-level next-token prediction | 1 layer, $d=16$, 4 heads | Minimal educational implementation |
| GPT-1 | Decoder-only Transformer | Generative pretraining + fine-tuning | 12 layers, $d=768$ | Transfer via pretraining |
| GPT-2 | Larger decoder-only Transformer | Large-scale next-token prediction | Up to 1.5B params | Zero-shot behavior emerges |
| GPT-3 | Much larger decoder-only Transformer | Massive next-token prediction | 175B params | Few-shot / in-context learning |

*References: Vaswani et al. (2017), Radford et al. (2018, 2019), Brown et al. (2020).*

### The Main Structural Trend

```
encoder–decoder Transformer  →  decoder-only GPT family  →  same skeleton, scaled up
```

- The original Transformer introduced the attention-based architecture.
- GPT keeps only the causal decoder side.
- microGPT strips that architecture down to its essentials.
- GPT-1, GPT-2, and GPT-3 mostly differ by **scale**, training data, and behavior induced by scale.

---

## References

1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. *Attention Is All You Need*. arXiv:1706.03762, 2017. <https://arxiv.org/abs/1706.03762>

2. Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. *Improving Language Understanding by Generative Pre-Training*. OpenAI technical report, 2018. <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>

3. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. *Language Models are Unsupervised Multitask Learners*. OpenAI technical report, 2019. <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>

4. Tom B. Brown et al. *Language Models are Few-Shot Learners*. arXiv:2005.14165, 2020. <https://arxiv.org/abs/2005.14165>

5. Andrej Karpathy. Minimal GPT-style educational code used as the microGPT reference.
