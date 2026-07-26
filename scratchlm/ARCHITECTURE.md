# ScratchLM architecture

ScratchLM is a causal decoder-only Transformer. For tokens
\(t_1,\ldots,t_T\), it models

\[
p(t_1,\ldots,t_T)=\prod_{i=1}^{T}p(t_i\mid t_1,\ldots,t_{i-1}).
\]

## 1. Tokenization

UTF-8 text is represented by bytes. IDs `0..255` are bytes, `256` is BOS,
and `257` is EOS. This is lossless and multilingual, although less
sequence-efficient than a learned BPE tokenizer.

## 2. Embedding

For token ID \(t_i\), the initial hidden state is

\[
x_i^{(0)}=E[t_i]\in\mathbb{R}^{d}.
\]

There is no learned absolute positional embedding. Position enters through
rotary position embedding (RoPE) applied to queries and keys.

## 3. Transformer block

Each pre-normalized residual block is

\[
\tilde x = x + \operatorname{Attention}(\operatorname{RMSNorm}(x)),
\]

\[
x' = \tilde x + \operatorname{SwiGLU}(\operatorname{RMSNorm}(\tilde x)).
\]

For head \(h\), causal self-attention is

\[
A_h=\operatorname{softmax}\!\left(
\frac{Q_hK_h^\top}{\sqrt{d_h}}+M
\right),\qquad O_h=A_hV_h,
\]

where \(M_{ij}=-\infty\) for \(j>i\). Thus position \(i\) cannot inspect a
future token.

The feed-forward map is

\[
\operatorname{SwiGLU}(x)=W_o\left[
\operatorname{SiLU}(W_gx)\odot(W_vx)
\right].
\]

## 4. Output and objective

The output logits are

\[
z_i=E^\top\operatorname{RMSNorm}(x_i^{(L)}),
\]

where the embedding matrix is weight-tied with the language-model head. The
training objective is next-token cross entropy:

\[
\mathcal L=-\frac1N\sum_{i=1}^{N}\log p_\theta(t_{i+1}\mid t_{\le i}).
\]

## 5. What "from scratch" means here

The tokenizer, model architecture, attention, optimizer schedule, training
loop, checkpointing, and sampler are implemented in this repository. PyTorch
is used only for tensors, automatic differentiation, and device execution.
There is no Hugging Face model, pretrained checkpoint, or Transformer module.
