# ScratchLM

A runnable decoder-only language model built directly in PyTorch. It contains:

- lossless UTF-8 byte tokenization written from scratch;
- RMSNorm, RoPE, causal multi-head self-attention, and SwiGLU;
- tied input/output embeddings;
- AdamW training with warmup, cosine decay, gradient accumulation, clipping,
  validation, perplexity, checkpointing, and resume;
- CPU, Apple Silicon MPS, and CUDA device selection;
- no Hugging Face, no pretrained model, and no tokenizer package.

This is the next step beyond your character-level microGPT: the same
next-token principle, but with a modern multi-layer Transformer and a general
UTF-8 corpus.

## Reality check

The code is an LLM architecture, but a laptop run produces a **small language
model**, not a competitive commercial LLM. Model quality depends mostly on
training tokens and compute. The default `tiny` model has about 3.4 million
parameters; `small` has about 10.8 million.

## 1. Create an environment

### macOS / Linux

```bash
cd scratchlm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
cd scratchlm
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run the smoke test

```bash
python tests/smoke_test.py
```

## 3. Get a first corpus

```bash
python download_tinyshakespeare.py
python prepare_data.py data/tinyshakespeare.txt --out-dir data/shakespeare
```

You can replace `data/tinyshakespeare.txt` with any UTF-8 `.txt` file. A useful
first experiment is 10-100 MB of clean text. A 1 MB corpus only demonstrates
that the system works.

## 4. Train

Good first command for your M1 Mac with 16 GB unified memory:

```bash
python train.py \
  --data-dir data/shakespeare \
  --out-dir out/shakespeare-tiny \
  --preset tiny \
  --batch-size 8 \
  --grad-accum 4 \
  --max-steps 5000
```

On Windows PowerShell, put the command on one line or replace each `\` with a
PowerShell backtick.

For the approximately 10.8M-parameter model:

```bash
python train.py --data-dir data/shakespeare --out-dir out/shakespeare-small --preset small --batch-size 4 --grad-accum 8 --max-steps 10000
```

The script automatically chooses CUDA, MPS, or CPU. To force Apple Silicon:

```bash
python train.py --data-dir data/shakespeare --device mps
```

To use the faster fused attention operation while retaining your own model
architecture:

```bash
python train.py --data-dir data/shakespeare --use-sdpa
```

Without `--use-sdpa`, attention is explicitly computed as
`softmax(QK^T/sqrt(d) + causal_mask)V` in `scratchlm/model.py`.

## 5. Generate text

```bash
python generate.py \
  --checkpoint out/shakespeare-tiny/best.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 400 \
  --temperature 0.8 \
  --top-k 50
```

## 6. Resume training

```bash
python train.py \
  --data-dir data/shakespeare \
  --out-dir out/shakespeare-tiny \
  --resume out/shakespeare-tiny/last.pt \
  --max-steps 10000
```

The new `--max-steps` is the final global step, not the number of extra steps.

## Project map

```text
scratchlm/
├── scratchlm/
│   ├── tokenizer.py   # UTF-8 bytes ↔ token IDs
│   ├── config.py      # model configuration and size presets
│   ├── model.py       # the decoder-only Transformer
│   └── data.py        # memory-mapped random training batches
├── prepare_data.py    # text → train.bin / val.bin
├── train.py           # optimization, evaluation, checkpointing
├── generate.py        # autoregressive sampling
├── tests/smoke_test.py
└── ARCHITECTURE.md     # mathematical description
```

## Scaling rules

Memory and training cost grow approximately as

- parameters: `O(n_layer * n_embd^2)`;
- attention compute and activation memory: `O(batch * n_layer * block_size^2 * n_embd)`;
- data requirement: at least tens of tokens per parameter for a serious run.

On an M1 with 16 GB, increase only one of `batch_size`, `block_size`, or model
width at a time. When memory is tight, reduce `--batch-size` and increase
`--grad-accum` to preserve the effective batch size.

## Deliberate limitations

- Byte tokens make sequences longer than BPE tokens.
- Generation recomputes the context and has no KV cache.
- Training is single-device.
- There is no instruction tuning, preference optimization, retrieval, or tool use.

Those are the appropriate next milestones after the base model trains correctly.
