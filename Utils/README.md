# safetensors → GGUF, on a standard Colab runtime

One notebook that converts a diffusion or LLM checkpoint to GGUF, quantizes it, and then
checks the result against the original before you publish anything.

Conversion runs on CPU and streams through disk, so peak memory tracks your largest single
tensor rather than the model. A 12 GB checkpoint converts inside a 12.7 GB runtime, and no
GPU is involved until you actually generate something.

Built on [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) and
[llama.cpp](https://github.com/ggerganov/llama.cpp).

---

## Contents

- [Quick start](#quick-start)
- [What happens when you run it](#what-happens-when-you-run-it)
- [Worked examples](#worked-examples)
- [Parameter reference](#parameter-reference)
- [What gets checked](#what-gets-checked)
- [Resources](#resources)
- [Already-quantized sources](#already-quantized-sources)
- [Known limits](#known-limits)
- [Fixed along the way](#fixed-along-the-way)
- [Troubleshooting](#troubleshooting)

---

## Quick start

1. Open the notebook in Colab. A standard CPU runtime is enough — you don't need a GPU.
2. Fill in **1a** (what you're doing) and **1b** (where the model is).
3. Runtime → Run all.

Everything else has working defaults.

```
1a   JOB      = convert and check
     TASK     = image generation
     OUT_DIR  = /content/gguf-out

1b   SRC_FROM = download link
     SRC_URL  = https://huggingface.co/<repo>/resolve/main/<file>.safetensors
```

Output lands in `OUT_DIR` along with `verification_report.md`.

---

## What happens when you run it

The cells run in this order. Each prints what it decided, so you can stop and change
something before the slow parts begin.

| # | Step | What it does |
|---|---|---|
| 1 | **Apply & check** | Reads your form settings, works out the pipeline, fails fast on anything contradictory |
| 2 | **Download** | Pulls the checkpoint with aria2c (16 connections, resumable), picking the right auth for the host |
| 3 | **The checkpoint** | Reads the safetensors header only — tensor count, dtypes, tensors over 4 dimensions, whether it's already quantized |
| 4 | **Architecture** | Fetches the family definitions from ComfyUI-GGUF and identifies the model from its tensor names |
| 5 | **What can be built** | Decides which requested outputs are actually possible, and drops the ones that aren't |
| 6 | **Toolchain** | Clones ComfyUI-GGUF, and builds `llama-quantize` only if a requested quant needs it |
| 7 | **Convert** | Streams the checkpoint one tensor at a time into the base GGUF |
| 8 | **Quantize** | Produces each quant from the base file |
| 9 | **Restore** | Puts back any tensors over 4 dimensions, which GGUF can't hold during conversion |
| 10 | **Relabel** *(optional)* | Changes the architecture string on files you already built |
| 11 | **Check** | Diffs every output against the source checkpoint |
| 12 | **Verdict & report** | Pass/fail summary, writes `verification_report.md`, optional upload |

Step 5 is the one worth reading. It prints a plan before anything is built:

```
plan:
  mix      build     written as Q8_0 - same size as the source
  Q8_0     build     the size-matched target for an 8-bit source
  Q4_K_M   build     4-bit from an 8-bit source stacks two rounding steps
  Q8_0 and mix are the same file here
```

---

## Worked examples

### A Flux-family model from Hugging Face

```
1a   JOB = convert and check      TASK = image generation
1b   SRC_FROM = download link
     SRC_URL  = https://huggingface.co/<repo>/resolve/main/model.safetensors
1c   WANT_MIX = on   WANT_8BIT = on   WANT_4BIT = on   RECIPE_4BIT = Q4_K_M
1e   NEED_VAE = on   NEED_TEXT_ENCODER = on
```

Detection finds `flux`, `llama-quantize` accepts it, you get BF16 + Q8_0 + Q4_K_M.

### A model from Civitai

```
1b   SRC_FROM = download link
     SRC_URL  = https://civitai.com/api/download/models/<id>?fileId=<id>
     CIVITAI_TOKEN = (or set a Colab secret named CIVITAI_TOKEN)
```

Civitai needs a token for most downloads. The filename comes from the server, since the URL
doesn't contain one.

### A video model

```
1a   TASK = video generation
```

Video checkpoints carry tensors with more than four dimensions. GGUF can't hold those during
conversion, so they're held back and restored after quantizing. The notebook does this
automatically and tells you how many.

### An already-quantized (int8/fp8) checkpoint

```
1d   IF_ALREADY_QUANTIZED = unpack and convert
     IF_TARGET_TOO_HIGH   = drop it
```

Float weights get rebuilt from the integers and their scales. The base file is written at
Q8_0 rather than BF16, and any target that can't beat the source is dropped. See
[Already-quantized sources](#already-quantized-sources).

### An architecture nothing supports yet

```
1d   ARCH_MODE       = detect
     IF_ARCH_UNKNOWN = quantize in python
```

Converts under a derived name, with a template worked out from the weights. Note the
[known limit](#known-limits): ComfyUI won't load an architecture it doesn't know.

### Krea 2, using the community fork

```
1d   GGUF_NODE_REPO = RealRebelAI/ComfyUI-GGUF_KREA-2
     ARCH_MODE      = force
     ARCH_NAME      = krea2
1e   NEED_VAE = on   NEED_TEXT_ENCODER = on
```

### Checking files you already have

```
1a   JOB     = check only
     OUT_DIR = /path/to/your/gguf/files
1b   SRC_FROM = local path
     SRC_PATH = /path/to/original.safetensors
```

---

## Parameter reference

### 1a — Task and files

| Parameter | Default | What it does |
|---|---|---|
| `JOB` | `convert and check` | `convert only` skips verification. `check only` skips to verifying whatever is already in `OUT_DIR` |
| `TASK` | `image generation` | Decides which converter runs and which checks apply. `auto-detect` guesses from the source |
| `WORK_DIR` | `/content/gguf-work` | Scratch space for the toolchain and downloads |
| `OUT_DIR` | `/content/gguf-out` | Where finished files go |
| `RUNTIME` | `auto` | Which architecture whitelist the output is judged against. `auto` = ComfyUI for image/video, llama.cpp otherwise |

### 1b — The model

| Parameter | Default | What it does |
|---|---|---|
| `SRC_FROM` | `download link` | `download link`, `local path`, or `Hugging Face repo` |
| `SRC_URL` | — | Direct URL. Several comma separated is fine; the first is the model, the rest are companion files |
| `SRC_PATH` | — | A `.safetensors` file, or a model folder for the LLM tasks |
| `SRC_REPO` | — | Hub repo id, for models that need a whole folder |
| `SRC_FILE` | — | One file inside that repo. Empty pulls the folder |
| `SRC_REVISION` | `main` | Branch, tag or commit |
| `HF_TOKEN` | — | Leave blank if you've set a Colab secret named `HF_TOKEN` |
| `CIVITAI_TOKEN` | — | Same, `CIVITAI_TOKEN`. Needed for most Civitai downloads |
| `CONNECTIONS` | `16` | Parallel connections per download |

Hugging Face takes an `Authorization` header, Civitai takes a `token` query parameter. The
notebook picks the right one per host, so a mix of links works.

### 1c — What to build

| Parameter | Default | What it does |
|---|---|---|
| `WANT_MIX` | `True` | The base file everything else is quantized from. Keeps the source's precision |
| `WANT_8BIT` | `True` | Build the 8-bit quant |
| `WANT_4BIT` | `True` | Build the 4-bit quant |
| `RECIPE_8BIT` | `Q8_0` | Which 8-bit format |
| `RECIPE_4BIT` | `Q4_K_M` | `Q4_K_M`, `Q4_K_S`, `Q4_0`, `IQ4_NL`, `IQ4_XS` |
| `EXTRA_QUANTS` | `none` | Anything else, comma separated: `Q6_K`, `Q5_K_M`, `Q3_K_M`, `Q2_K` |
| `IF_QUANT_UNAVAILABLE` | `substitute the nearest` | When a target can't be built, use the closest thing that can, or drop it |
| `EXPECT_MMPROJ` | `auto` | Export the vision/audio projector for multimodal LLMs |

### 1d — Architecture

| Parameter | Default | What it does |
|---|---|---|
| `ARCH_MODE` | `detect` | `detect` reads the family from the tensor names. `force` writes `ARCH_NAME` regardless |
| `ARCH_NAME` | — | Name to write. Under `detect`, used only for a checkpoint nothing matches. Empty derives one from the filename |
| `IF_ALREADY_QUANTIZED` | `unpack and convert` | `unpack` rebuilds float weights from the scales. `stop` refuses. `convert as-is` writes the packed integers, which produces noise |
| `IF_TARGET_TOO_HIGH` | `drop it` | For a target that can't improve on the source: drop, build anyway, or stop |
| `IF_ARCH_UNKNOWN` | `quantize in python` | When `llama-quantize` doesn't know the architecture: quantize here, drop the quants, or borrow a known name |
| `STAND_IN_ARCH` | `flux` | Which family to borrow, for `stand-in arch` |
| `PROGRESS` | `summary` | `every tensor` lists each one with dtype, shape and why it landed where it did |
| `GGUF_NODE_REPO` | `city96/ComfyUI-GGUF` | Which repo the architecture definitions come from. Point at a fork to pick up families it adds |
| `GGUF_NODE_REF` | `main` | Branch or tag of that repo |
| `LLAMA_TAG` | `b3962` | llama.cpp tag to build `llama-quantize` from |

### 1e — Checks

| Parameter | Default | What it does |
|---|---|---|
| `COMPARE_TO_SOURCE` | `True` | Diff tensor names and shapes against the original. The strongest check here |
| `COMPARE_QUANTS_TO_BASE` | `True` | Check each quant kept what the base file had |
| `NAN_CHECK` | `normal` | `off`, `quick` (20), `normal` (40), `thorough` (200), `every tensor` |
| `KEEP_IN_F32` | `auto` | Weights that must stay full precision. `auto` uses whatever the detected family protects. Or type comma separated patterns |
| `NEED_VAE` | `False` | Require a VAE file alongside |
| `NEED_TEXT_ENCODER` | `False` | Require a text encoder alongside |
| `NEED_CONNECTORS` | `False` | Require connector files alongside |
| `NEED_OTHER_FILES` | `none` | Any other required filenames or globs |
| `STRICTNESS` | `normal` | `strict` turns warnings into failures |
| `SAVE_REPORT` | `True` | Write `verification_report.md` into `OUT_DIR` |
| `UPLOAD_REPO` | — | Push the folder to a Hub repo. Only runs if the checks passed |

### Relabel cell

| Parameter | Default | What it does |
|---|---|---|
| `RELABEL_TO` | — | New architecture string. Empty skips the cell entirely |
| `RELABEL_FILES` | `*.gguf` | Which files, as a glob relative to `OUT_DIR` |

---

## What gets checked

- every requested file exists
- tensor names and shapes match the source checkpoint exactly
- nothing over 4 dimensions went missing
- no 1-D tensor was quantized
- weights that must stay full precision are still F32
- each quant kept the same tensors and shapes as the base
- vocab size matches the embedding matrix (LLM path)
- the projector has the tower the task needs (vision/audio)
- no NaN or Inf
- the architecture string matches what the weights actually look like

Ends with a verdict and writes `verification_report.md`.

**Why bother:** a correct file and a broken one can have exactly the same size and tensor
count. In testing, both had 5 tensors while the broken one had a wrong shape, a missing 5-D
tensor, two weights wrongly quantized, and a NaN.

---

## Resources

| | |
|---|---|
| RAM | ~0.3 GB above the runtime baseline, regardless of model size |
| Disk | roughly **2× the output size** free while writing |
| GPU | none needed for convert or quantize |
| Time | minutes for conversion; the first run also builds `llama-quantize` |

Measured peak anonymous memory, before and after the streaming rewrite:

| model size | before | after |
|---|---|---|
| 0.17 GB | 0.47 GB | 0.34 GB |
| 0.69 GB | 0.99 GB | 0.33 GB |

Before, peak grew with the model. After it's flat — that 0.3 GB is the Python baseline plus
the writer's spool buffer.

The output is spooled to a temp file and then copied, which is where the disk requirement
comes from. On a big model you'll hit `No space left on device` long before you hit a memory
limit.

`llama-quantize` is only built when you ask for a quant type that needs it.

---

## Already-quantized sources

An int8 or fp8 checkpoint stored next to its scale tensors is handled by rebuilding the
float weights (`weight × scale`) before conversion. Verified numerically: max error against
a correct manual unpack was 4.9e-4 on a 0.22 weight range, which is bf16 rounding.

Two rules follow, and the notebook applies them:

- **An 8-bit source stays 8-bit.** The base file is written at Q8_0, not BF16. Storing 8-bit
  data in 16-bit containers doubles the file for no gain.
- **You can go down, not up.** Q4 from an 8-bit source is a real size saving that stacks two
  rounding steps. BF16 from an 8-bit source is just a bigger file.

Supported scale layouts: `weight_scale`, `scale_weight`, `weight_scale_inv` (inverted) and
`.scales`, per-tensor or per-channel.

GPTQ/AWQ-style bit-packed weights (`qweight`, `qzeros`, `g_idx`) are refused — reversing
those needs the original quantization library, and guessing wrong is silent corruption.

---

## Known limits

**A derived architecture converts correctly but ComfyUI won't load it.** The node has its
own list of architectures it accepts. Producing a valid GGUF and having something able to
read it are separate problems, and only the first is solved here. Forcing a known name to
get past that list is how a file ends up loading and being wrong.

**K-quants need llama.cpp to know the architecture.** `lcpp.patch` teaches it eleven names.
Outside those, quantization falls back to Python, which can do Q8_0, Q5_1, Q5_0, Q4_1 and
Q4_0 but not the K or IQ types. `IF_ARCH_UNKNOWN = stand-in arch` works around it by
relabelling for the quantize step, at the cost of llama.cpp applying that family's
protected-tensor names, which won't match your model. Check the dtypes in the report if you
use it.

**The LLM path is less exercised.** Text, vision, audio and embedding go through llama.cpp's
`convert_hf_to_gguf.py`. It works the way it always has, but the diffusion path is the one
that has been tested end to end here.

---

## Fixed along the way

Things in the upstream tooling this works around:

- `GGUFWriter` keeps every converted tensor in RAM (`use_temp_file` defaults to `False`), so
  peak usage was input + output + a working copy
- `handle_nd_tensor` is implemented on 2 of 11 architectures and stores exactly one tensor;
  a second raises "fix file already exists", everything else raises `NotImplementedError`
- `fix_5d_tensors.py` crashes on numpy 2 with *only 0-dimensional arrays can be converted to
  Python scalars*
- `strip_prefix` silently drops keys that don't carry the prefix — the notebook reports how
  many
- `llama-quantize` refuses a quantized input unless given `--allow-requantize`
- forcing `keys_detect = [()]` disables architecture detection rather than satisfying it,
  since `all([])` is `True`

---

## Troubleshooting

**`no .gguf files in ...`** — you're pointing at the checkpoint rather than the converted
output, or nothing has been converted yet. The notebook lists what it found and looks for
`.gguf` files elsewhere on the machine.

**`unknown model architecture` from llama-quantize** — the architecture isn't one of the
eleven its patch knows. Set `IF_ARCH_UNKNOWN` in 1d.

**`arch is '...', not in ComfyUI's list`** — the file is fine, the name is wrong. Either
re-run with `ARCH_MODE = force` and the right `ARCH_NAME`, or use the relabel cell on the
files you already built. Relabelling copies the tensors untouched, so it's much quicker than
converting again.

**Output is twice the size of the source** — an 8-bit source should produce a Q8_0 base, not
BF16. Check `IF_TARGET_TOO_HIGH` is not set to `build it anyway`.

**`No space left on device`** — you need roughly twice the output size free while writing.

**Out of memory** — shouldn't happen; peak is a flat ~0.3 GB. If it does, `PROGRESS = every
tensor` will show which tensor it died on.

---

## Credits

- [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — the converter, the
  architecture templates, and the llama.cpp patch this depends on
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — `llama-quantize` and `gguf-py`
- [RealRebelAI/ComfyUI-GGUF_KREA-2](https://github.com/RealRebelAI/ComfyUI-GGUF_KREA-2) — a
  fork adding `krea2`

Model licences are the model author's. Converting a checkpoint doesn't change its terms, so
check before you redistribute.
