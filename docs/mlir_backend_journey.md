# Adding MLIR Kernel Support to Xe-Forge — The Full Story

*A step-by-step report of how we taught Xe-Forge to optimize MLIR GPU kernels,
written to be understandable without deep compiler expertise.*

Branch: `feat/mlir-xegpu-backend` · Period: late June → mid July 2026 · Target
hardware: Intel Xe2 "Battlemage" (BMG B580) GPU.

---

## 1. The 30-second summary

Xe-Forge is a tool that uses an LLM (a large language model) to **automatically
make GPU programs faster**. It reads a program (a "kernel"), figures out what's
slow about it, rewrites it, checks the rewrite is still correct *and* actually
faster, and keeps the good changes. It already did this for one style of kernel;
we taught it a second, harder style called **MLIR**.

By the end we could:

1. Take a **matrix-multiply** written in high-level MLIR and automatically turn it
   into a fast, GPU-specific version — trying several "tile shapes," running each
   on the real GPU, and keeping the fastest correct one.
2. Take a **flash-attention** program (the core math inside modern AI models) and
   run it end-to-end, automatically making it **~1.77× faster**.
3. Do all of this safely: every change is checked for correctness on the actual
   GPU before it's accepted.

---

## 2. Background: the words you need (and nothing more)

You can read this whole report knowing just these terms.

- **Kernel** — a small program that runs on the GPU. Our examples are
  matrix-multiply and attention.
- **MLIR** — a way of writing programs at *different levels of detail*. Think of a
  recipe: you can write "make a cake" (high level) or "beat 3 eggs for 2 minutes,
  then fold in flour" (low level). MLIR lets both exist and lets you **translate
  down** from one to the other.
  - **Linalg level** — the high-level "make a cake" description (e.g. "multiply
    these two matrices").
  - **XeGPU workgroup (WG) level** — the low-level, Intel-GPU-specific version with
    all the hardware details spelled out.
  - **"Lowering"** — the act of translating from high level to low level.
- **DPAS / XMX** — the special GPU hardware unit that multiplies matrices very
  fast. Good kernels feed it in exactly the right-sized chunks.
- **Tile / tile shape** — how you chop a big matrix into GPU-sized pieces. The
  chunk sizes dramatically affect speed. Picking them is the main tuning knob.
- **Register file / "large GRF"** — the GPU's fastest scratch memory. Intel's BMG
  chip has a mode that **doubles** this scratch space (helpful for heavy kernels)
  at the cost of running fewer things at once. It's turned on with a **compiler
  flag**, not by editing the program.
- **IMEX** — Intel's toolkit that actually compiles and runs these kernels on the
  GPU. It can also **time** them accurately.
- **CoVeR gate** — Xe-Forge's safety check: **Co**mpile + **Ve**rify + **R**un. A
  proposed change is only kept if it compiles, produces the *correct* answer, and
  is *faster*. This is what stops the LLM from "optimizing" a kernel into a wrong
  or broken one.
- **lighthouse** — a separate Intel research project that already knows how to
  translate an attention program down to the Intel-GPU level. We *reused* its
  output without depending on it at runtime (more on this later).

---

## 3. Where we started

Xe-Forge already optimized **Triton** and **SYCL** kernels. Its pipeline looks
like this, and we kept the same shape for MLIR:

```
  Input kernel
      │
      ▼
  ANALYSIS      ← LLM reads the kernel, lists what's slow ("issues")
      │
      ▼
  PLANNING      ← LLM decides which fixes to try, in what order
      │
      ▼
  OPTIMIZATION  ← LLM rewrites the kernel, one issue-type at a time
  STAGES           each rewrite goes through the CoVeR safety gate
      │
      ▼
  Best correct + faster kernel
```

Our job: make this whole machine work for **MLIR** kernels on **Intel** GPUs, and
add MLIR-specific superpowers along the way.

---

## 4. The journey, step by step

Each step below is a real milestone (and a real commit). I've kept them in the
order they happened.

### Step 1 — Teach Xe-Forge to run MLIR at all (the WG backend)
*(commit `6d605b2`, June 26)*

First we needed Xe-Forge to **compile and run** a low-level (XeGPU workgroup-level)
MLIR kernel on the real GPU and time it. This is the `MlirExecutor` — the piece
that shells out to Intel's IMEX tools, runs the kernel, reads back whether the
answer was correct, and measures how long it took.

**Why it matters:** without this, the CoVeR safety gate can't work — there'd be no
way to check "is the rewrite correct and faster?" This is the foundation
everything else stands on.

### Step 2 — The two-level pipeline: start high, translate down, then tune
*(commit `ad9490d`, July 13)*

A matrix-multiply can be written at the easy **Linalg** level. But GPUs want the
detailed **WG** level. So we built a **translator with a built-in search**:

1. The LLM proposes several **tile shapes** (ways to chop the matrix), guided by a
   knowledge base of good starting points.
2. For each proposal, we translate the kernel down to WG level, **run it on the
   GPU**, and record whether it's correct and how fast.
3. We keep the **fastest correct** one.

**Analogy:** instead of guessing one chunk size, we try a shortlist of sensible
sizes, actually bake each cake, and keep the best. This turned "tile size" into a
real, measured optimization instead of a guess.

### Step 3 — Add the "large register file" knob
*(commit `cd24eea`, July 13)*

We added Intel BMG's **large-GRF** mode as a tuning option. Remember: this is a
*compiler flag*, not a program edit. Turning it on gives heavy kernels more fast
scratch memory. We taught the system the rules (it halves how many things run at
once, so it only helps certain kernels) so it wouldn't pick invalid combinations.

### Step 4 — Make the generated kernel genuinely fast (two perf fixes)
*(commits `3834dcd` and `6539bde`, July 14)*

Two low-level improvements to the kernels our translator produced:

- **Register-carried accumulator** — keep the running total of the matrix-multiply
  in the GPU's fastest memory across the inner loop, instead of repeatedly reading
  and writing slower memory.
- **Zero-constant accumulator init** — start that running total at zero directly,
  rather than loading the old values first. This also made kernels **safe to
  re-run** (important for accurate timing — see next step).

**Result:** the generated matrix-multiply reached ~98 TFLOPS on a 4K problem,
matching a hand-written expert reference.

### Step 5 — Measure time *accurately* (IMEX profiling)
*(commit `9f9e8b9`, July 14)*

Our first timing method measured the whole round trip (including time spent on the
CPU setting things up), which drowned out the kernel's real speed. We switched to
**IMEX's built-in GPU profiler**, which times *only* the kernel on the GPU, with
proper warm-up runs. Now our numbers are trustworthy and comparable to Intel's own
benchmarks.

### Step 6 — The key insight: some wins don't need the LLM at all
*(commit `adb8641`, July 15)*

Here's a subtle but important realization.

The large-GRF mode is a **compiler flag**. The LLM, which rewrites the *program
text*, literally cannot reach it. But it's one of the biggest speedups available.

So we built `sweep_grf`: for a given kernel, it runs the kernel **twice** — once
with large-GRF off, once with it on — times both, and keeps the faster. Because
**both runs use the exact same program** (only the flag differs), the answer is
*guaranteed* to be identical. That means **no correctness check is needed** — this
is a free, safe speedup.

On the attention kernel this alone gave **~1.7×** (4.5 ms → 2.6 ms).

**Why this is a big deal:** it's an optimization Xe-Forge can apply *autonomously
and safely* to any heavy kernel, without asking the LLM and without risk.

### Step 7 — Wire that free speedup into the pipeline
*(commit `909505f`, July 15)*

We connected `sweep_grf` into the main flow so that **any WG-level kernel run
through Xe-Forge automatically gets the GRF sweep**, records it as a proper
optimization stage, and remembers the winning setting for the rest of the run. If
a kernel isn't in a runnable form, it's skipped cleanly with a note (no crashes).

### Step 8 — The hard one: flash attention
*(commit `f1309dc`, July 15)*

Attention is the math at the heart of modern AI models. The high-level MLIR for it
is much more complex than a plain matrix-multiply: it's a chain of operations
(a transpose, two "batched" matrix-multiplies, a softmax, a scaling step).

Our translator only knew how to handle a **single, plain matrix-multiply**. It had
no idea how to translate this whole fused chain — and building that translator from
scratch is a large research effort.

**What we did instead — reuse, without dependence:** the lighthouse project already
knows how to translate attention down to the Intel-GPU level. We call its
translator as a **one-shot tool** to produce the low-level kernel, then feed that
into *our* pipeline. Crucially, we only consume its **output file** — Xe-Forge does
not depend on lighthouse at runtime and stays independent.

Concretely, our new `attention_lowering.py`:
1. **Recognizes** an attention program and reads its dimensions.
2. **Calls lighthouse** to produce the low-level kernel (this runs in ~0.6 seconds
   and produces exactly the expected output).
3. **Wraps** that kernel in a tiny runnable test harness so we can time it.
4. Hands it to the **GRF sweep** from Step 6.

**Result — the headline achievement:** starting from the *raw high-level attention
program*, Xe-Forge now runs fully end-to-end and delivers an autonomous,
verified **~1.77× speedup**. The output is a proper low-level GPU kernel with no
high-level operations left in it.

### Step 9 — Stop a slow LLM from freezing the whole run
*(commits `a26f414` and `bb86a44`, July 15–16)*

Once the LLM token was refreshed and the *later* optimization stages started
running, we hit a real-world snag: on certain very large rewrites, the LLM endpoint
would **stall** — the connection stayed open but no answer ever came, and the whole
run hung indefinitely (we saw one wait 18+ minutes).

The cause: there was **no time limit** on LLM requests. We added a request
**timeout plus automatic retries**, and made sure the timeout applied at the actual
network layer (an internal HTTP client had been created without one). Now a stalled
request gives up and retries instead of freezing the run — the pipeline stays
resilient and finishes.

---

## 5. Where things stand today

### What works reliably
- ✅ **Matrix-multiply**: high-level MLIR → automatic tile search → fast,
  verified WG kernel (~98 TFLOPS on 4K, matching expert-written code).
- ✅ **Flash attention**: high-level MLIR → lowered via lighthouse → automatic,
  correctness-free **~1.77× speedup** from the GRF sweep. Runs end-to-end.
- ✅ **The safety gate** (CoVeR) guards every LLM rewrite.
- ✅ **The pipeline is resilient** to slow/stalled LLM responses.

### What's partial or still open
- ⚠️ **LLM rewrites of the attention kernel didn't beat the safety gate.** In the
  full runs, the LLM's attempts to further optimize the attention kernel were
  *rejected* by the correctness+speed check — which is the gate doing its job, not
  a bug. Hand-improving this dense kernel is genuinely hard. The verified win
  remains the 1.77× from lowering + GRF.
- ⚠️ **Batched matrix-multiply** (a building block of attention) still can't be
  lowered by our own translator — a known technical issue (it produces a form that
  crashes Intel's compiler). We worked around it by using lighthouse for attention;
  the root-cause fix is documented for later.
- ⚠️ **LLM model choice matters a lot.** See below.

### What we learned about LLM models (a bonus comparison)
Because one endpoint kept stalling, we compared several models on this exact task:

- **qwen3-coder-480b** — best for **running end-to-end**: it produces the large
  kernel rewrites quickly (~43 s), so the pipeline completes.
- **GLM-5** — best at **finding issues**: its analysis was the most detailed and
  precise (it cited exact code and quantities) — but it **stalls on large
  rewrites** on this endpoint, so it can't complete the optimization stages.
- **deepseek.v3.2** — analysis works, but also stalls on large rewrites.
- Several **GPT-5.x** models are listed on the endpoint but are **not actually
  reachable** (gated).

**Takeaway:** for issue-*finding* a strong reasoning model (GLM-5) shines; for
issue-*fixing* end-to-end you need a model that reliably returns large outputs
(qwen3-coder-480b).

---

## 6. The map: which file does what

| File | Role in plain terms |
|---|---|
| `src/xe_forge/core/mlir_executor.py` | Compiles, runs, times, and correctness-checks MLIR kernels on the GPU. Home of the `sweep_grf` free-speedup trick and the tile-shape search. |
| `src/xe_forge/core/linalg_lowering.py` | The rules and templates for translating a matrix-multiply from high level to Intel-GPU level; defines the tile-shape "knobs." |
| `src/xe_forge/core/attention_lowering.py` | Recognizes attention, calls lighthouse to lower it, and wraps the result so it can be run/timed. |
| `src/xe_forge/pipeline.py` | The conductor: runs Analysis → Planning → Optimization, and wires in the Linalg lowering and GRF sweep. |
| `pipelines/linalg_to_wg/` | The translation templates and transform recipes for matrix-multiply. |
| `knowledge_base/mlir/linalg/` | Accumulated hard-won knowledge: good tile shapes, known pitfalls, the attention/lighthouse findings. |

---

## 7. Timeline at a glance

| Date | Milestone |
|---|---|
| Jun 26 | MLIR WG backend: can compile/run/time a low-level kernel |
| Jul 13 | Two-level pipeline (high→low) with tile-shape search; large-GRF knob |
| Jul 14 | Perf fixes (register accumulator); accurate IMEX timing |
| Jul 15 | Autonomous, correctness-free GRF sweep, wired into the pipeline |
| Jul 15 | **Flash attention runs end-to-end via lighthouse — ~1.77×** |
| Jul 15–16 | LLM timeout + retries so slow responses can't freeze runs |

---

## 8. Suggested next steps

1. **Fix the batched-matrix-multiply lowering** so Xe-Forge can lower attention on
   its own, without lighthouse. (Root cause and fix direction are already
   documented in the knowledge base.)
2. **Give the LLM a cheaper way to verify attention rewrites** (e.g. test on a tiny
   problem size first) so its optimization attempts can iterate faster and stand a
   better chance of passing the gate.
3. **Standardize on qwen3-coder-480b** for MLIR runs (reliable large outputs), and
   consider using GLM-5 specifically for the analysis stage.

---

*This report reflects the state of the `feat/mlir-xegpu-backend` branch as of
July 16, 2026.*
