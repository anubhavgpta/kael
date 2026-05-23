# Kael — Attention Accelerator IP

**Layer 1 of the Circle inference silicon stack.**

Kael is a verified SystemVerilog IP block that accelerates scaled dot-product attention in hardware. It consumes K/V vectors streamed from [Vera](https://github.com/YOUR_USERNAME/vera) (the paged KV cache controller) and a Q vector loaded by the host, and produces a context vector entirely in hardware — no CPU involvement in the attention computation.

---

## The Problem Kael Solves

Agentic AI workloads re-attend over long, multi-session contexts repeatedly. The bottleneck is not SRAM bandwidth (Vera handles that) — it is the attention score computation itself: QK^T scaled dot-product over hundreds of tokens, in FP16, per head, per session, per forward pass.

Kael fuses the entire attention pipeline — dot product, scaling, softmax, weighted sum — into a single streaming hardware pipeline, eliminating the CPU/GPU bottleneck for this operation.

---

## Architecture

```
Q vectors (host)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                   attention_ctrl                     │
│                                                     │
│  ┌──────────────┐   ┌──────────────┐               │
│  │ qk_dot_engine│──▶│ score_scaler │               │
│  │  8-PE systolic│   │  >>3 shift  │               │
│  │  Q8.8 fixed  │   │  Q16.16→    │               │
│  │  HEAD_DIM=64 │   │  Q1.15      │               │
│  └──────────────┘   └──────┬───────┘               │
│                             ▼                       │
│                    ┌─────────────────┐              │
│                    │ softmax_engine  │              │
│                    │ online softmax  │              │
│                    │ Flash Attn style│              │
│                    └────────┬────────┘              │
│                             │ weight                │
│  K/V from Vera ────────────▼                       │
│                    ┌─────────────────┐              │
│                    │  v_accumulator  │              │
│                    │ weighted V sum  │              │
│                    │ + rescale path  │              │
│                    └────────┬────────┘              │
│                             │                       │
└─────────────────────────────┼───────────────────────┘
                              ▼
                       context vector
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Fixed-point format | Q8.8 inputs, Q1.15 scores | No Xilinx FP16 IP dependency — fully licensable |
| Dot product | 8-PE systolic fold, 8 cycles | 8 DSP48E1s, leaves budget for V accumulator |
| Score scaling | Arithmetic right-shift by 3 | 1/sqrt(64) = 0.125 = exact power of two, zero DSP cost |
| Softmax | Online (running max + rescale) | Single pass, no score buffer, streaming compatible |
| Exp approximation | 256-entry Q1.15 LUT ROM | Standard in fixed-point inference hardware |
| Batch support | MAX_BATCH=8 parallel instances | Forward-compatible with speculative decoding (Layer 2) |

---

## Module Breakdown

```
rtl/
  qk_dot_engine.v     8-PE folded systolic dot product
  score_scaler.v      Multiply by 1/sqrt(HEAD_DIM) via right-shift
  softmax_engine.v    Online streaming softmax with exp LUT
  v_accumulator.v     Weighted V sum, rescale path, context vector output
  attention_ctrl.v    Top-level: Vera interface, pipeline sequencing, batching
  exp_lut.mem         256-entry hex LUT for exp(x) over [-8, 0] in Q1.15
scripts/
  gen_exp_lut.py      Generates exp_lut.mem (Python 3, stdlib + numpy only)
tb/
  tb_qk_dot_engine.sv
  tb_score_scaler.sv
  tb_softmax_engine.sv
  tb_v_accumulator.sv
  tb_attention_ctrl.sv
vivado/
  setup_project.tcl   Loads all sources into existing Vivado project
```

---

## Verification

All modules verified in Vivado xsim 2018.2 on Artix-7 xc7a35tcpg236-1.

| Module | Tests | Result |
|---|---|---|
| qk_dot_engine | 6 | 6/6 PASS |
| score_scaler | 6 | 6/6 PASS |
| softmax_engine | 16 | 16/16 PASS |
| v_accumulator | 20 | 20/20 PASS |
| attention_ctrl | 14 | 14/14 PASS |
| **Total** | **62** | **62/62 PASS** |

Integration tests cover: single token, multi-token, batch=2, back-to-back passes, rescale path, stall handshake, batch ID routing, context vector correctness.

---

## Performance (estimated at 100MHz)

| Metric | Value |
|---|---|
| Cycles per attention head (256 tokens) | ~3,072 |
| Attention heads per second | ~32,500 |
| DSP48E1 usage | ~17 |
| Target clock | 100MHz (Artix-7) |

The efficiency-per-watt advantage over GPU attention is the primary licensing value proposition for SoC and edge inference customers.

---

## Interface

### Top-level ports (`attention_ctrl`)

```
HEAD_DIM   = 64    -- attention head dimension
MAX_BATCH  = 8     -- max Q vectors per pass (speculative decode)
DATA_WIDTH = 16    -- Q8.8 fixed-point

Q input:
  q_data[15:0], q_addr[5:0], q_batch_id[2:0], q_valid
  batch_size[2:0]   -- number of active Q vectors (1..8)

Session control:
  session_id[2:0], token_start[15:0], token_end[15:0]
  attn_start        -- one-cycle pulse to begin attention pass

Vera KV read interface:
  rd_req, rd_session_id, rd_token_start, rd_token_end  (outputs)
  rd_k_data[15:0], rd_v_data[15:0], rd_valid, rd_last, rd_busy  (inputs)

Context vector output:
  ctx_out[15:0], ctx_batch_id[2:0], ctx_valid, ctx_last

Status:
  attn_done         -- one-cycle pulse when context vector is ready
  attn_busy         -- high during attention pass
```

---

## Getting Started

### Prerequisites
- Vivado 2018.2 or later
- Python 3.8+ (for LUT generation only)

### Regenerate the exp LUT
```bash
cd scripts
python gen_exp_lut.py
```

### Load into Vivado
Open your Vivado project, then in the Tcl console:
```tcl
source C:/path/to/kael/vivado/setup_project.tcl
```

### Run all testbenches
```bat
D:\Vivado\2018.2\bin\xvlog.bat --sv rtl/qk_dot_engine.v rtl/score_scaler.v rtl/softmax_engine.v rtl/v_accumulator.v rtl/attention_ctrl.v tb/tb_attention_ctrl.sv
D:\Vivado\2018.2\bin\xelab.bat -top tb_attention_ctrl -snapshot ctrl_snap
D:\Vivado\2018.2\bin\xsim.bat ctrl_snap --runall
```

Expected output: `Results: 14 PASS, 0 FAIL`

---

## Circle Silicon Stack

```
Layer 3   Full SoC Reference Design          [planned]
Layer 2   Agentic Inference Subsystem        [planned]
            └─ speculative decode engine
Layer 1   Kael  Attention Accelerator        [THIS REPO]
Layer 0   Vera  Paged KV Cache Controller    [complete]
```

Kael is designed to be licensed independently or as part of the full Circle stack. Each layer exposes a clean interface to the layer above.

---

## License

Proprietary — Circle Inference Silicon IP. All rights reserved.
