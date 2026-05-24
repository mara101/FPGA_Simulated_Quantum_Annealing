# SQA FPGA Accelerator — Improvement Roadmap

## Status Key
- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete

---

## 1. Python QUBO Interface `[x]`

**Files:** `src/host/sqa_solver.py`, `tests/test_sqa_solver.py`

A clean Python API that accepts a QUBO matrix and handles all low-level register writes,
J-coupling padding, DMA transfers, and annealing schedule computation internally.
Includes a pure-Python `SQASimulator` for development and testing without FPGA hardware.

**Why:** The current notebooks require users to manually manage AXI-Lite addresses, pack
floats into register format, pad matrices to MAX_NSPIN, and re-implement the annealing loop
from scratch. A single `SQASolver(bitfile).solve(Q)` call should be enough.

**Effort:** Medium | **Impact on usability:** Very High

### QUBO → kernel mapping

The kernel (Opt5) maximizes H(σ) = Σᵢⱼ J[i,j]·σᵢ·σⱼ + Σᵢ h[i]·σᵢ with σ ∈ {−1,+1},
using Metropolis acceptance `exp(−β·dH/nTrot)`. A QUBO problem

    minimize  E(x) = xᵀ Q x ,  x ∈ {0,1}ⁿ

is equivalent (up to a constant) when we set:

    Q_sym    = (Q + Qᵀ) / 2
    J[i,j]   = −Q_sym[i,j] / 4   (i ≠ j, symmetric, zero diagonal)
    h[i]     = −row_sum(Q_sym)[i] / 2

Maximising H(σ) under these coefficients minimises E(x).

---

## 2. Parallel Tempering `[ ]`

**File:** `src/host/sqa_solver.py` (host-side annealing loop only, no kernel changes)

Run each of the 8 trotters at a slightly different temperature. At the end of each
iteration, propose replica-exchange swaps between adjacent-temperature trotters using the
standard Metropolis criterion on the QUBO energy difference.

**Why:** Single-temperature SQA can get trapped in local minima. Parallel tempering lets
high-temperature replicas escape barriers while low-temperature ones exploit good regions.

**Effort:** Low | **Impact on solution quality:** High

---

## 3. Per-trotter Xorshift32 PRNG `[x]`

**Files:** `src/include/prng.hpp`, `src/kernel_opt5/qmc_opt5.cpp`

Replaced the shared Park-Miller LCG with an inline xorshift32 PRNG, one independent state
per trotter, seeded from `seed_in` spread by the golden-ratio constant 2654435761. Keeps
one shared hardware `logf` unit via `#pragma HLS DEPENDENCE inter false` (no UNROLL) so
DSP cost stays low (~131 DSPs total on xc7z020).

**Why:** The old LCG shared one seed across all 8 trotters; by Marsaglia's lattice theorem
consecutive outputs are correlated, so the 8 trotters were not statistically independent.

---

## 3a. Staggered-pipeline offset bug fix `[x]`

**File:** `src/kernel_opt5/qmc_opt5.cpp`

The kernel computed `int offset = ctlStep - startStep[t]` but then assigned
`iPre[t] = (inside) ? ctlStep : 0` — the `offset` was discarded. Only trotter 0 worked
correctly (it had `offset == ctlStep`); trotters 1–7 applied the wrong J-row to every spin
update. Also wired up the previously unused `startStep/endStep/ctlStep` parameters in
`TrotterUnit5` as the inside guard, and fixed `dH[t]` init to `h[0]` for all trotters.

**Why:** All 8 trotters now run real SQA instead of 1 working trotter + 7 noise sources.

---

## 4. Internal iteration loop + AXI master J (Option B) `[x]`

**Files:** `src/kernel_opt5/qmc_opt5.cpp`, `src/host/sqa_solver.py`,
`C:\SQA\create_vivado_project.tcl`

Moved the entire annealing loop *inside* the HLS kernel. The host now makes
one kernel call per `solve()` (per restart) and the kernel runs all `iters`
sweeps internally, computing the β/γ/Jperp schedule via `tanhf`/`logf` once
per iter. J is read from DDR via an AXI master port (no DMA), so the only
per-iter host work is gone. Split the kernel into a DATAFLOW producer
(`read_J_stream`, m_axi → hls::stream) and consumer (`sqa_compute`,
hls::stream → SQA), matching the LUT footprint of the original streamed
design. NPC reduced 16 → 8 to fit on xc7z020 (47 487 / 53 200 LUT post-HLS;
20 945 LUT post-impl). Block diagram simplified: DMA removed, kernel's
`m_axi_gmem` wired directly to PS HP0 via `axi_mem_intercon`.

**Why:** Each annealing iteration was previously a host→FPGA round-trip:
~125 ms of Python + AXI-Lite + DMA setup overhead for ~0.7 ms of actual
kernel compute. Eliminating that loop is the only way to actually beat a
modern desktop CPU on PYNQ-Z2 hardware.

**Effort:** High | **Impact on throughput:** Very High (expected ~12× over previous build)

---

## 5. Problem Size Tiling `[ ]`

**Files:** `src/host/sqa_solver.py`, kernel sources

Tile problems larger than MAX_NSPIN=1024 by partitioning the J matrix into blocks and
making multiple kernel invocations per annealing step. Each block updates a subset of
spins using the current full spin state (read back between blocks).

**Why:** The 1024-spin hard limit rules out most industrial-scale problems (portfolio,
logistics, scheduling).

**Effort:** High | **Impact on usability:** High

---

## 6. Vitis HLS Migration `[x]`

**Files:** `src/kernel_opt5/qmc_opt5.cpp`, `C:\SQA\run_hls.tcl`, `C:\SQA\create_vivado_project.tcl`

Migrated `QuantumMonteCarloOpt5` to Vitis HLS 2025.2. Key changes required:
- `#pragma HLS PIPELINE II=1` must be placed on the **inner** `LOOP_CTRL_2` (64-iteration
  Metropolis loop), not the outer spin loop — the outer loop is unrolled automatically.
  Placing it on the outer loop caused Vitis 2025.2 to flatten both loops, breaking timing.
- AXI DMA `c_sg_length_width` set to 26 (max 64 MB) — the default 14-bit value caps
  transfers at 16 383 bytes, which is one byte short of the n=64 J matrix (64²×4=16 384 B).
- AXI4 (DMA) → AXI3 (PS HP0) bridged via `axi_mem_intercon`; direct connection is invalid.
- Register map corrected from generated `xquantummontecarloopt5_hw.h`:
  nTrot=0x10, nSpin=0x18, Jperp=0x20, Beta=0x28, seed_in=0x30.
  `h[MAX_NSPIN]` uses cyclic-8 AXI-Lite banks (h_0..h_7 at non-contiguous offsets).
- Working bitstream deployed to `impl_result/_hw/Opt5/SQA_Opt5.bit/.hwh`.
- Vivado project creation and bitstream flow fully automated in `create_vivado_project.tcl`.

**Why:** Vitis HLS has better pipelining heuristics, more reliable DATAFLOW support, and
enables modern hardware targets with 10–100× more resources.

**Effort:** Medium | **Impact on performance:** Medium + enables new hardware targets

---

## 7. Fixed-Point Energy Accumulation `[ ]`

**Files:** `src/include/sqa.hpp`, kernel sources

Replace `float` dH accumulators with `ap_fixed<32,8>` (24 fractional bits). Keep
`float` for Beta/Jperp since they require wider dynamic range.

**Why:** Float addition has 4-cycle latency on Zynq-7000. Fixed-point operations are
lower latency and use fewer DSP slices, tightening the reduction tree in `TrotterUnit5`.

**Effort:** Medium | **Impact on performance:** Low–Medium

---

## 8. Adaptive Annealing Schedule `[ ]`

**File:** `src/host/sqa_solver.py`

Monitor the per-iteration acceptance rate and slow the Beta ramp when acceptance drops
below a threshold. Requires reading back trotters each step (already done) and adjusting
`d_beta` dynamically.

**Why:** The current linear Beta schedule is problem-agnostic. Hard instances need slower
annealing near phase transitions, which a fixed schedule misses.

**Effort:** Low | **Impact on solution quality:** Medium
