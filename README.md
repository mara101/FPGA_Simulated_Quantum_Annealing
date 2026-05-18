# SQA QUBO Solver (PYNQ-Z2)

A Simulated Quantum Annealing (SQA) QUBO solver running on a PYNQ-Z2 FPGA, with a
clean Python interface. Forked from the NTU 2021 student project
[allen880117/Simulated-Quantum-Annealing](https://github.com/allen880117/Simulated-Quantum-Annealing)
and trimmed down to a single kernel variant (Opt5) plus host code.

## What you get

```python
from sqa_solver import SQASolver
solver = SQASolver("/home/xilinx/SQA_Opt5.bit")
result = solver.solve(Q, iters=200, restarts=3)
print(result.best_sample, result.best_energy)
```

`Q` is any QUBO matrix (n × n, n ≤ 1024). The solver minimises `E(x) = xᵀ Q x` over
`x ∈ {0,1}ⁿ` and returns the best sample found across all 8 trotters and restarts.

A pure-Python `SQASimulator` mirrors the kernel update rule for offline development
without the board.

## Project layout

```
src/include/         HLS kernel headers (sqa.hpp, prng.hpp)
src/kernel_opt5/     The HLS kernel (qmc_opt5.cpp → QuantumMonteCarloOpt5)
src/host/            Python solver + Jupyter benchmarks
  sqa_solver.py            SQASolver (FPGA) + SQASimulator (Python reference)
  SQA-QUBO-Demo.ipynb      Minimal usage demo
  SQA-Benchmark-FPGA.ipynb     Size / iters / restarts sweeps on the board
  SQA-Benchmark-Laptop.ipynb   Same sweeps with SQASimulator
  SQA-Benchmark-Compare.ipynb  Side-by-side comparison plots
tests/               pytest unit tests for sqa_solver.py
impl_result/_hw/Opt5/  Deployed bitstream + hardware handoff (SQA_Opt5.bit / .hwh)
```

## Running on the board

1. Copy `impl_result/_hw/Opt5/SQA_Opt5.bit` and `SQA_Opt5.hwh` to the PYNQ board
   (e.g. `/home/xilinx/jupyter_notebooks/sqa/`).
2. Copy `src/host/sqa_solver.py` and `src/host/SQA-QUBO-Demo.ipynb` to the same folder.
3. Open the notebook on the board; it walks through QUBO definition → solve → result.

## Rebuilding the bitstream

The kernel targets Vitis HLS 2025.2 + Vivado 2025.2 for `xc7z020clg400-1`. The build
TCL scripts are kept outside the repo (in `C:\SQA\`) because they reference absolute paths;
the kernel source in `src/kernel_opt5/qmc_opt5.cpp` is the only HLS-input artifact.

Build flow:
1. `vitis-run --tcl run_hls.tcl` → `ip_export.zip`
2. Extract IP, then `vivado -mode batch -source create_vivado_project.tcl` → `.bit` + `.hwh`
3. Copy outputs into `impl_result/_hw/Opt5/`.

Resources: ~131 DSPs, ~21 k FF, ~50 k LUT on xc7z020 (fits in the PYNQ-Z2 device).

## Algorithm

The Opt5 kernel maximises `H(σ) = Σᵢⱼ J[i,j]·σᵢσⱼ + Σᵢ h[i]·σᵢ` with σ ∈ {−1, +1},
using Metropolis acceptance `exp(−β · dH / nTrot)` over `MAX_NTROT = 8` Trotter
replicas. A QUBO problem maps to (J, h) via:

```
Q_sym  = (Q + Qᵀ) / 2
J[i,j] = −Q_sym[i,j] / 4    (i ≠ j, symmetric, zero diagonal)
h[i]   = −row_sum(Q_sym)[i] / 2
```

Maximising H under these coefficients minimises `xᵀ Q x` up to a constant.
PRNG is per-trotter xorshift32 (golden-ratio seed spread); see `src/include/prng.hpp`.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for outstanding improvements (parallel tempering,
DMA double buffering, fixed-point accumulation, problem-size tiling, etc.).

## License

See [LICENSE](LICENSE).
