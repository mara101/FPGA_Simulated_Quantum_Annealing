"""
SQA QUBO Solver
===============
High-level interface for the SQA FPGA kernel (Opt5, Option B build) and a
pure-Python simulator that mirrors the kernel update rule for offline
development.

QUBO convention
---------------
The user supplies a matrix Q (n×n, any combination of upper/lower triangular
or symmetric). The problem solved is:

    minimize  E(x) = xᵀ Q x ,  x ∈ {0,1}ⁿ

Internally Q is symmetrised as Q_sym = (Q + Qᵀ)/2.  The kernel mapping is:

    J[i,j] = −Q_sym[i,j] / 4   (i ≠ j, symmetric, zero diagonal)
    h[i]   = −row_sum(Q_sym)[i] / 2

This ensures the kernel's maximisation of H(σ) = Σ J·σᵢσⱼ + Σ h·σᵢ
is equivalent to minimising E(x) up to a constant.

Hardware limits
---------------
MAX_NTROT = 8, MAX_NSPIN = 1024  (compile-time constants in the HLS kernel)

Option B kernel
---------------
The kernel runs the entire annealing schedule internally — host calls it
once per ``solve()`` invocation. The kernel pulls J from DDR via an AXI
master port, so no DMA is involved at the host level.

Usage — simulator (no FPGA required)
-------------------------------------
    from sqa_solver import SQASimulator
    sim = SQASimulator(seed=42)
    result = sim.solve(Q, iters=500)
    print(result.best_sample, result.best_energy)

Usage — FPGA (PYNQ-Z2 board only)
-----------------------------------
    from sqa_solver import SQASolver
    solver = SQASolver("/home/xilinx/SQA_Opt5.bit")
    result = solver.solve(Q, iters=1000)
    print(result.best_sample, result.best_energy)
"""

from __future__ import annotations

import binascii
import struct
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Hardware constants (must match sqa.hpp)
# ---------------------------------------------------------------------------
MAX_NTROT: int = 8
MAX_NSPIN: int = 1024

# ---------------------------------------------------------------------------
# PYNQ import — optional; only needed on the FPGA board
# ---------------------------------------------------------------------------
try:
    from pynq import Overlay, allocate as pynq_allocate
    _PYNQ_AVAILABLE = True
except ImportError:
    _PYNQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SQAResult:
    """Returned by both SQASolver and SQASimulator."""
    best_sample: np.ndarray   # shape (n,), dtype bool — best binary vector found
    best_energy: float        # QUBO energy E = xᵀ Q x of best_sample
    all_samples: np.ndarray   # shape (n_trot, n), dtype bool — final trotter states
    all_energies: np.ndarray  # shape (n_trot,) — QUBO energies of all_samples
    timing_s: float           # wall-clock seconds for the annealing loop
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """Evaluate QUBO energy E = xᵀ Q x for binary vector x."""
    x = np.asarray(x, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    return float(x @ Q @ x)


def qubo_to_kernel(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a QUBO matrix to (J, h) coefficients for the SQA kernel.

    The kernel maximises H(σ) = Σᵢⱼ J[i,j]·σᵢ·σⱼ + Σᵢ h[i]·σᵢ with σ ∈ {−1,+1}.
    Setting J and h as below makes that equivalent to minimising xᵀ Q x.
    """
    Q = np.asarray(Q, dtype=np.float64)
    Q_sym = (Q + Q.T) / 2.0
    J = -Q_sym / 4.0
    np.fill_diagonal(J, 0.0)
    h = -Q_sym.sum(axis=1) / 2.0
    return J.astype(np.float32), h.astype(np.float32)


def jperp(gamma: float, n_trot: int, beta: float) -> float:
    """Inter-trotter coupling at given transverse field and temperature.

    Matches the kernel's internal schedule computation exactly. Used by
    SQASimulator to mirror the kernel behaviour offline.
    """
    arg = (gamma / n_trot) * beta
    arg = np.clip(arg, 1e-10, 700.0)
    tanh_val = np.tanh(arg)
    tanh_val = np.clip(tanh_val, 1e-15, 1.0 - 1e-15)
    return float(-0.5 * np.log(tanh_val) / beta)


def _float_to_reg(value: float) -> int:
    """Pack a Python float as a big-endian float32 register word."""
    return int(binascii.b2a_hex(struct.pack(">f", np.float32(value))), 16)


# ---------------------------------------------------------------------------
# Pure-Python simulator (mirrors the Opt5 kernel update rule exactly)
# ---------------------------------------------------------------------------

class SQASimulator:
    """
    Pure-Python SQA solver. No FPGA required — useful for development,
    testing, and problem formulation before deploying to hardware.

    The update rule matches the Opt5 HLS kernel:
      - Spins are Ising σ ∈ {−1, +1} internally (stored as int8).
      - Acceptance: flip if  −β·dHTmp/nTrot > log(uniform),
        where  dHTmp = 2·σᵢ·(local_field + tunnel_correction).
      - Tunnel correction applied only when both Trotter neighbours agree:
        correction = −Jperp·2·nTrot·σ_neighbour.
    """

    def __init__(self, num_trotters: int = MAX_NTROT, seed: Optional[int] = None):
        if not (1 <= num_trotters <= MAX_NTROT):
            raise ValueError(f"num_trotters must be in [1, {MAX_NTROT}]")
        self.num_trotters = num_trotters
        self.rng = np.random.default_rng(seed)

    def solve(
        self,
        Q: np.ndarray,
        iters: int = 500,
        beta_start: float = 1.0 / 4096.0,
        beta_end: float = 8.0,
        gamma_start: float = 8.0,
    ) -> SQAResult:
        Q = np.asarray(Q, dtype=np.float32)
        n = Q.shape[0]
        if n > MAX_NSPIN:
            raise ValueError(f"Problem size {n} exceeds MAX_NSPIN={MAX_NSPIN}")

        J_k, h_k = qubo_to_kernel(Q)
        nt = self.num_trotters

        sigma = self.rng.choice([-1, 1], size=(nt, n)).astype(np.int8)

        beta = beta_start
        d_beta = (beta_end - beta_start) / max(iters - 1, 1)

        best_x: Optional[np.ndarray] = None
        best_e = np.inf

        t0 = time.perf_counter()

        for step in range(iters):
            gamma = gamma_start * (1.0 - step / iters)
            jp = jperp(gamma, nt, beta)
            dH_tunnel_scale = jp * 2.0 * nt

            for t in range(nt):
                t_up = (t - 1) % nt
                t_dn = (t + 1) % nt

                for i in range(n):
                    local_f = float(J_k[i] @ sigma[t].astype(np.float32)) + h_k[i]

                    s_up = int(sigma[t_up, i])
                    s_dn = int(sigma[t_dn, i])
                    if s_up == s_dn:
                        local_f -= dH_tunnel_scale * s_up

                    s_i = int(sigma[t, i])
                    dHTmp = 2.0 * s_i * local_f

                    if (-beta * dHTmp / nt) > np.log(self.rng.random()):
                        sigma[t, i] = -s_i

            for t in range(nt):
                x = (sigma[t] + 1) // 2
                e = qubo_energy(Q, x)
                if e < best_e:
                    best_e = e
                    best_x = x.copy()

            beta += d_beta

        elapsed = time.perf_counter() - t0

        all_x = ((sigma + 1) // 2).astype(bool)
        all_e = np.array([qubo_energy(Q, all_x[t]) for t in range(nt)])

        return SQAResult(
            best_sample=best_x.astype(bool),
            best_energy=best_e,
            all_samples=all_x,
            all_energies=all_e,
            timing_s=elapsed,
            metadata={"iters": iters, "num_trotters": nt, "n_spins": n},
        )


# ---------------------------------------------------------------------------
# FPGA solver (requires PYNQ on the board)
# ---------------------------------------------------------------------------

# AXI-Lite register map for QuantumMonteCarloOpt5 (Option B + chunked schedule).
# Source: xquantummontecarloopt5_hw.h from Vitis HLS 2025.2 synthesis.
_REG_AP_CTRL     = 0x0000   # bit 0 = ap_start (W/COH), bit 1 = ap_done (R/COR), bit 2 = ap_idle (R)
_REG_NTROT       = 0x0010
_REG_NSPIN       = 0x0018
_REG_NITER       = 0x0020   # iters to run THIS kernel call (chunk size)
_REG_ITER_START  = 0x0028   # global iter offset of this chunk
_REG_TOTAL_ITERS = 0x0030   # full anneal length (schedule normalisation)
_REG_JCOUP_LO    = 0x0038   # 64-bit physical address of J buffer, low 32 bits
_REG_JCOUP_HI    = 0x003c   # 64-bit physical address of J buffer, high 32 bits
_REG_BETA_START  = 0x0044
_REG_BETA_END    = 0x004c
_REG_GAMMA_START = 0x0054
_REG_SEED_IN     = 0x005c

# h[MAX_NSPIN] — cyclic-8 partition by HLS; 8 banks of 128 float32 each.
# Bank k stores h[k], h[k+8], h[k+16], ..., h[k+1016]  (indices where i%8==k).
_REG_H_BANKS = [0x0200, 0x2400, 0x2600, 0x2800, 0x2a00, 0x2c00, 0x2e00, 0x3000]
_REG_H_BANK_DEPTH = 128

# trotters[8][1024] — 8 contiguous 0x400-byte banks, 4 bools per uint32 word.
_REG_TROT_BASE = 0x0400
_REG_TROT_END  = 0x2400


class SQASolver:
    """
    QUBO solver backed by the SQA FPGA kernel (Opt5, Option B build).

    The kernel runs the full annealing schedule internally — the host calls
    it exactly once per ``solve()`` (per restart). J is passed via DDR using
    a pre-allocated ``pynq.allocate`` buffer; the kernel reads from DDR
    directly via an AXI master port.

    Parameters
    ----------
    bitfile : str
        Absolute path to the .bit file on the PYNQ board.
    num_trotters : int
        Number of Trotter replicas to use (1–8).
    """

    def __init__(self, bitfile: str, num_trotters: int = MAX_NTROT):
        if not _PYNQ_AVAILABLE:
            raise ImportError(
                "pynq is not installed. Run this class on the PYNQ-Z2 board, "
                "or use SQASimulator for offline development."
            )
        if not (1 <= num_trotters <= MAX_NTROT):
            raise ValueError(f"num_trotters must be in [1, {MAX_NTROT}]")
        self.num_trotters = num_trotters
        self._ol = Overlay(bitfile)
        self._ip = self._ol.QuantumMonteCarloOpt5_0

        # Pre-allocate the J buffer at MAX_NSPIN² floats (4 MB). Reused across
        # every solve() call. cacheable=False so writes are visible to the
        # kernel without needing an explicit cache flush.
        self._J_buf = pynq_allocate(
            shape=(MAX_NSPIN * MAX_NSPIN,),
            dtype=np.float32,
            cacheable=False,
        )

    # ------------------------------------------------------------------
    def solve(
        self,
        Q: np.ndarray,
        iters: int = 1000,
        beta_start: float = 1.0 / 4096.0,
        beta_end: float = 8.0,
        gamma_start: float = 8.0,
        seed: Optional[int] = None,
        restarts: int = 1,
        snapshots: int = 20,
    ) -> SQAResult:
        """
        Solve a QUBO problem on the FPGA.

        ``iters`` is the total number of annealing sweeps in the schedule.
        The kernel runs the schedule in ``snapshots`` chunks; between chunks
        the host reads the trotter state and keeps the best energy seen at
        *any* point in the anneal (best-ever tracking — the final cold state
        is not always the best, so this matters).

        ``restarts`` runs the full schedule that many times from independent
        random initial states. Best result across all restarts is returned.

        ``snapshots`` controls how finely the anneal is sampled for best-ever
        tracking. More snapshots = better tracking, slightly more readback
        overhead. The schedule itself is unaffected by chunking — the kernel
        normalises against the global ``iters``.
        """
        Q = np.asarray(Q, dtype=np.float32)
        n = Q.shape[0]
        if n > MAX_NSPIN:
            raise ValueError(f"Problem size {n} exceeds MAX_NSPIN={MAX_NSPIN}")

        J_k, h_k = qubo_to_kernel(Q)
        Q_sym = (Q + Q.T) / 2.0
        nt = self.num_trotters

        rng = np.random.default_rng(seed)
        seed_rng = np.random.default_rng(seed if seed is not None else 0)

        # --- Stage J in the contiguous DDR buffer (row-major n×n) ---
        self._J_buf[: n * n] = J_k.flatten()

        # --- Write everything that's constant across restarts and chunks ---
        self._ip.write(_REG_NTROT,       nt)
        self._ip.write(_REG_NSPIN,       n)
        self._ip.write(_REG_TOTAL_ITERS, iters)
        self._ip.write(_REG_BETA_START,  _float_to_reg(beta_start))
        self._ip.write(_REG_BETA_END,    _float_to_reg(beta_end))
        self._ip.write(_REG_GAMMA_START, _float_to_reg(gamma_start))

        phys = int(self._J_buf.physical_address)
        self._ip.write(_REG_JCOUP_LO, phys & 0xFFFFFFFF)
        self._ip.write(_REG_JCOUP_HI, (phys >> 32) & 0xFFFFFFFF)

        h_padded = np.zeros(MAX_NSPIN, dtype=np.float32)
        h_padded[:n] = h_k
        self._write_h(h_padded)

        # Chunk boundaries for best-ever sampling.
        n_chunks = max(1, min(snapshots, iters))
        chunk = max(1, iters // n_chunks)

        best_x: Optional[np.ndarray] = None
        best_e = np.inf
        last_all_x: Optional[np.ndarray] = None
        last_all_e: Optional[np.ndarray] = None

        t0 = time.perf_counter()

        for _restart in range(max(restarts, 1)):
            # Fresh random initial trotter state for each restart.
            trotters = rng.integers(0, 2, size=(MAX_NTROT, MAX_NSPIN), dtype=np.uint8)
            self._write_trotters(trotters)

            iter_done = 0
            while iter_done < iters:
                this_chunk = min(chunk, iters - iter_done)

                # New PRNG seed each chunk (kernel re-seeds from seed_in per call).
                fpga_seed = int(seed_rng.integers(1, 2**31 - 1))
                self._ip.write(_REG_SEED_IN,    fpga_seed)
                self._ip.write(_REG_ITER_START, iter_done)
                self._ip.write(_REG_NITER,      this_chunk)

                # Run this chunk; trotter state persists in BRAM across calls.
                self._ip.write(_REG_AP_CTRL, 0x01)
                while (self._ip.read(_REG_AP_CTRL) & 0x02) == 0:
                    pass

                iter_done += this_chunk

                # Snapshot: read state, track best-ever across all trotters.
                snap = self._read_trotters(n)
                all_x = snap[:nt, :n].astype(bool)
                all_e = np.array([qubo_energy(Q_sym, all_x[t]) for t in range(nt)])
                last_all_x, last_all_e = all_x, all_e
                for t in range(nt):
                    if all_e[t] < best_e:
                        best_e = all_e[t]
                        best_x = all_x[t].copy()

        elapsed = time.perf_counter() - t0

        return SQAResult(
            best_sample=best_x,
            best_energy=float(best_e),
            all_samples=last_all_x,
            all_energies=last_all_e,
            timing_s=elapsed,
            metadata={
                "iters": iters,
                "restarts": restarts,
                "snapshots": n_chunks,
                "num_trotters": nt,
                "n_spins": n,
            },
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Free the pre-allocated J buffer. Call before deleting the solver."""
        if getattr(self, "_J_buf", None) is not None:
            try:
                self._J_buf.freebuffer()
            except Exception:
                pass
            self._J_buf = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_h(self, h_padded: np.ndarray) -> None:
        """Write h[MAX_NSPIN] to the 8 cyclic AXI-Lite banks.
        HLS cyclic-8 maps h[i] → bank (i%8), element (i//8).
        """
        for bank in range(8):
            base = _REG_H_BANKS[bank]
            for j in range(_REG_H_BANK_DEPTH):
                self._ip.write(base + j * 4, _float_to_reg(float(h_padded[j * 8 + bank])))

    def _write_trotters(self, trotters: np.ndarray) -> None:
        """Write trotters[MAX_NTROT][MAX_NSPIN] via AXI-Lite (4 bools / word)."""
        flat = trotters.flatten().astype(np.uint8)
        k = 0
        for addr in range(_REG_TROT_BASE, _REG_TROT_END, 4):
            word = (int(flat[k+3]) << 24) | (int(flat[k+2]) << 16) | \
                   (int(flat[k+1]) <<  8) |  int(flat[k])
            self._ip.write(addr, word)
            k += 4

    def _read_trotters(self, n_spin: int = MAX_NSPIN) -> np.ndarray:
        """Read trotters back via AXI-Lite. Returns shape (MAX_NTROT, MAX_NSPIN).
        Reads only the first ceil(n_spin/4)*4 words per bank for speed.
        """
        words_per_trot = (n_spin + 3) // 4
        flat = np.zeros(MAX_NTROT * MAX_NSPIN, dtype=np.uint8)
        for t in range(MAX_NTROT):
            base = _REG_TROT_BASE + t * 0x400
            k = t * MAX_NSPIN
            for w in range(words_per_trot):
                word = self._ip.read(base + w * 4)
                flat[k]   =  word        & 0x01
                flat[k+1] = (word >>  8) & 0x01
                flat[k+2] = (word >> 16) & 0x01
                flat[k+3] = (word >> 24) & 0x01
                k += 4
        return flat.reshape(MAX_NTROT, MAX_NSPIN)
