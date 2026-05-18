"""
SQA QUBO Solver
===============
High-level interface for the SQA FPGA kernel (Opt5) and a pure-Python
simulator that mirrors the kernel's update rule for offline development.

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
    result = solver.solve(Q, iters=500)
    print(result.best_sample, result.best_energy)
"""

from __future__ import annotations

import struct
import binascii
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

    Parameters
    ----------
    Q : ndarray, shape (n, n)
        QUBO matrix (upper triangular, lower triangular, or symmetric).

    Returns
    -------
    J : ndarray, shape (n, n), float32 — symmetric, zero diagonal
    h : ndarray, shape (n,),   float32
    """
    Q = np.asarray(Q, dtype=np.float64)
    Q_sym = (Q + Q.T) / 2.0
    J = -Q_sym / 4.0
    np.fill_diagonal(J, 0.0)
    h = -Q_sym.sum(axis=1) / 2.0
    return J.astype(np.float32), h.astype(np.float32)


def jperp(gamma: float, n_trot: int, beta: float) -> float:
    """Compute the inter-trotter coupling at given transverse field and temperature."""
    arg = (gamma / n_trot) * beta
    # Guard against numerical underflow/overflow in tanh
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
    Pure-Python SQA solver.  No FPGA required — useful for development, testing,
    and problem formulation before deploying to hardware.

    The update rule matches the Opt5 HLS kernel:
      - Spins are Ising σ ∈ {−1, +1} internally (stored as int8).
      - Acceptance: flip if  −β·dHTmp/nTrot > log(uniform),
        where  dHTmp = 2·σᵢ·(local_field + tunnel_correction).
      - Tunnel correction applied only when both Trotter neighbours agree:
        correction = −Jperp·2·nTrot·σ_neighbour.

    Parameters
    ----------
    num_trotters : int
        Number of Trotter replicas (1–8).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, num_trotters: int = MAX_NTROT, seed: Optional[int] = None):
        if not (1 <= num_trotters <= MAX_NTROT):
            raise ValueError(f"num_trotters must be in [1, {MAX_NTROT}]")
        self.num_trotters = num_trotters
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def solve(
        self,
        Q: np.ndarray,
        iters: int = 500,
        beta_start: float = 1.0 / 4096.0,
        beta_end: float = 8.0,
        gamma_start: float = 8.0,
    ) -> SQAResult:
        """
        Solve a QUBO problem using simulated quantum annealing.

        Parameters
        ----------
        Q : ndarray, shape (n, n)
            QUBO matrix.  Symmetrised internally.
        iters : int
            Number of annealing iterations (each sweeps all spins once per trotter).
        beta_start, beta_end : float
            Inverse-temperature schedule endpoints.
        gamma_start : float
            Initial transverse field strength G₀.

        Returns
        -------
        SQAResult
        """
        import time

        Q = np.asarray(Q, dtype=np.float32)
        n = Q.shape[0]
        if n > MAX_NSPIN:
            raise ValueError(f"Problem size {n} exceeds MAX_NSPIN={MAX_NSPIN}")

        J_k, h_k = qubo_to_kernel(Q)   # kernel coefficients
        nt = self.num_trotters

        # Initialise trotters as random Ising spins {−1, +1}
        sigma = self.rng.choice([-1, 1], size=(nt, n)).astype(np.int8)

        beta = beta_start
        d_beta = (beta_end - beta_start) / max(iters - 1, 1)

        best_x: Optional[np.ndarray] = None
        best_e = np.inf

        t0 = time.perf_counter()

        for step in range(iters):
            gamma = gamma_start * (1.0 - step / iters)
            jp = jperp(gamma, nt, beta)
            dH_tunnel_scale = jp * 2.0 * nt   # = Jperp * 2 * nTrot  (matches kernel)

            for t in range(nt):
                t_up = (t - 1) % nt
                t_dn = (t + 1) % nt

                for i in range(n):
                    # Classical Ising local field: Σⱼ J[i,j]·σⱼ + h[i]
                    local_f = float(J_k[i] @ sigma[t].astype(np.float32)) + h_k[i]

                    # Tunnel correction: only when both neighbours agree (matches kernel)
                    s_up = int(sigma[t_up, i])
                    s_dn = int(sigma[t_dn, i])
                    if s_up == s_dn:
                        local_f -= dH_tunnel_scale * s_up  # s_up ∈ {−1, +1}

                    # dHTmp = 2·σᵢ·local_f  (kernel: *= 2; if !spin: negate)
                    s_i = int(sigma[t, i])
                    dHTmp = 2.0 * s_i * local_f

                    # Accept with probability min(1, exp(−β·dHTmp/nTrot))
                    if (-beta * dHTmp / nt) > np.log(self.rng.random()):
                        sigma[t, i] = -s_i

            # Track best solution across trotters
            for t in range(nt):
                x = (sigma[t] + 1) // 2   # σ→x: +1→1, −1→0
                e = qubo_energy(Q, x)
                if e < best_e:
                    best_e = e
                    best_x = x.copy()

            beta += d_beta

        elapsed = time.perf_counter() - t0

        # Final trotter states as binary
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

# AXI-Lite register map for QuantumMonteCarloOpt5
# Source: xquantummontecarloopt5_hw.h from Vitis HLS 2025.2 synthesis
_REG_CTRL  = 0x0000   # control: write 0x01 to start; bit 2 = done
_REG_NTROT = 0x0010
_REG_NSPIN = 0x0018
_REG_JPERP = 0x0020
_REG_BETA  = 0x0028
_REG_SEED  = 0x0030   # seed_in (confirmed from xquantummontecarloopt5_hw.h)

# h[MAX_NSPIN] — cyclic-8 partition by HLS; 8 banks of 128 float32 each.
# Bank k stores h[k], h[k+8], h[k+16], ..., h[k+1016]  (indices where i%8==k).
_REG_H_BANKS     = [0x0200, 0x2400, 0x2600, 0x2800, 0x2a00, 0x2c00, 0x2e00, 0x3000]
_REG_H_BANK_DEPTH = 128   # floats per bank (MAX_NSPIN / 8)

# trotters[8][1024] — 8 contiguous 0x400-byte banks, 4 bools per uint32 word.
# Word n: bit 0 = spin[4n], bit 8 = spin[4n+1], bit 16 = spin[4n+2], bit 24 = spin[4n+3]
_REG_TROT_BASE = 0x0400
_REG_TROT_END  = 0x2400


class SQASolver:
    """
    QUBO solver backed by the SQA FPGA kernel (Opt5, PYNQ-Z2).

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
                "pynq is not installed.  Run this class on the PYNQ-Z2 board, "
                "or use SQASimulator for offline development."
            )
        if not (1 <= num_trotters <= MAX_NTROT):
            raise ValueError(f"num_trotters must be in [1, {MAX_NTROT}]")
        self.num_trotters = num_trotters
        self._ol = Overlay(bitfile)
        self._ip = self._ol.QuantumMonteCarloOpt5_0
        self._dma = self._ol.axi_dma_0

    # ------------------------------------------------------------------
    def solve(
        self,
        Q: np.ndarray,
        iters: int = 200,
        beta_start: float = 1.0 / 4096.0,
        beta_end: float = 8.0,
        gamma_start: float = 8.0,
        seed: Optional[int] = None,
        restarts: int = 1,
    ) -> SQAResult:
        """
        Solve a QUBO problem on the FPGA.

        For the current Jacobi-update kernel, the sweet spot is iters=100–300.
        Use ``restarts > 1`` to compensate: each restart runs a full annealing
        schedule from a fresh random trotter state and the best result is
        returned.  Because the FPGA is fast, multiple restarts are cheap.
        """
        import time

        Q = np.asarray(Q, dtype=np.float32)
        n = Q.shape[0]
        if n > MAX_NSPIN:
            raise ValueError(f"Problem size {n} exceeds MAX_NSPIN={MAX_NSPIN}")

        J_k, h_k = qubo_to_kernel(Q)
        Q_sym = (Q + Q.T) / 2.0
        nt = self.num_trotters
        rng = np.random.default_rng(seed)
        seed_rng = np.random.default_rng(seed if seed is not None else 0)

        # --- Build the DMA buffer: exactly n² values, n-stride (row-major) ---
        if _PYNQ_AVAILABLE:
            J_buf = pynq_allocate(shape=(n * n,), dtype=np.float32)
        else:
            J_buf = np.zeros(n * n, dtype=np.float32)
        J_buf[:] = J_k.flatten()

        # --- Write problem constants (same across restarts) ---
        h_padded = np.zeros(MAX_NSPIN, dtype=np.float32)
        h_padded[:n] = h_k
        self._write_h(h_padded)
        self._ip.write(_REG_NTROT, nt)
        self._ip.write(_REG_NSPIN, n)

        # Intra-run snapshots only for long runs (>500 iters) where the Jacobi
        # update can oscillate and drive trotters away from a good intermediate
        # state.  For short runs the final readback is sufficient and we avoid
        # adding MMIO overhead that would slow the benchmark.
        do_snapshots = iters > 500
        snapshot_interval = max(10, min(50, iters // 20)) if do_snapshots else iters

        best_x: Optional[np.ndarray] = None
        best_e = np.inf
        last_all_x: Optional[np.ndarray] = None
        last_all_e: Optional[np.ndarray] = None

        t0 = time.perf_counter()

        for _restart in range(max(restarts, 1)):
            # Fresh random trotter initialisation for each restart
            trotters = rng.integers(0, 2, size=(MAX_NTROT, MAX_NSPIN), dtype=np.uint8)
            self._write_trotters(trotters)

            beta = beta_start
            d_beta = (beta_end - beta_start) / max(iters - 1, 1)

            for step in range(iters):
                gamma = gamma_start * (1.0 - step / iters)
                jp = jperp(gamma, nt, beta)

                self._ip.write(_REG_JPERP, _float_to_reg(jp))
                self._ip.write(_REG_BETA,  _float_to_reg(beta))
                fpga_seed = int(seed_rng.integers(1, 2**31 - 1))
                self._ip.write(_REG_SEED, fpga_seed)

                self._ip.write(_REG_CTRL, 0x01)
                self._dma.sendchannel.transfer(J_buf)
                self._dma.sendchannel.wait()

                while (self._ip.read(_REG_CTRL) & 0x04) == 0:
                    pass

                beta += d_beta

                # Intra-run snapshot: only taken for long runs
                if do_snapshots and (step + 1) % snapshot_interval == 0:
                    snap = self._read_trotters(n)
                    for t in range(nt):
                        x_t = snap[t, :n].astype(bool)
                        e_t = qubo_energy(Q_sym, x_t)
                        if e_t < best_e:
                            best_e = e_t
                            best_x = x_t.copy()

            # Always read the final state once per restart to track best-ever
            trotters_out = self._read_trotters(n)
            last_all_x = trotters_out[:nt, :n].astype(bool)
            last_all_e = np.array([qubo_energy(Q_sym, last_all_x[t]) for t in range(nt)])
            for t in range(nt):
                if last_all_e[t] < best_e:
                    best_e = last_all_e[t]
                    best_x = last_all_x[t].copy()

        elapsed = time.perf_counter() - t0

        return SQAResult(
            best_sample=best_x,
            best_energy=float(best_e),
            all_samples=last_all_x,
            all_energies=last_all_e,
            timing_s=elapsed,
            metadata={"iters": iters, "restarts": restarts, "num_trotters": nt, "n_spins": n},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_h(self, h_padded: np.ndarray) -> None:
        """Write h[MAX_NSPIN] to the 8 cyclic AXI-Lite banks.

        The HLS cyclic-8 partition maps h[i] → bank (i%8), element (i//8).
        Bank k base addresses are in _REG_H_BANKS.
        """
        for bank in range(8):
            base = _REG_H_BANKS[bank]
            for j in range(_REG_H_BANK_DEPTH):
                self._ip.write(base + j * 4, _float_to_reg(float(h_padded[j * 8 + bank])))

    def _write_trotters(self, trotters: np.ndarray) -> None:
        """Write trotters[MAX_NTROT][MAX_NSPIN] to AXI-Lite registers (4 bools / word)."""
        flat = trotters.flatten().astype(np.uint8)
        k = 0
        for addr in range(_REG_TROT_BASE, _REG_TROT_END, 4):
            word = (int(flat[k+3]) << 24) | (int(flat[k+2]) << 16) | \
                   (int(flat[k+1]) <<  8) |  int(flat[k])
            self._ip.write(addr, word)
            k += 4

    def _read_trotters(self, n_spin: int = MAX_NSPIN) -> np.ndarray:
        """Read trotters back from AXI-Lite registers; returns shape (MAX_NTROT, MAX_NSPIN).

        Only reads the first ceil(n_spin/4)*4 words per trotter bank, which is
        much faster for small problem sizes.
        """
        words_per_trot = (n_spin + 3) // 4   # ceil(n_spin / 4)
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
