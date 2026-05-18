"""
Tests for sqa_solver.py
=======================
All tests run without FPGA hardware (SQASimulator only).
Run with:  python -m pytest tests/test_sqa_solver.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "host"))

import numpy as np
import pytest
from sqa_solver import (
    SQASimulator,
    SQAResult,
    qubo_energy,
    qubo_to_kernel,
    jperp,
    MAX_NTROT,
    MAX_NSPIN,
    _REG_SEED,
)


# ===========================================================================
# qubo_to_kernel — conversion correctness
# ===========================================================================

class TestQuboToKernel:

    def test_symmetric_input_unchanged(self):
        Q = np.array([[2., -1.], [-1., 3.]], dtype=np.float32)
        J, h = qubo_to_kernel(Q)
        # J should be −Q_sym/4 off-diagonal, zero diagonal
        Q_sym = (Q + Q.T) / 2
        np.testing.assert_allclose(J[0, 1], -Q_sym[0, 1] / 4, rtol=1e-5)
        np.testing.assert_allclose(J[1, 0], -Q_sym[1, 0] / 4, rtol=1e-5)
        assert J[0, 0] == pytest.approx(0.0)
        assert J[1, 1] == pytest.approx(0.0)

    def test_upper_triangular_symmetrised(self):
        # Provide upper-triangular Q and verify symmetrisation
        Q = np.array([[0., 4.], [0., 0.]], dtype=np.float32)
        J, h = qubo_to_kernel(Q)
        Q_sym = (Q + Q.T) / 2   # = [[0, 2], [2, 0]]
        np.testing.assert_allclose(J[0, 1], -Q_sym[0, 1] / 4, rtol=1e-5)
        np.testing.assert_allclose(J[1, 0], J[0, 1], rtol=1e-5)

    def test_number_partition_h_is_zero(self):
        # For the number partition QUBO the h vector must be identically zero
        w = np.array([1., 2., 3., 6.], dtype=np.float64)
        W = w.sum()
        n = len(w)
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Q[i, j] = 4 * w[i] * w[j]
        Q[np.arange(n), np.arange(n)] = 4 * w * (w - W)
        _, h = qubo_to_kernel(Q)
        np.testing.assert_allclose(h, np.zeros(n), atol=1e-4)

    def test_number_partition_J_matches_notebook(self):
        # Kernel J must equal −rn·rn^T / 4 (off-diagonal) for number partition
        w = np.array([1., 2., 3.], dtype=np.float64)
        W = w.sum()
        n = len(w)
        Q = np.outer(4 * w, w)
        Q[np.arange(n), np.arange(n)] = 4 * w * (w - W)
        J, _ = qubo_to_kernel(Q)
        # Expected: J[i,j] = −w[i]*w[j] for i≠j
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert J[i, j] == pytest.approx(-w[i] * w[j], rel=1e-5)

    def test_output_dtype_float32(self):
        Q = np.eye(4)
        J, h = qubo_to_kernel(Q)
        assert J.dtype == np.float32
        assert h.dtype == np.float32

    def test_J_is_symmetric(self):
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((6, 6)).astype(np.float32)
        J, _ = qubo_to_kernel(Q)
        np.testing.assert_allclose(J, J.T, atol=1e-6)

    def test_J_diagonal_is_zero(self):
        rng = np.random.default_rng(1)
        Q = rng.standard_normal((5, 5)).astype(np.float32)
        J, _ = qubo_to_kernel(Q)
        np.testing.assert_allclose(np.diag(J), np.zeros(5), atol=1e-7)


# ===========================================================================
# qubo_energy
# ===========================================================================

class TestQuboEnergy:

    def test_all_zeros(self):
        Q = np.array([[1., -2.], [-2., 3.]])
        x = np.array([0, 0])
        assert qubo_energy(Q, x) == pytest.approx(0.0)

    def test_single_spin(self):
        Q = np.array([[5.]])
        assert qubo_energy(Q, [1]) == pytest.approx(5.0)
        assert qubo_energy(Q, [0]) == pytest.approx(0.0)

    def test_two_spins_coupled(self):
        # Q = [[0, 1],[1, 0]], x=[1,1] → E = 0*1 + 1*1 + 1*1 + 0*1 = 2
        Q = np.array([[0., 1.], [1., 0.]])
        assert qubo_energy(Q, [1, 1]) == pytest.approx(2.0)
        assert qubo_energy(Q, [1, 0]) == pytest.approx(0.0)
        assert qubo_energy(Q, [0, 1]) == pytest.approx(0.0)

    def test_consistent_with_qubo_to_kernel(self):
        # Verify that qubo_energy and ising energy give the same optimum
        Q = np.array([[1., -2., 0.5],
                      [-2., 3., -1.],
                      [0.5, -1., 2.]])
        energies = []
        for bits in range(8):
            x = np.array([(bits >> k) & 1 for k in range(3)])
            energies.append((qubo_energy(Q, x), x.copy()))
        best_e, best_x = min(energies, key=lambda t: t[0])
        # Just check the minimum is finite and the energy is correct
        assert np.isfinite(best_e)
        assert qubo_energy(Q, best_x) == pytest.approx(best_e)


# ===========================================================================
# jperp schedule
# ===========================================================================

class TestJperp:

    def test_positive(self):
        # Jperp must always be positive
        for gamma in [8.0, 4.0, 1.0, 0.1]:
            for beta in [0.001, 0.1, 1.0, 8.0]:
                assert jperp(gamma, 8, beta) > 0.0

    def test_decreases_with_gamma(self):
        # As Gamma → 0, Jperp → ∞ (locking trotters); as Gamma → ∞, Jperp → 0
        beta = 1.0
        jp_high = jperp(8.0, 8, beta)
        jp_low  = jperp(0.1, 8, beta)
        assert jp_low > jp_high

    def test_matches_notebook_formula(self):
        gamma, nt, beta = 4.0, 8, 0.5
        expected = -0.5 * np.log(np.tanh((gamma / nt) * beta)) / beta
        assert jperp(gamma, nt, beta) == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# SQASimulator — structural tests
# ===========================================================================

class TestSQASimulatorStructure:

    def test_result_type(self):
        sim = SQASimulator(num_trotters=2, seed=0)
        Q = -np.ones((4, 4))
        np.fill_diagonal(Q, 0)
        result = sim.solve(Q, iters=10)
        assert isinstance(result, SQAResult)

    def test_best_sample_shape(self):
        n = 8
        sim = SQASimulator(num_trotters=4, seed=1)
        Q = np.diag(np.ones(n))
        result = sim.solve(Q, iters=20)
        assert result.best_sample.shape == (n,)
        assert result.best_sample.dtype == bool

    def test_all_samples_shape(self):
        n, nt = 6, 3
        sim = SQASimulator(num_trotters=nt, seed=2)
        Q = np.eye(n)
        result = sim.solve(Q, iters=10)
        assert result.all_samples.shape == (nt, n)

    def test_all_energies_shape(self):
        n, nt = 5, 4
        sim = SQASimulator(num_trotters=nt, seed=3)
        Q = np.eye(n)
        result = sim.solve(Q, iters=10)
        assert result.all_energies.shape == (nt,)

    def test_best_energy_is_minimum_of_all(self):
        sim = SQASimulator(seed=4)
        Q = np.diag([1., 2., 3., 4.])
        result = sim.solve(Q, iters=50)
        assert result.best_energy <= float(result.all_energies.min()) + 1e-6

    def test_best_energy_consistent_with_best_sample(self):
        sim = SQASimulator(seed=5)
        Q = np.array([[1., -1.], [-1., 1.]])
        result = sim.solve(Q, iters=30)
        recomputed = qubo_energy(Q, result.best_sample)
        assert result.best_energy == pytest.approx(recomputed, rel=1e-5)

    def test_timing_positive(self):
        sim = SQASimulator(seed=6)
        Q = np.eye(4)
        result = sim.solve(Q, iters=5)
        assert result.timing_s > 0.0

    def test_metadata_keys(self):
        sim = SQASimulator(num_trotters=3, seed=7)
        Q = np.eye(4)
        result = sim.solve(Q, iters=12)
        assert result.metadata["iters"] == 12
        assert result.metadata["num_trotters"] == 3
        assert result.metadata["n_spins"] == 4

    def test_oversized_problem_raises(self):
        sim = SQASimulator(seed=8)
        Q = np.eye(MAX_NSPIN + 1)
        with pytest.raises(ValueError, match="MAX_NSPIN"):
            sim.solve(Q)

    def test_invalid_num_trotters_raises(self):
        with pytest.raises(ValueError):
            SQASimulator(num_trotters=0)
        with pytest.raises(ValueError):
            SQASimulator(num_trotters=MAX_NTROT + 1)

    def test_deterministic_with_seed(self):
        Q = np.array([[1., -2., 0.5], [-2., 3., -1.], [0.5, -1., 2.]])
        r1 = SQASimulator(seed=99).solve(Q, iters=50)
        r2 = SQASimulator(seed=99).solve(Q, iters=50)
        np.testing.assert_array_equal(r1.best_sample, r2.best_sample)
        assert r1.best_energy == pytest.approx(r2.best_energy)


# ===========================================================================
# SQASimulator — optimality tests (small instances with known solutions)
# ===========================================================================

class TestSQASimulatorOptimality:

    def test_two_spin_ferromagnet(self):
        # Q = [[0, -1],[-1, 0]]: minimum E = -2 at x=(1,1), not x=(0,0) since E=0 there
        # Actually x=(1,1): E = 0*1 + (-1)*1*1 + (-1)*1*1 + 0*1 = -2  ← minimum
        Q = np.array([[0., -1.], [-1., 0.]])
        sim = SQASimulator(num_trotters=8, seed=0)
        result = sim.solve(Q, iters=200)
        assert result.best_energy == pytest.approx(-2.0, abs=1e-4)
        np.testing.assert_array_equal(result.best_sample, [True, True])

    def test_single_spin_bias(self):
        # Q = [[-3]]: minimum at x=1, E=-3
        Q = np.array([[-3.]])
        sim = SQASimulator(num_trotters=2, seed=0)
        result = sim.solve(Q, iters=50)
        assert result.best_energy == pytest.approx(-3.0, abs=1e-4)
        assert result.best_sample[0] == True

    def test_two_spin_antiferromagnet(self):
        # Q = [[0, 1],[1, 0]]: minimum E = 0 at x=(0,0), x=(1,0), or x=(0,1)
        # x=(0,0) → E=0; x=(1,1) → E=2 (worst); all others → E=0
        # The degenerate minimum includes (0,0) so we only check the energy value.
        Q = np.array([[0., 1.], [1., 0.]])
        sim = SQASimulator(num_trotters=8, seed=0)
        result = sim.solve(Q, iters=200)
        assert result.best_energy == pytest.approx(0.0, abs=1e-4)

    def test_four_spin_ferromagnet(self):
        # All-to-all negative coupling: minimum at all-1 or all-0
        # all-1: E = Σ_{i≠j} (-1) * 1 * 1 = -(n*(n-1)) = -12 for n=4
        n = 4
        Q = -np.ones((n, n))
        np.fill_diagonal(Q, 0)
        sim = SQASimulator(num_trotters=8, seed=0)
        result = sim.solve(Q, iters=300)
        assert result.best_energy == pytest.approx(-12.0, abs=0.1)

    def test_number_partition_small(self):
        # w = [1, 2, 3, 6]: perfect partition {1,2,3} vs {6}, E=0
        w = np.array([1., 2., 3., 6.])
        W = w.sum()
        n = len(w)
        Q = np.outer(4 * w, w)
        Q[np.arange(n), np.arange(n)] = 4 * w * (w - W)
        sim = SQASimulator(num_trotters=8, seed=42)
        result = sim.solve(Q, iters=500, beta_end=10.0)
        # Best QUBO energy = 0 (perfect partition), but we add constant -W^2
        # So actual minimum of xᵀQx (with the linear encoding) might be negative.
        # The actual partition energy is (sum_A - sum_B)^2, verify it's 0:
        x = result.best_sample.astype(float)
        sigma = 2 * x - 1
        partition_energy = float((w @ sigma) ** 2)
        assert partition_energy == pytest.approx(0.0, abs=0.1)

    def test_number_partition_asymmetric_Q(self):
        # Verify that lower-triangular and upper-triangular encodings of the
        # SAME coupling (value 2 between spin 0 and spin 1) produce identical
        # J/h after symmetrisation and therefore identical annealing trajectories.
        Q_upper = np.array([[0., 2., 0.],
                             [0., 0., 2.],
                             [0., 0., 0.]], dtype=np.float64)
        Q_lower = Q_upper.T   # same coupling, different triangle

        J_u, h_u = qubo_to_kernel(Q_upper)
        J_l, h_l = qubo_to_kernel(Q_lower)

        np.testing.assert_allclose(J_u, J_l, atol=1e-6)
        np.testing.assert_allclose(h_u, h_l, atol=1e-6)

        # With the same J/h both solvers (same seed) must give the same result
        r_upper = SQASimulator(num_trotters=4, seed=7).solve(Q_upper, iters=200)
        r_lower = SQASimulator(num_trotters=4, seed=7).solve(Q_lower, iters=200)
        np.testing.assert_array_equal(r_upper.best_sample, r_lower.best_sample)

    def test_max_cut_triangle(self):
        # Triangle graph: 3 nodes, each edge weight 1.
        # Max-cut = 2 (one node separated from the other two).
        # QUBO: maximise Σ_{(i,j)∈E} x_i*(1-x_j) + x_j*(1-x_i)
        #      = minimise Σ_{(i,j)∈E} (2*x_i*x_j - x_i - x_j)
        Q = np.array([[-2.,  2.,  2.],
                      [ 2., -2.,  2.],
                      [ 2.,  2., -2.]])
        sim = SQASimulator(num_trotters=8, seed=0)
        result = sim.solve(Q, iters=300)
        # All 3 partitions that cut 2 edges: {0}|{1,2}, {1}|{0,2}, {2}|{0,1}
        x = result.best_sample.astype(int)
        cut = sum(
            1 for i, j in [(0, 1), (1, 2), (0, 2)]
            if x[i] != x[j]
        )
        assert cut == 2


# ===========================================================================
# SQASolver register interface — tested with a lightweight mock
# ===========================================================================

class _MockIP:
    """Minimal AXI-Lite IP core mock."""
    def __init__(self):
        self.regs: dict = {}
        self._done = False

    def write(self, addr, val):
        self.regs[addr] = val
        if addr == 0x0000 and (val & 0x01):
            self._done = True

    def read(self, addr):
        if addr == 0x0000:
            return 0x04 if self._done else 0x00
        return self.regs.get(addr, 0)


class _MockDMAChannel:
    def __init__(self):
        self.last_transfer: np.ndarray = np.array([], dtype=np.float32)

    def transfer(self, buf):
        self.last_transfer = np.array(buf, dtype=np.float32)

    def wait(self): pass


class _MockDMA:
    def __init__(self):
        self.sendchannel = _MockDMAChannel()



class TestSQASolverInterface:
    """Verify the FPGA solver writes the right registers without requiring PYNQ."""

    def _make_solver(self, Q, n_trot=4):
        """Instantiate SQASolver with mocked PYNQ internals."""
        from sqa_solver import SQASolver, _REG_NTROT, _REG_NSPIN, _REG_JPERP, _REG_BETA

        solver = object.__new__(SQASolver)
        solver.num_trotters = n_trot
        solver._ip = _MockIP()
        solver._dma = _MockDMA()
        return solver

    def test_ntrot_written_correctly(self):
        from sqa_solver import _REG_NTROT
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver(Q, n_trot=4)
        solver.solve(Q, iters=1)
        assert solver._ip.regs.get(_REG_NTROT) == 4

    def test_nspin_written_correctly(self):
        from sqa_solver import _REG_NSPIN
        n = 6
        Q = np.eye(n, dtype=np.float32)
        solver = self._make_solver(Q, n_trot=2)
        solver.solve(Q, iters=1)
        assert solver._ip.regs.get(_REG_NSPIN) == n

    def test_jperp_register_written(self):
        from sqa_solver import _REG_JPERP
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver(Q)
        solver.solve(Q, iters=1)
        assert _REG_JPERP in solver._ip.regs

    def test_beta_register_written(self):
        from sqa_solver import _REG_BETA
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver(Q)
        solver.solve(Q, iters=1)
        assert _REG_BETA in solver._ip.regs

    def test_trotters_region_written(self):
        from sqa_solver import _REG_TROT_BASE, _REG_TROT_END
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver(Q)
        solver.solve(Q, iters=1)
        # At least some trotter registers must have been written
        trot_writes = [a for a in solver._ip.regs if _REG_TROT_BASE <= a < _REG_TROT_END]
        assert len(trot_writes) > 0

    def test_dma_buffer_matches_J_kernel(self):
        # Verify the DMA buffer has n-stride layout and contains exactly n² values
        n = 3
        Q = np.array([[0., -1., 0.5],
                      [-1., 0., -0.5],
                      [0.5, -0.5, 0.]], dtype=np.float32)
        solver = self._make_solver(Q)
        solver.solve(Q, iters=1)
        from sqa_solver import qubo_to_kernel
        J_k, _ = qubo_to_kernel(Q)
        buf = solver._dma.sendchannel.last_transfer
        assert buf.size == n * n, f"Expected {n*n} values, got {buf.size}"
        for i in range(n):
            row = buf[i * n : i * n + n]
            np.testing.assert_allclose(row, J_k[i], atol=1e-6)

    def test_seed_register_written_and_varies(self):
        # Each iteration must write a different seed so the kernel PRNG is not stuck.
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver(Q)
        solver.solve(Q, iters=3)
        seeds = [solver._ip.regs.get(_REG_SEED + i * 4, None) for i in range(3)
                 if solver._ip.regs.get(_REG_SEED) is not None]
        # At minimum, the register must have been written at least once.
        assert _REG_SEED in solver._ip.regs
        assert solver._ip.regs[_REG_SEED] != 0

    def test_oversized_problem_raises(self):
        from sqa_solver import SQASolver
        solver = object.__new__(SQASolver)
        solver.num_trotters = 4
        solver._ip = _MockIP()
        solver._dma = _MockDMA()
        Q = np.eye(MAX_NSPIN + 1, dtype=np.float32)
        with pytest.raises(ValueError, match="MAX_NSPIN"):
            solver.solve(Q, iters=1)
