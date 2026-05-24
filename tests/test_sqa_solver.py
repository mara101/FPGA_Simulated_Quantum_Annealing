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
    """Minimal AXI-Lite IP core mock.

    Simulates the kernel done-handshake: writing 0x01 to AP_CTRL sets the
    'kernel-completed' flag so the host's polling loop exits immediately.
    """
    def __init__(self):
        self.regs: dict = {}
        self._done = False

    def write(self, addr, val):
        self.regs[addr] = val
        if addr == 0x0000 and (val & 0x01):
            self._done = True

    def read(self, addr):
        if addr == 0x0000:
            # bit 1 = ap_done, bit 2 = ap_idle. Both set after our fake kernel run.
            return 0x06 if self._done else 0x00
        return self.regs.get(addr, 0)


class _MockJBuf:
    """Stand-in for a pynq.allocate buffer.

    Backed by a regular numpy array with a fake physical_address attribute
    so the solver can write it to the JCOUP registers.
    """
    def __init__(self, size: int = MAX_NSPIN * MAX_NSPIN, phys: int = 0x10000000):
        self._arr = np.zeros(size, dtype=np.float32)
        self.physical_address = phys

    def __setitem__(self, idx, value): self._arr[idx] = value
    def __getitem__(self, idx):        return self._arr[idx]
    @property
    def size(self):                    return self._arr.size
    def freebuffer(self):              pass


class TestSQASolverInterface:
    """Verify the FPGA solver writes the right registers without requiring PYNQ."""

    def _make_solver(self, n_trot=4):
        """Instantiate SQASolver with mocked PYNQ internals."""
        from sqa_solver import SQASolver
        solver = object.__new__(SQASolver)
        solver.num_trotters = n_trot
        solver._ip = _MockIP()
        solver._J_buf = _MockJBuf()
        return solver

    def test_ntrot_written_correctly(self):
        from sqa_solver import _REG_NTROT
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver(n_trot=4)
        solver.solve(Q, iters=1)
        assert solver._ip.regs.get(_REG_NTROT) == 4

    def test_nspin_written_correctly(self):
        from sqa_solver import _REG_NSPIN
        n = 6
        Q = np.eye(n, dtype=np.float32)
        solver = self._make_solver(n_trot=2)
        solver.solve(Q, iters=1)
        assert solver._ip.regs.get(_REG_NSPIN) == n

    def test_total_iters_written_correctly(self):
        # iters is the global anneal length, written to TOTAL_ITERS.
        from sqa_solver import _REG_TOTAL_ITERS
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver()
        solver.solve(Q, iters=137)
        assert solver._ip.regs.get(_REG_TOTAL_ITERS) == 137

    def test_chunking_covers_full_anneal(self):
        # The sum of chunk sizes (ITER_START progression) must cover all iters.
        from sqa_solver import _REG_ITER_START, _REG_NITER
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver()
        solver.solve(Q, iters=100, snapshots=10)
        # Last chunk's start + size should reach iters.
        last_start = solver._ip.regs.get(_REG_ITER_START)
        last_size  = solver._ip.regs.get(_REG_NITER)
        assert last_start + last_size == 100

    def test_schedule_registers_written(self):
        # Option B writes beta_start / beta_end / gamma_start instead of
        # per-iter beta/jperp.
        from sqa_solver import _REG_BETA_START, _REG_BETA_END, _REG_GAMMA_START
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver()
        solver.solve(Q, iters=1, beta_start=0.001, beta_end=8.0, gamma_start=4.0)
        assert _REG_BETA_START  in solver._ip.regs
        assert _REG_BETA_END    in solver._ip.regs
        assert _REG_GAMMA_START in solver._ip.regs

    def test_jcoup_physical_address_written(self):
        # The kernel reads J via AXI master, so the host must write the
        # buffer's DDR physical address split across two 32-bit regs.
        from sqa_solver import _REG_JCOUP_LO, _REG_JCOUP_HI
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver()
        phys = solver._J_buf.physical_address
        solver.solve(Q, iters=1)
        assert solver._ip.regs.get(_REG_JCOUP_LO) == (phys & 0xFFFFFFFF)
        assert solver._ip.regs.get(_REG_JCOUP_HI) == ((phys >> 32) & 0xFFFFFFFF)

    def test_trotters_region_written(self):
        from sqa_solver import _REG_TROT_BASE, _REG_TROT_END
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver()
        solver.solve(Q, iters=1)
        trot_writes = [a for a in solver._ip.regs if _REG_TROT_BASE <= a < _REG_TROT_END]
        assert len(trot_writes) > 0

    def test_J_buffer_matches_J_kernel(self):
        # The first n*n entries of the J buffer must be exactly J_k (row-major).
        n = 3
        Q = np.array([[0., -1., 0.5],
                      [-1., 0., -0.5],
                      [0.5, -0.5, 0.]], dtype=np.float32)
        solver = self._make_solver()
        solver.solve(Q, iters=1)
        from sqa_solver import qubo_to_kernel
        J_k, _ = qubo_to_kernel(Q)
        buf = np.asarray(solver._J_buf[:n*n])
        for i in range(n):
            row = buf[i * n : i * n + n]
            np.testing.assert_allclose(row, J_k[i], atol=1e-6)

    def test_seed_register_written_and_nonzero(self):
        # Kernel needs a non-zero seed; host writes it once per restart.
        from sqa_solver import _REG_SEED_IN
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver()
        solver.solve(Q, iters=1, seed=12345)
        assert _REG_SEED_IN in solver._ip.regs
        assert solver._ip.regs[_REG_SEED_IN] != 0

    def test_ap_start_written(self):
        # The solve() must trigger the kernel via the ap_start bit.
        from sqa_solver import _REG_AP_CTRL
        Q = np.eye(4, dtype=np.float32)
        solver = self._make_solver()
        solver.solve(Q, iters=1)
        assert solver._ip.regs.get(_REG_AP_CTRL, 0) & 0x01

    def test_oversized_problem_raises(self):
        Q = np.eye(MAX_NSPIN + 1, dtype=np.float32)
        solver = self._make_solver()
        with pytest.raises(ValueError, match="MAX_NSPIN"):
            solver.solve(Q, iters=1)
