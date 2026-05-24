#include "../include/prng.hpp"
#include "../include/sqa.hpp"

#include <cmath>

#include "ap_int.h"
#include "hls_stream.h"

#define DEP 1
#define NPC (8)

/* Reduction */
template <int SIZE> void reductionFP5(fp_t fpBuffer[NPC]) {
#pragma HLS INLINE
    reductionFP5<SIZE / 2>(fpBuffer);
    for (int i = 0; i < NPC; i += SIZE) {
#pragma HLS UNROLL
        fpBuffer[i] += fpBuffer[i + SIZE / 2];
    }
}

template <> void reductionFP5<2>(fp_t fpBuffer[NPC]) {
#pragma HLS INLINE
    for (int i = 0; i < NPC; i += 2) {
#pragma HLS UNROLL
        fpBuffer[i] += fpBuffer[i + 1];
    }
}

/* Trotter Unit (unchanged from the iPre-fix version: inside-guard wired up
 * via startStep/endStep/ctlStep, dH-write only when active). */
template <int t>
void TrotterUnit5(const int nTrot, const int nSpin, const int ctlStep,
                  const int i, const int j, const int startStep,
                  const int endStep, spin_t trotters[MAX_NSPIN],
                  const spin_t up_trotter, const spin_t down_trotter, fp_t &dH,
                  const fp_t hNext, const fp_t Beta, const fp_t dHTunnel,
                  const fp_t JcoupLocal[MAX_NSPIN], const fp_t logRandNumber) {

#pragma HLS INLINE off

    fp_t fpBuffer[NPC];
#pragma HLS ARRAY_PARTITION variable = fpBuffer complete dim = 1

    if (t < nTrot) {
        if (startStep <= ctlStep && ctlStep < endStep && 0 <= i && i < nSpin) {
            fp_t dHTmp = dH;

            for (int k = 0; k < NPC; k++) {
#pragma HLS UNROLL
                if (j + k < nSpin) {
                    if (trotters[j + k]) {
                        fpBuffer[k] = JcoupLocal[j + k];
                    } else {
                        fpBuffer[k] = -JcoupLocal[j + k];
                    }
                } else {
                    fpBuffer[k] = 0;
                }
            }

            reductionFP5<NPC>(fpBuffer);
            dHTmp += fpBuffer[0];

            if (j + NPC >= nSpin) {
                spin_t this_spin = trotters[i];

                if (up_trotter == down_trotter) {
                    if (up_trotter)
                        dHTmp -= dHTunnel;
                    else
                        dHTmp += dHTunnel;
                }

                dHTmp *= 2.0f;
                if (!this_spin) {
                    dHTmp = -dHTmp;
                }

                if ((-Beta * dHTmp) > logRandNumber) {
                    trotters[i] = (!this_spin);
                }

                dHTmp = hNext;
            }

            dH = dHTmp;
        }
    }
}

template <int NTROT>
void DuplicateTrotterUnits5(
    const int nTrot, const int nSpin, const int ctlStep,
    const int iPre[MAX_NTROT], const int j, const int startStep[MAX_NTROT],
    const int endStep[MAX_NTROT], spin_t trotters[MAX_NTROT][MAX_NSPIN],
    const spin_t up_trotter[MAX_NTROT], const spin_t down_trotter[MAX_NTROT],
    fp_t dH[MAX_NTROT], const fp_t h[MAX_NTROT], const fp_t Beta,
    const fp_t dHTunnel, const fp_t JcoupLocal[MAX_NTROT][MAX_NSPIN],
    const fp_t logRandNumber[MAX_NTROT]) {
#pragma HLS INLINE
    DuplicateTrotterUnits5<NTROT - 1>(nTrot, nSpin, ctlStep, iPre, j, startStep,
                                      endStep, trotters, up_trotter,
                                      down_trotter, dH, h, Beta, dHTunnel,
                                      JcoupLocal, logRandNumber);

    fp_t hNext = (iPre[NTROT - 1] != nSpin - 1) ? h[iPre[NTROT - 1] + 1] : 0.0f;
    TrotterUnit5<NTROT - 1>(
        nTrot, nSpin, ctlStep, iPre[NTROT - 1], j, startStep[NTROT - 1],
        endStep[NTROT - 1], trotters[NTROT - 1], up_trotter[NTROT - 1],
        down_trotter[NTROT - 1], dH[NTROT - 1], hNext, Beta, dHTunnel,
        JcoupLocal[NTROT - 1], logRandNumber[NTROT - 1]);
};

template <>
void DuplicateTrotterUnits5<1>(
    const int nTrot, const int nSpin, const int ctlStep,
    const int iPre[MAX_NTROT], const int j, const int startStep[MAX_NTROT],
    const int endStep[MAX_NTROT], spin_t trotters[MAX_NTROT][MAX_NSPIN],
    const spin_t up_trotter[MAX_NTROT], const spin_t down_trotter[MAX_NTROT],
    fp_t dH[MAX_NTROT], const fp_t h[MAX_NTROT], const fp_t Beta,
    const fp_t dHTunnel, const fp_t JcoupLocal[MAX_NTROT][MAX_NSPIN],
    const fp_t logRandNumber[MAX_NTROT]) {
#pragma HLS INLINE
    fp_t hNext_0 = (iPre[0] != nSpin - 1) ? h[iPre[0] + 1] : 0.0f;
    TrotterUnit5<0>(nTrot, nSpin, ctlStep, iPre[0], j, startStep[0], endStep[0],
                    trotters[0], up_trotter[0], down_trotter[0], dH[0], hNext_0,
                    Beta, dHTunnel, JcoupLocal[0], logRandNumber[0]);
}

/* =============================================================================
 * DATAFLOW producer: reads J from DDR via AXI master, feeds a hls::stream.
 *
 * Runs nIter × nSpin × nSpin reads total. Each inner k-loop is pipelined
 * II=1 — HLS should infer a burst read of consecutive addresses.
 * ===========================================================================*/
static void read_J_stream(const fp_t *Jcoup_ddr,
                          hls::stream<fp_t> &Jcoup_out,
                          const int nIter, const int nSpin) {
ITER_LOOP_R:
    for (int iter = 0; iter < nIter; iter++) {
#pragma HLS LOOP_TRIPCOUNT max = 10000
    CTLSTEP_R:
        for (int ctlStep = 0; ctlStep < MAX_NSPIN; ctlStep++) {
#pragma HLS LOOP_TRIPCOUNT max = 1024
            if (ctlStep >= nSpin) break;
        ROW_READ:
            for (int k = 0; k < MAX_NSPIN; k++) {
#pragma HLS LOOP_TRIPCOUNT max = 1024
#pragma HLS PIPELINE II = 1
                if (k >= nSpin) break;
                Jcoup_out.write(Jcoup_ddr[ctlStep * nSpin + k]);
            }
        }
    }
}

/* =============================================================================
 * DATAFLOW consumer: SQA compute kernel. Reads J from the input stream just
 * like the original (pre-Option-B) design did, so LUT footprint matches that
 * of the streamed version.
 * ===========================================================================*/
static void sqa_compute(hls::stream<fp_t> &Jcoup_in,
                        spin_t trotters[MAX_NTROT][MAX_NSPIN],
                        const fp_t h[MAX_NSPIN],
                        const int  nTrot,
                        const int  nSpin,
                        const int  nIter,
                        const int  iter_start,
                        const int  total_iters,
                        const fp_t beta_start,
                        const fp_t beta_end,
                        const fp_t gamma_start,
                        const int  seed_in) {
#pragma HLS ARRAY_PARTITION variable = trotters complete dim = 1
#pragma HLS ARRAY_PARTITION variable = h cyclic factor = 8 dim = 1

    fp_t JcoupLocal[MAX_NTROT][MAX_NSPIN];
#pragma HLS ARRAY_PARTITION variable = JcoupLocal complete dim = 1

    fp_t dH[MAX_NTROT];
    int  startStep[MAX_NTROT];
    int  endStep[MAX_NTROT];
    int  up[MAX_NTROT];
    int  down[MAX_NTROT];
#pragma HLS ARRAY_PARTITION variable = dH complete dim = 1
#pragma HLS ARRAY_PARTITION variable = startStep complete dim = 1
#pragma HLS ARRAY_PARTITION variable = endStep complete dim = 1
#pragma HLS ARRAY_PARTITION variable = up complete dim = 1
#pragma HLS ARRAY_PARTITION variable = down complete dim = 1

LOOP_INIT:
    for (int t = 0; t < MAX_NTROT; t++) {
#pragma HLS UNROLL
        startStep[t] = t;
        endStep[t]   = t + nSpin;
        up[t]        = (t == 0)         ? (nTrot - 1) : (t - 1);
        down[t]      = (t == nTrot - 1) ? 0           : (t + 1);
    }

    unsigned int seeds[MAX_NTROT];
#pragma HLS ARRAY_PARTITION variable = seeds complete dim = 1
    for (int t = 0; t < MAX_NTROT; t++) {
#pragma HLS UNROLL
        unsigned int s = (unsigned int)seed_in + (unsigned int)t * 2654435761u;
        seeds[t] = (s == 0u) ? 1u : s;
    }

    spin_t up_trotter[MAX_NTROT];
    spin_t down_trotter[MAX_NTROT];
    int    iPre[MAX_NTROT];
    fp_t   logRandomNumber[MAX_NTROT];
#pragma HLS ARRAY_PARTITION variable = up_trotter complete dim = 1
#pragma HLS ARRAY_PARTITION variable = down_trotter complete dim = 1
#pragma HLS ARRAY_PARTITION variable = iPre complete dim = 1
#pragma HLS ARRAY_PARTITION variable = logRandomNumber complete dim = 1

    /* Schedule slope/normaliser keyed to the GLOBAL anneal length so the host
     * can run the schedule in chunks. iter_start is this chunk's global offset;
     * total_iters is the full anneal length. */
    fp_t d_beta = (total_iters > 1)
                      ? (beta_end - beta_start) / (fp_t)(total_iters - 1)
                      : 0.0f;
    fp_t inv_total = (total_iters > 0) ? 1.0f / (fp_t)total_iters : 0.0f;

ITER_LOOP:
    for (int iter = 0; iter < nIter; iter++) {
#pragma HLS LOOP_TRIPCOUNT max = 10000

        /* Schedule for this iter, using the global iteration index. */
        int  global_iter = iter_start + iter;
        fp_t beta  = beta_start + (fp_t)global_iter * d_beta;
        fp_t gamma = gamma_start * (1.0f - (fp_t)global_iter * inv_total);

        fp_t arg = (gamma / (fp_t)nTrot) * beta;
        if (arg < 1e-10f) arg = 1e-10f;
        if (arg > 700.0f) arg = 700.0f;
        fp_t tanh_val = tanhf(arg);
        if (tanh_val < 1e-15f)        tanh_val = 1e-15f;
        if (tanh_val > 1.0f - 1e-15f) tanh_val = 1.0f - 1e-15f;
        fp_t Jperp    = -0.5f * logf(tanh_val) / beta;
        fp_t dHTunnel = Jperp * 2.0f * (fp_t)nTrot;
        fp_t Beta     = beta;

        /* Reset dH for this iter — all trotters start at spin 0 with h[0]. */
        for (int t = 0; t < MAX_NTROT; t++) {
#pragma HLS UNROLL
            dH[t] = h[0];
        }

    LOOP_CTRL:
        for (int ctlStep = 0; ctlStep < (MAX_NSPIN + MAX_NTROT - 1); ctlStep++) {
#pragma HLS LOOP_TRIPCOUNT max = 1031
            if (ctlStep >= nSpin + nTrot - 1) break;

            for (int t = 0; t < MAX_NTROT; t++) {
#pragma HLS UNROLL
#if DEP
#pragma HLS DEPENDENCE variable = startStep inter false
#pragma HLS DEPENDENCE variable = endStep inter false
#pragma HLS DEPENDENCE variable = iPre inter false
#endif
                int  offset = ctlStep - startStep[t];
                bool inside = (startStep[t] <= ctlStep && ctlStep < endStep[t]);
                iPre[t]     = (inside) ? offset : (0);
            }

            for (int t = 0; t < MAX_NTROT; t++) {
#pragma HLS UNROLL
#if DEP
#pragma HLS DEPENDENCE variable = trotters inter false
#pragma HLS DEPENDENCE variable = up inter false
#pragma HLS DEPENDENCE variable = down inter false
#pragma HLS DEPENDENCE variable = iPre inter false
#pragma HLS DEPENDENCE variable = up_trotter inter false
#pragma HLS DEPENDENCE variable = down_trotter inter false
#endif
                up_trotter[t]   = trotters[up[t]][iPre[t]];
                down_trotter[t] = trotters[down[t]][iPre[t]];
            }

            for (int t = 0; t < MAX_NTROT; t++) {
#pragma HLS DEPENDENCE variable = logRandomNumber inter false
#pragma HLS DEPENDENCE variable = seeds inter false
                logRandomNumber[t] = logf(uniform01_xorshift(seeds[t])) * (fp_t)nTrot;
            }

        LOOP_CTRL_2:
            for (int j = 0; j < MAX_NSPIN; j += NPC) {
#pragma HLS LOOP_TRIPCOUNT max  = 64
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = h inter false
                if (j >= nSpin) break;

                for (int t = 0; t < MAX_NTROT; t++) {
#pragma HLS UNROLL
#if DEP
#pragma HLS DEPENDENCE variable = startStep inter false
#pragma HLS DEPENDENCE variable = endStep inter false
#pragma HLS DEPENDENCE variable = iPre inter false
#endif
                    int  offset = ctlStep - startStep[t];
                    bool inside = (startStep[t] <= ctlStep && ctlStep < endStep[t]);
                    iPre[t]     = (inside) ? offset : (0);
                }

                for (int t = MAX_NTROT - 1; t > 0; t--) {
#pragma HLS UNROLL
                    for (int k = 0; k < NPC; k++) {
#pragma HLS UNROLL
                        if (j + k < nSpin) {
                            JcoupLocal[t][j + k] = JcoupLocal[t - 1][j + k];
                        }
                    }
                }

                /* Pop J row chunk from the producer stream. */
                if (ctlStep < endStep[0]) {
                    for (int k = 0; k < NPC; k++) {
#pragma HLS UNROLL
                        if (j + k < nSpin) {
                            JcoupLocal[0][j + k] = Jcoup_in.read();
                        }
                    }
                }

                DuplicateTrotterUnits5<MAX_NTROT>(
                    nTrot, nSpin, ctlStep, iPre, j, startStep, endStep, trotters,
                    up_trotter, down_trotter, dH, h, Beta, dHTunnel, JcoupLocal,
                    logRandomNumber);
            }
        }
    }
}

/* =============================================================================
 * Top-level: DATAFLOW splits the m_axi reader and the SQA compute into
 * two concurrent processes, communicating via Jcoup_stream.
 *
 * The host writes the physical base address of a contiguous J buffer to the
 * Jcoup AXI-Lite register, the schedule scalars, then starts the kernel.
 * ===========================================================================*/
void QuantumMonteCarloOpt5(
    const int  nTrot,
    const int  nSpin,
    const int  nIter,
    const int  iter_start,
    const int  total_iters,
    spin_t     trotters[MAX_NTROT][MAX_NSPIN],
    const fp_t *Jcoup,
    const fp_t h[MAX_NSPIN],
    const fp_t beta_start,
    const fp_t beta_end,
    const fp_t gamma_start,
    const int  seed_in
) {
#pragma HLS INTERFACE m_axi      port = Jcoup offset = slave bundle = gmem depth = 1048576 num_read_outstanding = 8 max_read_burst_length = 16
#pragma HLS INTERFACE s_axilite  port = Jcoup
#pragma HLS INTERFACE s_axilite  port = trotters
#pragma HLS INTERFACE s_axilite  port = h
#pragma HLS INTERFACE s_axilite  port = nTrot
#pragma HLS INTERFACE s_axilite  port = nSpin
#pragma HLS INTERFACE s_axilite  port = nIter
#pragma HLS INTERFACE s_axilite  port = iter_start
#pragma HLS INTERFACE s_axilite  port = total_iters
#pragma HLS INTERFACE s_axilite  port = beta_start
#pragma HLS INTERFACE s_axilite  port = beta_end
#pragma HLS INTERFACE s_axilite  port = gamma_start
#pragma HLS INTERFACE s_axilite  port = seed_in
#pragma HLS INTERFACE s_axilite  port = return

#pragma HLS DATAFLOW

    hls::stream<fp_t> Jcoup_stream;
#pragma HLS STREAM variable = Jcoup_stream depth = 256

    read_J_stream(Jcoup, Jcoup_stream, nIter, nSpin);
    sqa_compute(Jcoup_stream, trotters, h,
                nTrot, nSpin, nIter, iter_start, total_iters,
                beta_start, beta_end, gamma_start,
                seed_in);
}
