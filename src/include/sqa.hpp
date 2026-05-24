#ifndef _SQA_HPP_
#define _SQA_HPP_

#include "ap_int.h"
#include "hls_stream.h"
#include <cmath>

/* Hardware sizes. Must match MAX_NTROT / MAX_NSPIN in sqa_solver.py. */
#define MAX_NTROT 8
#define MAX_NSPIN 1024

#define LOG2_MAX_NTROT 3
#define LOG2_MAX_NSPIN 10

typedef float fp_t;
typedef bool  spin_t;

/* Quantum Monte-Carlo (Option B kernel: internal iter loop, AXI master J).
 *
 * Runs nIter full SQA sweeps internally. Schedule (beta, gamma, Jperp) is
 * computed inside the kernel from the three scalar parameters. J is read
 * from DDR via an AXI master interface — no DMA needed.
 */
void QuantumMonteCarloOpt5(
    const int  nTrot,        /* number of active trotters (<= MAX_NTROT) */
    const int  nSpin,        /* number of active spins    (<= MAX_NSPIN) */
    const int  nIter,        /* annealing iterations to run THIS call (chunk) */
    const int  iter_start,   /* global iter offset of this chunk */
    const int  total_iters,  /* full anneal length (for schedule normalisation) */
    spin_t     trotters[MAX_NTROT][MAX_NSPIN],
    const fp_t *Jcoup,       /* AXI master: row-major nSpin × nSpin */
    const fp_t h[MAX_NSPIN],
    const fp_t beta_start,
    const fp_t beta_end,
    const fp_t gamma_start,
    const int  seed_in       /* xorshift32 base seed; spread per-trotter inside */
);

#endif
