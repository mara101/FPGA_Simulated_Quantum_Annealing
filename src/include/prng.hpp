#ifndef _PRNG_HPP_
#define _PRNG_HPP_

#include "sqa.hpp"

/* Xorshift32 — one independent state per trotter.
 * State must be non-zero; advances in place and returns a value in (0, 1]. */
inline fp_t uniform01_xorshift(unsigned int &state) {
#pragma HLS INLINE
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    /* Multiply by 2^-32 to map to (0, 1] */
    return (fp_t)(state) * 2.3283064365386963e-10f;
}

#endif