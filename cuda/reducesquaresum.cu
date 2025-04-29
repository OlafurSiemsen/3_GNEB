#include "reduce.h"
#include "sum.h"
#include "float3.h"

#define load_square(i) pow2(src[i])

extern "C" __global__ void
reducesquaresum(float* __restrict__ src, float*__restrict__  dst, float initVal, int n) {
    reduce(load_square, sum, atomicAdd)
}

