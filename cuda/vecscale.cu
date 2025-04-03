#include "float3.h"

extern "C" __global__ void
vecscale(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
           float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
           float* fac, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        // float3 A = {ax[i], ay[i], az[i]};
        // float3 facxA = fac*A;
        // dstx[i] = facxA.x;
        // dsty[i] = facxA.y;
        // dstz[i] = facxA.z;
        dstx[i] = fac[i] * ax[i];
        dsty[i] = fac[i] * ay[i];
        dstz[i] = fac[i] * az[i];
    }
}