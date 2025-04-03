#include "float3.h"

// dst[i] = veclen(a[i])
extern "C" __global__ void
veclen(float* __restrict__ dst,
        float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
        int N) {
    
    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float3 A = {ax[i], ay[i], az[i]};
        
        dst[i] = len(A);
    }
}