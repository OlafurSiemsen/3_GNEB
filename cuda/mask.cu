#include "float3.h"
#include "stdio.h"

// Mask out vector when volume is zero.
extern "C" __global__ void
mask(float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
	float* __restrict__ vol,
	int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		bool cell_empty = vol[i] == 0.0f; 
		if (cell_empty)
		{
			// Mask out zero volume cell
			vx[i] = 0.0f;
			vy[i] = 0.0f;
			vz[i] = 0.0f;
		}
	}
}

