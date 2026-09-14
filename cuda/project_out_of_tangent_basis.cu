#include "float3.h"
#include <iostream>

// Projects the vectors in src into space spanned by basis0 and basis1,
// copies the result into dst.
extern "C" __global__ void
project_out_of_tangent_basis(
			float* __restrict__  basis0_x, float* __restrict__  basis0_y, float* __restrict__  basis0_z,
			float* __restrict__  basis1_x, float* __restrict__  basis1_y, float* __restrict__  basis1_z,
			float* __restrict__ src_η, float* __restrict__ src_ξ,
			float* __restrict__ dst_x, float* __restrict__ dst_y, float* __restrict__ dst_z,
			int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

	if (i < N) {
		
		dst_x[i] = basis0_x[i]*src_η[i] + basis1_x[i]*src_ξ[i];
		dst_y[i] = basis0_y[i]*src_η[i] + basis1_y[i]*src_ξ[i];
		dst_z[i] = basis0_z[i]*src_η[i] + basis1_z[i]*src_ξ[i];
	}
}