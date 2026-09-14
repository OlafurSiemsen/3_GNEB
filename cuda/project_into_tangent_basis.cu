#include "float3.h"
#include <iostream>

// Projects the vectors in src into space spanned by basis0 and basis1,
// copies the result into dst.
extern "C" __global__ void
project_into_tangent_basis(
			float* __restrict__  basis0_x, float* __restrict__  basis0_y, float* __restrict__  basis0_z,
			float* __restrict__  basis1_x, float* __restrict__  basis1_y, float* __restrict__  basis1_z,
			float* __restrict__ src_x, float* __restrict__ src_y, float* __restrict__ src_z,
			float* __restrict__ dst_η, float* __restrict__ dst_ξ,
			int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

	if (i < N) {
		float3 basis0 = {basis0_x[i],basis0_y[i],basis0_z[i]};
		float3 basis1 = {basis1_x[i],basis1_y[i],basis1_z[i]};
		float3 src = {src_x[i],src_y[i],src_z[i]};
		
		dst_η[i] = dot(basis0,src);
		dst_ξ[i] = dot(basis1,src);
	}
}