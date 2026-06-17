#include "float3.h"
#include "stdio.h"

// Calculates the angle between corresponging vectors in neighbouring images in
// src, copying the angles into dst
extern "C" __global__ void
interimageangles(float* __restrict__ dst,
				 float* __restrict__ vec_x, float* __restrict__ vec_y, float* __restrict__ vec_z, 
				 float* __restrict__ vol, int N, int N_cells) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		bool cell_empty = vol[i%N_cells] == 0.0f; 
		if (cell_empty)
		{
			// Skip the calculation for empty cells
			dst[i] = 0.0f;
		}
		else
		{
			float3 v_n = {vec_x[i], vec_y[i], vec_z[i]};
			float3 v_np1 = {vec_x[i + N_cells], vec_y[i + N_cells], vec_z[i + N_cells]};
			float crossprod_norm = len(cross(v_n, v_np1));
			float dotprod = dot(v_n, v_np1);

			dst[i] = atan2(crossprod_norm, dotprod);
		}
	}
}

