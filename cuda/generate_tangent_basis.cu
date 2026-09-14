#include "float3.h"
#include <iostream>

// Creates a basis that spans the tangent bundle at the point of the magnetic
// configuration m. Sets basis0 and basis1 as the basis vectors in embedding space.
extern "C" __global__ void
generate_tangent_basis(
			float* __restrict__  basis0_x, float* __restrict__  basis0_y, float* __restrict__  basis0_z,
			float* __restrict__  basis1_x, float* __restrict__  basis1_y, float* __restrict__  basis1_z,
			float* __restrict__ m_x, float* __restrict__ m_y, float* __restrict__ m_z,
			int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

	if (i < N) {
		if (abs(m_z[i]) > 0.9f) {
			basis0_x[i] = 1.0f;
			basis0_y[i] = 0.0f;
			basis0_z[i] = 0.0f;
		} else {
			basis0_x[i] = 0.0f;
			basis0_y[i] = 0.0f;
			basis0_z[i] = 1.0f;
		}

		float3 basis0 = {basis0_x[i],basis0_y[i],basis0_z[i]};
		float3 m = {m_x[i],m_y[i],m_z[i]};
		float basis0_dot_m = dot(basis0,m);
		
		basis0 = basis0 - basis0_dot_m * m;
		float recipnorm = 1/sqrt(dot(basis0,basis0));
		basis0 = recipnorm * basis0; // TODO-olafur: Check if this works: reference/value

		float3 basis1 = cross(basis0,m);

		basis1_x[i] = basis1.x;
		basis1_y[i] = basis1.y;
		basis1_z[i] = basis1.z;
	}
}