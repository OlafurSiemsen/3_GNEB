
#include <stdint.h>
#include "float3.h"

// TODO: Rename and refactor signature
// Rotates a vector v0 from the cotangent space of m0 on the N unit sphere
// to the cotangent space of m, outputting in v0.
// This assumes several facts: |m0|=|m|=1, v0 ∈ span(m0, m), v0 ⟂ m0

extern "C" __global__ void
cotangentspacerotation(
        float* __restrict__ vx,  float* __restrict__  vy,  float* __restrict__ vz,
        float* __restrict__ v0x,  float* __restrict__  v0y,  float* __restrict__ v0z,
        float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz,
        float* __restrict__ m0x,  float* __restrict__  m0y,  float* __restrict__ m0z,
        int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m  = { mx[i],  my[i],  mz[i]};
        float3 m0 = {m0x[i], m0y[i], m0z[i]};
        float3 v0 = {v0x[i], v0y[i], v0z[i]};

        const float m_DOT_v0 = dot(m,v0);
        const float m_DOT_m0 = dot(m,m0);

        vx[i] = m_DOT_m0*v0x[i]-m_DOT_v0*m0x[i];
        vy[i] = m_DOT_m0*v0y[i]-m_DOT_v0*m0y[i];
        vz[i] = m_DOT_m0*v0z[i]-m_DOT_v0*m0z[i];

    }
}
