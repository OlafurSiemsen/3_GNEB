#include "float3.h"
#include <iostream>

extern "C" __global__ void
rotationvectors(float* __restrict__  dst_x, float* __restrict__  dst_y, float* __restrict__  dst_z,
           float* __restrict__ a_x, float* __restrict__ a_y, float* __restrict__ a_z,
           float* __restrict__ cross_x, float* __restrict__ cross_y, float* __restrict__ cross_z,
           float* __restrict__ cross_norm,
           float* __restrict__ r_x, float* __restrict__ r_y, float* __restrict__ r_z,
           int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    float eps = 1.0e-06; // Slightly above std::numeric_limits<float>::epsilon()
    // float eps = 1.0e-09; 

    if (i < N) {
        float3 AxB = {cross_x[i], cross_y[i], cross_z[i]};
        float AxBnorm = cross_norm[i];
        
        if (AxBnorm > eps) { 
            // A and B are not anti-parallel
            float recipnorm = 1/AxBnorm;
            dst_x[i] = AxB.x * recipnorm;
            dst_y[i] = AxB.y * recipnorm;
            dst_z[i] = AxB.z * recipnorm;
        }
        else {
            float3 A = {a_x[i], a_y[i], a_z[i]};
            float3 R = {r_x[0], r_y[0], r_z[0]};
            float AdotR = dot(A, R);
            if (abs(abs(AdotR) - 1) > eps) {
                // Orthogonalize R to A
                float3 RorthA = R - AdotR*A;
                RorthA = normalized(RorthA);
                dst_x[i] = RorthA.x;
                dst_y[i] = RorthA.y;
                dst_z[i] = RorthA.z;
            } else { 
                // A is too parallel to the first r vector and therefore already
                // othrogonal to the second r vector
                dst_x[i] = r_x[1];
                dst_y[i] = r_y[1];
                dst_z[i] = r_z[1];
            }
        }
    }
}

