#include <stdint.h>
#include "float3.h"

// Rotates a towards b by an angle of dt*norm(b)
// TODO: rename, refactor final assignment to cancel dt?
extern "C" __global__ void
rotatevectors(
            float* __restrict__ ax,  float* __restrict__  ay,  float* __restrict__ az,
            float* __restrict__ bx,  float* __restrict__  by,  float* __restrict__ bz,
            float dt, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 A = {ax[i], ay[i], az[i]};
        float3 B = {bx[i], by[i], bz[i]};
        // TODO: Check if modulo is necessary or relevant to performance
        const float theta = fmod(dt*len(B), 2* M_PIf);

        float pref;
        if(theta<=1e-1){
            //pref = 1.0 - theta*theta*(1.0-theta*theta/20.0)/6.0;
            // don't divide when using float...
            const float theta2 = theta*theta;
            pref = 1.0 - 0.166667*theta2*(1.0-0.05*theta2);
        }else{
            pref = sin(theta)/theta;
        }
        // update m and normalize but check if it is within the sample
        if(!is0(A))
            A = normalized(A*cos(theta) + dt*pref*B);

        ax[i] = A.x;
        ay[i] = A.y;
        az[i] = A.z;

    }
}
