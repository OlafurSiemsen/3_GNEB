// dst[i] = a[i] * b[i]
extern "C" __global__ void
pointwise_mul(float* __restrict__  dst, float* __restrict__  a, float* __restrict__ b, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        if (b[i] != 0.0f) {
            dst[i] = a[i] * b[i];
        } else {
            dst[i] = 0.0f;
        }
    }
}

