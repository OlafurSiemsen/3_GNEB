// dst[i] = a[i] / b[i]
extern "C" __global__ void
pointwise_recip(float* __restrict__  dst, float* __restrict__  a, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if(i < N) {
        if (a[i] != 0.0f) {
            dst[i] = 1.0f / a[i];
        } else {
            dst[i] = 0.0f;
        }
    }
}

