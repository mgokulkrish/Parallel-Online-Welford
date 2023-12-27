#include <iostream>
using namespace std;

// setting the default values, change them per requirment.
// TODO: need to stress test if grid dimension occurs
#define N 10
const int threadsPerBlock = 256;
const int blocksPerGrid = min(32, (N + threadsPerBlock-1)/threadsPerBlock);

__global__
void stddv(float *a, float* mean, float* stddv){
    __shared__ float cache_mean[threadsPerBlock];
    __shared__ float cache_var[threadsPerBlock];
    __shared__ int cache_count[threadsPerBlock];

    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int cacheIndex = threadIdx.x;

    float tMean = 0.0f;
    float tVar = 0.0f;
    int cnt = 1;

    while(tid < N){
        float x = a[tid];
        float oldMean = tMean;
        tMean = tMean + (x-tMean)/(cnt);
        tVar = tVar + (x-tMean)*(x-oldMean);
        cnt = cnt + 1;
        tid += blockDim.x * gridDim.x;
    }

    cache_mean[cacheIndex] = tMean;
    cache_var[cacheIndex] = tVar;
    cache_count[cacheIndex] = cnt-1;
    __syncthreads();

    int total_i = min(blockDim.x, N-blockDim.x*blockIdx.x);
    int i = (total_i==1) ? total_i/2 : (total_i+1)/2;

    while(i != 0){
        if(cacheIndex < i){
            int idx1 = cacheIndex; int idx2 = cacheIndex + i;
            int n1 = cache_count[idx1]; int n2 = cache_count[idx2];
            float mean1 = cache_mean[idx1]; float mean2 = cache_mean[idx2];
            float var1 = cache_var[idx1]; float var2 = cache_var[idx2];
            if(!(total_i%2 && cacheIndex == i-1)){
                int n = n1 + n2;
                float delta = mean2 - mean1;
                float combined_mean = mean1 + (delta*((float)n2/(float)n));
                float combined_var = var2 + var1 + ((delta*delta)*(float)n1*(float)n2/(float)n);
                cache_count[idx1] = n;
                cache_mean[idx1] = combined_mean;
                cache_var[idx1] = combined_var;
            }
        }
        __syncthreads();
        total_i = i;
        i = (i==1) ? i/2 : (i+1)/2;
    }

    if(cacheIndex == 0){
        mean[blockIdx.x] = cache_mean[0];
        int size = min(blockDim.x, N-blockDim.x*blockIdx.x);
        stddv[blockIdx.x] = cache_var[0];
    }
    __syncthreads();

    // TODO: parallelize this part.
    // computing mean and variance across
    // the block in a parallel way
    if(blockIdx.x==0 && cacheIndex==0){
        int n1 = min(blockDim.x, N-blockDim.x*blockIdx.x);
        float mean1 = mean[0];
        float var1 = stddv[0];
        for(int i=1; i<gridDim.x; i++){
            float mean2 = mean[i];
            float var2 = stddv[i];
            int n2 = min(blockDim.x, N-blockDim.x*i);
            int n = n1 + n2;
            float delta = mean2 - mean1;
            float combined_mean = mean1 + (delta*((float)n2/(float)n));
            float combined_var = var2 + var1 + ((delta*delta)*(float)n1*(float)n2/(float)n);
            mean1 = combined_mean;
            var1 = combined_var;
            n1 = n;
        }
        mean[0] = mean1;
        stddv[0] = sqrt(var1/(float)(n1-1));
    }

    return;
}

int main(void){
    int output_size = blocksPerGrid;
    float a[N], b[output_size], c[output_size];
    float *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, sizeof(float)*N);
    cudaMalloc((void**)&dev_b, sizeof(float)*output_size);
    cudaMalloc((void**)&dev_c, sizeof(float)*output_size);

    for(int i=0; i<N; i++){
        a[i] = (float)i;
    }

    cudaMemcpy(dev_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);

    //call the kernel
    printf("bpg, tpb = (%d, %d)\n", blocksPerGrid, threadsPerBlock);
    stddv<<<blocksPerGrid, threadsPerBlock>>> (dev_a, dev_b, dev_c);


    cudaMemcpy(b, dev_b, sizeof(float)*output_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(c, dev_c, sizeof(float)*output_size, cudaMemcpyDeviceToHost);
    printf("mean (a.k.a b[0]) = %f\n", b[0]);
    printf("stddv (a.k.a c[0]) = %f\n", c[0]);


    cudaFree(dev_a); 
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}