#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_init(int *bucket) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bucket[i] = 0;
}

__global__ void bucket_add(int *key, int *bucket) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&bucket[key[i]], 1);
}

__global__ void bucket_scan( int *bucket,int *a, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    // prefix sum     
    for(int j=1; j<range; j<<=1){
        a[i] = bucket[i];
        __syncthreads();
        bucket[i] += a[i-j];
        __syncthreads();
    }
}

__global__ void bucket_assignment(int *key, int *bucket, int n, int range){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(int j=0; j<range; j++){
        if(i < bucket[j] && i >= bucket[j-1]){
            key[i] = j;
            return;
        }
    }
}

int main() {
    int n = 50;
    int range = 5;
    int *key;
    cudaMallocManaged(&key, n*sizeof(int));

    for (int i=0; i<n; i++) {
        key[i] = rand() % range;
        printf("%d ",key[i]);
    }
    printf("\n");

    int *bucket, *a;
    cudaMallocManaged(&bucket, range*sizeof(int));
    cudaMallocManaged(&a, range*sizeof(int));

    bucket_init<<<1,range>>>(bucket);
    cudaDeviceSynchronize();

    bucket_add<<<1,n>>>(key, bucket);
    cudaDeviceSynchronize();

    bucket_scan<<<1,range>>>(bucket ,a, range);
    cudaDeviceSynchronize();

    bucket_assignment<<<1,n>>>(key, bucket, n, range);
    cudaDeviceSynchronize();

    for (int i=0; i<n; i++) {
        printf("%d ",key[i]);
    }
    printf("\n");

    cudaFree(key);
    cudaFree(bucket);
}
