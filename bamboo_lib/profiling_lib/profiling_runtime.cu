#include <stdio.h>
#include <cuda.h>

const int MAX_THREAD_NUMBER = 1000000;
__device__ long long counterArray[MAX_THREAD_NUMBER] = {0};

extern "C" __device__ void bambooProfile(long bambooIndex)
{
	int blockId = blockIdx.x 
		+ blockIdx.y * gridDim.x 
		+ gridDim.x * gridDim.y * blockIdx.z; 
	long long index = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x)
		+ threadIdx.x;

	counterArray[index]++;
}
