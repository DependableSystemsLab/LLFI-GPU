#include <stdio.h>
#include <cuda.h>
#include <stdint.h>
#include <math.h>

__device__ bool deviceFiFlag = false;
__device__ long long deviceFiInstCount;
__device__ long long deviceFiThreadIndex;
__device__ double deviceSeedFactor; // Should be from [0,1)
__device__ int deviceFiBit;
__device__ long deviceBambooIndex;

/////////////////////////////////////////////////
__device__ long long currentCycle = 0;

extern "C" __device__ void injectFault(char* buf, int size, long bambooIndex)
{
	int blockId = blockIdx.x 
		+ blockIdx.y * gridDim.x 
		+ gridDim.x * gridDim.y * blockIdx.z; 
	long long index = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x)
		+ threadIdx.x;

	//////////////////////////////////////////////////////////////////////////////

	if(deviceFiFlag == true && index == deviceFiThreadIndex){
		//printf("Cycle: %lld\n", currentCycle);
		// There is not instCount=0, instCount means Nth inst executed.
		currentCycle++;
	}

	if(deviceFiFlag == true && deviceFiInstCount == currentCycle && index == deviceFiThreadIndex){
		deviceFiFlag = false;

		//printf("\n\\\\\\\\\\\\\\\\\\/////////////////\nInjecting fault ... \n\n");

		unsigned char oldbuf;
		unsigned fiBit, fiBytepos, fiBitpos;
		int randint = deviceSeedFactor*size;
		fiBit = (unsigned) randint;
		fiBytepos = fiBit / 8;
		fiBitpos = fiBit % 8;
		memcpy(&oldbuf, &buf[fiBytepos], 1);

		printf("	|** fiBit: %d, fiBambooIndex: %ld\n", fiBit, bambooIndex);
		printf("	|** FI Config Runtime ** deviceFiThreadIndex: %lld, deviceFiInstCount: %lld, fiBit: %u(size %d)\n", deviceFiThreadIndex, deviceFiInstCount, fiBit, size);
		printf("	|** Original Value: %#010x **\n", oldbuf);
		buf[fiBytepos] ^= 0x1 << fiBitpos;
		printf("	|** Corrupted Value: %#010x **\n", buf[fiBytepos]);

		deviceFiBit = fiBit;
		deviceBambooIndex = bambooIndex;

	}

}

