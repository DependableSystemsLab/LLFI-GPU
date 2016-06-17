#include <stdio.h> 
#include <cuda.h> 
#include <stdlib.h> 
#include <time.h>

#define B 1 // blocks in the grid 
#define T 10 // threads in a block 

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#ifdef BAMBOO_PROFILING
#include "bamboo_profiling.cu"
#else
#include "bamboo_injection.cu"
#endif

__global__ void gpu_mult(int *a,int *b, int *c, int N) { 

	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	if( col < N && row < N) {
		for(int i = 0; i < N; i++) {
			sum += a[row * N + i] * b[i * N + col];
		}
		c[row * N + col] = sum;
	}
} 

void cpu_matrixmult(int * cpu_a, int* cpu_b, int* cpu_c, int N) {
	int row, col, k, sum;

	for(row = 0; row < N; row++) {
		for(col = 0; col < N; col++) {
			sum = 0;
			for(k = 0; k < N; k++) {
				sum += cpu_a[row * N + k] * cpu_b[k * N + col];
			}
			cpu_c[row * N + col] = sum;
		}
	}
}
int getInput(int* N) {
	printf("Please input size of array(NxN): ");
	scanf("%d", N);
	return 0;
}

void printArray(char* name, int* result, int N) {
	FILE * fp;
	fp = fopen("result.txt", "a");
	fprintf(fp, "%s\n", name);
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			fprintf(fp, "%d ", result[i * N + j]);
		}
		fprintf(fp, "\n");
	}
}

int main(void) { 

	int N;							// Size of the array in each dimension
	int *a, *b, *c, *d; 			// arrays
	int *dev_a, *dev_b, *dev_c; 
	int size;						// number of bytes in arrays
	cudaEvent_t start, stop;		// Time measuring events
	float elapsed_time_ms;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Get array dimensions from the user
	//getInput(&N);
	N = 16;

	int Grid_Dim_x = N;				// Grid structure values
	int Grid_Dim_y = N;
	int Block_Dim_x = 1; 			// Block structure values
	int Block_Dim_y = 1;

	// Grid and block structures
	dim3 Grid(Grid_Dim_x, Grid_Dim_y); 
	dim3 Block(Block_Dim_x, Block_Dim_y);



	// Total size of arrays in bytes
	size = N * N * sizeof(int);

	// Set size of arrays
	a = (int*) malloc(size);
	b = (int*) malloc(size);
	c = (int*) malloc(size);
	d = (int*) malloc(size);

	// Initialize matrices with random numbers
	srand(time(NULL));
	for (int i = 0; i < N; i++) { 
		for(int j = 0 ; j < N; j++) {

			int valA = rand() % 10;
			a[i * N + j] = valA;
			int valB = rand() % 10;
			b[i * N + j] = valB;
			c[i * N + j] = 0;
			d[i * N + j] = 0;
		}
	} 

	// Allocate memory on the device
	cudaMalloc((void**)&dev_a, N * N * sizeof(int)); 
	cudaMalloc((void**)&dev_b, N * N * sizeof(int)); 
	cudaMalloc((void**)&dev_c, N * N * sizeof(int)); 

	// Copy the array from the host to the device
	cudaMemcpy(dev_a, a , N * N * sizeof(int),cudaMemcpyHostToDevice); 
	cudaMemcpy(dev_b, b , N * N * sizeof(int),cudaMemcpyHostToDevice); 
	cudaMemcpy(dev_c, c , N * N * sizeof(int),cudaMemcpyHostToDevice); 
	// Start timing the cuda computation
	cudaEventRecord(start, 0);

	bambooLogKernelBegin(0);
	gpu_mult<<<Grid, Block>>>(dev_a, dev_b, dev_c, N);
	bambooLogKernelEnd(0);

	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

	// Measure end of compuatation
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);

	printf("Time spent by GPU: %f ms.\n", elapsed_time_ms);

	// CPU computation
	cudaEventRecord(start, 0);

	cpu_matrixmult(a, b, d, N);

	// Measure end of compuatation
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("Time spent by CPU: %f ms.\n", elapsed_time_ms);

	printArray("CPU", d, N);
	printArray("GPU", c, N);

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			if(c[i * N + j] != d[i * N + j]){
				printf("Matrices are not the same\n");
			}	
		}
	}

	free(a); 
	free(b); 
	free(c); 
	free(d);
	cudaFree(dev_a); 
	cudaFree(dev_b); 
	cudaFree(dev_c); 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0; 
}
