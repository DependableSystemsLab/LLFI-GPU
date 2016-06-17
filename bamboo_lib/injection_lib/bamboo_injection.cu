#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib> // for rand() and srand()
#include <ctime> // for time()
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>


extern "C" __device__ bool deviceFiFlag;
extern "C" __device__ long long deviceFiInstCount;
extern "C" __device__ long long deviceFiThreadIndex;
extern "C" __device__ double deviceSeedFactor;
extern "C" __device__ unsigned deviceFiBit;
extern "C" __device__ long deviceBambooIndex;

using namespace std;

long long dynamicKernelIndex = 0;
long long fiThreadIndex, fiInstCount, fiDynamicKernelIndex, fiStaticKernelIndex;


///////////////////////////////////////////////////////////////////
// e.g. of fiConfigLine:
//threadIndex=0 instCount=81 dynamicKernelIndex=0 staticKernelIndex=0 
//0 81 0 0

long long getThreadIndex(string fiConfigLine){
	string lineArr[4];
	int i = 0;
	stringstream ssin(fiConfigLine);
	while (ssin.good() && i < 4){
		ssin >> lineArr[i];
		++i;
	}
	char lineChars[1024];
	strncpy(lineChars, lineArr[0].c_str(), sizeof(lineChars));
	lineChars[sizeof(lineChars) - 1] = 0;
	return atoll(lineChars);
}

long long getInstCount(string fiConfigLine){
	string lineArr[4];
	int i = 0;
	stringstream ssin(fiConfigLine);
	while (ssin.good() && i < 4){
		ssin >> lineArr[i];
		++i;
	}
	char lineChars[1024];
	strncpy(lineChars, lineArr[1].c_str(), sizeof(lineChars));
	lineChars[sizeof(lineChars) - 1] = 0;
	return atoll(lineChars);
}

long long getDynamicKernelIndex(string fiConfigLine){
	string lineArr[4];
	int i = 0;
	stringstream ssin(fiConfigLine);
	while (ssin.good() && i < 4){
		ssin >> lineArr[i];
		++i;
	}
	char lineChars[1024];
	strncpy(lineChars, lineArr[2].c_str(), sizeof(lineChars));
	lineChars[sizeof(lineChars) - 1] = 0;
	return atoll(lineChars);
}

long long getStaticKernelIndex(string fiConfigLine){
	string lineArr[4];
	int i = 0;
	stringstream ssin(fiConfigLine);
	while (ssin.good() && i < 4){
		ssin >> lineArr[i];
		++i;
	}
	char lineChars[1024];
	strncpy(lineChars, lineArr[3].c_str(), sizeof(lineChars));
	lineChars[sizeof(lineChars) - 1] = 0;
	return atoll(lineChars);
}

double fRand(){
	srand(static_cast<unsigned int>(time(0)));
	double r = ((double)rand() / (double)(RAND_MAX));
	return r;
}

//////////////////////////////////////////////////////////////////


void bambooLogKernelBegin(int staticKernelIndex) {
	// Read profiled line for configuring FI
	ifstream t("bamboo_fi/bamboo.fi.config.txt");
	string fiConfigLine((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());
	fiThreadIndex = getThreadIndex(fiConfigLine);
	fiInstCount = getInstCount(fiConfigLine);
	fiDynamicKernelIndex = getDynamicKernelIndex(fiConfigLine); 
	fiStaticKernelIndex = getStaticKernelIndex(fiConfigLine);
	double seedFactor = fRand();

	// Set kernel fi flag
	if( staticKernelIndex == fiStaticKernelIndex && dynamicKernelIndex == fiDynamicKernelIndex ){
		// debug
		printf("	|-- FI Config Read -- fiThreadIndex: %lld, fiInstCount: %lld, fiDynamicKernelIndex: %lld, fiStaticKernelIndex: %lld, seedFactor: %f\n", fiThreadIndex, fiInstCount, fiDynamicKernelIndex, fiStaticKernelIndex, seedFactor);
		bool fiFlag = true;
		// Set fi config to runtime_lib
		cudaMemcpyToSymbol(deviceFiFlag, &fiFlag, sizeof(bool), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(deviceFiInstCount, &fiInstCount, sizeof(long long), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(deviceFiThreadIndex, &fiThreadIndex, sizeof(long long), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(deviceSeedFactor, &seedFactor, sizeof(double), 0, cudaMemcpyHostToDevice);
	}

#ifdef KERNELTRACE
	/*
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		ofstream errf;
		errf.open ("bamboo.error.log.txt");
		errf << "CUDA Error: " << cudaGetErrorString(error) << "\ndynamicKernelIndex: " << dynamicKernelIndex << "\nstaticKernelIndex: " << staticKernelIndex << "\n";
		errf.close();
	}
	*/
	ofstream ktracef("bamboo.ktrace.log.txt", std::ios_base::app | std::ios_base::out);
	//ktracef.open("bamboo.ktrace.log.txt");
	ktracef << " -- Start: Static Kernel Index: " << staticKernelIndex << " - Dynamic Kernel Index: "<< dynamicKernelIndex << "\n";
	ktracef.close();
#endif

}

void bambooLogKernelEnd(int staticKernelIndex) {

	// Dump fiBit and bambooIndex
	if( staticKernelIndex == fiStaticKernelIndex && dynamicKernelIndex == fiDynamicKernelIndex ){
		int fiBit;
		long fiBambooIndex;
		cudaMemcpyFromSymbol(&fiBit, deviceFiBit, sizeof(int), 0, cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(&fiBambooIndex, deviceBambooIndex, sizeof(long), 0, cudaMemcpyDeviceToHost);
		ofstream logf;
		logf.open ("bamboo_fi/bamboo.fi.runtime.log.txt");
		logf << "fiBit: " << fiBit << "\nbambooIndex: " << fiBambooIndex;
		logf.close();
	}

	// First check if there is last error
#ifdef KERNELTRACE
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		ofstream errf;
		errf.open ("bamboo.error.txt", std::ios_base::app | std::ios_base::out);
		errf << "Error Detected: " << cudaGetErrorString(error) << "\ndynamicKernelIndex: " << dynamicKernelIndex << "\nstaticKernelIndex: " << staticKernelIndex << "\n";
		errf.close();
		exit(-20);
	}
	ofstream ktracef("bamboo.ktrace.log.txt", std::ios_base::app | std::ios_base::out);
	//ktracef.open("bamboo.ktrace.log.txt");
	ktracef << " -- End: Static Kernel Index: " << staticKernelIndex << " - Dynamic Kernel Index: "<< dynamicKernelIndex << "\n";
	ktracef.close();
#endif

	dynamicKernelIndex++;
}

__device__ void capturePoint(void){}
