#! /usr/bin/python

import sys, os, subprocess

#############################################################################
# FI Config
staticInstIndex = "" # fiInstIndex=20 or ""
staticKernelIndex = "" # 5 or ""
dynamicKernelIndex = "" # 3 or ""
bitIndex = "" # 63 or ""	
#############################################################################
flagHeader = staticInstIndex + " CICC_MODIFY_OPT_MODULE=1 LD_PRELOAD=./libnvcc.so nvcc -arch=sm_30 -rdc=true -dc -g -G -Xptxas -O0 -D BAMBOO_PROFILING"
ktraceFlag = " -D KERNELTRACE"
linkFlags = ""
optFlags = ""
#############################################################################
makeCommand1 = flagHeader + " example.cu -o example.o " + ktraceFlag
makeCommand2 = ""
makeCommand3 = ""
makeCommand4 = ""
linkList = " example.o "
outputExeFile = "example.out"
#############################################################################
bmName = "example"
inputParameters = ""
#############################################################################


def runProfile():
	# Prepare lib and files
	# TODO: mkdir for bamboo_fi and bamboo_lib
	originalFileList = os.listdir("./")
	os.system("mkdir bamboo_fi")
	os.system("mkdir bamboo_fi/baseline")
	os.system("cp bamboo_lib/profiling_lib/* .")

	# Compile to profiling pass and cuda program
	print ("***[GPGPU-BAMBOO]*** Generating Profiling Pass ... ")
	os.system(makeCommand1)
	os.system(makeCommand2)
	os.system(makeCommand3)
	os.system(makeCommand4)
	os.system("nvcc -arch=sm_30 profiling_runtime.cu -c -dc -O0")
	os.system("nvcc -arch=sm_30 profiling_runtime.o " + linkList + " -o " + outputExeFile + " -O0 " + linkFlags)

	# Run profiling pass and dump bamboo.profile.txt
	print ("***[GPGPU-BAMBOO]*** Generating Profiling Traces ... ")
	os.system("rm bamboo.profile.txt")
	
	goldenOutput = subprocess.check_output("./" +outputExeFile+ " " + inputParameters, shell=True)
	print goldenOutput
	goldenOutputFile = open("bamboo_fi/baseline/golden_output", "w")
	goldenOutputFile.write(goldenOutput)
	goldenOutputFile.close()
	
	# Clean obj files
	os.system("rm bamboo_profiling.cu profiling_runtime.o profiling_runtime.cu " + linkList + " libcicc.so libnvcc.so " + outputExeFile)
	
	# TODO: Move output files to bamboo_fi
	os.system("mv bamboo.profile.txt bamboo_fi/")
	os.system("mv opt_bamboo_before.ll bamboo_fi/" +bmName+ "_kernels.ll")
	os.system("mv opt_bamboo_after.ll bamboo_fi/" +bmName+ "_profile.ll")

	updatedFileList = os.listdir("./")
	newFileList = set(updatedFileList).difference(originalFileList)
	for newFileName in newFileList:
		if newFileName != "bamboo_fi":
			os.system("mv " + newFileName + " bamboo_fi/baseline")

	print ("***[GPGPU-BAMBOO]*** Done! ")




def main():
	runProfile();


##############################################################################
main()
			
	
	


