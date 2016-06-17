#! /usr/bin/python

import sys, os, subprocess, random, time
#import pexpect, re

#############################################################################
totalFiNumber = 30
#############################################################################
# FI Config
staticInstIndex = "" # fiInstIndex=20 or ""
staticKernelIndex = "" # 5 or ""
dynamicKernelIndex = "" # 3 or ""
bitIndex = "" # 63 or ""	
useGdb = False
#############################################################################
# Make commands
flagHeader = staticInstIndex + " CICC_MODIFY_OPT_MODULE=1 LD_PRELOAD=./libnvcc.so nvcc -arch=sm_30 -rdc=true -dc -g -G -Xptxas -O0 "
ktraceFlag = " -D KERNELTRACE"
linkFlags = ""
optFlags = ""
#############################################################################
makeCommand1 = flagHeader + " example.cu -o example.o" + ktraceFlag
makeCommand2 = ""
makeCommand3 = ""
makeCommand4 = ""
linkList = " example.o "
outputExeFile = "example.out"
#############################################################################
bmName = "example"
inputParameters = ""
#############################################################################


# For grouping
#groupedProfileLineList = []
groupedProfileDic = {}

def compileInjectionPass():
	# Prepare lib and files
	os.system("cp bamboo_lib/injection_lib/* .")

	# Compile to injection pass and cuda program
	print ("***[GPGPU-BAMBOO]*** Generating Injection Pass ... ")
	os.system(makeCommand1)
	os.system(makeCommand2)
	os.system(makeCommand3)
	os.system(makeCommand4)
	os.system("nvcc -arch=sm_30 injection_runtime.cu -c -dc -O0 -g -G")
	os.system("nvcc -arch=sm_30 injection_runtime.o " + linkList + " -o " + outputExeFile + " -O0 " + linkFlags)

	# Clean obj files
	os.system("rm bamboo_injection.cu injection_runtime.o injection_runtime.cu " + linkList + " libcicc.so libnvcc.so ")
	os.system("rm opt_bamboo_before.ll")
	os.system("mv opt_bamboo_after.ll bamboo_fi/" + bmName + "_injection.ll")


# Take bamboo.profile.txt and generate global groupedProfileLineList
# This grouping algorithm is to profile thread based dynamic kernel launches.
def groupProfileLines():
	profileLineList = []
	with open("bamboo_fi/bamboo.profile.txt", "r") as ins:
		for line in ins:
			profileLineList.append(line)
	
	# Grouping
	profileDic = {}
	for line in profileLineList:
		instCount = getInstCountFromProfileLine(line)
		#staticKernelIndex = getStaticKernelFromProfileLine(line)
		dynamicKernelIndex = getDynamicKernelFromProfileLine(line)
		if dynamicKernelIndex not in profileDic:
			profileDic[dynamicKernelIndex] = []
			groupedProfileDic[dynamicKernelIndex] = []
		#if instCount not in profileDic[dynamicKernelIndex]:
		profileDic[dynamicKernelIndex].append(instCount)
		groupedProfileDic[dynamicKernelIndex].append(line)
			#groupedProfileLineList.append(line)
	
	print ("***[GPGPU-BAMBOO]*** Done grouping: ")
	#for staticKernelIndex in profileDic:
	#	print ("***[GPGPU-BAMBOO]*** staticKernelIndex: " + `staticKernelIndex` + " size: " + `len(profileDic[staticKernelIndex])`)


def getInstCountFromProfileLine(line):
	return int(line.split(" -- ")[2].split()[1])


def getStaticKernelFromProfileLine(line):
	return int(line.split(" -- ")[4].split()[1])


def getDynamicKernelFromProfileLine(line):
	return int(line.split(" -- ")[3].split()[1])


def getThreadIndexFromProfileLine(line):
	return int(line.split(" -- ")[1].split()[1])


# Randomly select a FI point from groups
# This grouping algorithm is to profile thread based dynamic kernel launches.
def generateFiPointFromGroupedLines():
	# Choose dynamic kernel
	groupSize = len(groupedProfileDic)
	#print "Total Dyn Kernels: " + `groupSize`
	random.seed(time.time())
	selectedDynamicKernelIndex = 0
	while(True):
		selectedDynamicKernelIndex = random.randint(0, groupSize-1)
		if selectedDynamicKernelIndex in groupedProfileDic:
			break
	groupedLinesInDynamicKernel = groupedProfileDic[selectedDynamicKernelIndex]

	# choose thread
	groupedLineSize = len(groupedProfileDic[selectedDynamicKernelIndex])
	#print "Total Unique Threads: " + `groupedLineSize`
	random.seed(time.time())
	selectedLineNumber = random.randint(0, groupedLineSize-1)
	selectedLine = groupedLinesInDynamicKernel[selectedLineNumber]
	
	
	# Append to bamboo.fi.log.txt
	with open("bamboo_fi/bamboo.fi.log.txt", "a") as logf:
		logf.write(selectedLine)
	
	# Overwrite bamboo.fi.config.txt
	with open("bamboo_fi/bamboo.fi.config.txt", "w") as configf:
		fiThreadIndex = getThreadIndexFromProfileLine(selectedLine)
		totalInstCount = getInstCountFromProfileLine(selectedLine)
		fiInstCount = random.randint(1, totalInstCount)
		fiDynamicKernelIndex = getDynamicKernelFromProfileLine(selectedLine)
		fiStaticKernelIndex = getStaticKernelFromProfileLine(selectedLine)
		configf.write(`fiThreadIndex` + " " + `fiInstCount` + " " + `fiDynamicKernelIndex` + " " + `fiStaticKernelIndex`)
	
	print "***[GPGPU-BAMBOO]*** Injecting to threadID:" + `fiThreadIndex` + ", fiInstCount:" + `fiInstCount` + ", fiDynKernel:" + `fiDynamicKernelIndex` + ", fiStaticKernel:" + `fiStaticKernelIndex`


def runInjection(totalFiNumber):
	# mkdir for output files of FI
	os.system("mkdir bamboo_fi/std_output")
	os.system("mkdir bamboo_fi/err_output")
	os.system("mkdir bamboo_fi/prog_output")
	os.system("mkdir bamboo_fi/ktrace")

	for fiNumber in range(1, totalFiNumber+1):

		originalFileList = os.listdir("./")

		# Select FI point and generate bamboo.fi.config.txt for this round of FI
		generateFiPointFromGroupedLines()		

		# Run injection pass
		print ("***[GPGPU-BAMBOO]*** Running Injection Executable ... (" + `fiNumber` + "/" + `totalFiNumber` + ")")

		# Use getLastError instead of cuda-gdb
		if useGdb == False:
			try:
				stdOutput = subprocess.check_output("timeout 500 ./" +outputExeFile+ " " + inputParameters, stderr=subprocess.STDOUT, shell=True)
				stdOutputFileName = "std_output-" + `fiNumber`
				stdOutputFile = open(stdOutputFileName, "w")
				stdOutputFile.write(stdOutput)
				stdOutputFile.close()
				os.system("mv " + stdOutputFileName + " bamboo_fi/std_output/")
			except subprocess.CalledProcessError as grepexc:
				if os.path.isfile("bamboo.error.txt") == False:
                                	os.system("echo \"Maybe crash in host as there is no error log dumped at runtime.\n\" >> bamboo.error.txt")
				os.system("echo \"Return Code: " + `grepexc.returncode` + ", Output: " + `grepexc.output` + "\" >> bamboo.error.txt")
				os.system("mv bamboo.error.txt bamboo_fi/err_output/err_output-"+`fiNumber`)
			#except:
			#	print "It's hang!"
		else:
			# Get std output
			try:
				p = pexpect.spawn("cuda-gdb " +outputExeFile)
				p.timeout = 5000
				p.expect("cuda-gdb")
				p.sendline("break injection_runtime.cu:39")
				p.expect("cuda-gdb")
				p.sendline("run " + inputParameters)
		
				# Break at FI point
				p.expect("cuda-gdb")
				p.sendline("continue")

				# Measure execution time
				startTime = time.time()
				p.expect("cuda-gdb")

				# Log crash time
				elapsedTime = time.time() - startTime
		
				if "CUDA Exception:" in p.before:
					p.terminate(force=True)
					p.close()
					errOutput = p.before
					errOutput += "\n\n***" + `elapsedTime` + "***\n\n"
					errOutputFileName = "err_output-" + `fiNumber`
					errOutputFile = open(errOutputFileName, "w")
					errOutputFile.write(errOutput)
					errOutputFile.close()
					os.system("mv " + errOutputFileName + " bamboo_fi/err_output/")

					print "***[GPGPU-BAMBOO]*** Exception Thrown ... "
				else:
					# Move stdOutputFile	
					stdOutput = p.before
					stdOutputFileName = "std_output-" + `fiNumber`
					stdOutputFile = open(stdOutputFileName, "w")
					stdOutputFile.write(stdOutput)
					stdOutputFile.close()
					os.system("mv " + stdOutputFileName + " bamboo_fi/std_output/")
			except KeyboardInterrupt:
				sys.exit(0)
			except:
				# TODO: Hangle hangs
				print "It's hang!"

		# Move bamboo.ktrace.log.txt
		if os.path.isfile("bamboo.ktrace.log.txt"):
			os.system("mv bamboo.ktrace.log.txt bamboo_fi/ktrace/ktrace.log-" + `fiNumber` + ".txt")	

		# Update fiBit and fiBambooIndex from runtime log
		if os.path.isfile("bamboo_fi/bamboo.fi.runtime.log.txt"):
			fiBit = 0
			fiBambooIndex = 0
			with open("bamboo_fi/bamboo.fi.runtime.log.txt") as runtimef:
				runtimeLogLines = runtimef.readlines()
				fiBit = int(runtimeLogLines[0].replace("fiBit: ", ""))
				fiBambooIndex = int(runtimeLogLines[1].replace("bambooIndex: ", ""))

			# If index is 0, it is crash
			if fiBambooIndex == 0:
				# Update error report
				errOutputFileName = "bamboo_fi/err_output/err_output-" + `fiNumber`
				with open(errOutputFileName, "a") as errOutputFile:
					errOutputFile.write("Error when getting fiBit and bambooIndex\n")
					errOutputFile.close()

			# Update fi log file
                        logf = open("bamboo_fi/bamboo.fi.log.txt", "rb")
                        logLines = logf.readlines()
                        lastLine = logLines[-1]
                        logLines[-1] = lastLine.replace("\n", "") + "fiBit " + `fiBit` + " -- fiBambooIndex " + `fiBambooIndex` + "\n"
                        os.system("rm bamboo_fi/bamboo.fi.log.txt")
                        for logLine in logLines:
                                with open("bamboo_fi/bamboo.fi.log.txt", "a") as updatedLogf:
                                        updatedLogf.write(logLine)
                        os.system("rm bamboo_fi/bamboo.fi.runtime.log.txt")

		# Move program output file
		updatedFileList = os.listdir("./")
	        newFileList = set(updatedFileList).difference(originalFileList)

		for newFileName in newFileList:
			os.system("mv " + newFileName + " bamboo_fi/prog_output/" + newFileName.replace("./", "") + "-" + `fiNumber`)

		# zip checkpoint
                #os.system("zip kernel_checkpoint-" + `fiNumber` + ".zip bamboo_fi/prog_output/kernel_checkpoint.txt-" + `fiNumber`)
                #os.system("mv kernel_checkpoint-" + `fiNumber` + ".zip bamboo_fi/prog_output/")
                #os.system("rm bamboo_fi/prog_output/kernel_checkpoint.txt-" + `fiNumber`)

def main():
	compileInjectionPass()
	groupProfileLines()
	runInjection(totalFiNumber)
	


##############################################################################
main()
			
	
	


