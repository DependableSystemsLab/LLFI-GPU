LLFI-GPU
====

LLFI-GPU is an LLVM based fault injection tool, that injects faults into the LLVM IR of the application source code of GPU CUDA kernels.  The faults can be injected into specific program points, and the effect can be easily tracked back to the source code.  LLFI-GPU is typically used to map fault characteristics back to source code, and hence understand source level or program characteristics for various kinds of fault outcomes.


INSTALLATION
===

Dependencies (Tested):

1. Python 2.7
2. NVCC v6.0.1
3. CUDA SDK v6.0.37 
4. LLVM 3.0
5. Ubuntu 12.04 LTS x64

Steps:

1. Modify 'makeCommands'to compile target benchmark in profile.py and inject.py. Put benchmark input in 'inputParameters'.
2. Configure fault injection parameters in '# FI Config' section in profile.py and inject.py.
3. Add headers in target benchmark source code. This is shown in line 9-17 in example.cu. Label GPU kernel calls so that you can trace it. This is added in line 124 and 126 in exmple.cu.

That's it. Run "python profile.py" and "python inject.py" to start fault injections. All fault injection logs and IR files are located under folder called 'bamboo_fi'.


We tested on NVIDIA K20 and GTX960 GPUs.

Have fun!


PAPER
===
[Understanding Error Propagation in GPGPU Applications (SC'16)](http://blogs.ubc.ca/karthik/2016/06/20/understanding-error-propagation-in-gpgpu-applications/)


An example of the command line output by running the above scripts is as follow:

------------------------------------------------------------------
gpli@karthik-pc00:~/Workplace/test/LLFI-GPU$ python profile.py 
***[GPGPU-BAMBOO]*** Generating Profiling Pass ... 
Profiling pass installed ... 
***[GPGPU-BAMBOO]*** Generating Profiling Traces ... 
Time spent by GPU: 21.590336 ms.
Time spent by CPU: 0.037888 ms.

***[GPGPU-BAMBOO]*** Done! 
gpli@karthik-pc00:~/Workplace/test/LLFI-GPU$ python inject.py 
***[GPGPU-BAMBOO]*** Generating Injection Pass ... 
***[GPGPU-BAMBOO]*** Done grouping: 
***[GPGPU-BAMBOO]*** Injecting to threadID:52, fiInstCount:39, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (1/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:213, fiInstCount:61, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (2/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:176, fiInstCount:180, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (3/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:170, fiInstCount:36, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (4/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:113, fiInstCount:59, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (5/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:223, fiInstCount:141, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (6/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:236, fiInstCount:124, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (7/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:175, fiInstCount:49, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (8/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:145, fiInstCount:67, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (9/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:104, fiInstCount:5, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (10/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:59, fiInstCount:103, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (11/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:123, fiInstCount:64, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (12/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:14, fiInstCount:178, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (13/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:106, fiInstCount:181, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (14/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:194, fiInstCount:95, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (15/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:54, fiInstCount:143, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (16/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:138, fiInstCount:22, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (17/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:251, fiInstCount:15, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (18/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:20, fiInstCount:42, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (19/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:105, fiInstCount:113, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (20/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:10, fiInstCount:162, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (21/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:149, fiInstCount:110, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (22/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:159, fiInstCount:142, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (23/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:218, fiInstCount:20, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (24/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:186, fiInstCount:12, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (25/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:46, fiInstCount:167, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (26/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:52, fiInstCount:195, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (27/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:30, fiInstCount:53, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (28/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:104, fiInstCount:41, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (29/30)
***[GPGPU-BAMBOO]*** Injecting to threadID:24, fiInstCount:42, fiDynKernel:0, fiStaticKernel:0
***[GPGPU-BAMBOO]*** Running Injection Executable ... (30/30)
------------------------------------------------------------------

