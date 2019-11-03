#include "pch.h"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <iostream>
#include <string>
#include <chrono>


const char *source =
"kernel void matmul(global float *a, global float *b, global float *c) {\n"
"  int row = get_local_id(0);\n"
"  int col = get_local_id(1);\n"
"  const int globalRow = BS*get_group_id(0) + row;\n"
"  const int globalCol = BS*get_group_id(1) + col;\n"
"  int n = get_global_size(0);\n"
"  local float Asub[BS][BS];\n"
"  local float Bsub[BS][BS];\n"
""
"  float acc = 0.0f;\n"
"  const int numTiles = n/BS;\n"
"  for (int t = 0; t < numTiles; t++) {\n"
"    const int tiledRow = BS*t+row;\n"
"    const int tiledCol = BS*t+col;\n"
"    Asub[col][row] = a[tiledCol*n + globalRow];\n"
"    Bsub[col][row] = b[globalCol*n + tiledRow];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int k=0; k < BS; k++) {\n"
"      acc = mad(Asub[k][row], Bsub[col][k], acc);\n"
"     }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"  c[globalCol*n+globalRow] = acc;\n"
"}\n";
/*
const char *source =
"kernel void matmul(global float *a, global float *b, global float *c) {\n"
"  int x = get_global_id(0);\n"
"  int y = get_global_id(1);\n"
"  int n = get_global_size(0);\n"
""
"  float acc = 0.0f;\n"
"  for (int i = 0; i < n; i++) {\n"
"    acc += a[x*n+i] * b[y*n+i];\n"
"  }\n"
"  c[x*n+y] = acc;\n"
"}\n";*/

void matmult(float *a, float *b, float *c, int SIZE) {
	int BS = 16;
	float sum;
	for (int kk = 0; kk < SIZE; kk += BS) {
		for (int jj = 0; jj < SIZE; jj += BS) {
			for (int i = 0; i < SIZE; i++) {
				for (int j = jj; j < jj + BS; j++) {
					sum = c[i*SIZE + j];
					for (int k = kk; k < kk + BS; k++) {
						sum += a[i*SIZE + k] * b[SIZE*k + j];
					}
					c[i*SIZE + j] = sum;
				}
			}
		}
	}
}

void matmultOMP(float *a, float *b, float *c, int SIZE) {

		int BS = 16;
	float sum;
#pragma omp parallel for
	for (int kk = 0; kk < SIZE; kk += BS) {
//#pragma omp parallel for
		for (int jj = 0; jj < SIZE; jj += BS) {
#pragma omp parallel for
			for (int i = 0; i < SIZE; i++) {
				for (int j = jj; j < jj + BS; j++) {
					sum = c[i*SIZE + j];
					for (int k = kk; k < kk + BS; k++) {
						sum += a[i*SIZE + k] * b[SIZE*k + j];
					}
					c[i*SIZE + j] = sum;
				}
			}
		}
	}

	
}



//const unsigned int SIZE = 512;

int main() {
	float *a, *b, *c;
	int SIZE;
	std::cout << "   This program produces matrix multiplication. " << std::endl <<
		"      CPU---OpenMP---GPU" << std::endl;
	std::cout << "Enter the matrix size (Size must be a multiple of 16)" << std::endl;
	std::cin >> SIZE;

	a = new float[SIZE*SIZE];
	b = new float[SIZE*SIZE];
	c = new float[SIZE*SIZE];

	for (int i = 0; i < SIZE*SIZE; i++) {
		a[i] = 2;
		b[i] = 3;
		c[i] = 0;
	}
	auto startSys = std::chrono::steady_clock::now();
	matmult(a, b, c, SIZE);
	auto endSys = std::chrono::steady_clock::now();
	std::cout << "Time CPU: " << std::chrono::duration_cast<std::chrono::nanoseconds>(endSys - startSys).count()*(1e-09) << std::endl;

	memset(c, 0, sizeof(float)*SIZE*SIZE);

	startSys = std::chrono::steady_clock::now();
	matmultOMP(a, b, c, SIZE);
	endSys = std::chrono::steady_clock::now();
	std::cout << "Time OpenMP: " << std::chrono::duration_cast<std::chrono::nanoseconds>(endSys - startSys).count()*(1e-09) << std::endl;

	cl_int error = 0;

	cl_uint num_platforms = 0;
	clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id platform = NULL;
	if (0 < num_platforms) {
		cl_platform_id *platforms = new cl_platform_id[num_platforms];
		clGetPlatformIDs(num_platforms, platforms, NULL);
		platform = platforms[0];

		char platform_name[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);
		std::cout << platform_name << std::endl;

		delete[] platforms;
	}

	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM,
										   (cl_context_properties)platform, 0 };

	cl_context context =
		clCreateContextFromType((NULL == platform) ? NULL : properties,
			CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create context from type failed" << std::endl;
	}

	size_t size = 0;

	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);

	cl_device_id device = 0;
	if (size > 0) {
		cl_device_id *devices = (cl_device_id *)alloca(size);
		clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
		device = devices[0];

		char device_name[128];
		clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
	}

	cl_command_queue queue =
		clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed" << std::endl;
	}

	size_t srclen[] = { strlen(source) };

	cl_program program =
		clCreateProgramWithSource(context, 1, &source, srclen, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create program with source failed" << std::endl;
	}

	//clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	std::string buildOpts = "-DBS=" + std::to_string(16);
	error = clBuildProgram(program, 1, &device, buildOpts.c_str(), nullptr, nullptr);

	if (error != CL_SUCCESS) {
		std::cout << "Build prog failed" << std::endl;
		size_t logSize = 0;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
			nullptr, &logSize);
		char *log = new char[logSize];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize,
			log, nullptr);
		std::cout << log;
	}


	cl_kernel kernel = clCreateKernel(program, "matmul", &error);
	if (error != CL_SUCCESS) {
		std::cout << "Create kernel failed" << std::endl;
		std::cout << error << std::endl;
	}

	cl_mem aBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, SIZE*SIZE * sizeof(float), a, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Err creating buffer: " << error << std::endl;
	}

	cl_mem bBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, SIZE*SIZE * sizeof(float), b, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Err creating buffer: " << error << std::endl;
	}

	cl_mem cBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE*SIZE * sizeof(int), NULL, &error);
	if (error != CL_SUCCESS) {
		std::cout << "Err creating buffer: " << error << std::endl;
	}

	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuf);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuf);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuf);

	if (error != CL_SUCCESS) {
		std::cout << "Failed to set kernel args: " << error << std::endl;
	}

	const size_t offsets[] = { 0, 0 };
	const size_t sizes[] = { SIZE, SIZE };
	const size_t local_size[] = { 16, 16 };

	cl_event evt;
	startSys = std::chrono::steady_clock::now();
	error = clEnqueueNDRangeKernel(queue, kernel, 2, offsets, sizes, local_size, 0,
		0, &evt);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue failed: " << error << std::endl;
	}

	clWaitForEvents(1, &evt);
	endSys = std::chrono::steady_clock::now();

	cl_ulong start = 0, end = 0;
	error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
	if (error != CL_SUCCESS) {
		std::cout << "Error getting start time: " << error << std::endl;
	}
	error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
	if (error != CL_SUCCESS) {
		std::cout << "Error getting end time: " << error << std::endl;
	}

	std::cout << "Time: " << (cl_double)(end - start)*(cl_double)(1e-09) << std::endl;
	


	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	return 0;
}
