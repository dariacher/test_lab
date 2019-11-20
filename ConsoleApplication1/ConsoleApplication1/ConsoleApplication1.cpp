#include "pch.h"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <iostream>
#include <random>
#include <chrono>

using namespace std;

inline int idx(size_t i, size_t j, size_t size) {
	return i * size + j;
}

void matrix_print(double *&M, const size_t size) {
	for (size_t i = 0; i < 10; i++) {
		for (size_t j = 0; j < 10; j++)
			cout << M[idx(i, j, size)] << " ";
		cout << endl;
	}
	cout << endl;
}

void matrix_mult(double *&A, double *&B, double *&C, size_t size) {
	for (size_t i = 0; i < size; i++)
		for (size_t j = 0; j < size; j++) {
			C[idx(i, j, size)] = 0;
			for (size_t k = 0; k < size; k++)
				C[idx(i, j, size)] += A[idx(i, k, size)] * B[idx(k, j, size)];
		}
}

bool is_equal(double x, double y) {
	return std::fabs(x - y) < std::numeric_limits<double>::epsilon();
}

void matrix_generate(double *&M, const size_t size) {
	std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<double> distribution(-10, 10);
	for (size_t i = 0; i < size; i++)
		for (size_t j = 0; j < size; j++)
			M[idx(i, j, size)] = distribution(generator);
}

void matrix_mult_openmp(double *&A, double *&B, double *&C, size_t n) {
	{
		int i, j, k;
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				double sum = 0;
#pragma omp parallel for
				for (k = 0; k < n; k++) {
					sum += A[i*n + k] * B[k*n + j];
				}
				C[i*n + j] = sum;
			}
		}

	}
}



void matmultOMP(double *a, double *b, double *c, int SIZE) {

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


int main() {
	int err;
	size_t matrix_size;

	cout << "Enter the matrix size: ";
	cin >> matrix_size;
	cout << endl;

/*	const char* matrix_mult =
		"__kernel void matrix_mult(__global double* A, __global double* B, __global double* C, int n) {\n" \
		"	double sum = 0;\n" \
		"	int row = get_global_id(0);\n" \
		"	int col = get_global_id(1);\n" \
		"	for (int k = 0; k < n; k++) {\n" \
		"		sum += A[row * n + k] * B[k * n + col];\n" \
		"	}\n" \
		"	C[row * n + col] = sum;\n" \
		"}";*/

	const char* matrix_mult =
		"__kernel void matrix_mult(__global double* a, __global double* b, __global double* c, int n) {\n" \
		"  int BS = 16;\n" \
		"  int row = get_local_id(0);\n" \
		"  int col = get_local_id(1);\n" \
		"  const int globalRow = BS*get_group_id(0) + row;\n" \
		"  const int globalCol = BS*get_group_id(1) + col;\n" \
		"  local float Asub[16][16];\n" \
		"  local float Bsub[16][16];\n"\
		""\
		"  float acc = 0.0f;\n"\
		"  const int numTiles = n/16;\n"\
		"  for (int t = 0; t < numTiles; t++) {\n"\
		"    const int tiledRow = 16*t+row;\n"\
		"    const int tiledCol = 16*t+col;\n"\
		"    Asub[col][row] = a[tiledCol*n + globalRow];\n"\
		"    Bsub[col][row] = b[globalCol*n + tiledRow];\n"\
		"    barrier(CLK_LOCAL_MEM_FENCE);\n"\
		"    for (int k=0; k < 16; k++) {\n"\
		"      acc = mad(Asub[k][row], Bsub[col][k], acc);\n"\
		"     }\n"\
		"    barrier(CLK_LOCAL_MEM_FENCE);\n"\
		"  }\n"\
		"  c[globalCol*n+globalRow] = acc;\n"\
		"}";

	
	cl_uint num_platforms = 0;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id platform = NULL;
	if (num_platforms > 0) {
		cl_platform_id* platforms = new cl_platform_id[num_platforms];
		clGetPlatformIDs(num_platforms, platforms, NULL);
		platform = platforms[0];
		delete[] platforms;
	}

	struct
	{
		cl_device_type type;
		const char* name;
		cl_uint count;
		cl_device_id device_id;
		double time;
		double* C;
	}
	devices[] =
	{
		{ CL_DEVICE_TYPE_CPU, "GPU", 0, NULL, 0, NULL },
		{ CL_DEVICE_TYPE_GPU, "CPU", 0, NULL, 0, NULL },
	};

	const int NUM_OF_DEVICE_TYPES = sizeof(devices) / sizeof(devices[0]);

	for (int i = 0; i < NUM_OF_DEVICE_TYPES; ++i)
	{
		devices[i].C = (double*)malloc(matrix_size * matrix_size * sizeof(double));

		for (int k = 0; k < matrix_size * matrix_size; k++) {
			devices[i].C[k] = 0;
		}

		err = clGetDeviceIDs(
			platform,
			devices[i].type,
			1,
			&devices[i].device_id,
			&devices[i].count
		);

		if (CL_DEVICE_NOT_FOUND == err) {
			devices[i].count = 0;
			err = CL_SUCCESS;
		}
	}

	double* data_A = (double*)malloc(matrix_size * matrix_size * sizeof(double));
	double* data_B = (double*)malloc(matrix_size * matrix_size * sizeof(double));

	matrix_generate(data_A, matrix_size);
	matrix_generate(data_B, matrix_size);
	/*
	cout << "matrix A:" << endl;
	matrix_print(data_A, matrix_size);
	cout << "matrix B:" << endl;
	matrix_print(data_B, matrix_size);*/
	double* omp_C = (double*)malloc(matrix_size * matrix_size * sizeof(double));

	clock_t start = clock();

	

	matmultOMP( data_A, data_B, omp_C, 1024);

	clock_t finish = clock();

	double openmp_time = (double)(finish - start) / CLOCKS_PER_SEC;

	for (int i = 0; i < NUM_OF_DEVICE_TYPES; ++i) {

		cl_device_id device_id = devices[i].device_id;

		cl_context context = clCreateContext(0, devices[i].count, &device_id, NULL, NULL, &err);

		cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);



		cl_program program = clCreateProgramWithSource(context, 1, &matrix_mult, NULL, &err);

		err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
		if (err != CL_SUCCESS) {
			size_t len;
			char buffer[2048];
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
			printf("%s\n", buffer);
			system("pause");
			//exit(1);
		}

		cl_kernel kernel = clCreateKernel(program, "matrix_mult", &err);
		if (err != CL_SUCCESS) {
			printf("Error create kernel\n");
			system("pause");
			//exit(1);
		}
		cl_mem A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * matrix_size * matrix_size, NULL, &err);
		if (err != CL_SUCCESS) {
			printf("Error create buffer\n");
			system("pause");
		}
		cl_mem B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * matrix_size * matrix_size, NULL, &err);
		if (err != CL_SUCCESS) {
			printf("Error create buffer\n");
			system("pause");
		}
		cl_mem C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * matrix_size * matrix_size, NULL, &err);
		if (err != CL_SUCCESS) {
			printf("Error create buffer\n");
			system("pause");
		}
		

		clEnqueueWriteBuffer(queue, A, CL_TRUE, 0, sizeof(double) * matrix_size * matrix_size, data_A, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, B, CL_TRUE, 0, sizeof(double) * matrix_size * matrix_size, data_B, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, C, CL_TRUE, 0, sizeof(double) * matrix_size * matrix_size, devices[i].C, 0, NULL, NULL);

		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
		if (err != CL_SUCCESS) {
			printf("Error set kernel arg 1\n");
			system("pause");
		}
		err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
		if (err != CL_SUCCESS) {
			printf("Error set kernel arg 2\n");
			system("pause");
		}
		err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
		if (err != CL_SUCCESS) {
			printf("Error set kernel arg 3\n");
			system("pause");
		}
		err = clSetKernelArg(kernel, 3, sizeof(int), &matrix_size);
		if (err != CL_SUCCESS) {
			printf("Error set kernel arg 4\n");
			system("pause");
		}

		size_t group_size[2] = { 16, 16 };
		size_t global_work_size[2] = { matrix_size, matrix_size };

		//clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group_size, NULL);


		clock_t start = clock();

		clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, group_size, 0, NULL, NULL);

		clFinish(queue);

		clock_t finish = clock();

		devices[i].time = (double)(finish - start) / CLOCKS_PER_SEC;

		clEnqueueReadBuffer(queue, C, CL_TRUE, 0, sizeof(double) * matrix_size * matrix_size, devices[i].C, 0, NULL, NULL);

	//	std::cout << devices[i].name << ":" << std::endl;

	//	matrix_print(devices[i].C, matrix_size);
	//	std::cout << std::endl;

		clReleaseMemObject(A);
		clReleaseMemObject(B);
		clReleaseMemObject(C);
		clReleaseProgram(program);
		clReleaseKernel(kernel);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
	}



	int error = 0;
	for (int i = 0; i < matrix_size * matrix_size; i++) {
		if (devices[0].C[i] != omp_C[i] || devices[1].C[i] != omp_C[i]) error = 1;

		if (is_equal(devices[0].C[i],omp_C[i]) || is_equal(devices[1].C[i], omp_C[i]) ) error = 1;
	}

	if (error) {
		std::cout << "error occurred! check mistakes." << std::endl;
	}
	else {
		std::cout << "cpu/gpu answer is correct." << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Time on CPU_OpenCL: " << devices[0].time << std::endl;
	std::cout << "Time on GPU_OpenCl: " << devices[1].time << std::endl;
	std::cout << "Time on OpenMP: " << openmp_time << std::endl;

	system("pause");
	return 0;
}
