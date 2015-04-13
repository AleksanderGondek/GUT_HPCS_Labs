/*
* Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
* This sample implements a conjugate graident solver on GPU
* using CUBLAS and CUSPARSE
*
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

// Utilities and system includes
#include "device_launch_parameters.h" // helper for shared functions common to CUDA SDK samples
//#include <helper_cuda.h>       // helper function CUDA error checking and intialization



//zad 1
__global__ void copyKernel(int n, const float* d_source, float* d_destination) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid < n) {
		d_destination[gid] = d_source[gid];
	}
}

__global__ void scaleKernel(int n, float* d_source, float alpha) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid < n) {
		d_source[gid] *= alpha;
	}
}

__global__ void axpyKernel(int n, const float* d_source, float alpha, float* d_destination) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid < n) {
		d_destination[gid] += alpha * d_source[gid];
	}
}

cublasStatus_t my_cublasScopy(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy) {
	//	return cublasScopy( handle, n, x, incx, y, incy );

	const int threads = 1024;
	const int blocks = (n + threads - 1) / threads;
	copyKernel << <blocks, threads >> >(n, x, y);
	return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t my_cublasSscal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
	//	return cublasSscal( handle, n, alpha, x, incx );

	const int threads = 1024;
	const int blocks = (n + threads - 1) / threads;
	scaleKernel << <blocks, threads >> >(n, x, *alpha);
	return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t my_cublasSaxpy(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy) {
	//	return cublasSaxpy( handle, n, alpha, x, incx, y, incy );

	const int threads = 1024;
	const int blocks = (n + threads - 1) / threads;
	axpyKernel << <blocks, threads >> >(n, x, *alpha, y);
	return CUBLAS_STATUS_SUCCESS;
}

//zad 2
// WPROWADZA MINIMALNA ROZBIEZNOSC, KTORA WYGLADA NA WYNIKAJACA Z BLEDU NUMERYCZNEGO
__global__ void MatVecKernel(int n, const float* d_val, const int* d_row, const int* d_col, const float* d_source_vec, float* d_destination) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;

	if (gid < n) {
		float sub = 0.f;
		int j;

		for (j = d_row[gid]; j < d_row[gid + 1]; ++j) {
			sub += d_val[j] * d_source_vec[d_col[j]];
		}

		d_destination[gid] = sub;
	}
}

cusparseStatus_t my_cusparseScsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const float *alpha, const cusparseMatDescr_t descrA,
	const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *x, const float *beta, float *y) {
	//	return cusparseScsrmv( handle, transA, m, n, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, x, beta, y );
	const int threads = 1024;
	const int blocks = (n + threads - 1) / threads;
	MatVecKernel << <blocks, threads >> >(n, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, x, y);
	return CUSPARSE_STATUS_SUCCESS;

}

//zad 3
__global__ void elementwiseMultiplicationKernel(int n, const float* d_x, const float* d_y, float* d_sub_sums) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	int i;
	extern __shared__ float intermediate[];

	if (gid < n) {
		intermediate[tid] = d_x[gid] * d_y[gid];
	}
	else {
		intermediate[tid] = 0.f;
	}
	__syncthreads();

	for (i = blockDim.x / 2; i > 0; i >>= 1) {
		if (tid < i) {
			intermediate[tid] += intermediate[tid + i];
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_sub_sums[blockIdx.x] = intermediate[0];
	}
}

// UWAGA, source musi miescic sie w obrebie jednego bloku !!!
__global__ void reduceKernel(float* d_source, float* d_result) {
	int tid = threadIdx.x;
	int i;

	for (i = blockDim.x / 2; i > 0; i >>= 1) {
		if (tid < i) {
			d_source[tid] += d_source[tid + i];
		}
		__syncthreads();
	}

	if (tid == 0) {
		*d_result = d_source[0];
	}
}

cublasStatus_t my_cublasSdot(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result) {
	//	return cublasSdot( handle, n, x, incx, y, incy, result );
	const int max_blocks = 1024;
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	float* d_block_sub_sums, *d_result;

	if (blocks > max_blocks) {
		return CUBLAS_STATUS_EXECUTION_FAILED; // hehe :D
	}

	checkCudaErrors(cudaMalloc(&d_block_sub_sums, sizeof(float) * max_blocks));
	checkCudaErrors(cudaMemset(d_block_sub_sums, 0, sizeof(float) * max_blocks));
	checkCudaErrors(cudaMalloc(&d_result, sizeof(float)));

	elementwiseMultiplicationKernel << <blocks, threads, threads * sizeof(float) >> >(n, x, y, d_block_sub_sums);
	reduceKernel << < 1, max_blocks >> >(d_block_sub_sums, d_result);
	checkCudaErrors(cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

	return CUBLAS_STATUS_SUCCESS;
}





const char *sSDKname = "conjugateGradient";

double mclock() {
	struct timeval tp;

	double sec, usec;
	gettimeofday(&tp, NULL);
	sec = double(tp.tv_sec);
	usec = double(tp.tv_usec) / 1E6;
	return sec + usec;
	return 0.l;
}


#define dot_BS     32
#define kernel_BS  32

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz) {
	double RAND_MAXi = 1e6;
	double val_r = 12.345 * 1e5;

	I[0] = 0, J[0] = 0, J[1] = 1;
	val[0] = (float)val_r / RAND_MAXi + 10.0f;
	val[1] = (float)val_r / RAND_MAXi;
	int start;

	for (int i = 1; i < N; i++) {
		if (i > 1) {
			I[i] = I[i - 1] + 3;
		}
		else {
			I[1] = 2;
		}

		start = (i - 1) * 3 + 2;
		J[start] = i - 1;
		J[start + 1] = i;

		if (i < N - 1) {
			J[start + 2] = i + 1;
		}

		val[start] = val[start - 1];
		val[start + 1] = (float)val_r / RAND_MAXi + 10.0f;

		if (i < N - 1) {
			val[start + 2] = (float)val_r / RAND_MAXi;
		}
	}

	I[N] = nz;
}


void cgs_basic(int argc, char **argv, int N, int M) {

	//int M = 0, N = 0, 
	int nz = 0, *I = NULL, *J = NULL;
	float *val = NULL;
	const float tol = 1e-10f;
	const int max_iter = 1000;
	float *x;
	float *rhs;
	float a, b, na, r0, r1;
	int *d_col, *d_row;
	float *d_val, *d_x, dot;
	float *d_r, *d_p, *d_Ax;
	int k;
	float alpha, beta, alpham1;

	// This will pick the best possible CUDA capable device
	cudaDeviceProp deviceProp;
	int devID = findCudaDevice(argc, (const char **)argv);

	if (devID < 0) {
		printf("exiting...\n");
		exit(EXIT_SUCCESS);
	}

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	// Statistics about the GPU device
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
		deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	int version = (deviceProp.major * 0x10 + deviceProp.minor);

	if (version < 0x11) {
		printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}

	/* Generate a random tridiagonal symmetric matrix in CSR format */
	//M = N = 32*64;//10; //1048576;
	printf("M = %d, N = %d\n", M, N);
	nz = (N - 2) * 3 + 4;
	I = (int *)malloc(sizeof(int)*(N + 1));
	J = (int *)malloc(sizeof(int)*nz);
	val = (float *)malloc(sizeof(float)*nz);
	genTridiag(I, J, val, N, nz);

	/*
	for (int i = 0; i < nz; i++){
	printf("%d\t", J[i]);
	}
	printf("\n");
	for (int i = 0; i < nz; i++){
	printf("%2f\t", val[i]);
	}
	*/

	x = (float *)malloc(sizeof(float)*N);
	rhs = (float *)malloc(sizeof(float)*N);

	for (int i = 0; i < N; i++) {
		rhs[i] = 1.0;
		x[i] = 0.0;
	}

	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

	checkCudaErrors(cublasStatus);

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	checkCudaErrors(cusparseStatus);

	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);

	checkCudaErrors(cusparseStatus);

	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (N + 1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));

	cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);

	alpha = 1.0;
	alpham1 = -1.0;
	beta = 0.0;
	r0 = 0.;


	double t_start = mclock();
	cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

	cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);                                // PODMIEN FUNCKJE (I)
	cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                        // PODMIEN FUNCKJE (II)

	k = 1;

	while (r1 > tol*tol && k <= max_iter) {
		if (k > 1) {
			b = r1 / r0;
			cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);                        // PODMIEN FUNCKJE (I)
			cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);            // PODMIEN FUNCKJE (I)
		}
		else {
			cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);                    // PODMIEN FUNCKJE (I)
		}

		cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax); // PODMIEN FUNCKJE (III)
		cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);                  // PODMIEN FUNCKJE (II)
		a = r1 / dot;

		cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);                    // PODMIEN FUNCKJE (I)
		na = -a;
		cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);                  // PODMIEN FUNCKJE (I)

		r0 = r1;
		cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                    // PODMIEN FUNCKJE (II)
		cudaThreadSynchronize();
		printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
		k++;
	}
	printf("TIME OF CGS_BASIC = %f\n", mclock() - t_start);

	cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

	float rsum, diff, err = 0.0;

	for (int i = 0; i < N; i++) {
		rsum = 0.0;

		for (int j = I[i]; j < I[i + 1]; j++) {
			rsum += val[j] * x[J[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err) {
			err = diff;
		}
	}

	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	free(I);
	free(J);
	free(val);
	free(x);
	free(rhs);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);

	cudaDeviceReset();

	printf("Test Summary:  Error amount = %e\n", err);
	//exit((k <= max_iter) ? 0 : 1);


}




void cgs_TODO(int argc, char **argv, int N, int M) {

	//int M = 0, N = 0, 
	int nz = 0, *I = NULL, *J = NULL;
	float *val = NULL;
	const float tol = 1e-10f;
	const int max_iter = 1000;
	float *x;
	float *rhs;
	float a, b, na, r0, r1;
	int *d_col, *d_row;
	float *d_val, *d_x, dot;
	float *d_r, *d_p, *d_Ax;
	int k;
	float alpha, beta, alpham1;

	// This will pick the best possible CUDA capable device
	cudaDeviceProp deviceProp;
	int devID = findCudaDevice(argc, (const char **)argv);

	if (devID < 0) {
		printf("exiting...\n");
		exit(EXIT_SUCCESS);
	}

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	// Statistics about the GPU device
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
		deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	int version = (deviceProp.major * 0x10 + deviceProp.minor);

	if (version < 0x11) {
		printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}

	/* Generate a random tridiagonal symmetric matrix in CSR format */
	//M = N = 32*64;//10; //1048576;
	printf("M = %d, N = %d\n", M, N);
	nz = (N - 2) * 3 + 4;
	I = (int *)malloc(sizeof(int)*(N + 1));
	J = (int *)malloc(sizeof(int)*nz);
	val = (float *)malloc(sizeof(float)*nz);
	genTridiag(I, J, val, N, nz);

	/*
	for (int i = 0; i < nz; i++){
	printf("%d\t", J[i]);
	}
	printf("\n");
	for (int i = 0; i < nz; i++){
	printf("%2f\t", val[i]);
	}
	*/

	x = (float *)malloc(sizeof(float)*N);
	rhs = (float *)malloc(sizeof(float)*N);

	for (int i = 0; i < N; i++) {
		rhs[i] = 1.0;
		x[i] = 0.0;
	}

	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

	checkCudaErrors(cublasStatus);

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	checkCudaErrors(cusparseStatus);

	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);

	checkCudaErrors(cusparseStatus);

	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (N + 1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));

	cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);

	alpha = 1.0;
	alpham1 = -1.0;
	beta = 0.0;
	r0 = 0.;


	// sparse matrix vector product: d_Ax = A * d_x
	my_cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);  // PODMIEN FUNCKJE (ZADANIE-II)


	//azpy: d_r = d_r + alpham1 * d_Ax
	my_cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);        			    // PODMIEN FUNCKJE (ZADANIE-I)
	//dot:  r1 = d_r * d_r
	cublasStatus = my_cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                        // PODMIEN FUNCKJE (ZADANIE-III)

	k = 1;

	while (r1 > tol*tol && k <= max_iter) {
		if (k > 1) {
			b = r1 / r0;
			//scal: d_p = b * d_p
			cublasStatus = my_cublasSscal(cublasHandle, N, &b, d_p, 1);                        // PODMIEN FUNCKJE (ZADANIE-I)
			//axpy:  d_p = d_p + alpha * d_r
			cublasStatus = my_cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);            // PODMIEN FUNCKJE (ZADANIE-I)
		}
		else {
			//cpy: d_p = d_r
			cublasStatus = my_cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);                    // PODMIEN FUNCKJE (ZADANIE-I)
		}

		//sparse matrix-vector product: d_Ax = A * d_p
		my_cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax); // PODMIEN FUNCKJE (ZADANIE-II)
		cublasStatus = my_cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);                  // PODMIEN FUNCKJE (ZADANIE-III)
		a = r1 / dot;

		//axpy: d_x = d_x + a*d_p
		cublasStatus = my_cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);                    // PODMIEN FUNCKJE (ZADANIE-I)
		na = -a;

		//axpy:  d_r = d_r + na * d_Ax
		cublasStatus = my_cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);                  // PODMIEN FUNCKJE (ZADANIE-I)

		r0 = r1;

		//dot: r1 = d_r * d_r
		cublasStatus = my_cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                    // PODMIEN FUNCKJE (ZADANIE-III)
		cudaThreadSynchronize();
		printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
		k++;
	}

	cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

	float rsum, diff, err = 0.0;

	for (int i = 0; i < N; i++) {
		rsum = 0.0;

		for (int j = I[i]; j < I[i + 1]; j++) {
			rsum += val[j] * x[J[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err) {
			err = diff;
		}
	}

	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	free(I);
	free(J);
	free(val);
	free(x);
	free(rhs);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);

	cudaDeviceReset();

	printf("Test Summary:  Error amount = %e\n", err);
	//exit((k <= max_iter) ? 0 : 1);


}







int main(int argc, char **argv) {
	//int N = 1e6;//1 << 20;
	//int N = 256 * (1<<10)  -10 ; //1e6;//1 << 20;
	int N = 1e5;
	int M = N;

	cgs_basic(argc, argv, N, M);

	cgs_TODO(argc, argv, N, M);
}