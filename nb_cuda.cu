#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include "io.hpp"
#include "evaluate.h"

#define CUDA_CHECK_RETURN(value) { \
               cudaError_t _m_cudaStat = value;\
               if (_m_cudaStat != cudaSuccess) {\
                       fprintf(stderr, "Error %s at line %d in file %s\n",\
                                       cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
                                       exit(1);\
               }}
#define SIZE_TRAIN 128

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


__global__ void trainning_kernel2(int *freqClassVector, double *matrixTermFreq,
		double* totalFreqClassVector, int *docTestIndexVector,
		int *docTestVector, double *docTestFreqVector, int *docClasse,
		int numClasses, int numTerms, int numDocsTest, double *freqTermVector,
		double totalTermFreq, int totalTerms, double lambda, double alpha,
		int numDocs) {

	int vecs, len, term, bestClass;
	double freq;
	double prob, nt, highestProb;
	extern __shared__ double temp[]; // used to hold segment of the vector (size nthreads)
	// plus 3 integers (vecs, len, partial sum) at the end
	int tid = threadIdx.x;

	if (tid == 0) {  // thread 0 calculates vecs and len
		//vecs - vector size
		temp[blockDim.x + 1] = (docTestIndexVector[blockIdx.x + 1]
				- docTestIndexVector[blockIdx.x]);
		// len - number of segments (size nthreads) of the vector
		if(temp[blockDim.x + 1] > blockDim.x)
			temp[blockDim.x + 2] = ceil(temp[blockDim.x + 1] / (float) blockDim.x);
		else
			temp[blockDim.x + 2] = 1.0;
		// partial sum initialization
		//temp[blockDim.x + 3] = 0.0;
	}
	__syncthreads();

	vecs = temp[blockDim.x + 1]; // communicate vecs and len's values to other threads
	len = (int) temp[blockDim.x + 2];

	for (int c = 0; c < numClasses; c++) {
		if (tid == 0) {
			// partial sum initialization
			temp[blockDim.x + 3] = log(
					(freqClassVector[c] + alpha)
							/ (numDocs + alpha * numClasses));
		}
		__syncthreads();
		for (int b = 0; b < len; b++) { // loop through 'len' segments
			// first, each thread loads data into shared memory
			if ((b * blockDim.x + tid) >= vecs) // check if outside 'vec' boundary
				temp[tid] = 0.0;
			else {
				term = docTestVector[docTestIndexVector[blockIdx.x]
						+ b * blockDim.x + tid];
				if(freqTermVector[term] != 0){
					freq = docTestFreqVector[docTestIndexVector[blockIdx.x]
						+ b * blockDim.x + tid];
					prob = (matrixTermFreq[c * numTerms + term] + alpha)
						/ (totalFreqClassVector[c] + alpha * totalTerms);
					nt = freqTermVector[term] / totalTermFreq;
					prob = lambda * nt + (1.0 - lambda) * prob;
					temp[tid] = freq * log(prob);
				}
				else{
					temp[tid] = 0.0;
				}
			}
			__syncthreads();

			// next, perform binary tree reduction on shared memory
			for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
				if (tid < d)
					temp[tid] += (tid + d) >= vecs ? 0.0 : temp[tid + d];
				__syncthreads();
			}

			// first thread puts partial result into shared memory
			if (tid == 0) {
				temp[blockDim.x + 3] += temp[0];
			}
			__syncthreads();
		}
		// finally, first thread puts result into global memory
		if (tid == 0) {
			if (c == 0) {
				highestProb = temp[blockDim.x + 3];
				bestClass = 0;
			} else if (temp[blockDim.x + 3] > highestProb) {
				highestProb = temp[blockDim.x + 3];
				bestClass = c;
			}
		}
		__syncthreads();
	}
	if (tid == 0) {
		docClasse[blockIdx.x] = bestClass;
	}
}

extern "C"{
void nb_gpu(const char* filenameTreino, const char* filenameTeste,
		int numDocs, int numClasses, int numTerms, int numDocsTest,
		int numTermsTest, double alpha, double lambda, int cudaDevice) {

	double begin, end;
	begin=get_wall_time();
	int block_size, n_blocks;
	int *docTestIndexVector = (int*) malloc((numDocsTest + 1) * sizeof(int));
	int *docTestVector = NULL;
	double *docTestFreqVector = NULL;

	int *freqClassVector = (int*) malloc(numClasses * sizeof(int));
	double *totalFreqClassVector = (double*) malloc(
			numClasses * sizeof(double));
	double *matrixTermFreq = (double*) malloc(
			(numTerms * numClasses) * sizeof(double));
	double *freqTermVector = (double*) malloc((numTerms) * sizeof(double));
	double totalTermFreq = 0.0;
	int totalTerms = 0;

	for (int i = 0; i < numClasses; i++) {
		totalFreqClassVector[i] = 0.0;
		freqClassVector[i] = 0;
		for (int j = 0; j < numTerms; j++) {
			matrixTermFreq[i * numTerms + j] = 0.0;
		}
	}
	for (int j = 0; j < numTerms; j++) {
		freqTermVector[j] = 0.0;
	}
	
	readTrainData(filenameTreino, freqClassVector, totalFreqClassVector, freqTermVector, &totalTermFreq, 
		numClasses, numTerms, &totalTerms, matrixTermFreq);

	double *matrixTermFreq_D;
	cudaMalloc((void **) &matrixTermFreq_D,
			sizeof(double) * (numTerms * numClasses));
	cudaMemcpy(matrixTermFreq_D, matrixTermFreq,
			sizeof(double) * (numTerms * numClasses), cudaMemcpyHostToDevice);

	int *freqClassVector_D;
	cudaMalloc((void **) &freqClassVector_D, sizeof(int) * numClasses);
	cudaMemcpy(freqClassVector_D, freqClassVector, sizeof(int) * numClasses,
			cudaMemcpyHostToDevice);

	double *totalFreqClassVector_D;
	cudaMalloc((void **) &totalFreqClassVector_D, sizeof(double) * numClasses);
	cudaMemcpy(totalFreqClassVector_D, totalFreqClassVector,
			sizeof(double) * numClasses, cudaMemcpyHostToDevice);

	double *freqTermVector_D;
	cudaMalloc((void **) &freqTermVector_D, sizeof(double) * numTerms);
	cudaMemcpy(freqTermVector_D, freqTermVector, sizeof(double) * numTerms,
			cudaMemcpyHostToDevice);

	int *realClass = (int*) malloc((numDocsTest + 1) * sizeof(int));

	
	docTestVector = readTestData(filenameTeste, docTestIndexVector, realClass,
			&docTestFreqVector);

	end=get_wall_time();
	cerr << "read test time " << end - begin << endl;
	begin=get_wall_time();

	int *docTestIndexVector_D;
	cudaMalloc((void **) &docTestIndexVector_D,
			sizeof(int) * (numDocsTest + 1));
	cudaMemcpy(docTestIndexVector_D, docTestIndexVector,
			sizeof(int) * (numDocsTest + 1), cudaMemcpyHostToDevice);

	int *docTestVector_D;
	cudaMalloc((void **) &docTestVector_D,
			sizeof(int) * docTestIndexVector[numDocsTest]);
	cudaMemcpy(docTestVector_D, docTestVector,
			sizeof(int) * docTestIndexVector[numDocsTest],
			cudaMemcpyHostToDevice);
	
	double *docTestFreqVector_D;
	cudaMalloc((void **) &docTestFreqVector_D,
			sizeof(double) * docTestIndexVector[numDocsTest]);
	cudaMemcpy(docTestFreqVector_D, docTestFreqVector,
			sizeof(double) * docTestIndexVector[numDocsTest],
			cudaMemcpyHostToDevice);

	int *docClasse = (int*) malloc((numDocsTest) * sizeof(int));	

	int *docClasse_D;
	cudaError_t status = cudaMallocHost((void **) &docClasse_D, sizeof(int) * (numDocsTest));
    if (status != cudaSuccess)
		printf("Error allocating pinned host memoryn");
    double *valor = (double*) malloc(2*sizeof(double));
   
    block_size = SIZE_TRAIN;
    n_blocks = numDocsTest;
    trainning_kernel2<<<n_blocks, block_size, (block_size + 3) * sizeof(double)>>>(
	  freqClassVector_D, matrixTermFreq_D, totalFreqClassVector_D,
	  docTestIndexVector_D, docTestVector_D, docTestFreqVector_D,
	  docClasse_D, numClasses, numTerms, numDocsTest, freqTermVector_D,
	  totalTermFreq, totalTerms, lambda, alpha, numDocs);

    cudaMemcpy(docClasse, docClasse_D, sizeof(int) * (numDocsTest),
	  cudaMemcpyDeviceToHost);

    valor[0] = evaluate(realClass, docClasse, numDocsTest, 1);
    valor[1] = evaluate(realClass, docClasse, numDocsTest, 0);

    std::cout << alpha << " " << lambda << " " << valor[0]*100 << " " << valor[1]*100 << std::endl;

	end=get_wall_time();
	cerr << "classification and evaluation times " << end - begin << endl;

	cudaFreeHost(docClasse_D);
	cudaFree(docTestIndexVector_D);
	cudaFree(docTestVector_D);
	cudaFree(docTestFreqVector_D);
	cudaFree(freqTermVector_D);
	cudaFree(matrixTermFreq_D);
	cudaFree(freqClassVector_D);
	cudaFree(totalFreqClassVector_D);
	free(totalFreqClassVector);	
	free(matrixTermFreq);
	free(freqTermVector);
	free(freqClassVector);
	free(realClass);
	free(docTestIndexVector);
	free(docTestVector);
	free(docClasse);
	free(docTestFreqVector);
	free(valor);
	return;
}
}
