#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include "io.hpp"
#include "nb_cuda.h"

int main(int argc, char *argv[]) {

	int cudaDevice;
	// double inicio, final;
	int numDocs, numClasses, numTerms;
	int numDocsTest, numTermsTest;
	double alpha = 1.0, lambda = 0.3;
	/***************************************/

	if (argc != 19) {
		printf(
				"\n\n./nb -nd [NumDocs] -nc [numClasses] -nt [numTerms] -fl [fileTrainning] -ndT [NumDocsTest] -ntT [numTermsTest] -ft [fileTest] -a [alpha] -l [lambda]\n\n");
		exit(0);
	}

	//Parametros
	numDocs = atoi(argv[2]);
	numClasses = atoi(argv[4]);
	numTerms = atoi(argv[6]);

	numDocsTest = atoi(argv[10]);
	numTermsTest = atoi(argv[12]);

	alpha = atof(argv[16]);
	lambda = atof(argv[18]);

	cout << numDocs << " " << numClasses << " " << numTerms << " " << numTermsTest << " " << alpha << " " << lambda << endl;

	nb_gpu(argv[8], argv[14], numDocs, numClasses, numTerms,
			numDocsTest, numTermsTest, alpha, lambda);

	return 0;
}
