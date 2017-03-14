#ifndef IO_H__
#define IO_H__

#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <utility>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <sys/time.h>
#include <sys/resource.h>
#include <queue>


using namespace std;


void stringTokenize(const std::string& str, std::vector<std::string>& tokens,
		const std::string& delimiters);
void temposExecucao(double *utime, double *stime, double *total_time);
double tempoAtual();
void readTrainData(const char* filename, int* freqClassVector, double *totalFreqClassVector,
		double *freqTermVector, double *totalTermFreq, int numClasses, int numTerms, int *totalT, 
		double* matrixTermFreq);
int* readTestData(const char* filename, int *docTestIndexVector, int *realClass,
		double *(*docTestFreqVector));

#endif
