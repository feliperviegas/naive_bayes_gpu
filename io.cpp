#include "io.hpp"
#include <string>

#define print if(DEBUG) printf
#define DEBUG 1


struct rusage resources;
struct rusage ru;
struct timeval tim;
struct timeval tv;

/*=============================================================================================*/
void temposExecucao(double *utime, double *stime, double *total_time) {
	int rc;

	if ((rc = getrusage(RUSAGE_SELF, &resources)) != 0)
		perror("getrusage Falhou");

	*utime = (double) resources.ru_utime.tv_sec
			+ (double) resources.ru_utime.tv_usec * 1.e-6;
	*stime = (double) resources.ru_stime.tv_sec
			+ (double) resources.ru_stime.tv_usec * 1.e-6;
	*total_time = *utime + *stime;

}

double tempoAtual() {

	gettimeofday(&tv, 0);

	return tv.tv_sec + tv.tv_usec / 1.e6;
}

void stringTokenize(const std::string& str, std::vector<std::string>& tokens,
		const std::string& delimiters) {

	std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	std::string::size_type pos = str.find_first_of(delimiters, lastPos);
	while (std::string::npos != pos || std::string::npos != lastPos) {
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		lastPos = str.find_first_not_of(delimiters, pos);
		pos = str.find_first_of(delimiters, lastPos);
	}

}

void readTrainData(const char* filename, int* freqClassVector, double *totalFreqClassVector,
		double *freqTermVector, double *totalTermFreq, int numClasses, int numTerms, int *totalT, 
		double* matrixTermFreq) {
	std::ifstream file(filename);
	std::string line;
	set<int> vocabulary;
	int docId = 0;
	*totalTermFreq = 0;
	if (file) {
		while (file >> line) {
			vector < std::string > tokens;
			stringTokenize(line, tokens, ";");
			// cerr << tokens[0] << " " << tokens[2] << " ";
			int docClass = atoi(tokens[2].replace(0, 6, "").c_str());            
			int totalTerms = (int) ceil((tokens.size() - 3) / 2.0);

			freqClassVector[docClass] += 1;
			for (int i = 3; i < (int) tokens.size(); i = i + 2) {
				int term = (atoi(tokens[i].c_str())) - 1;
				double freq = atof(tokens[i + 1].c_str());
				vocabulary.insert(term);
				(*totalTermFreq) += freq;
				freqTermVector[term] += freq;
				totalFreqClassVector[docClass] += freq;
				matrixTermFreq[docClass * numTerms + term] += freq;
			}
			docId++;
		}
		(*totalT) = vocabulary.size();
		file.close();
		
		return;
	} else {
		std::cout << "Error while opening vertex fadile." << std::endl;
		exit(1);
	}
	return;
}

int* readTestData(const char* filename, int *docTestIndexVector, int *realClass,
		double *(*docTestFreqVector)) {

	std::ifstream file(filename);
	std::string line;

	int *docTestVector = NULL;
	(*docTestFreqVector) = NULL;

	int tamDocVector = 0;
	int termPosition = 0;
	int docId = 0;
	if (file) {
		while (file >> line) {

			vector < std::string > tokens;
			stringTokenize(line, tokens, ";");

			int docClass = atoi(tokens[2].replace(0, 6, "").c_str());
			int totalTerms = (int) ceil((tokens.size() - 3) / 2.0);
			realClass[docId] = docClass;
			docTestVector = (int*) realloc(docTestVector,
					(tamDocVector + totalTerms) * sizeof(int));
			(*docTestFreqVector) = (double*) realloc((*docTestFreqVector),
					(tamDocVector + totalTerms) * sizeof(double));
			docTestIndexVector[docId] = tamDocVector;
			tamDocVector += totalTerms;
			for (int i = 3; i < (int) tokens.size(); i = i + 2) {
				int term = (atoi(tokens[i].c_str())) - 1;
				double freq = atof(tokens[i + 1].c_str());
				docTestVector[docTestIndexVector[docId] + termPosition] = term;
				(*docTestFreqVector)[docTestIndexVector[docId] + termPosition] =
						freq;
				termPosition += 1;
			}
			termPosition = 0;
			docId++;
		}
		docTestIndexVector[docId] = tamDocVector;
		file.close();
		return docTestVector;
	} else {
		std::cout << "Error while opening vertex fadile." << std::endl;
		exit(1);
	}

	return NULL;
}
