#!/bin/bash
treino=$1;
teste=$2;
alpha=$3;
lambda=$4;

echo "Parsing folds..."
awk -f libsvm2tsalles.awk ${treino} > "treino.txt";
awk -f libsvm2tsalles.awk ${teste} > "teste.txt";

echo "Counting number of attributes and classes..."
numAttributes=`awk -f attributes.awk "treino.txt" "teste.txt"`;
numClasses=`awk -F";" '{print $3;}' "treino.txt" "teste.txt" | sort | uniq -c | wc -l`;
echo "# Attributes: "${numAttributes};
echo "# Classes: "${numClasses};

numDocTreino=`awk -F";" 'END{print NR;}' "treino.txt"`;
numDocTeste=`awk -F";" 'END{print NR;}' "teste.txt"`;
echo "# Training Docs: ${numDocTreino}"; 
echo "# Test Docs: ${numDocTeste}";

echo "Naive Bayes..."
./nb -nd ${numDocTreino} -nc ${numClasses} -nt ${numAttributes} -fl "treino.txt" -ndT ${numDocTeste} -ntT ${numAttributes} -ft "teste.txt" -a ${alpha} -l ${lambda};
#done
