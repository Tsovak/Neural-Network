#ifndef NN_DATA_H
#define NN_DATA_H

#include <vector>
#include <string>
#include "net.h"

std::vector<unsigned> getTopology(std::string input);
double getEta(std::string input);
double getMomentum(std::string input);
std::string getTransferFunction(std::string input);

unsigned getNextInputs(std::string input, std::vector<double> &inputVals);
unsigned getTargetOutputs(std::string input, std::vector<double> &targetOutputVals);

std::vector<double> readWeights(std::string input, const int x, const int y);

void saveNetwork(Net input, std::string filename);

void loadData(std::string input, std::vector<unsigned> &topology, std::vector<std::vector<double>> &inputVals, std::vector<std::vector<double>> &targetValues, double &eta, double &momentum, std::string &transferFunction);

Net loadnetwork(std::string input);

#endif // defined(NN_DATA_H)
