#ifndef NET_H
#define NET_H

#include <vector>
#include <string>
#include "layer.h"

class Net{

    double error;
    double recentAverageError;
    double eta;
    double momentum;
    std::string transferfunction;
    std::vector<Layer> layers ;// layers[layerNumber][neuronNumber]

public:
    Net(std::string input);
    Net(const std::vector<unsigned> &topology, const std::string &transferFunction, double eta = 0.1, double momentum = 0.5);

    //feedForward - operation to train the network
    void feedForward(const std::vector<double> &inputValues);

    //backPropagation learning
    void backPropagation(const std::vector<double> &targetValues);

    void getResults(std::vector<double> &resultValues) const;

    double getRecentAverageError() const;

    void setLayer(std::vector<double> values, int row);

    std::vector<double> getLayerValues(int row) const;

    std::vector<double> getWeights(int x, int y) const;

    int getTotalLayers() const;

    int getLayerSize(int x) const;

    std::string getTransferFunction() const;

    double getEta() const;

    double getMomentum() const;

    //Set the weight of neuron x,y to input...
    void setWeight(int x, int y, std::vector<double> input);

    Net& operator=(const Net &rhs);

};

#endif
