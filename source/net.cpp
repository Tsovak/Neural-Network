#include <cassert>
#include "net.h"
#include "data.h"
#include <iostream>
using namespace std;

Net::Net(const vector<unsigned> &topology,const string &transferFunction, double eta, double momentum)
{
    // size
    unsigned numberOfLayers = topology.size();
    //create a new Layer on each interation
    for (unsigned layerNumber = 0; layerNumber < numberOfLayers; layerNumber++)
    {
        layers.push_back(Layer());
        //number of outputs to a neuron
        unsigned numberOutputs = (layerNumber == topology.size()-1) ? 0: topology[layerNumber+1];
        // fill layer with neurons and add bias neuron to the layer;
        if(layerNumber != numberOfLayers  - 1)
        {
            for (unsigned neuronNumber = 0; neuronNumber <= topology[layerNumber] ; neuronNumber++)
            {
                layers.back().push_back(Neuron(numberOutputs,neuronNumber,transferFunction));
            }
            // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
            layers.back().back().setOutputValue(1.0);
        }
        else
        {
            for (unsigned neuronNumber = 0; neuronNumber < topology[layerNumber] ; neuronNumber++)
            {
                layers.back().push_back(Neuron(numberOutputs,neuronNumber,transferFunction));
            }
        }
    }
    this->eta = eta;
    this->momentum = momentum;
    this->transferfunction = transferFunction;
}

Net::Net(string input)
{
    *this = loadnetwork(input);
}

//feedForward - operation to train the network
void Net::feedForward(const vector<double> &inputValues)
{
    assert(inputValues.size() == layers[0].size() - 1 && " the inputValues size needs to be the same of the first layer - bias");
    for (unsigned i = 0; i < inputValues.size(); i++)
    {
        layers[0][i].setOutputValue(inputValues[i]);
    }
    //forward propagation
    //loop each layer and each neuron inside the layer
    for (unsigned layerNumber = 1; layerNumber < layers.size(); layerNumber++)
    {
        Layer &prevLayer = layers[layerNumber-1];
        for (unsigned n = 0; n < layers[layerNumber].size(); n++)
        {
            layers[layerNumber][n].feedForward(prevLayer);
        }
    }
}
// backPropagation learning
void Net::backPropagation(const vector<double> &targetValues)
{
    //Calculate overall net error (RMS-root mean square error - of output neuron errors)
    Layer &outputLayer = layers.back();
    //overwall net error
    error=0.0;
    for(unsigned n = 0; n < outputLayer.size(); n++)
    {
        double delta = targetValues[n] - outputLayer[n].getOutputValue();
        error += delta*delta;
    }
    error *= 0.5;

    recentAverageError = error;
    //Calculate output layer gradients
    for(unsigned n = 0; n < outputLayer.size(); n++)
    {
        outputLayer[n].calculateOutputGradients(targetValues[n]);
    }
    //Calculate hidden layer gradients
    for(unsigned layerNumber = layers.size() -2; layerNumber > 0; layerNumber--)
    {
        Layer &hiddenLayer = layers[layerNumber];
        Layer &nextLayer = layers[layerNumber+1];
        for(unsigned n = 0; n < hiddenLayer.size() - 1; n++)
        {
            hiddenLayer[n].calculateHiddenGradients(nextLayer);
        }
    }
    //For all layers from outputs to first hidden layer,
    //update connection weights
    for(unsigned layerNumber = layers.size() - 1; layerNumber > 0; layerNumber--)
    {
        Layer &layer = layers[layerNumber];
        Layer &prevLayer = layers[layerNumber-1];
        //cout << layer.size() << endl;
        for(unsigned n = 0; n < layer.size(); n++)
        {
            layer[n].updateInputWeights(prevLayer,this->eta,this->momentum);
        }
    }
}
void Net::getResults(vector<double> &resultValues) const
{
    resultValues.clear();
    for (unsigned n = 0; n < layers.back().size(); n++)
    {
        resultValues.push_back(layers.back()[n].getOutputValue());
    }
}

double Net::getRecentAverageError(void) const
{
    return recentAverageError;
}

vector<double> Net::getLayerValues(int row) const
{
    vector<double> values;
    for(unsigned int i = 0; i < layers[row].size() ;i++)
    {
        //hurray for object oreinted programming
        values.push_back(layers[row][i].getOutputValue());
    }
    return values;
}

void Net::setLayer(vector<double> values, int row)
{
    for(unsigned int i = 0; i < layers[row].size(); i++)
    {
        layers[row][i].setOutputValue(values[i]);
    }
}

int Net::getTotalLayers() const
{
    return layers.size();
}

int Net::getLayerSize(int x) const
{
    return layers[x].size();
}

vector<double> Net::getWeights(int x, int y) const
{
    return layers[x][y].getConnections();
}

void Net::setWeight(int x, int y, vector<double> input)
{
    layers[x][y].setWeights(input);
}

string Net::getTransferFunction() const
{
    return transferfunction;
}

double Net::getEta() const
{
    return eta;
}

double Net::getMomentum() const
{
    return momentum;
}

Net& Net::operator = (const Net &rhs)
{
    this->layers = rhs.layers;
    this->transferfunction = rhs.transferfunction;
    this->eta = rhs.eta;
    this->momentum = rhs.momentum;
    for(int x = 0; x < this->getTotalLayers(); x++)
    {
        for(int y = 0; y < this->getLayerSize(x); y++)
        {
            this->setWeight(x, y, rhs.getWeights(x,y));
        }
    }
    return *this;
}
