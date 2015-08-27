#include "neuron.h"

using namespace std;

//double Neuron::eta = 0.15; // overall net learning rate, [0.0..1.0]
//double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..1.0]

Neuron::Neuron(unsigned numberOutputs, unsigned id)
{
    for (unsigned c = 0; c < numberOutputs; c++)
    {
        outputWeights.push_back(Connection());
    }
    index = id;
}

Neuron::Neuron(unsigned numberOutputs, unsigned id, const string &transferfunction)
{
    for (unsigned c = 0; c < numberOutputs; c++)
    {
        outputWeights.push_back(Connection());
    }
    index =  id;
    transferFunction = transferfunction;
}

void Neuron::setOutputValue(double value)
{
    outputValue = value;
}

double Neuron::getOutputValue() const
{
    return outputValue;
}

double Neuron::transferFunctionTanH(double x)
{
    // tanh - output range [-1.0..1.0]
    return tanh(x);
}

double Neuron::transferFunctionTanHDerivative(double x)
{
    // tanh derivative aproximation
    //return 1.0 - x * x;
    return 1.0 - tanh(x) * tanh(x);
}

double Neuron::transferFunctionSig(double x)
{
    // sig - output range [-1.0..1.0]
    return 1/(1+exp(-1*x));
}

double Neuron::transferFunctionSigDerivative(double x)
{
    // sig derivative aproximation
    return exp(-1*x)/pow((1+exp(-1*x)),2);
}

double Neuron::transferFunctionStep(double x)
{
    return (x<0)?0:x;
}

double Neuron::transferFunctionStepDerivative()
{
    // step derivative aproximation
    return 1.0;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.
    for (unsigned n = 0; n < prevLayer.size(); n++)
    {
        sum += prevLayer[n].getOutputValue() * prevLayer[n].outputWeights[index].weight;
    }
    // activate function or transfer /sig /gaussian /linear/ step
    if (transferFunction == "th")
        outputValue = transferFunctionTanH(sum);
    else if (transferFunction == "sig")
        outputValue = transferFunctionSig(sum);
}

void Neuron::calculateOutputGradients(double targetValue)
{
    double delta = targetValue - outputValue;
    if (transferFunction == "th")
        gradient = delta * transferFunctionTanHDerivative(outputValue);
    else if (transferFunction == "sig")
        gradient = delta * transferFunctionSigDerivative(outputValue);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    // Sum our contributions of the errors at the nodes we feed.
    for(unsigned n = 0; n < nextLayer.size(); n++)
    {
        sum += outputWeights[n].weight * nextLayer[n].gradient;
    }
    return sum;
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    if (transferFunction == "th")
        gradient = dow * transferFunctionTanHDerivative(outputValue);
    else if (transferFunction == "sig")
        gradient = dow * transferFunctionSigDerivative(outputValue);
}

void Neuron::updateInputWeights(Layer &prevLayer, const double &eta, const double &alpha)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for(unsigned n = 0; n < prevLayer.size(); n++)
    {
        Neuron &neuron = prevLayer[n];
        Connection &conn = neuron.outputWeights[index];
        double oldDeltaWeight = conn.deltaWeight;
        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta
                * neuron.getOutputValue()
                * gradient
                // Also add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;
        conn.deltaWeight = newDeltaWeight;
        conn.weight += newDeltaWeight;
    }
}

vector<double> Neuron::getConnections() const
{
    vector<double> weights;
    for(unsigned int i = 0; i < outputWeights.size(); i++)
    {
        weights.push_back(outputWeights[i].weight);
    }
    return weights;
}

void Neuron::setWeights(vector<double> input)
{
    for(unsigned int i = 0; i < this->outputWeights.size(); i++)
    {
        outputWeights[i].weight = input[i];
    }
}
