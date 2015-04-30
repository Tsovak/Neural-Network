#include "../include/data.h"

using namespace std;

Data::Data(const string filename)
{
    DataFile.open(filename.c_str());
}

bool Data::isEof()
{
    return DataFile.eof();
}

void Data::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(DataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0)
    {
        abort();
    }

    while (!ss.eof())
    {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

void Data::getEta(double &eta)
{
    string line;
    string label;

    getline(DataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("eta:") != 0)
    {
        abort();
    }

    ss >> eta;

    return;
}

void Data::getTransferFunction(string &transferFunction)
{
    string line;
    string label;

    getline(DataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("transfer_function:") != 0)
    {
        abort();
    }

    ss >> transferFunction;

    return;
}

void Data::getMomentum(double &momentum)
{
    string line;
    string label;

    getline(DataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("momentum:") != 0)
    {
        abort();
    }

    ss >> momentum;

    return;
}

unsigned Data::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(DataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0)
    {
        double oneValue;
        while (ss >> oneValue)
        {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned Data::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(DataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0)
    {
        double oneValue;
        while (ss >> oneValue)
        {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}

void Data::saveNetwork(Net input, string filename, double eta, double momentum)
{
    ofstream output;
    output.open(filename + ".txt");

    //record the architecture of the network first
    output << "topology: ";
    for(int i = 0; i < input.getTotalLayers(); i++)
    {
            output << input.getLayerSize(i) - 1 << " ";
    }

    output << endl;

    //record learning parameters
    output << "eta: " << eta << endl;
    output << "momentum: " << momentum << endl;
    output << "transfer_function: " << input.getTransferFunction() << endl;

    //record weights
    //loop through each layer (minus the output layer)
    for(int x = 0; x < input.getTotalLayers() - 1; x++)
    {
        vector<double> weights;
        //loop through each node in the layer
        for(int y = 0; y < input.getLayerSize(x); y++)
        {
            weights = input.getWeights(x, y);

            output << "(" << x << "," << y << "): ";

            for(vector<double>::const_iterator k = weights.begin(); k != weights.end(); k++)
            {
                if(k != weights.end() - 1)
                {
                    output << *k << " ";
                }
                else
                {
                    output << *k;
                }
            }

            output << endl;
        }
    }
    output.close();
}

void Data::loadData(vector<unsigned> &topology, vector<vector<double>> &inputVals, vector<vector<double>> &targetValues, double &eta, double &momentum, string &transferFunction)
{
    getTopology(topology);
    getEta(eta);
    getMomentum(momentum);
    getTransferFunction(transferFunction);

    vector<double> tempInput, tempOutput;

    //Load trainning data from file
    while (!this->isEof())
    {
        // Get new input data and feed it forward:
        if (getNextInputs(tempInput) != topology[0])
            break;
        inputVals.push_back(tempInput);

        //Get the data for the correct outputs for the inputs just recorded
        getTargetOutputs(tempOutput);

        assert(tempOutput.size() == topology.back());
        targetValues.push_back(tempOutput);
    }
}

vector<Connection> Data::readWeights(const int x, const int y)
{
    string line;
    string label;
    vector<Connection> weights;

    getline(DataFile, line);
    stringstream ss(line);
    cout << line << endl;
    ss >> label;
    if (this->isEof() || label.compare("(" + to_string(x)  + ','+ to_string(y) + ')' + ':') != 0)
    {
        abort();
    }

    while (!ss.eof())
    {
        double n;
        ss >> n;
        Connection tmp(n, 0);
        weights.push_back(tmp);
    }

    return weights;
}

Net Data::loadnetwork()
{
    vector<unsigned> topology;
    double eta;
    double momentum;
    string transferFunction;

    //get
    getTopology(topology);
    getEta(eta);
    getMomentum(momentum);
    getTransferFunction(transferFunction);

    Net net(topology, transferFunction);

    //read weights, minus the output layer
    //loop through every layer, except the output layer
    for(int x = 0; x < net.getTotalLayers() - 2; x++)
    {
        //loop through every node in layer
        for(int y = 0; y < net.getLayerSize(x); y++)
        {
            net.setWeight(x, y, readWeights(x, y));
        }
    }

    return net;
}
