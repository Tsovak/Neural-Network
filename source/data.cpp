#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include "../include/data.h"

using namespace std;

vector<unsigned> getTopology(string input)
{
    ifstream file;
    file.open(input);
    string line;
    string label;
    vector<unsigned> topology;

    getline(file, line);
    stringstream ss(line);
    ss >> label;
    while (!file.eof() && label.compare("topology:") != 0)
    {
        getline(file, line);
        stringstream ss(line);
        ss >> label;
    }

    while (!ss.eof())
    {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    file.close();
    return topology;
}

double getEta(string input)
{
    ifstream file;
    file.open(input);
    string line;
    string label;
    double eta;

    getline(file, line);
    stringstream ss(line);
    ss >> label;
    while (!file.eof() && label.compare("eta:") != 0)
    {
        getline(file, line);
        stringstream ss(line);
        ss >> label;
    }

    stringstream s(line);

    s >> label;
    s >> label;

    eta = stod(label);

    file.close();
    return eta;
}

string getTransferFunction(string input)
{
    ifstream file;
    file.open(input);
    string line;
    string label;
    string transferFunction;

    getline(file, line);
    stringstream ss(line);
    ss >> label;
    while (!file.eof() && label.compare("transfer_function:") != 0)
    {
        getline(file, line);
        stringstream ss(line);
        ss >> label;
    }

    stringstream s(line);

    s >> label;
    s >> transferFunction;

    file.close();
    return transferFunction;
}

double getMomentum(string input)
{
    ifstream file;
    file.open(input);
    string line;
    string label;
    double momentum;

    getline(file, line);
    stringstream ss(line);
    ss >> label;
    while (!file.eof() && label.compare("momentum:") != 0)
    {
        getline(file, line);
        stringstream ss(line);
        ss >> label;
    }

    stringstream s(line);

    s >> label;
    s >> label;

    momentum = stod(label);

    file.close();

    return momentum;
}

unsigned getNextInputs(string input, vector<double> &inputVals, int iter)
{
    inputVals.clear();
    ifstream file;
    file.open(input);

    string line;
    getline(file, line);
    stringstream ss(line);

    string label;
    ss >> label;

    int counter = 0;
    while (!file.eof() && counter <= iter)
    {
        getline(file, line);
        stringstream ss(line);
        ss >> label;
        if(label.compare("in:") == 0)
        {
            counter++;
        }
        if(label.compare("in:") == 0 && counter == iter)
        {
            stringstream s(line);
            s >> label;
            while (!s.eof())
            {
                string n;
                double m;
                s >> n;
                m = stod(n.c_str());
                inputVals.push_back(m);
            }
        }
    }

    file.close();
    return inputVals.size();
}

unsigned getTargetOutputs(string input, vector<double> &targetOutputVals, int iter)
{
    targetOutputVals.clear();

    ifstream file;
    file.open(input);
    string line;
    getline(file, line);
    stringstream ss(line);

    string label;
    ss >> label;

    int counter = 0;

    while (!file.eof() && counter <= iter)
    {
        getline(file, line);
        stringstream ss(line);
        ss >> label;
        if(label.compare("out:") == 0)
        {
            counter++;
        }
        if(label.compare("out:") == 0 && counter == iter)
        {
            stringstream s(line);
            s >> label;
            while (!s.eof())
            {
                string n;
                double m;
                s >> n;
                m = stod(n.c_str());
                targetOutputVals.push_back(m);
            }
        }
    }

    file.close();

    return targetOutputVals.size();
}

void saveNetwork(Net input, string filename)
{
    ofstream output(filename.c_str());

    //record the architecture of the network first
    output << "topology: ";
    for(int i = 0; i < input.getTotalLayers(); i++)
    {
        if( i != input.getTotalLayers() - 1)
        {
            output << input.getLayerSize(i) - 1 << " ";
        }
        else
        {
            output << input.getLayerSize(i)<< endl;
        }
    }

    //record learning parameters
    output << "eta: " << input.getEta() << endl;
    output << "momentum: " << input.getMomentum() << endl;
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

void loadData(string input, vector<unsigned> &topology, vector<vector<double>> &inputVals, vector<vector<double>> &targetValues, double &eta, double &momentum, string &transferFunction)
{
    ifstream file;
    file.open(input);

    topology = getTopology(input);

    eta = getEta(input);
    momentum = getMomentum(input);
    transferFunction = getTransferFunction(input);

    vector<double> tempInput, tempOutput;

    int iter = 1;

    //Load trainning data from file
    while (!file.eof())
    {
        // Get new input data and feed it forward:
        if (getNextInputs(input, tempInput, iter) != topology[0])
            break;
        inputVals.push_back(tempInput);

        //Get the data for the correct outputs for the inputs just recorded
        getTargetOutputs(input, tempOutput, iter);

        assert(tempOutput.size() == topology.back());
        targetValues.push_back(tempOutput);
        iter++;
    }

    file.close();
}

vector<double> readWeights(string input, const int x, const int y)
{
    ifstream file;
    file.open(input);
    string line;
    string label;
    vector<double> weights;

    getline(file, line);
    stringstream ss(line);
    ss >> label;
    while (file.eof() || label.compare("(" + to_string(x)  + ','+ to_string(y) + ')' + ':') != 0)
    {
        getline(file, line);
        stringstream ss(line);
        ss >> label;
    }

    stringstream s(line);
    s >> label;
    while (!s.eof())
    {
        string n;
        double m;
        s >> n;
        m = stof(n.c_str());
        weights.push_back(m);
    }
    file.close();
    return weights;
}

Net loadnetwork(string input)
{
    Net net(getTopology(input),getTransferFunction(input), getEta(input), getMomentum(input));

    //read weights, minus the output layer
    //loop through every layer, except the output layer
    for(int x = 0; x < net.getTotalLayers() - 1; x++)
    {
        //loop through every node in layer
        for(int y = 0; y < net.getLayerSize(x); y++)
        {
            net.setWeight(x, y, readWeights(input, x, y));
        }
    }
    return net;
}
