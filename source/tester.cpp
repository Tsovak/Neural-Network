#include <iostream>
#include <fstream>

#include "../include/net.h"
#include "../include/data.h"

using namespace std;

int main()
{
    srand(time(NULL));

    vector<unsigned> topology;

    vector<vector<double>> inputValsA, targetValuesA;
    vector<double> resultValues;
    double minError = .000001;

    int epochs = 0;
    int maxEpochs = 10000;

    ofstream finalOutput;
    finalOutput.open("./data/finalOutput.txt");

    double eta;
    double momentum;

    string transferFunction;

    loadData("./data/xordata.txt",
             topology,
             inputValsA,
             targetValuesA,
             eta,
             momentum,
             transferFunction);

    Net net(topology,transferFunction, eta, momentum);
    double recentAverageError;

    double currentError = 999;
    double prevErr = 999;

    //continous feed the training data through the network
    //until it is perfect or after n iterations
    while(prevErr > minError && epochs < maxEpochs)
    {
        prevErr = currentError;
        currentError = 0;
        for(unsigned int i = 0; i < inputValsA.size(); i++)
        {
            // Get new input data and feed it forward:
            net.feedForward(inputValsA[i]);

            // Collect the net's actual output results:
            net.getResults(resultValues);

            // Train the net what the outputs should have been:
            net.backPropagation(targetValuesA[i]);

            // Report how well the training is working, average over recent samples:
            recentAverageError = net.getRecentAverageError();
            currentError += recentAverageError;
        }
        epochs++;
    }

    cout << "Error: "<< prevErr << endl;

    //see how well the network trained
    for(unsigned int i = 0; i < inputValsA.size(); i++)
    {
        // Get new input data and feed it forward:
        net.feedForward(inputValsA[i]);

        // Collect the net's actual output results:
        net.getResults(resultValues);

        //record results of the network
        finalOutput << "Inputs: ";
        for(unsigned int j = 0; j < inputValsA[i].size(); j++)
        {
            if(j != inputValsA[i].size() -1)
                finalOutput << inputValsA[i][j] << " ";
            else
                finalOutput << inputValsA[i][j];
        }

        finalOutput << endl;

        finalOutput << "Expected Output: ";
        for(unsigned int j = 0; j < resultValues.size(); j++)
        {
            if(j != resultValues.size() - 1)
                finalOutput << targetValuesA[i][j] << " ";
            else
                finalOutput << targetValuesA[i][j];
        }
        finalOutput << endl;

        finalOutput << "Trained Results: ";
        for(unsigned int j = 0; j < resultValues.size(); j++)
        {
            if(j != resultValues.size() -1)
                finalOutput << resultValues[j] << " ";
            else
                finalOutput << resultValues[j];
        }

        finalOutput << endl;
        finalOutput << endl;
    }

    finalOutput.close();

    //save, load, and resave the network as a test
    saveNetwork(net, "data/weights.txt");

    Net example("./data/weights.txt");

    saveNetwork(example, "./data/copiedweights.txt");

    return 0;
}
