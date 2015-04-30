#include "../include/net.h"
#include "../include/data.h"

int main()
{
    srand(time(NULL));
    Data trainData("xordata.txt");

    vector<unsigned> topology;

    vector<vector<double>> inputValsA, targetValuesA;
    vector<double> resultValues;
    double minError = .001;

    int epochs = 0;
    int maxEpochs = 500;

    ofstream finalOutput;
    finalOutput.open("finalOutput.txt");

    double eta;
    double momentum;

    string transferFunction;

    trainData.loadData(topology,
                       inputValsA,
                       targetValuesA,
                       eta,
                       momentum,
                       transferFunction);

    Net net(topology,transferFunction);
    double recentAverageError;

    double globalError = 999;
    double prevErr = 999;

    while(prevErr > minError && epochs < maxEpochs)
    {
        prevErr = globalError;
        globalError = 0;
        for(unsigned int i = 0; i < inputValsA.size(); i++)
        {
            // Get new input data and feed it forward:
            net.feedForward(inputValsA[i]);

            // Collect the net's actual output results:
            net.getResults(resultValues);

            // Train the net what the outputs should have been:
            net.backPropagation(targetValuesA[i], eta, momentum);

            // Report how well the training is working, average over recent samples:
            recentAverageError = net.getRecentAverageError();
            globalError += recentAverageError;
        }
        epochs++;
    }

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
            finalOutput << inputValsA[i][j] << " ";
        }

        finalOutput << endl;

        finalOutput << "Expected Output: ";
        for(unsigned int j = 0; j < resultValues.size(); j++)
        {
            finalOutput << targetValuesA[i][j] << " ";
        }
        finalOutput << endl;

        finalOutput << "Results: ";
        for(unsigned int j = 0; j < resultValues.size(); j++)
        {
            finalOutput << resultValues[j] << " ";
        }

        finalOutput << endl;
        finalOutput << endl;
    }

    finalOutput.close();

    //test saving and loading a network
    trainData.saveNetwork(net, "weights1",.1, .5);

    //Net example = trainData.loadnetwork();

    //trainData.saveNetwork(example, "weights2", .1, .5);

    return 0;
}
