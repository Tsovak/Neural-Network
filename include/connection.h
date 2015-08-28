#ifndef CONNECTION_H
#define CONNECTION_H

class Connection{ 
public:
    static double randomWeight();
    double weight;
    double deltaWeight;

    Connection();
    Connection(double w, double d);
};

#endif
