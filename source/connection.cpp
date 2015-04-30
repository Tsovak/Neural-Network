#include "../include/connection.h"
Connection::Connection()
{
	weight=randomWeight();
	deltaWeight=0;
}

Connection::Connection(double w, double d)
{
    weight = w;
    deltaWeight = d;
}

