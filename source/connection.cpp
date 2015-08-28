#include "../include/connection.h"
#include <iostream>
using namespace std;

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


double Connection::randomWeight(){
	return rand()/double(RAND_MAX);
}


