#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <string>
#include <vector>
#include <fstream>

void showVectorValues(std::string label, std::vector<double> &v,std::ofstream &file);

void showTestVectorValues(std::string label, std::vector<double> &v,std::ofstream &file);

double round(double f,double pres);

#endif // FUNCTIONS_H
