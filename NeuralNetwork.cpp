//
//  NeuralNetwork.cpp
//  SingleHiddenNN
//
//  Created by Timothy S. Lewkow
//
#include <math.h>
#include "NeuralNetwork.h"
using namespace std;

NN::NN(vector<double> & X, int HN)
{
  eta = 0.01; // step size parameter
  d = (int)X.size();
  h = HN;
  p = h*d + 2*h + 1; // total number of parameters
  tempf.resize(h);
  tempfPrime.resize(h);
  initializeWeights();
}
void NN::initializeWeights()
{
  W.resize(p);
  for(int i = 0; i < p; i++)
    W[i] = (double) rand() / RAND_MAX;
}
double NN::F(vector<double> & X) // return F evaluated at current weights
{
  double sum = 0;
  // Evaluate f_i
  for(int i = 0; i < h; i++)
    tempf[i] = f(X,i);
  // Evaluate F
  for(int i = 0; i < h; i++)
    sum += W[(d*h)+h+i]*tempf[i]; // V_i*f_i
  sum += W[(d*h)+(2*h)];
  return sum;
}
double NN::f(vector<double> & X, int index)
{
  double sum = 0;
  for(int j = 0; j < d; j++)
    sum += X[j]*W[(d*index)+j];
  sum += W[(d*h)+index];
  // evaluate sigmoid logistic function
  sum = 1.0/(1.0 + exp(-1.0*sum));
  return sum;
}
double NN::fPrime(vector<double> & X, int index)
{
  double sum = 0;
  for(int j = 0; j < d; j++)
    sum += X[j]*W[(d*index)+j];
  sum += W[(d*h)+index];
  // evaluate sigmoid logistic function
  sum = exp(-1.0*sum)/(pow(1.0 + exp(-1.0*sum),2));
  return sum;
}
void NN::updateWeights(vector<double> & X, double target)
{
  /*
   Use back propagation algorithm to update network weights
  */
  // Calculate (t-z) : difference between answer and expected
  double dif = (target-F(X));
  // Evaluate all litte fPrime
  for(int i = 0; i < h; i++)
    tempfPrime[i] = fPrime(X, i);
  // Evaluate all little f
  for(int i = 0; i < h; i++)
    tempf[i] = f(X,i);
  double updateFactor;
  // Update input to hidden weights
  for(int i = 0; i < h; i++)
  {
    for(int j = 0; j < d; j++)
    {
      updateFactor = W[(d*h)+h+i]*X[j]*tempfPrime[i]; // V_i*X_j*f'
      W[(d*i)+j] = W[(d*i)+j] + (updateFactor*dif*eta);
    }
  }
  // Update hidden bias weights
  for(int i = 0; i < h; i++)
    W[(d*h)+i] = W[(d*h)+i] + (W[(d*h)+h+i]*tempfPrime[i]*eta*dif); // V_i*f_i
  // Update hidden to output weights
  for(int i = 0; i < h; i++)
    W[(d*h)+h+i] = W[(d*h)+h+i] + (tempf[i]*eta*dif);
  // Update output bias
  W[(d*h)+(2*h)] = W[(d*h)+(2*h)] + (1.0*eta*dif);
}
double NN::evalError(vector<double> & X, double target)
{
  double retVal = 0.5*pow((F(X)-target),2);
  return retVal;
}
void NN::displayW()
{ // Function used for debugging purposes
  cout << "\nWeight Vector" << endl;
  for(int i = 0; i < p; i++)
    cout << "\nW[" << i << "] = " << W[i];
}
