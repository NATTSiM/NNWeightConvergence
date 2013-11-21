//
//  main.cpp
//  SingleHiddenNN
//
//  Created by Timothy S. Lewkow
//
/*
 
 A demonstration of a single hidden layer neural network
  with sigmoid logistic activation function
 Network is a mapping from R^n to R
 
 Goal: Update neural network weights so that input vector to converges to output value

 User selects the number of hidden nodes & number of network iterations
 
 */


#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "NeuralNetwork.h"

using namespace std;
int main()
{
  srand((int)time(NULL)); // Seed random numbers
  /*************************************************/
  /*             Set values for demo               */
  /*************************************************/
  double demoNetworkOutput = 12.2; // Set desired network output
  int HN = 50; // Number of hidden nodes in network
  int demoVecSize = 4;
  vector<double> demoX(demoVecSize); // Network input vector
  for(int i = 0; i < demoVecSize; i++)
    demoX[i] = (double) rand()/RAND_MAX;

  NN myNet(demoX, HN);   // Build the network
  /*************************************************/
  /*             Demo weight iterations            */
  /*************************************************/
  int wIterations = 30; // Set max number of weight iterations
  double currentError;
  bool displayVerboseError = true; // Set true to display result of each iteration
  for(int i = 0; i < wIterations; i++)
  {
    myNet.updateWeights(demoX, demoNetworkOutput);
    currentError = myNet.evalError(demoX, demoNetworkOutput);
    if(displayVerboseError)
    {
      cout << "\n Iternation number " << i+1;
      cout << "\nNetwork error : " << currentError;
      cout << "\nNetwork output : " << setprecision(10) << fixed << myNet.F(demoX) << endl;
    }
  }
  /*************************************************/
  /*             Display results of demo           */
  /*************************************************/
  cout << "\n\n\n\t\tCode Results";
  cout << "\n\nThe input vector was : ";
  for(int i = 0; i < demoVecSize; i++)
    cout << demoX[i] << "\t\t";
  cout << "\nThe desired network output was : " << demoNetworkOutput;
  cout << "\nAfter " << wIterations << " weight iterations,";
  cout << " the network output is " << myNet.F(demoX) << endl;
  
  cout << "\n\t\tProgram Complete\n\n" << endl;
  
  return 0;
}

