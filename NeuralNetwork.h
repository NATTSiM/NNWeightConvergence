//
//  NeuralNetwork.h
//  SingleHiddenNN
//
//  Created by Timothy S. Lewkow
//

#ifndef __SingleHiddenNN__NeuralNetwork__
#define __SingleHiddenNN__NeuralNetwork__

#include <iostream>
#include <vector>

class NN
{
public:
  NN(std::vector<double> &, int); // Constructor takes
  
  double F(std::vector<double> &); // Evaluates network with current weights
  void updateWeights(std::vector<double> &, double); // training vec and target
  double evalError(std::vector<double> &, double); // Done under 2 norm
  
  void displayW();
private:
  int d; // dimension of input
  int h; // number hidden nodes
  int p; // total number of parameters
  double eta; // step size parameter
  
  std::vector<double> W; // Vector of network weights
  std::vector<double> tempf; // Used for network evaluation
  std::vector<double> tempfPrime; // Used for network evaluation

  void initializeWeights();
  double fPrime(std::vector<double> &, int); // Used for network evalueaiton
  double f(std::vector<double> &, int); // Used for network evaluation
};

#endif /* defined(__SingleHiddenNN__NeuralNetwork__) */
