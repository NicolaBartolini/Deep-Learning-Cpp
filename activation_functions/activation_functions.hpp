#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <eigen3/Eigen/Dense>
using namespace Eigen;

double sigmoid(double x);
double ReLu(double x);

MatrixXd sigmoid_activation(const MatrixXd& X );
MatrixXd ReLu_activation(const MatrixXd& X );


#endif