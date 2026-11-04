#ifndef CONVOLUTIONS_HPP
#define CONVOLUTIONS_HPP

#include <eigen3/Eigen/Dense>

using namespace Eigen;

MatrixXd Conv2D(const MatrixXd& X, const MatrixXd& K);
MatrixXd NaivePaddingConv2D(const MatrixXd& X, const MatrixXd& K, int padding);

#endif