#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <functional> // for : std::ref (this create a pointer to an l-value)
#include <eigen3/Eigen/Dense>
#include "activation_functions/activation_functions.hpp"

using namespace Eigen;

template<typename activation>
MatrixXd calc_layer(const MatrixXd& X, const MatrixXd& Weights, const VectorXd& bias, activation ActFun){
    // bias is a column vector 
    MatrixXd result = ActFun((Weights * X).colwise() + bias);

    return result;
}

// template <typename F>
class DenseLayer{
    private:
        MatrixXd weights;
        MatrixXd bias;
        std::function<MatrixXd(const MatrixXd&)> activation_fun;

    public:

        // Constructor
        DenseLayer(const MatrixXd& weights, const VectorXd& bias, std::function<MatrixXd(const MatrixXd&)> activation_fun) : weights(weights), bias(bias), activation_fun(activation_fun) {}

        ~DenseLayer() = default;

        // Copy constructor
        DenseLayer(const DenseLayer& other) : weights(other.weights), bias(other.bias), activation_fun(other.activation_fun){}

        // Copy assignment operator
        DenseLayer& operator=(const DenseLayer& other) {

            if (this != &other) {

                weights = other.weights;
                bias = other.bias;
                activation_fun = other.activation_fun;

            }

            return *this;
        }

        MatrixXd evaluate(const MatrixXd& X) const {

            MatrixXd res = calc_layer(X, this->weights, this->bias, this->activation_fun); 
            return res;
        }
};

#endif