#include "activation_functions.hpp"
#include <cmath>
#include <functional> // for : std::ref (this create a pointer to an l-value)


double sigmoid(double x){
    double result;

    if (x>= 45.f){
        return 1.0;
    }
    else if (x<=-45.f){
        return 0.0;
    }
    else {
        result = (double) 1.0 / (1.0 + std::exp(-x));
    }
    return result;
}

double ReLu(double x){
    if(x<=0.0){
        return 0.0;
    }
    return x;
}


MatrixXd sigmoid_activation(const MatrixXd& X ){

    MatrixXd result(X.rows(), X.cols());
    result = X.unaryExpr(std::ref(sigmoid)); 

    return result;
}

MatrixXd ReLu_activation(const MatrixXd& X ){

    MatrixXd result(X.rows(), X.cols());
    result = X.unaryExpr(std::ref(ReLu)); 

    return result;
}


