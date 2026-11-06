#include "convolutions.hpp"
#include <eigen3/Eigen/Dense>

MatrixXd Conv2D(const MatrixXd& X, const MatrixXd& K){

    // compute output dimension for the output matrix 
    // rows = X_rows - K_rows + 1
    // cols = X_cols - K_cols + 1
    const int k_rows = K.rows();
    const int k_cols = K.cols(); 
    const int rows = X.rows() - k_rows + 1;
    const int cols = X.cols() - k_cols +1; 

    // define output matrix result 

    MatrixXd result = MatrixXd::Zero(rows, cols);

    // convolution operation
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double x = X.block(i, j, k_rows, k_cols).cwiseProduct(K).sum();
            result(i,j) = x;
        }
    }
    return result;
}


MatrixXd NaivePaddingConv2D(const MatrixXd& X, const MatrixXd& K, int padding){
    // compute output dimension for the output matrix 
    // rows = X_rows - K_rows + 1
    // cols = X_cols - K_cols + 1
    const int k_rows = K.rows();
    const int k_cols = K.cols(); 
    const int rows = X.rows() - k_rows + 2*padding + 1;
    const int cols = X.cols() - k_cols + 2*padding + 1; 

    // memory preallocation padding matrix of X
    MatrixXd padX = MatrixXd::Zero(X.rows() + 2*padding, X.cols() + 2*padding);
    padX.block(padding, padding, X.rows(), X.cols()) = X; // setting the internal part of padX = X 

    MatrixXd res = MatrixXd::Zero(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double x = padX.block(i,j, k_rows, k_cols).cwiseProduct(K).sum();
            res(i, j) = x;
        }
        
    }
    return res;
}