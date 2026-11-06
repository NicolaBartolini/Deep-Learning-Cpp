#define BOOST_TEST_MODULE MyLibraryTest
#include <boost/test/included/unit_test.hpp>
#include "MLP/dense_layer.hpp"
#include "activation_functions/activation_functions.hpp"
#include <eigen3/Eigen/Dense>
#include <functional> // for : std::ref (this create a pointer to an l-value)

using namespace Eigen;

BOOST_AUTO_TEST_CASE(test_ReLu){
    MatrixXd X(2,2);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            X(i,j) = i+j - 1;
        }
        
    }

    MatrixXd Y = ReLu_activation(X);

    BOOST_TEST(Y(0,0) == 0);
    BOOST_TEST(Y(0,1) == 0);
    BOOST_TEST(Y(1,0) == 0);
    BOOST_TEST(Y(1,1) == 1);   
    
}

BOOST_AUTO_TEST_CASE(test_dense_layer){

    MatrixXd X(2,2);
    MatrixXd Weights(2,2);
    int x = 1.0;
    
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            X(i,j) = x;
            x+=1.0;
            Weights(i,j) = .5;
        }
        
    }

    VectorXd bias(2,1);
    
    for (int i = 0; i < 2; i++)
    {
        bias(i, 0) = 1.;
    }
    
    MatrixXd Y = calc_layer(X, Weights, bias, ReLu_activation );

    BOOST_TEST(Y(0,0) == 3.0);
    BOOST_TEST(Y(0,1) == 4.0);
    BOOST_TEST(Y(1,0) == 3.0);
    BOOST_TEST(Y(1,1) == 4.0);  

}


BOOST_AUTO_TEST_CASE(test_dense_layer_class1){
    MatrixXd X(2,2);
    MatrixXd Weights(2,2);
    int x = 1.0;
    
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            X(i,j) = x;
            x+=1.0;
            Weights(i,j) = .5;
        }
        
    }

    VectorXd bias(2,1);
    
    for (int i = 0; i < 2; i++)
    {
        bias(i, 0) = 1.;
    } 

    DenseLayer my_layer(Weights, bias, ReLu_activation);

    MatrixXd Y = my_layer.evaluate(X);

    BOOST_TEST(Y(0,0) == 3.0);
    BOOST_TEST(Y(0,1) == 4.0);
    BOOST_TEST(Y(1,0) == 3.0);
    BOOST_TEST(Y(1,1) == 4.0);
}