#define BOOST_TEST_MODULE MyLibraryTest
#include <boost/test/included/unit_test.hpp>
#include "convolutions/convolutions.hpp"
#include <eigen3/Eigen/Dense>

using namespace Eigen;

BOOST_AUTO_TEST_CASE(test_case1)
{
    MatrixXd X(3,3);
    MatrixXd K(2,2);
    int n = 1;
    
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            X(i,j) = n; 
            n+=1;
        }
    }

    for(int i=0; i<2; i++){
        for(int j=0; j<2; j++){
            K(i,j) = 1;
        }
    }

    MatrixXd conv_matrix = Conv2D(X,K);

    BOOST_TEST(conv_matrix(0,0) == 12);
    BOOST_TEST(conv_matrix(0,1) == 16);
    BOOST_TEST(conv_matrix(1,0) == 24);
    BOOST_TEST(conv_matrix(1,1) == 28);
}



BOOST_AUTO_TEST_CASE(test_padding1)
{
    MatrixXd X(2,2);
    MatrixXd K(2,2);
    int n = 1;
    
    for(int i=0; i<2; i++){
        for(int j=0; j<2; j++){
            X(i,j) = n; 
            n+=1;
        }
    }

    for(int i=0; i<2; i++){
        for(int j=0; j<2; j++){
            K(i,j) = 1;
        }
    }

    MatrixXd conv_matrix = NaivePaddingConv2D(X,K,1);

    BOOST_TEST(conv_matrix(0,0) == 1);
    BOOST_TEST(conv_matrix(0,1) == 3);
    BOOST_TEST(conv_matrix(0,2) == 2);

    BOOST_TEST(conv_matrix(1,0) == 4);
    BOOST_TEST(conv_matrix(1,1) == 10);
    BOOST_TEST(conv_matrix(1,2) == 6);

    BOOST_TEST(conv_matrix(2,0) == 3);
    BOOST_TEST(conv_matrix(2,1) == 7);
    BOOST_TEST(conv_matrix(2,2) == 4);
}