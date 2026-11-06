#ifndef TENSOR_DENSE_LAYER_HPP
#define TENSOR_DENSE_LAYER_HPP

#include <functional>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

using Tensor2D = Tensor<double, 2>;

class DenseLayer {

private:

    Tensor2D weights;
    Tensor2D bias;

    std::function<Tensor2D(const Tensor2D&)> activation_fun;

public:

    DenseLayer(const Tensor2D& weights, const Tensor2D& bias, std::function<Tensor2D(const Tensor2D&)> activation_fun) :
        weights(weights),
        bias(bias),
        activation_fun(activation_fun)
    {}

    // Copy constructor
    DenseLayer(const DenseLayer& other) = default;

    // Assignment operator
    DenseLayer& operator=(const DenseLayer& other) = default;

    // Evaluation
    Tensor2D evaluate(const Tensor2D& X) const {

        Tensor2D Z = weights.contract(X, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)});

        Z = Z + bias.broadcast(Eigen::array<Eigen::Index, 2>{1, X.dimension(1)});

        return activation_fun(Z);
    }

};

#endif