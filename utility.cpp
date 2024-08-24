//
// Created by Gun woo Kim on 8/22/24.
//
#include "NeuralNetwork.h"
#include <vector>

using namespace std;

double linear(double x) {
    return x;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double relu(double x) {
    return std::max(0.0, x);
}

std::vector<double> softmax(const std::vector<double>& logits) {
    std::vector<double> exp_values(logits.size());
    double max_logit = *std::max_element(logits.begin(), logits.end());
    for (size_t i = 0; i < logits.size(); ++i)
        exp_values[i] = std::exp(logits[i] - max_logit);
    double sum_exp_values = std::accumulate(exp_values.begin(), exp_values.end(), 0.0);
    for (double &val : exp_values)
        val /= sum_exp_values;
    return exp_values;
}

double loss_MSE(const double& y_hat, const double& y) {
    return (y-y_hat)*(y-y_hat)/2;
}

double multi_output_MSE(const vector<double>& y_hats, const vector<double>& ys) {
    unsigned int m = ys.size();
    double squaredError = 0;
    for (int i = 0; i < ys.size(); i++) {
        squaredError += loss_MSE(y_hats[i], ys[i]);
    }
    return squaredError/m;
}

double loss_BinaryCrossEntropy(double y_hat, double y) {
    y_hat = std::max(std::min(y_hat, 1.0 - 1e-15), 1e-15); // preventing log(0)
    return -(y * std::log(y_hat) + (1 - y) * std::log(1 - y_hat));
}

double loss_CategoricalCrossEntropy(const std::vector<double> &y_hat, const std::vector<double> &y) {
    double categorical_ce = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        categorical_ce += y[i] * std::log(std::max(y_hat[i], 1e-15));  // preventing log(0)
    }
    return -categorical_ce;
}

double lossFunctionDerivative(LossFxn loss_fxn, const double &y_hat, const double &y) { // del(C)/del(a)
    switch (loss_fxn) {
        case LossFxn::MSE:
            return (y_hat-y); // (y_hat - y)^2 --> 2(y_hat - y) but (y_hat-y) for simplicity
        case LossFxn::BinaryCrossEntropy:
            return - (y / y_hat) + (1 - y) / (1 - y_hat);
        case LossFxn::CategoricalCrossEntropy:
            return y_hat - y; // used with softmax output
        default:
            return -1;
    }
}

double activationFxnDerivative(ActivationType activationType, const double& a) {
    switch (activationType) {
        case LINEAR:
            return 1;
        case SIGMOID:
            return a * (1-a);
        case RELU:
            return a > 0 ? 1 : 0;
        case TANH:
            return 1-a*a;
        default:
            return 0;
    }
}


