//
// Created by Gun woo Kim on 8/21/24.
//

#ifndef UTILITY_H
#define UTILITY_H

enum LossFxn {
    MSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
};

enum ActivationType {
    LINEAR,
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX,
};

enum GradientDescentType {
    SGD,
    MiniBatch,
    Batch,
};

double linear(double x);
double sigmoid(double x);
double relu(double x);
std::vector<double> softmax(const std::vector<double>& logits);
double loss_MSE(const double& y_hat, const double& y);
double multi_output_MSE(const std::vector<double>& y_hats, const std::vector<double>& ys);
double loss_BinaryCrossEntropy(double y_hat, double y);
double loss_CategoricalCrossEntropy(const std::vector<double>& y_hat, const std::vector<double>& y);
int randomNumber(const int& min_val, const int& max_val);

double lossFunctionDerivative(LossFxn loss_fxn, const double &y_hat, const double &y);
double activationFxnDerivative(ActivationType, const double &a);


#endif //UTILITY_H
