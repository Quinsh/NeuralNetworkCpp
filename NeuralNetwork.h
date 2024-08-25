//
// Created by Gun woo Kim on 8/21/24.
//

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include "Layer.h"
#include "NetDrawer.h"
#include "utility.h"

class NetDrawer;
class Layer;

class NeuralNetwork {
    std::vector<Layer> layers;
    unsigned int input_size; // this should be adjusted with adjustFirstLayer when training. 1 is default, when it hasn't been trained yet
    double eta;
    double epsilon; // (for automatic convergence
    GradientDescentType gradient_descent_type;
    double mini_batch_size;

public:
    NeuralNetwork()
        : input_size(1), eta(0.05), epsilon(0), gradient_descent_type(Batch), mini_batch_size(0.1) {
    }

    explicit NeuralNetwork(const std::vector<int>& layer_configuration)
        : input_size(1), eta(0.05), epsilon(0), gradient_descent_type(Batch), mini_batch_size(0.1) {
        layers.emplace_back(1, layer_configuration[0]);
        // temporary first layer (needs to change depending on the input layer size later)
        for (int i = 1; i < layer_configuration.size(); ++i) {
            layers.emplace_back(layer_configuration[i - 1], layer_configuration[i]);
        }
    }

    const std::vector<Layer>& getLayerReadOnly() const;
    unsigned long getMaxNeuronInLayer() const;
    void addLayer(int size, ActivationType);
    void adjustFirstLayer(int input_size, ActivationType = SIGMOID);
    void forwardProp(const std::vector<double> &); // the difference btw forwardProp and predict is only that predict returns a the last layer's output (y_hat).
    std::vector<double> predict(const std::vector<double>&);
    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>&);
    double cost_compute(const std::vector<std::vector<double>> &X_train, const std::vector<std::vector<double>> &Y_train, LossFxn loss_fxn = MSE);
    double getLearningRate() const;
    void setLearningRate(const double&);
    void clearAllDeltas();
    void clearAllWeightBiasGradients();
    void fit(const std::vector<std::vector<double>> &X_train, const std::vector<std::vector<double>> &Y_train, int epoch, LossFxn, NetDrawer *drawer = nullptr);
    void gradientDescent(); // prereq: gradients are alrdy calculated
    void setGradientDescentType(GradientDescentType);
    void setMiniBatchSize(double ratio);
    friend void NetDrawer::drawNetwork(const NeuralNetwork&, int epoch, double cost);

    void printDeltaAndWeights() const;

};

#endif //NEURALNETWORK_H
