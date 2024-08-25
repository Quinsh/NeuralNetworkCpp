//
// Created by Gun woo Kim on 8/21/24.
//

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <random>
#include <functional>
#include <iostream>
#include <iomanip>
#include "Neuron.h"
#include "utility.h"

class Neuron;

class Layer {
    std::vector<Neuron> neurons;
    ActivationType activationType;
    double (*activation_fxn)(double); // for softmax processing, we use different logic

public:
    Layer(unsigned int prev_n, unsigned int cur_n, ActivationType _activationType = RELU);
    explicit Layer(const std::vector<double>& input_vec);

    void XavierInitialization(unsigned int prev_n);
    void KaimingInitialization(unsigned int prev_n);
    const std::vector<Neuron>& getNeuronsReadOnly() const; // read-only access of neurons. const l-val refernece
    std::vector<Neuron>& getNeurons();
    double maxWeightAmongAllNeurons() const;
    void set_z(const std::vector<double>&& new_zs);
    void set_a(const std::vector<double>&& new_as);
    unsigned long getNeuronCount() const;
    void forward(const Layer& prev_layer);
    std::vector<double> compute_z_vector(const Layer &prev_layer);
    std::vector<double> getOutputVector();
    void setActivationFxn(ActivationType);
    ActivationType getActivationType() const;
    void computeDelta(const Layer &next_layer);
    void clearDeltas();
    void clearWeightGradients();
    void clearBiasGradients();
    void computeLastLayerDelta(const std::vector<double>& Y_train, LossFxn);
    void computeWeightGradient(const Layer& prev_layer, int sample_size);
    void computeWeightGradient(const std::vector<double>& prev_layer, int sample_size);
    void gradientDescent(const double eta);

    void printWeights(); // just for testing
    void printOutput(); // just for testing

};

#endif //LAYER_H
