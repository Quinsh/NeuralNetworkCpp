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
    const std::vector<Neuron>& getNeuronsReadOnly() const; // read-only access of neurons. const l-val refernece
    std::vector<Neuron>& getNeurons();
    double maxWeightAmongAllNeurons() const;
    void set_a(const std::vector<double>&& new_as);
    unsigned long getNeuronCount() const;
    void forward(const Layer& prev_layer);
    std::vector<double> compute_z_vector(const Layer &prev_layer);
    std::vector<double> getOutputVector();
    void setActivationFxn(ActivationType);
    ActivationType getActivationType() const;
    void backward(const Layer &prev_layer, const Layer &next_layer, const double &eta);
    void computeDelta(const Layer &next_layer);
    void computeAndApplyWeightGradient(const Layer& prev_layer, const double &eta); // prereq: delta is computed
    void firstLayerBatchGD(const std::vector<std::vector<double>> &input_layers, const double &eta);

    void printWeights(); // just for testing
    void printOutput(); // just for testing

};

#endif //LAYER_H
