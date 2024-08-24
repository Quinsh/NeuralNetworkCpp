//
// Created by Gun woo Kim on 8/21/24.
//

#include "Neuron.h"

void Neuron::deleteWeights() {
    weights.clear();
}

void Neuron::initWeights(std::vector<double>&& _weights) {
    weights = std::move(_weights);
}

const std::vector<double> & Neuron::getWeightsReadOnly() const {
    return weights;
}

double Neuron::maxWeight() const {
    double maxWeight = 0;
    for (const auto& weight : weights) {
        maxWeight = std::max(maxWeight, weight);
    }
    return maxWeight;
}

double Neuron::getBias() const {
    return bias;
}

void Neuron::computeOutput(const Layer& prev_layer, const std::function<double(double)>& activation) {
    double _a = 0;
    int i = 0;
    for (const auto& neuron : prev_layer.getNeuronsReadOnly()) {
        _a += neuron.a * weights[i++];
    }
    _a += bias;
    _a = activation(_a);
    a = _a;
}
