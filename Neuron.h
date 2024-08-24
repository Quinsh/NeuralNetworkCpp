//
// Created by Gun woo Kim on 8/21/24.
//

#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <functional>

#include "Layer.h"

class Layer; // forward decl

class Neuron {

public:
    std::vector<double> weights;
    double bias;
    double a; // the output value
    double delta;

    Neuron(): bias(0), a(0), delta(0) {
    };

    void deleteWeights();
    void initWeights(std::vector<double>&& weights);
    const std::vector<double>& getWeightsReadOnly() const;
    double maxWeight() const;
    double getBias() const;
    void computeOutput(const Layer& prev_layer, const std::function<double(double)>&);

};



#endif //NEURON_H
