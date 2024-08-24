//
// Created by Gun woo Kim on 8/21/24.
//

#include "Layer.h"
#include "utility.h"

Layer::Layer(unsigned int prev_n, unsigned int cur_n, ActivationType _activationType)
    : activationType(_activationType)
{
    setActivationFxn(_activationType);

    neurons.resize(cur_n);
    XavierInitialization(prev_n);
}

Layer::Layer(const std::vector<double>& input_vec)
    : activationType(LINEAR) , activation_fxn(&relu)
{ // for making input layer
    setActivationFxn(activationType);
    neurons.resize(input_vec.size());
    for (int i=0; i<input_vec.size(); ++i) {
        neurons[i].a = input_vec[i];
    }
}

void Layer::XavierInitialization(unsigned int prev_n) {
    double mean = 0;
    double stdev = std::sqrt(1.0/prev_n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(mean, stdev);

    for (auto& neuron : neurons) {
        neuron.deleteWeights();
        std::vector<double> _weights;
        _weights.reserve(prev_n);
        for (int j=0; j<prev_n; ++j) {
            _weights.push_back(dist(gen));
        }
        neuron.initWeights(std::move(_weights));
    }
}

const std::vector<Neuron>& Layer::getNeuronsReadOnly() const {
    return neurons;
}

std::vector<Neuron> & Layer::getNeurons() {
    return neurons;
}

double Layer::maxWeightAmongAllNeurons() const {
    double maxWeight = 0;
    for (const auto& neuron : neurons) {
        maxWeight = std::max(maxWeight, neuron.maxWeight());
    }
    return maxWeight;
}

void Layer::set_a(const std::vector<double> &&new_as) {
    for(int i=0; i<neurons.size(); ++i) {
        neurons[i].a = new_as[i];
    }
}

unsigned long Layer::getNeuronCount() const {
    return neurons.size();
}

void Layer::forward(const Layer& prev_layer) {
    if (activationType == SOFTMAX) {
        std::vector<double> z = compute_z_vector(prev_layer);
        set_a(std::move(softmax(z)));
        return;
    }
    for (auto& neuron : neurons) {
        neuron.computeOutput(prev_layer, activation_fxn);
    }
}

std::vector<double> Layer::compute_z_vector(const Layer &prev_layer) {
    std::vector<double> z(neurons.size());
    const auto& prevlayer_neurons = prev_layer.getNeuronsReadOnly();

    for(int i=0; i<neurons.size(); ++i) {
        double _z = 0;
        int j = 0;
        for (const auto& neuron : prevlayer_neurons) {
            _z += neuron.a * neurons[i].weights[j++];
        }
        _z += neurons[i].bias;
        z[i] = _z;
    }
    return z;
}

std::vector<double> Layer::getOutputVector() {
    std::vector<double> output;
    output.resize(getNeuronCount());
    for (int i=0; i<getNeuronCount(); ++i) {
        output[i] = neurons[i].a;
    }
    return output;
}

void Layer::setActivationFxn(ActivationType) {
    switch (activationType) {
        case LINEAR:
            activation_fxn = &linear;
            break;
        case SIGMOID:
            activation_fxn = &sigmoid;
            break;
        case RELU:
            activation_fxn = &relu;
            break;
        case TANH:
            activation_fxn = &std::tanh;
            break;
        case SOFTMAX:
            activation_fxn = nullptr; // we use different logic for softmax
        default:
            break;
    }
}

ActivationType Layer::getActivationType() const {
    return activationType;
}

void Layer::backward(const Layer &prev_layer, const Layer &next_layer, const double &eta) {
    computeDelta(next_layer);
    computeAndApplyWeightGradient(prev_layer, eta);
}

void Layer::computeDelta(const Layer &next_layer) {
    double delCdelA = 0, activationDerivative = 0;
    int next_layer_size = next_layer.neurons.size();

    for(int i=0; i<neurons.size(); ++i) {
        delCdelA = 0;
        for (int j=0; j<next_layer_size; ++j) {
            delCdelA += (next_layer.neurons[j].delta * next_layer.neurons[j].weights[i]);
        }
        activationDerivative = activationFxnDerivative(activationType, neurons[i].a);

        neurons[i].delta = delCdelA * activationDerivative;
    }
}

void Layer::computeAndApplyWeightGradient(const Layer &prev_layer, const double &eta) {
    double gradient = 0;

    for (auto & neuron : neurons) {
        // weights update
        for (int j=0; j<prev_layer.neurons.size(); ++j) {
            gradient = neuron.delta * prev_layer.neurons[j].a;
            neuron.weights[j] -= gradient * eta;
        }
        // bias update
        neuron.bias -= neuron.delta * eta;
    }
}

/*
// weightGradient computation for Batch Gradient Descent using all samples in one backprop
void Layer::firstLayerBatchGD(const std::vector<std::vector<double>> &input_layers, const double &eta) {
    int sampleSize = input_layers.size();
    double adjustedEta = eta/sampleSize;

    for (int i=0; i<sampleSize; ++i) { // do gradient descent for all samples with weaker eta.
        auto input_layer = Layer(input_layers[i]);
        computeAndApplyWeightGradient(input_layer, adjustedEta);
    }
}
*/

// weightGradient computation for Batch Gradient Descent using all samples in one backprop
void Layer::firstLayerBatchGD(const std::vector<std::vector<double>> &input_layers, const double &eta) {
    int sampleSize = input_layers.size();
    std::vector<double> gradientSum;
    gradientSum.resize(neurons[0].weights.size(), 0.0);

    std::vector<std::vector<double>> gradientSums(neurons.size(), gradientSum);
    std::vector<double> biasSums(neurons.size(), 0.0);

    for (int i = 0; i < sampleSize; ++i) { // Accumulate gradients over all samples
        const std::vector<double>& input_layer = input_layers[i];
        for (int n = 0; n < neurons.size(); ++n) {
            for (int j = 0; j < input_layer.size(); ++j) {
                gradientSums[n][j] += neurons[n].delta * input_layer[j];
            }
            biasSums[n] += neurons[n].delta;
        }
    }

    for (int n = 0; n < neurons.size(); ++n) {
        for (int j = 0; j < neurons[n].weights.size(); ++j) {
            neurons[n].weights[j] -= (eta / sampleSize) * gradientSums[n][j];
        }
        neurons[n].bias -= (eta / sampleSize) * biasSums[n];
    }
}

void Layer::printWeights() {
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Weights and Bias for each neuron: " << std::endl;
    for (const auto& neuron : neurons) {
        std::cout << "Weights: ";
        for (const auto& w : neuron.getWeightsReadOnly()) {
            std::cout << w << " ";
        }
        std::cout << std::endl << "Bias: " << neuron.getBias() << std::endl;
    }
}

void Layer::printOutput() {
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Output: ";
    for (const auto& neuron : neurons) {
        std::cout << neuron.a << " ";
    }
    std::cout << std::endl;
}

