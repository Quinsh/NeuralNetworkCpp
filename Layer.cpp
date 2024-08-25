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
    double stdev = std::sqrt(1.0/(prev_n+static_cast<int>(neurons.size())));
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

void Layer::KaimingInitialization(unsigned int prev_n) {
    double mean = 0;
    double stdev = std::sqrt(2.0/prev_n);
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

void Layer::set_z(const std::vector<double> &&new_zs) {
    for(int i=0; i<neurons.size(); ++i) {
        neurons[i].z = new_zs[i];
    }
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
        set_z(std::move(z));
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

void Layer::computeDelta(const Layer &next_layer) {
    double delCdelA = 0, activationDerivative = 0;
    int next_layer_size = next_layer.neurons.size();

    for(int i=0; i<neurons.size(); ++i) {
        delCdelA = 0;
        for (int j=0; j<next_layer_size; ++j) {
            delCdelA += (next_layer.neurons[j].delta * next_layer.neurons[j].weights[i]);
        }
        activationDerivative = activationFxnDerivative(activationType, neurons[i].z);

        neurons[i].delta = delCdelA * activationDerivative;
    }
}

void Layer::clearDeltas() {
    for (auto &neuron : neurons) {
        neuron.delta = 0;
    }
}

void Layer::clearWeightGradients() {
    for (auto &neuron : neurons) {
        neuron.weightGradient.clear();
        neuron.weightGradient.resize(neuron.weights.size());
    }
}

void Layer::clearBiasGradients() {
    for (auto &neuron : neurons) {
        neuron.biasGradient = 0;
    }
}

void Layer::computeLastLayerDelta(const std::vector<double> &Y_train, LossFxn loss_fxn) {
    for (int i=0; i<getNeuronCount(); ++i) {
        neurons[i].delta = 0;
        neurons[i].delta += activationFxnDerivative(activationType, neurons[i].z);
        neurons[i].delta *= lossFunctionDerivative(loss_fxn, neurons[i].a, Y_train[i]);
    }
}

void Layer::computeWeightGradient(const Layer &prev_layer, int sample_size) {
    for (auto & neuron : neurons) {
        // weights gradient
        for (int j=0; j<prev_layer.neurons.size(); ++j) {
            neuron.weightGradient[j] += (neuron.delta * prev_layer.neurons[j].a) / sample_size;
        }
        // bias gradient
        neuron.biasGradient += neuron.delta / sample_size;
    }
}

void Layer::computeWeightGradient(const std::vector<double> &prev_layer, int sample_size) {
    for (auto & neuron : neurons) {
        // weights gradient
        for (int j=0; j<prev_layer.size(); ++j) {
            neuron.weightGradient[j] += (neuron.delta * prev_layer[j]) / sample_size;
        }
        // bias gradient
        neuron.biasGradient += neuron.delta / sample_size;
    }
}

void Layer::gradientDescent(const double eta) {
    for (auto& neuron : neurons) {
        neuron.gradientDescent(eta);
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

