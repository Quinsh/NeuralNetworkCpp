//
// Created by Gun woo Kim on 8/21/24.
//
#include "NeuralNetwork.h"
#include <thread>

const std::vector<Layer> & NeuralNetwork::getLayerReadOnly() const {
    return layers;
}

unsigned long NeuralNetwork::getMaxNeuronInLayer() const {
    unsigned long maxNeurons = 0;
    for (int i=0; i<layers.size(); ++i) {
        maxNeurons = std::max(maxNeurons, layers[i].getNeuronCount());
    }
    return maxNeurons;
}

void NeuralNetwork::addLayer(int size, ActivationType _activationType) {
    if (!layers.empty()) {
        layers.emplace_back(layers[layers.size()-1].getNeuronCount(), size, _activationType);
    }
    else {
        layers.emplace_back(1, size, _activationType);
    }
}

void NeuralNetwork::adjustFirstLayer(int _input_size, ActivationType _activationType) {
    try {
        int firstLayerSize = layers[0].getNeuronCount();
        layers.erase(layers.begin()); // delete first element
        layers.insert(layers.begin(), Layer(_input_size, firstLayerSize, _activationType));
        input_size = _input_size;
    }
    catch (...) {
        std::cerr << "Cannot adjust first layer (check if it has a first layer)" << std::endl;
    }
}

void NeuralNetwork::forwardProp(const std::vector<double> &input_vector) {
    try {
        if (input_vector.size() != input_size) throw std::runtime_error("input vector size doesn't match with trained network's input size");

        auto input_layer = Layer(input_vector); // craete input_layer from input,
        layers[0].forward(input_layer);
        for (int i=1; i<layers.size(); ++i) { // forward every layers for forward propagation
            layers[i].forward(layers[i-1]);
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double> &input_vector) {
    forwardProp(input_vector);
    return layers[layers.size()-1].getOutputVector();
}

std::vector<std::vector<double>> NeuralNetwork::predict(const std::vector<std::vector<double>> &input_vectors) {
    std::vector<std::vector<double>> predictions;
    predictions.reserve(input_vectors.size());
    for (const auto& input_sample : input_vectors) {
        predictions.push_back(predict(input_sample));
    }
    return predictions;
}

double NeuralNetwork::cost_compute(const std::vector<std::vector<double>> &X_train, const std::vector<std::vector<double>> &Y_train, LossFxn loss_fxn) {
    size_t sample_size = X_train.size();
    double cost = 0;
    for (int i=0; i<sample_size; ++i) {
        double loss = 0;
        auto Yhat_vector = predict(X_train[i]);
        auto& Y_vector = Y_train[i];
        switch (loss_fxn) {
            case MSE:
                if (Yhat_vector.size() == 1)
                    loss = loss_MSE(Yhat_vector[0], Y_vector[0]);
                else
                    loss = multi_output_MSE(Yhat_vector, Y_vector);
                break;
            case BinaryCrossEntropy:
                if (Yhat_vector.size() == 1)
                    loss = loss_BinaryCrossEntropy(Yhat_vector[0], Y_vector[0]);
                break;
            case CategoricalCrossEntropy:
                if (Yhat_vector.size() > 1)
                    loss = loss_CategoricalCrossEntropy(Yhat_vector, Y_vector);
                break;
            default:
                break;
        }
        cost += loss;
    }
    cost /= static_cast<double>(sample_size);
    return cost;
}

double NeuralNetwork::getLearningRate() const {
    return eta;
}

void NeuralNetwork::setLearningRate(const double &_eta) {
    eta = _eta;
}

void NeuralNetwork::fit(const std::vector<std::vector<double>> &X_train, const std::vector<std::vector<double>> &Y_train, int epoch, LossFxn loss_fxn = MSE, NetDrawer *drawer) {
    size_t sample_size = X_train.size();
    size_t inputlayer_size = X_train[0].size();
    size_t layer_size  = layers.size();

    auto _activationType = layers[0].getActivationType();
    adjustFirstLayer(static_cast<int>(inputlayer_size), _activationType); // change the shape of the first layer according to the input layer shape.

    // TODO: implement some automatic convergence using epsilon = 0.05
    for (int _=0; _<epoch; ++_) {

        // comptute the cost
        double cost = cost_compute(X_train, Y_train, loss_fxn);
        std::cout << "Cost is: " << cost << std::endl;

        // computing last layer delta
        auto& lastLayer = layers[layer_size-1];
        const auto& lastLayerNeuronsRO = lastLayer.getNeuronsReadOnly();
        ActivationType activation = lastLayer.getActivationType();
        double singleDelta = 0; // delta for one train sample
        std::vector<double> delta; delta.resize(lastLayerNeuronsRO.size());

        for (int k=0; k<sample_size; ++k) {
            forwardProp(X_train[k]);
            for (int i=0; i<lastLayer.getNeuronCount(); ++i) {
                singleDelta = 0;
                singleDelta += activationFxnDerivative(activation, lastLayerNeuronsRO[i].a);
                singleDelta *= lossFunctionDerivative(loss_fxn, lastLayerNeuronsRO[i].a, Y_train[k][i]);
                delta[i] += singleDelta;
            }
        }
        for (double & d : delta) {d /= static_cast<double>(sample_size);}
        auto& lastLayerNeurons = lastLayer.getNeurons();
        for (int i=0; i<delta.size(); ++i) {
            lastLayerNeurons[i].delta = delta[i];
        }
        // compute weight gradient, and gradient descent for last layer
        if (layer_size > 1) {
            lastLayer.computeAndApplyWeightGradient(layers[layer_size-2], eta);
        }
        else { // when last layer is the first layer
            lastLayer.firstLayerBatchGD(X_train, eta);
        }

        // BACKPROPAGATION starting from the penultimate layer
        for (int i=layer_size-2; i>0; --i) {
            layers[i].backward(layers[i-1], layers[i+1], eta);
        }
        // first layer gradient descent
        if (layer_size > 1) {
            layers[0].computeDelta(layers[1]);
            layers[0].firstLayerBatchGD(X_train, eta);
        }

        if (drawer && _%5==0) {
            drawer->drawNetwork(*this, _);
            drawer->handleEvents(); // keep window responsive
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

    }
}

