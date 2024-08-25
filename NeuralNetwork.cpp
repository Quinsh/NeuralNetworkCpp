//
// Created by Gun woo Kim on 8/21/24.
//
#include "NeuralNetwork.h"
#include <thread>
#include <unordered_set>

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

void NeuralNetwork::clearAllDeltas() {
    for (auto& layer : layers) {
        layer.clearDeltas();
    }
}

void NeuralNetwork::clearAllWeightBiasGradients() {
    for (auto& layer : layers) {
        layer.clearWeightGradients();
        layer.clearBiasGradients();
    }
}

void NeuralNetwork::fit(const std::vector<std::vector<double>> &X_train, const std::vector<std::vector<double>> &Y_train, int epoch, LossFxn loss_fxn = MSE, NetDrawer *drawer) {
    size_t sample_size = X_train.size();
    size_t inputlayer_size = X_train[0].size();
    size_t layer_size  = layers.size();
    size_t batch_size = sample_size;
    switch(gradient_descent_type) {
        case SGD:
            batch_size = 1;
        break;
        case MiniBatch:
            batch_size = mini_batch_size;
        break;
        case Batch:
            break;
        default:
            break;
    }
    std::vector<std::vector<double>> X_train_use(batch_size, std::vector<double>(inputlayer_size));
    std::vector<std::vector<double>> Y_train_use(batch_size, std::vector<double>(Y_train[0].size()));


    auto _activationType = layers[0].getActivationType();
    adjustFirstLayer(static_cast<int>(inputlayer_size), _activationType); // change the shape of the first layer according to the input layer shape.

    // TODO: implement some automatic convergence using epsilon = 0.05
    for (int _=0; _<epoch; ++_) {
        // comptute the cost and print
        double cost = cost_compute(X_train, Y_train, loss_fxn);
        std::cout << "Cost is: " << cost << std::endl;

        // logic for selecting the training set for each epoch (depending on if it's SDG, mini-batch, batch)
        // by default, batch:
        switch(gradient_descent_type) {
            case SGD: // randomly select one
                int randomIndex = randomNumber(0, sample_size-1);
                X_train_use[0] = X_train[randomIndex];
                Y_train_use[0] = Y_train[randomIndex];
                break;
            case MiniBatch: // randomly form a mini-batch
                std::unordered_set<int> usedIndices;
                for (int i=0; i<batch_size; ++i) {
                    randomIndex = randomNumber(0, sample_size-1);
                    if (usedIndices.find(randomIndex) == usedIndices.end()) {
                        X_train_use[i] = X_train[randomIndex];
                        Y_train_use[i] = Y_train[randomIndex];
                        usedIndices.insert(randomIndex);
                    }
                    else {
                        --i;
                    }
                }
                break;
            case Batch:
                X_train_use = X_train;
                Y_train_use = Y_train;
                break;
            default:
                break;
        }

        clearAllWeightBiasGradients();

        // for all sample size, do backprop
        for (int i=0; i<batch_size; ++i) {
            // 1, forward prop
            forwardProp(X_train_use[i]);
            // 2. compute deltas and weights (averaged by doing sample_size)
            // compute delta and gradients in the last layer
            layers[layer_size-1].computeLastLayerDelta(Y_train_use[i], loss_fxn);
            if (layer_size == 1)// last layer is first
                layers[layer_size-1].computeWeightGradient(X_train_use[i], batch_size);
            else
                layers[layer_size-1].computeWeightGradient(layers[layer_size-2], batch_size);
            // compute delta and weight gradients in further layers.
            for (int l=layer_size-2; l>0; --l) {
                layers[l].computeDelta(layers[l+1]);
                layers[l].computeWeightGradient(layers[l-1], batch_size);
            }
            // compute delta and weight gradients in first layer
            if (layer_size != 1) {
                layers[0].computeDelta(layers[1]);
                layers[0].computeWeightGradient(X_train_use[i], batch_size);
            }
        }
        // 3. subtract the weigths for all neurons ()
        gradientDescent();

        if (drawer && _%5==0) {
            drawer->drawNetwork(*this, _, cost);
            drawer->handleEvents(); // keep window responsive
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // printDeltaAndWeights(); // for testing
    }
}

void NeuralNetwork::gradientDescent() {
    for (auto & layer : layers) {
        layer.gradientDescent(eta);
    }
}

void NeuralNetwork::setGradientDescentType(GradientDescentType gd) {
    gradient_descent_type = gd;
}

void NeuralNetwork::setMiniBatchSize(double size) {
    mini_batch_size = size;
}

void NeuralNetwork::printDeltaAndWeights() const {
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        std::cout << "Layer " << layer_idx + 1 << ":\n";
        const auto& neurons = layers[layer_idx].getNeuronsReadOnly();
        for (size_t neuron_idx = 0; neuron_idx < neurons.size(); ++neuron_idx) {
            const auto& neuron = neurons[neuron_idx];
            std::cout << "  Neuron " << neuron_idx + 1 << ":\n";
            std::cout << "    Delta: " << neuron.delta << "\n";
            std::cout << "    Weights: ";
            for (const auto& weight : neuron.weights) {
                std::cout << weight << " ";
            }
            std::cout << "\n    Bias: " << neuron.bias << "\n";
        }
        std::cout << std::endl;
    }
}


