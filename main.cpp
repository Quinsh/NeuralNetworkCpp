#include <iostream>
#include "Neuron.h"
#include "Layer.h"
#include "NeuralNetwork.h"
#include "utility.h"
#include <vector>

using namespace std;

int main() {
    // X_train data with two input features
    std::vector<std::vector<double>> X_train = {
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0},
        {5.0, 6.0},
        {6.0, 7.0},
    };

    // Y_train data based on the linear relationship with noise
    std::vector<std::vector<double>> Y_train;

    for (const auto& x : X_train) {
        double y = 10 * x[0] + 5 * x[1] + ((rand() % 10) / 10.0) - 0.5; // Adding noise
        Y_train.push_back({y});  // Applying ReLU (max(0, y))
    }

    // drawing tool
    NetDrawer drawer(2200, 1100);

    NeuralNetwork model;
    model.addLayer(10, RELU);
    model.addLayer(10, RELU);
    model.addLayer(5, RELU);
    model.addLayer(1, LINEAR);
    // model.adjustFirstLayer(2, LINEAR);
    model.setLearningRate(0.00001);
    model.fit(X_train, Y_train, 1000, MSE, &drawer);

    auto predictions = model.predict(X_train);

    cout << "Prediction: " << endl;
    for (const auto& prediction : predictions) {
        for (const auto& pred : prediction) {
            cout << pred << endl;
        }
    }
    cout << "Real: " << endl;
    for (const auto& Y : Y_train) {
        for (const auto& y : Y) {
            cout << y << endl;
        }
    }


    return 0;
}
