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
        Y_train.push_back({y});
    }

    // drawing tool
    NetDrawer drawer(1600, 800);

    NeuralNetwork model;
    // model.addLayer(10, RELU);
    // model.addLayer(10, RELU);
    // model.addLayer(5, RELU);
    // model.addLayer(1, LINEAR);
    // // model.adjustFirstLayer(2, LINEAR);
    // model.setLearningRate(0.00001);
    // model.fit(X_train, Y_train, 1000, MSE, &drawer);
    //
    // auto predictions = model.predict(X_train);
    //
    // cout << "Prediction: " << endl;
    // for (const auto& prediction : predictions) {
    //     for (const auto& pred : prediction) {
    //         cout << pred << endl;
    //     }
    // }
    // cout << "Real: " << endl;
    // for (const auto& Y : Y_train) {
    //     for (const auto& y : Y) {
    //         cout << y << endl;
    //     }
    // }

    // X_train data with two features
    // std::vector<std::vector<double>> X_train = {
    //     {1, 3},
    //     {4, 5},
    //     {7, 8},
    //     {9, 1},
    //     {12, 15},
    //     {15, 13},
    //     {18, 19},
    //     {20, 21},
    //     {23, 24},
    //     {26, 27}
    // };

    // Y_train data (labels), corresponding to X_train
    // std::vector<std::vector<double>> Y_train = {
    //     {0},  // Class 0
    //     {0},  // Class 0
    //     {0},  // Class 0
    //     {0},  // Class 0
    //     {1},  // Class 1
    //     {1},  // Class 1
    //     {1},  // Class 1
    //     {1},  // Class 1
    //     {1},  // Class 1
    //     {1}   // Class 1
    // };

    // std::vector<std::vector<double>> X_train = {
    //     {0.25, 0.19}, {0.26, 0.35}, {0.18, 0.18}, {0.36, 0.28}, {0.15, 0.25},
    //     {0.30, 0.35}, {0.24, 0.19}, {0.19, 0.23}, {0.22, 0.29}, {0.21, 0.28},
    //     {0.20, 0.19}, {0.23, 0.26}, {0.25, 0.18}, {0.28, 0.23}, {0.19, 0.24},
    //     {0.80, 0.80}, {0.87, 0.86}, {0.73, 0.71}, {0.81, 0.87}, {0.92, 0.84},
    //     {0.75, 0.70}, {0.83, 0.79}, {0.78, 0.75}, {0.84, 0.73}, {0.82, 0.77},
    //     {0.50, 0.52}, {0.47, 0.48}, {0.55, 0.53}, {0.51, 0.49}, {0.49, 0.50}
    // };
    //
    // std::vector<std::vector<double>> Y_train = {
    //     {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
    //     {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
    //     {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
    //     {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
    //     {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
    //     {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}
    // };

    model.addLayer(12, LINEAR);
    model.addLayer(8, LINEAR);
    model.addLayer(4, LINEAR);
    model.addLayer(1, LINEAR);
    model.setLearningRate(0.0001);
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
