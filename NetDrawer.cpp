//
// Created by Gun woo Kim on 8/23/24.
//

#include "NetDrawer.h"
#include "NeuralNetwork.h"

void NetDrawer::drawNetwork(const NeuralNetwork &net, int epoch, double cost) {

    window.clear(sf::Color::Black);

    unsigned long layer_size = net.layers.size();
    unsigned long max_neuron_size = net.getMaxNeuronInLayer();
    auto window_size = window.getSize();
    auto width = window_size.x;
    auto height = window_size.y;
    unsigned int marginTopBottom = 0.2 * height; // change this to make margin
    unsigned int marginLeftRight = 0.3 * width; //
    double heightBtwNeuronRatio = 1; // ratio of (empty space btw neuron) : neuron
    double widthBtwNeuronRatio = 1.5; //

    unsigned int circleWidth = (width - marginLeftRight) / (1+widthBtwNeuronRatio) / layer_size;
    unsigned int circleHeight = (height - marginTopBottom) / (1+heightBtwNeuronRatio) / max_neuron_size;
    unsigned int circleDiameter = std::min(circleWidth, circleHeight);
    unsigned int circleRadius = circleDiameter/2;
    unsigned int heightInterval = circleDiameter * heightBtwNeuronRatio;
    unsigned int widthInterval = (width - marginLeftRight - (circleDiameter*layer_size)) / (layer_size-1);

    NodePoints.resize(layer_size, std::vector<std::pair<unsigned int, unsigned int>>());
    unsigned int x = 0, y = 0;
    x += marginLeftRight/2 + circleDiameter/2;
    for (int i = 0; i < net.layers.size(); i++) {
        NodePoints[i].resize(net.layers[i].getNeuronCount());
        auto neuron_size = net.layers[i].getNeuronCount();
        y = 0;
        y += marginTopBottom/2;
        y += ((max_neuron_size-neuron_size)*(circleDiameter + heightInterval))/2;
        y += circleDiameter/2;
        for (int j=0; j<net.layers[i].getNeuronCount(); ++j) {
            NodePoints[i][j] = {x, y};
            y += circleDiameter + heightInterval;
        }
        x += circleDiameter + widthInterval;
    }

    for (int i=0; i<NodePoints.size(); ++i) {
        for (int j=0; j<NodePoints[i].size(); j++) {
            // Node Drawing
            sf::CircleShape node(circleRadius);
            node.setFillColor(sf::Color::White);
            node.setOrigin(circleRadius, circleRadius);
            node.setPosition(NodePoints[i][j].first, NodePoints[i][j].second);
            window.draw(node);
        }
    }

    // Weight Drawing
    double zero = 0;
    for (int i = 0; i < net.layers.size() - 1; ++i) {
        const auto maxWeightInLayer = net.layers[i].maxWeightAmongAllNeurons();
        for (int j = 0; j < net.layers[i].getNeuronCount(); ++j) {
            for (int k = 0; k < net.layers[i + 1].getNeuronCount(); ++k) {
                // get the weight btw neurons
                const auto& neuronsRO = net.layers[i].getNeuronsReadOnly();
                double weight = neuronsRO[j].weights[k];

                // Normalize weight to a brightness value
                double normalizedWeight = std::abs(weight) / maxWeightInLayer;
                int brightness;
                if (normalizedWeight < 0.4) {
                    brightness = static_cast<int>(normalizedWeight * 2 * 127.5);  // Scale to [0, 127.5]
                } else {
                    brightness = static_cast<int>((normalizedWeight * 255) - 127.5);  // Scale to [127.5, 255]
                }
                // exponential scaling
                // brightness = static_cast<int>((std::exp(normalizedWeight) - 1) / (std::exp(1) - 1) * 255);



                // Draw
                sf::Vertex line[] = {
                    sf::Vertex(sf::Vector2f(NodePoints[i][j].first, NodePoints[i][j].second), sf::Color(brightness, brightness, brightness)),
                    sf::Vertex(sf::Vector2f(NodePoints[i + 1][k].first, NodePoints[i + 1][k].second), sf::Color(brightness, brightness, brightness))
                };
                window.draw(line, 2, sf::Lines);
            }
        }
    }

    // Text Drawing for Epoch
    sf::Font font;
    if (!font.loadFromFile("../arial.ttf")) {
    }
    sf::Text epochText, costText;
    epochText.setFont(font);
    epochText.setString("Epoch: " + std::to_string(epoch));
    epochText.setCharacterSize(80);
    epochText.setFillColor(sf::Color::White);
    epochText.setPosition(10, 10);
    window.draw(epochText);

    costText.setFont(font);
    costText.setString("Cost: " + std::to_string(cost));
    costText.setCharacterSize(60);
    costText.setFillColor(sf::Color::Red);
    costText.setPosition(width-0.3*width, 10);
    window.draw(costText);

    window.display();
}

void NetDrawer::displayWindow() {
    handleEvents();
}

void NetDrawer::closeWindow() {
    window.close();
}

bool NetDrawer::isOpen() const {
    return window.isOpen();
}

void NetDrawer::handleEvents() {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            closeWindow();
        }
    }
}