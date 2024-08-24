//
// Created by Gun woo Kim on 8/23/24.
//

#ifndef NETDRAWER_H
#define NETDRAWER_H

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

class NeuralNetwork;

class NetDrawer {
    sf::RenderWindow window;
    std::vector<std::vector<std::pair<unsigned int, unsigned int>>> NodePoints;
public:
    NetDrawer(int width, int height)
        : window(sf::VideoMode(width, height), "Neural Network Visualization") {
        window.clear(sf::Color(0, 0, 0));
    }

    void drawNetwork(const NeuralNetwork&, int epoch);
    void displayWindow();
    void closeWindow();
    bool isOpen() const;
    void handleEvents();

};



#endif //NETDRAWER_H
