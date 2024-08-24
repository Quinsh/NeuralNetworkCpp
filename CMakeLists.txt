cmake_minimum_required(VERSION 3.28)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 17)

# Find SFML package
set(SFML_DIR /opt/homebrew/opt/sfml/lib/cmake/SFML)
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# SFML directories
include_directories(${SFML_INCLUDE_DIR})

add_executable(neuralnetwork main.cpp
        neuralnetwork.txt
        Neuron.cpp
        Neuron.h
        Layer.cpp
        Layer.h
        utility.h
        NeuralNetwork.cpp
        NeuralNetwork.h
        utility.cpp
        NetDrawer.cpp
        NetDrawer.h
)

target_link_libraries(neuralnetwork sfml-graphics sfml-window sfml-system)
