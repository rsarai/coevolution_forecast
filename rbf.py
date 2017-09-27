import math
import random
import numpy as np
from functions import radial_based_gaussian_function


class RBF():
    ciclos = 1000
    num_training = 7  # Atribui o número de exemplos de treinamento.

    def __init__(self, hidden_neurons_number):
        self.hiddenNeuronsNumber = hidden_neurons_number
        self.centers = []
        self.radius = []
        self.weights = []
        self.input_values = list(range(10))

    def initialize(self):
        self.centers = np.random.randn(self.hidden_neurons_number)
        self.radius = np.random.randn(self.hidden_neurons_number)
        self.weights = np.random.randn(self.hidden_neurons_number)

    def radial_based_activation_function(self, input_value, center, radius):
        return radial_based_gaussian_function(input_value, center, radius)

    def calculate_activation(self, input_values):
        # calculates activations of hidden layer of RBFs
        inputs_hidden_layer = []
        for index, input_value in enumerate(input_values):
            inputs_hidden_layer.append(
                radial_based_gaussian_function(
                    input_value, self.centers[index], self.radius[index]
                )
            )
        return inputs_hidden_layer

    def calculate_output(self, inputs_hidden_layer):
        return np.dot(inputs_hidden_layer, self.weights)

    def predict(self):
        inputs_hidden_layer = self.calculate_activation(self.input_values)
        output = self.calculate_output(inputs_hidden_layer)
        net2 = sum(output)
        return round((1 / (1 + math.exp(-round(net2, 5)))), 5)

    # Mudar função de saída
    def calculate_fitness(self):
        # get training data
        training_values = self.input_values
        net = self.calculate_activation(training_values)
        net2 = self.calculate_output(net)
        print(net2)
        output = round((1 / (1 + math.exp(-net2))), 5)
        print(output)
        error = self.get_output(training_values) - output
        return error

    def get_output(self, training_values):
        return sum(training_values)/random.uniform(0, 1)
