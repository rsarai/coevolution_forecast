import copy
from rbf import RBF


class Particle():

    def __init__(self, positions, strategy_parameters):
        self.current_position = positions
        self.strategy_parameters = strategy_parameters
        self.best_particle = self.current_position
        self.fitness = 1000000000.0
        self.rbf = RBF(10)

    def aply_function_on_current_position(self, hidden_neurons_number=10):
        self.rbf.centers = self.current_position[0:hidden_neurons_number]
        self.rbf.radius = self.current_position[hidden_neurons_number:hidden_neurons_number*2]
        self.rbf.weights = self.current_position[hidden_neurons_number*2:hidden_neurons_number*3]
        self.fitness = self.rbf.calculate_fitness()

    def clone_particle(self):
        clone_object = copy.copy(self)
        clone_object.current_position = copy.deepcopy(self.current_position)
        clone_object.strategy_parameters = copy.deepcopy(self.strategy_parameters)
        clone_object.fitness = copy.deepcopy(self.fitness)
        clone_object.rbf = copy.deepcopy(self.rbf)
        return clone_object
