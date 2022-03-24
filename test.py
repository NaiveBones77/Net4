import core
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


network = core.NeuralNetwork([784, 40, 10])
network.fit(training_data, 30, 10, 3.0)
print('Accuracy: {} %'.format((network.accuracy(test_data) / len(test_data)) * 100))