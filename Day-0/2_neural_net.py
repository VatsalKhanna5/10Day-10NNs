import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, random_seed=None):
        """
        Initializes the neural network.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        self.weights_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias_output = np.zeros((1, self.output_size))

    def _sigmoid(self, z):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, sigmoid_output):
        """
        Derivative of the sigmoid activation function.
        """
        return sigmoid_output * (1 - sigmoid_output)

    def forward(self, X):
        """
        Performs the forward pass.
        """
        self.z_hidden = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.activation_hidden = self._sigmoid(self.z_hidden)

        self.z_output = np.dot(self.activation_hidden, self.weights_output) + self.bias_output
        # For regression, we might not use an activation here, or use a linear one.
        # Let's start without an activation on the output for simplicity in regression.
        self.output_prediction = self.z_output

        return self.activation_hidden, self.output_prediction

    def backward(self, X, y, activation_hidden, output_prediction):
        """
        Performs the backward pass.
        """
        # Output layer error (derivative of MSE with respect to output)
        output_error = output_prediction - y

        # Gradients for the output layer
        d_weights_output = np.dot(activation_hidden.T, output_error)
        d_bias_output = np.sum(output_error, axis=0, keepdims=True)

        # Error propagated to the hidden layer
        hidden_error = np.dot(output_error, self.weights_output.T) * self._sigmoid_derivative(activation_hidden)

        # Gradients for the hidden layer
        d_weights_hidden = np.dot(X.T, hidden_error)
        d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

        return d_weights_output, d_bias_output, d_weights_hidden, d_bias_hidden

    def update_weights(self, dw_output, db_output, dw_hidden, db_hidden):
        """
        Updates the weights and biases.
        """
        self.weights_output -= self.learning_rate * dw_output
        self.bias_output -= self.learning_rate * db_output
        self.weights_hidden -= self.learning_rate * dw_hidden
        self.bias_hidden -= self.learning_rate * db_hidden

    def train(self, X_train, y_train, epochs=1000, verbose=False):
        """
        Trains the neural network.
        """
        history = []
        for epoch in range(epochs):
            activation_hidden, output_prediction = self.forward(X_train)
            dw_output, db_output, dw_hidden, db_hidden = self.backward(X_train, y_train, activation_hidden, output_prediction)
            self.update_weights(dw_output, db_output, dw_hidden, db_hidden)

            if verbose and epoch % 100 == 0:
                loss = np.mean((output_prediction - y_train) ** 2)
                history.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return history

    def predict(self, X):
        """
        Makes predictions.
        """
        _, output_prediction = self.forward(X)
        return output_prediction