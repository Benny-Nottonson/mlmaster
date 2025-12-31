import numpy as np
from data.levels import ActivationLayer


class NeuralNetwork:
    def __init__(self, level_id, dataset):
        self.level_id = level_id
        self.dataset = dataset
        self.weights = []
        self.biases = []
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self._generate_training_data()

    def _generate_training_data(self):
        """Generate training and test data using the dataset's generator."""
        self.X_train, self.y_train, self.X_test, self.y_test = self.dataset.generate_data()

    def initialize_network(self, sorted_layers):
        """Initialize network weights and biases based on layer configuration."""
        self.weights = []
        self.biases = []
        
        layer_sizes = [self.dataset.input_features]
        for layer in sorted_layers:
            layer_sizes.append(layer.output_size)
        layer_sizes.append(self.dataset.output_classes)
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def apply_activation(self, x, activation):
        if activation == ActivationLayer.RELU:
            return np.maximum(0, x)
        elif activation == ActivationLayer.TANH:
            return np.tanh(x)
        elif activation == ActivationLayer.SOFTMAX:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        elif activation == ActivationLayer.SIGMOID:
            return 1 / (1 + np.exp(-x))
        return x
    
    def activation_derivative(self, x, activation):
        if activation == ActivationLayer.RELU:
            return (x > 0).astype(float)
        elif activation == ActivationLayer.TANH:
            return 1 - np.tanh(x) ** 2
        elif activation == ActivationLayer.SIGMOID:
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        return np.ones_like(x)
    
    def train_step(self, sorted_layers, learning_rate=0.05, batch_size=32):
        indices = np.random.choice(len(self.X_train), batch_size, replace=False)
        X_batch = self.X_train[indices]
        y_batch = self.y_train[indices]
        
        activations = [X_batch]
        zs = []
        
        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            zs.append(z)
            if i < len(sorted_layers):
                a = self.apply_activation(z, sorted_layers[i].activation)
            else:
                a = z
            activations.append(a)
        
        predictions = activations[-1]
        mse_loss = np.mean((predictions - y_batch) ** 2)
        l1_loss = np.mean(np.abs(predictions - y_batch))
        
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        y_clipped = np.clip(y_batch, 1e-7, 1 - 1e-7)
        ce_loss = -np.mean(y_clipped * np.log(predictions_clipped) + (1 - y_clipped) * np.log(1 - predictions_clipped))
        
        delta = 2 * (predictions - y_batch) / batch_size
        for i in range(len(self.weights) - 1, -1, -1):
            self.weights[i] -= learning_rate * (activations[i].T @ delta)
            self.biases[i] -= learning_rate * np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                if i - 1 < len(sorted_layers):
                    delta = (delta @ self.weights[i].T) * self.activation_derivative(zs[i-1], sorted_layers[i-1].activation)
                else:
                    delta = delta @ self.weights[i].T
        
        return mse_loss, l1_loss, ce_loss
    
    def evaluate(self, sorted_layers):
        """Evaluate the model on test data and return accuracy."""
        activations = self.X_test
        for i in range(len(self.weights)):
            z = activations @ self.weights[i] + self.biases[i]
            if i < len(sorted_layers):
                activations = self.apply_activation(z, sorted_layers[i].activation)
            else:
                activations = z
        
        predictions = activations
        
        if self.dataset.output_classes > 1:
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(self.y_test, axis=1)
            accuracy = np.mean(predicted_classes == true_classes)
        else:
            mse = np.mean((predictions - self.y_test) ** 2)
            accuracy = max(0, 1 - mse)
        
        return accuracy
    
    def forward(self, X, sorted_layers):
        activations = X
        for i in range(len(self.weights)):
            z = activations @ self.weights[i] + self.biases[i]
            if i < len(sorted_layers):
                activations = self.apply_activation(z, sorted_layers[i].activation)
            else:
                activations = z
        return activations
