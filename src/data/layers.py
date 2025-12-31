from enum import Enum

class ActivationLayer(Enum):
    LINEAR = "Linear"
    SOFTMAX = "Softmax"
    RELU = "ReLU"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"

class OperationLayer(Enum):
    ADD = "Add"
    MULTIPLY = "Multiply"

class ReshapeLayer(Enum):
    CONV = "Conv"
    MAXPOOL = "MaxPool"
    FLATTEN = "Flatten"
    DENSE = "Dense"
