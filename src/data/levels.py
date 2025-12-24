from dataclasses import dataclass
from enum import Enum

class ActivationLayer(Enum):
    LINEAR = "Linear"
    SOFTMAX = "Softmax"
    RELU = "ReLU"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"

class OperationType(Enum):
    ADD = "Add"
    MULTIPLY = "Multiply"

class LayerType(Enum):
    INPUT = "Input"
    HIDDEN = "Hidden"
    OUTPUT = "Output"
    CONV = "Conv"
    MAXPOOL = "MaxPool"
    FLATTEN = "Flatten"
    DENSE = "Dense"

@dataclass
class LevelGoals:
    accuracy_target: float
    time_limit_seconds: int
    cost_limit: float

@dataclass
class DatasetInfo:
    name: str
    train_samples: int
    test_samples: int
    input_features: int
    output_classes: int

@dataclass
class Level:
    id: str
    name: str
    description: str
    position: tuple
    dataset: DatasetInfo
    goals: LevelGoals
    available_activations: list
    available_layers: list
    available_operations: list
    is_completed: bool
    is_unlocked: bool

LEVELS = {
    "level_1": Level(
        id="level_1",
        name="Getting Started",
        description="Learn the basics with a simple classification task",
        position=(400, 400),
        dataset=DatasetInfo(
            name="MNIST_Simple",
            train_samples=10000,
            test_samples=2000,
            input_features=1,
            output_classes=1
        ),
        goals=LevelGoals(
            accuracy_target=0.85,
            time_limit_seconds=120,
            cost_limit=50.0
        ),
        available_activations=[ActivationLayer.LINEAR, ActivationLayer.TANH],
        available_layers=[LayerType.INPUT, LayerType.OUTPUT],
        available_operations=[],
        is_completed=False,
        is_unlocked=True
    ),
    "level_2": Level(
        id="level_2",
        name="Probability Distribution",
        description="Model probability distributions across multiple classes",
        position=(900, 400),
        dataset=DatasetInfo(
            name="MultiClass",
            train_samples=800,
            test_samples=200,
            input_features=2,
            output_classes=3
        ),
        goals=LevelGoals(
            accuracy_target=0.80,
            time_limit_seconds=120,
            cost_limit=60.0
        ),
        available_activations=[ActivationLayer.LINEAR, ActivationLayer.TANH, ActivationLayer.SOFTMAX],
        available_layers=[LayerType.INPUT, LayerType.OUTPUT],
        available_operations=[],
        is_completed=False,
        is_unlocked=False
    ),
    "level_3": Level(
        id="level_3",
        name="Advanced Regression",
        description="Approximate a complex non-linear function",
        position=(1400, 400),
        dataset=DatasetInfo(
            name="ComplexFunc",
            train_samples=1200,
            test_samples=300,
            input_features=1,
            output_classes=1
        ),
        goals=LevelGoals(
            accuracy_target=0.88,
            time_limit_seconds=150,
            cost_limit=80.0
        ),
        available_activations=[ActivationLayer.LINEAR, ActivationLayer.TANH, ActivationLayer.SOFTMAX],
        available_layers=[LayerType.INPUT, LayerType.HIDDEN, LayerType.OUTPUT],
        available_operations=[OperationType.ADD],
        is_completed=False,
        is_unlocked=False
    )
}

CHECKPOINTS = {
    "checkpoint_1": {
        "id": "checkpoint_1",
        "name": "Softmax Unlocked",
        "description": "You now have access to Softmax activation",
        "position": (650, 400),
        "unlocks": [ActivationLayer.SOFTMAX],
        "required_level": "level_1",
        "is_unlocked": False
    },
    "checkpoint_2": {
        "id": "checkpoint_2",
        "name": "Add Block Unlocked",
        "description": "You now have access to Add operation blocks",
        "position": (1150, 400),
        "unlocks": [OperationType.ADD],
        "required_level": "level_2",
        "is_unlocked": False
    }
}
