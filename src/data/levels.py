from dataclasses import dataclass
from typing import List, Callable, Tuple
import numpy as np
from data.layers import *

@dataclass
class LevelTarget:
    accuracy: float
    cost: float

@dataclass
class DatasetInfo:
    name: str
    train_samples: int
    test_samples: int
    input_features: int
    output_classes: int
    generator: Callable[[int, int, int], Tuple[np.ndarray, np.ndarray]]
    
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(42)
        X_train, y_train = self.generator(self.train_samples, self.input_features, self.output_classes)
        X_test, y_test = self.generator(self.test_samples, self.input_features, self.output_classes)
        return X_train, y_train, X_test, y_test

@dataclass
class Level:
    name: str
    position: tuple
    dataset: DatasetInfo
    goals: LevelTarget
    available_layers: List[Enum]
    completed: bool
    unlocked: bool

LEVELS = {
    "level_1": Level(
        name="Getting Started",
        position=(400, 400),
        dataset=DatasetInfo(
            name="SinWave",
            train_samples=10000,
            test_samples=2000,
            input_features=1,
            output_classes=1,
            generator=lambda n_samples, n_features, n_classes: (
                lambda X: (X, np.sin(X))
            )(np.random.uniform(-np.pi, np.pi, (n_samples, n_features)))
        ),
        goals=LevelTarget(
            accuracy=0.85,
            cost=50.0
        ),
        available_layers=[ActivationLayer.LINEAR, ActivationLayer.TANH],
        completed=False,
        unlocked=True
    ),
    "level_2": Level(
        name="Probability Distribution",
        position=(900, 400),
        dataset=DatasetInfo(
            name="GaussianClusters",
            train_samples=800,
            test_samples=200,
            input_features=2,
            output_classes=3,
            generator=lambda n_samples, n_features, n_classes: (
                lambda X, centers, labels: (
                    X + centers[labels] + np.random.randn(n_samples, n_features) * 0.5,
                    np.eye(n_classes)[labels]
                )
            )(
                np.random.randn(n_samples, n_features),
                np.array([
                    (lambda angle: np.concatenate([
                        [3 * np.cos(angle), 3 * np.sin(angle)],
                        np.random.randn(max(0, n_features - 2))
                    ])[:n_features])(2 * np.pi * i / n_classes)
                    for i in range(n_classes)
                ]),
                np.random.randint(0, n_classes, n_samples)
            )
        ),
        goals=LevelTarget(
            accuracy=0.80,
            cost=60.0
        ),
        available_layers=[ActivationLayer.LINEAR, ActivationLayer.TANH, ActivationLayer.SOFTMAX],
        completed=False,
        unlocked=False
    )
}