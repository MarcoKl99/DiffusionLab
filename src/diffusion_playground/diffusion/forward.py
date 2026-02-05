import numpy as np


class ForwardDiffusion:
    def __init__(self, time_steps: int = 50, beta: float = 0.02):
        self.time_steps = time_steps
        self.beta = beta

    def diffuse(self, x0: np.ndarray) -> np.ndarray:
        x = x0.copy()
        trajectory = [x]

        for t in range(self.time_steps):
            # Create Gaussian noise
            epsilon = np.random.randn(*x.shape)
            x = np.sqrt(1 - self.beta) * x + np.sqrt(self.beta) * epsilon
            trajectory.append(x)

        return np.array(trajectory)
