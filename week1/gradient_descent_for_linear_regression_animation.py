import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def generate_data(
    num_samples: int, noise: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates random linear data for testing gradient descent.

    Parameters:
    num_samples (int): Number of data points to generate.
    noise (float): Standard deviation of Gaussian noise added to target values.

    Returns:
    x_train (np.ndarray): Generated feature values.
    y_train (np.ndarray): Generated target values with noise.
    #"""
    np.random.seed(0)  # for reproducibility
    x_train = np.random.rand(num_samples) * 10  # Scale features to a range [0, 10]
    true_w, true_b = 2.0, 5.0  # Define an arbitrary linear relationship
    y_train = true_w * x_train + true_b + np.random.randn(num_samples) * noise
    return x_train, y_train


x_train, y_train = generate_data(num_samples=100, noise=5.0)

m = x_train.shape[0]


def compute_f_wb(x: float, w: float, b: float) -> float:
    return w * x + b


def compute_gradient(
    x_train: np.ndarray, y_train: np.ndarray, w: float, b: float
) -> tuple[float, float]:
    m = x_train.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        x_i, y_i = x_train[i], y_train[i]
        f_wb = compute_f_wb(x_i, w, b)

        dj_dw += (f_wb - y_i) * x_i
        dj_db += f_wb - y_i

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def compute_gradient_descent_epoch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w: float,
    b: float,
    learning_rate: float,
) -> tuple[float, float]:
    dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)
    w = w - learning_rate * dj_dw
    b = b - learning_rate * dj_db
    return w, b


def gradient_descent(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w: float,
    b: float,
    learning_rate: float,
    epochs: int,
) -> tuple[float, float]:
    for _ in range(epochs):
        w, b = compute_gradient_descent_epoch(x_train, y_train, w, b, learning_rate)

    return w, b


learning_rate = 0.01
epochs = 1000
w, b = 0, 0

fig, ax = plt.subplots()
ax.set_title("Gradient Descent for Linear Regression")
ax.set_xlabel("Features")
ax.set_ylabel("Target Values")
ax.scatter(x_train, y_train, label="Actual Values", color="red")
(line,) = ax.plot([], [], label="Prediction Line")
epoch_text = ax.text(x=max(x_train) / 2, y=max(y_train), s="")
ax.legend()


def update(epoch: int):
    global w, b
    w, b = compute_gradient_descent_epoch(x_train, y_train, w, b, learning_rate)
    tmp_f_wb = [compute_f_wb(x, w, b) for x in x_train]

    line.set_data(x_train, tmp_f_wb)
    epoch_text.set_text(f"epoch: {epoch}")

    return line, epoch_text


animation = FuncAnimation(
    fig=fig, func=update, frames=range(epochs), interval=10, blit=True
)

plt.show()
