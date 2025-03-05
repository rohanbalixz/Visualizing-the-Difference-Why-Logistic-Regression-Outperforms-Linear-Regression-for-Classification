import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic data:
n_points = 50
cats = np.random.randn(n_points, 2) * 0.5 + np.array([1, 1])
dogs = np.random.randn(n_points, 2) * 0.5 + np.array([3, 3])
X = np.vstack([cats, dogs])
y = np.hstack([np.zeros(n_points), np.ones(n_points)])

# Initialize parameters for both models randomly
w_linear = np.random.randn(2)
b_linear = np.random.randn()

# For logistic regression (using cross-entropy)
w_logistic = np.random.randn(2)
b_logistic = np.random.randn()

# Learning rates
lr_linear = 0.01
lr_logistic = 0.1

# Sigmoid function for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Create a grid to display prediction heatmaps
grid_res = 100
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Prepare figure with three subplots:
fig = plt.figure(figsize=(14, 10))
ax_linear = fig.add_subplot(2, 2, 1)
ax_logistic = fig.add_subplot(2, 2, 2)
ax_loss = fig.add_subplot(2, 1, 2)

# Scatter plot for the data on both decision boundary plots
for ax in [ax_linear, ax_logistic]:
    ax.scatter(cats[:, 0], cats[:, 1], color='blue', label='Cat')
    ax.scatter(dogs[:, 0], dogs[:, 1], color='red', label='Dog')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()

ax_linear.set_title("Linear Regression (MSE)")
ax_logistic.set_title("Logistic Regression (Cross-Entropy)")
ax_loss.set_title("Training Loss Over Iterations")
ax_loss.set_xlim(0, 200)
ax_loss.set_xlabel("Iteration")
ax_loss.set_ylabel("Loss")

# Initial heatmap predictions for each model (reshaped to grid shape)
preds_linear = (grid_points.dot(w_linear) + b_linear).reshape(xx.shape)
preds_logistic = sigmoid(grid_points.dot(w_logistic) + b_logistic).reshape(xx.shape)

# Display heatmaps using imshow
heat_linear = ax_linear.imshow(preds_linear, extent=[x_min, x_max, y_min, y_max],
                               origin='lower', alpha=0.5, cmap='coolwarm')
heat_logistic = ax_logistic.imshow(preds_logistic, extent=[x_min, x_max, y_min, y_max],
                                   origin='lower', alpha=0.5, cmap='coolwarm')

# Initialize decision boundary lines (where prediction = 0.5)
def get_boundary_line(w, b, x_range):
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
    else:
        y_vals = np.full_like(x_vals, -b)
    return x_vals, y_vals

line_linear, = ax_linear.plot([], [], 'k-', lw=2, label='Boundary at 0.5')
line_logistic, = ax_logistic.plot([], [], 'k-', lw=2, label='Boundary at 0.5')

# Initialize lists to store loss history
loss_history_linear = []
loss_history_logistic = []
loss_line_linear, = ax_loss.plot([], [], 'b-', lw=2, label='Linear MSE Loss')
loss_line_logistic, = ax_loss.plot([], [], 'r-', lw=2, label='Logistic CE Loss')
ax_loss.legend()

# Number of iterations for the animation
n_iters = 500

def update(frame):
    global w_linear, b_linear, w_logistic, b_logistic
    y_pred_linear = X.dot(w_linear) + b_linear
    error_linear = y_pred_linear - y
    loss_linear = np.mean(error_linear**2)
    grad_w_linear = (2 / len(y)) * X.T.dot(error_linear)
    grad_b_linear = (2 / len(y)) * np.sum(error_linear)
    w_linear -= lr_linear * grad_w_linear
    b_linear -= lr_linear * grad_b_linear

    z = X.dot(w_logistic) + b_logistic
    y_pred_logistic = sigmoid(z)
    epsilon = 1e-8
    loss_logistic = -np.mean(y * np.log(y_pred_logistic + epsilon) + (1 - y) * np.log(1 - y_pred_logistic + epsilon))
    error_logistic = y_pred_logistic - y
    grad_w_logistic = (1 / len(y)) * X.T.dot(error_logistic)
    grad_b_logistic = (1 / len(y)) * np.sum(error_logistic)
    w_logistic -= lr_logistic * grad_w_logistic
    b_logistic -= lr_logistic * grad_b_logistic

    # Record losses
    loss_history_linear.append(loss_linear)
    loss_history_logistic.append(loss_logistic)

    # Update heatmaps:
    preds_linear = (grid_points.dot(w_linear) + b_linear).reshape(xx.shape)
    preds_logistic = sigmoid(grid_points.dot(w_logistic) + b_logistic).reshape(xx.shape)
    heat_linear.set_data(preds_linear)
    heat_logistic.set_data(preds_logistic)

    # Update decision boundary lines at prediction value 0.5:
    x_vals, y_vals = get_boundary_line(w_linear, b_linear - 0.5, (x_min, x_max))
    line_linear.set_data(x_vals, y_vals)
    x_vals2, y_vals2 = get_boundary_line(w_logistic, b_logistic - np.log(0.5/0.5), (x_min, x_max))
    x_vals2, y_vals2 = get_boundary_line(w_logistic, b_logistic, (x_min, x_max))
    line_logistic.set_data(x_vals2, y_vals2)

    # Update loss plot
    loss_line_linear.set_data(range(len(loss_history_linear)), loss_history_linear)
    loss_line_logistic.set_data(range(len(loss_history_logistic)), loss_history_logistic)
    ax_loss.set_xlim(0, n_iters)
    ax_loss.set_ylim(0, max(max(loss_history_linear), max(loss_history_logistic)) * 1.1)

    return heat_linear, heat_logistic, line_linear, line_logistic, loss_line_linear, loss_line_logistic

# Create the animation
anim = FuncAnimation(fig, update, frames=n_iters, interval=100, blit=True)

plt.tight_layout()
plt.show()
