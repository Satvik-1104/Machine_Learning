import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Step 1: Load and Preprocess Data
def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1
    return X, y_one_hot, y


def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Step 2: Define Single-Layer Perceptron Model
class SingleLayerPerceptron:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        return self.softmax(np.dot(X, self.weights) + self.bias)

    def compute_loss(self, y_pred, y_true):
        n_samples = y_true.shape[0]
        if y_true.ndim == 1:
            y_true_one_hot = np.zeros_like(y_pred)
            y_true_one_hot[np.arange(n_samples), y_true] = 1
        else:
            y_true_one_hot = y_true
        logp = -np.log(y_pred[np.arange(n_samples), np.argmax(y_true_one_hot, axis=1)])
        return np.sum(logp) / n_samples

    def backprop(self, X, y_pred, y_true):
        n_samples = y_true.shape[0]
        dW = np.dot(X.T, (y_pred - y_true)) / n_samples
        dB = np.sum(y_pred - y_true, axis=0) / n_samples
        return dW, dB

    def update_weights(self, dW, dB):
        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * dB

    def train(self, X, y, X_val=None, y_val=None, epochs=1000, print_freq=500):
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            y_pred = self.forward(X)
            train_loss = self.compute_loss(y_pred, y)
            train_losses.append(train_loss)

            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val_pred, y_val)
                val_losses.append(val_loss)

            dW, dB = self.backprop(X, y_pred, y)
            self.update_weights(dW, dB)

            if epoch % print_freq == 0:
                # Print only every 'print_freq' epochs (default is 100)
                print(
                    f'Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss if X_val is not None else "N/A"}')

        return train_losses, val_losses

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)


# Step 3: Implement Custom K-Fold Cross-Validation
def k_fold_split(X, y, k=5, shuffle=True):
    assert len(X) == len(y), "X and y must have the same number of samples"
    if shuffle:
        indices = np.random.permutation(len(X))
    else:
        indices = np.arange(len(X))

    fold_sizes = np.full(k, len(X) // k, dtype=int)
    fold_sizes[:len(X) % k] += 1
    current = 0
    folds = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_index = indices[start:stop]
        train_index = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_index, test_index))
        current = stop

    return folds


# Step 4: Define Evaluation Metrics
def compute_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    precision, recall = {}, {}
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0

    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return precision, recall, accuracy


# Step 5: Perform Cross-Validation with Custom K-Fold
def cross_validation(X, y_one_hot, y, k=5, learning_rate=0.01, epochs=1000, print_freq=100):
    folds = k_fold_split(X, y, k=k, shuffle=True)
    overall_accuracy = []
    class_precision, class_recall = [], []
    avg_train_losses = np.zeros(epochs)
    avg_val_losses = np.zeros(epochs)

    for fold, (train_index, test_index) in enumerate(folds):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_one_hot[train_index], y[test_index]
        y_test_labels = y[test_index]

        model = SingleLayerPerceptron(input_dim=X.shape[1], output_dim=y_one_hot.shape[1], learning_rate=learning_rate)
        train_losses, val_losses = model.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=epochs,
                                               print_freq=print_freq)

        avg_train_losses += np.array(train_losses)
        avg_val_losses += np.array(val_losses)

        y_pred = model.predict(X_test)
        precision, recall, accuracy = compute_metrics(y_test_labels, y_pred)

        class_precision.append(precision)
        class_recall.append(recall)
        overall_accuracy.append(accuracy)

        print(f"Fold {fold + 1} - Accuracy: {accuracy}")

    avg_train_losses /= k
    avg_val_losses /= k

    # Plot average losses
    plt.plot(range(epochs), avg_train_losses, label='Average Train Loss')
    plt.plot(range(epochs), avg_val_losses, label='Average Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    avg_accuracy = np.mean(overall_accuracy)
    avg_precision = {cls: np.mean([p[cls] for p in class_precision]) for cls in np.unique(y)}
    avg_recall = {cls: np.mean([r[cls] for r in class_recall]) for cls in np.unique(y)}

    print("\n--- Cross-Validation Results ---")
    print("Average Accuracy:", avg_accuracy)
    print("Average Precision per Class:", avg_precision)
    print("Average Recall per Class:", avg_recall)
    return avg_accuracy


# Hyperparameter Tuning
def tune_hyperparameters(X, y_one_hot, y, learning_rates, epochs_list, k=5, print_freq=100):
    best_accuracy = 0
    best_params = {}
    for lr in learning_rates:
        for epochs in epochs_list:
            print(f"Testing learning_rate={lr}, epochs={epochs}")
            avg_accuracy = cross_validation(X, y_one_hot, y, k=k, learning_rate=lr, epochs=epochs,
                                            print_freq=print_freq)
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_params = {'learning_rate': lr, 'epochs': epochs}
    print("Best hyperparameters:", best_params)
    return best_params


# Load data, normalize, and tune hyperparameters
filepath = 'iris.csv'  # Replace with the path to your CSV file
X, y_one_hot, y = load_data(filepath)
X = normalize(X)

# Define ranges for hyperparameters
learning_rates = [0.01, 0.05, 0.1]
epochs_list = [500, 1000, 1500]
best_params = tune_hyperparameters(X, y_one_hot, y, learning_rates, epochs_list, k=5, print_freq=100)
