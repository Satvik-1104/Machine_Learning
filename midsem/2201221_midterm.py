import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the dataset
data = pd.read_csv('iris.csv')

# Extract features and labels
X = data.iloc[:, :-1].values  # Exclude the label column
y = data.iloc[:, -1].values  # Labels

print("Dataset loaded successfully.")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)


# Step 2: Implement K-means Clustering
def kmeans(X, k, max_iters=100):
    np.random.seed(42)
    random_indices = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X[random_indices]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return clusters, centroids


# Step 3: Elbow Method to Determine Optimal k
def elbow_method(X, max_k=10):
    sse = []
    for k in range(1, max_k + 1):
        clusters, centroids = kmeans(X, k)
        sse.append(np.sum((X - centroids[clusters]) ** 2))

    # Plot SSE for each value of k
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid()
    plt.show()

    # Finding the elbow point
    optimal_k = np.argmax(np.diff(np.diff(sse))) + 2  # Adding 2 due to the way np.diff works
    print(f"Optimal number of clusters (k): {optimal_k}")
    return optimal_k


# Find optimal number of clusters
optimal_k = elbow_method(X)

# Run K-means with the determined k
clusters, centroids = kmeans(X, optimal_k)
print("K-means clustering completed.")


# Step 4: Select samples closest to each centroid
def select_closest_samples(X, clusters, centroids, n_samples=25):
    selected_samples = []
    selected_labels = []

    for i in range(len(centroids)):
        cluster_samples = X[clusters == i]
        cluster_labels = y[clusters == i]

        distances = np.linalg.norm(cluster_samples - centroids[i], axis=1)
        closest_indices = np.argsort(distances)[:n_samples]

        selected_samples.extend(cluster_samples[closest_indices])
        selected_labels.extend(cluster_labels[closest_indices])

    return np.array(selected_samples), np.array(selected_labels)


selected_samples, selected_labels = select_closest_samples(X, clusters, centroids)
print(f"Selected {len(selected_samples)} samples for training/validation.")


# Step 5: Implement Multi-Class Logistic Regression
class MultiClassLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        # One-hot encoding of the target labels
        y_encoded = np.zeros((num_samples, num_classes))
        for idx, label in enumerate(y):
            y_encoded[idx, np.where(self.classes == label)[0][0]] = 1

        # Initialize weights and bias
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros(num_classes)

        # Gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.softmax(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y_encoded))
            db = (1 / num_samples) * np.sum(y_predicted - y_encoded, axis=0)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.softmax(linear_model)
        return self.classes[np.argmax(y_predicted, axis=1)]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)


# Train Multi-Class Logistic Regression on the selected samples
log_reg = MultiClassLogisticRegression()
log_reg.fit(selected_samples, selected_labels)

# Evaluate model performance on the training/validation set
predictions = log_reg.predict(selected_samples)
accuracy = np.mean(predictions == selected_labels)  # Update according to your label transformation
print(f'Accuracy on training/validation set: {accuracy:.2f}')

# Step 6: Evaluate the model on the remaining test set
remaining_indices = np.isin(range(len(X)), np.array(selected_samples))
test_samples = X[~remaining_indices]
test_labels = y[~remaining_indices]

# Predict on test set
test_predictions = log_reg.predict(test_samples)
test_accuracy = np.mean(test_predictions == test_labels)  # Update according to your label transformation
print(f'Accuracy on test set: {test_accuracy:.2f}')


# Step 7: Output Analysis
def output_analysis(clusters, test_labels, test_predictions):
    # Calculate the confusion matrix
    confusion_matrix = pd.crosstab(test_labels, test_predictions, rownames=['Actual'], colnames=['Predicted'],
                                   margins=True)

    # Summarize results
    summary = {
        'Training Accuracy': accuracy,
        'Test Accuracy': test_accuracy,
        'Confusion Matrix': confusion_matrix
    }
    return summary


summary_results = output_analysis(clusters, test_labels, test_predictions)


# Step 8: Save results to Excel file
def save_to_excel(summary, filename='output_analysis.xlsx'):
    with pd.ExcelWriter(filename) as writer:
        # Save confusion matrix
        summary['Confusion Matrix'].to_excel(writer, sheet_name='Confusion Matrix')
        # Save summary metrics
        pd.DataFrame({'Metric': ['Training Accuracy', 'Test Accuracy'],
                      'Value': [summary['Training Accuracy'], summary['Test Accuracy']]}).to_excel(writer,
                                                                                                   sheet_name='Summary')

    print(f"Results saved to {filename}")


# Save the output analysis to an Excel file
save_to_excel(summary_results)

print("Output analysis completed and saved to Excel.")
