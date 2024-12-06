import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime


def load_and_prepare_data(dataset_path):
    # Read the dataset
    df = pd.read_csv(dataset_path)

    if 'breast' in dataset_path.lower():
        # For Breast Cancer dataset
        X = df.drop(['id', 'diagnosis'], axis=1)  # Remove id and diagnosis columns
        y = df['diagnosis']
        # Convert M/B to 1/0
        le = LabelEncoder()
        y = le.fit_transform(y)
        dataset_title = "Breast Cancer"
    else:
        # For Iris dataset
        X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]  # Select feature columns
        y = df['Species']
        # Encode species names to numbers
        le = LabelEncoder()
        y = le.fit_transform(y)
        dataset_title = "Iris"

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, dataset_title


def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_title):
    results = {}

    # 1. Single Layer Perceptron (SLP)
    print(f"Training SLP for {dataset_title}...")
    slp_params = {
        'max_iter': [1000, 2000],
        'eta0': [0.1, 0.01, 0.001],
        'penalty': [None, 'l2']
    }
    slp = Perceptron()
    slp_grid = GridSearchCV(slp, slp_params, cv=3, scoring='accuracy')
    slp_grid.fit(X_train, y_train)

    # 2. Multi-Layer Perceptron (MLP)
    print(f"Training MLP for {dataset_title}...")
    mlp_params = {
        'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01]
    }
    mlp = MLPClassifier(max_iter=1000)
    mlp_grid = GridSearchCV(mlp, mlp_params, cv=3, scoring='accuracy')
    mlp_grid.fit(X_train, y_train)

    # 3. K-Nearest Neighbors (KNN)
    print(f"Training KNN for {dataset_title}...")
    knn_params = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn, knn_params, cv=3, scoring='accuracy')
    knn_grid.fit(X_train, y_train)

    # Store best models
    models = {
        'SLP': slp_grid.best_estimator_,
        'MLP': mlp_grid.best_estimator_,
        'KNN': knn_grid.best_estimator_
    }

    # Store hyperparameter tuning results
    results['hyperparameter_tuning'] = {
        'SLP': {
            'best_params': slp_grid.best_params_,
            'best_score': slp_grid.best_score_
        },
        'MLP': {
            'best_params': mlp_grid.best_params_,
            'best_score': mlp_grid.best_score_
        },
        'KNN': {
            'best_params': knn_grid.best_params_,
            'best_score': knn_grid.best_score_
        }
    }

    # Check overfitting and create learning curves
    plot_learning_curves(models, X_train, y_train, dataset_title)

    # Perform 3-fold cross-validation and get metrics
    results['cross_validation'] = {}
    class_metrics = {}

    for name, model in models.items():
        print(f"Evaluating {name} for {dataset_title}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=3)
        y_pred = model.predict(X_test)

        # Get class-wise metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        class_metrics[name] = report

        # Calculate overall metrics
        precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        results['cross_validation'][name] = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy_score(y_test, y_pred),
            'class_metrics': class_metrics[name]
        }

    return results


def plot_learning_curves(models, X, y, dataset_title):
    plt.figure(figsize=(15, 5))

    for i, (name, model) in enumerate(models.items(), 1):
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=3, n_jobs=-1)

        plt.subplot(1, 3, i)
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Cross-validation score')
        plt.title(f'{name} Learning Curve\n{dataset_title} Dataset')
        plt.xlabel('Training Size')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'learning_curves_{dataset_title}_{timestamp}.png')
    plt.close()


def save_results_to_excel(results, dataset_title):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'ml_results_{dataset_title}_{timestamp}.xlsx'

    with pd.ExcelWriter(filename) as writer:
        # Save hyperparameter tuning results
        hp_tuning_data = []
        for model, data in results['hyperparameter_tuning'].items():
            hp_tuning_data.append({
                'Model': model,
                'Best Parameters': str(data['best_params']),
                'Best Score': data['best_score']
            })
        pd.DataFrame(hp_tuning_data).to_excel(writer, sheet_name='Hyperparameter_Tuning', index=False)

        # Save cross-validation results with class-wise metrics
        cv_data = []
        class_metrics_data = []

        for model, data in results['cross_validation'].items():
            # Overall metrics
            cv_data.append({
                'Model': model,
                'CV Mean Score': data['cv_mean'],
                'CV Std': data['cv_std'],
                'Overall Precision': data['precision'],
                'Overall Recall': data['recall'],
                'Overall Accuracy': data['accuracy']
            })

            # Class-wise metrics
            for class_label, metrics in data['class_metrics'].items():
                if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                    class_metrics_data.append({
                        'Model': model,
                        'Class': class_label,
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1-score'],
                        'Support': metrics['support']
                    })

        pd.DataFrame(cv_data).to_excel(writer, sheet_name='Overall_Metrics', index=False)
        pd.DataFrame(class_metrics_data).to_excel(writer, sheet_name='Class_Metrics', index=False)


def main():
    # Process Breast Cancer dataset
    print("Processing Breast Cancer dataset...")
    X_train_bc, X_test_bc, y_train_bc, y_test_bc, title_bc = load_and_prepare_data('breast-cancer.csv')
    results_bc = train_and_evaluate_models(X_train_bc, X_test_bc, y_train_bc, y_test_bc, title_bc)
    save_results_to_excel(results_bc, title_bc)

    # Process Iris dataset
    print("Processing Iris dataset...")
    X_train_iris, X_test_iris, y_train_iris, y_test_iris, title_iris = load_and_prepare_data('iris.csv')
    results_iris = train_and_evaluate_models(X_train_iris, X_test_iris, y_train_iris, y_test_iris, title_iris)
    save_results_to_excel(results_iris, title_iris)


if __name__ == "__main__":
    main()