import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, make_scorer
import matplotlib.pyplot as plt
from datetime import datetime


def make_overfitting_scorer():
    """Create a custom scorer that penalizes overfitting"""

    def overfitting_scorer(estimator, X, y):
        # Use a fixed validation size that ensures at least 2 samples per class
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        estimator.fit(X_train, y_train)
        train_score = estimator.score(X_train, y_train)
        val_score = estimator.score(X_val, y_val)

        # Calculate penalty for gap between training and validation scores
        gap_penalty = abs(train_score - val_score)
        return val_score - gap_penalty

    return make_scorer(overfitting_scorer)


def load_and_prepare_data(dataset_path):
    """Load and prepare dataset for training"""
    df = pd.read_csv(dataset_path)

    if 'breast' in dataset_path.lower():
        X = df.drop(['id', 'diagnosis'], axis=1)
        y = df['diagnosis']
        dataset_title = "Breast Cancer"
    else:
        X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = df['Species']
        dataset_title = "Iris"

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, dataset_title


def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_title):
    """Train and evaluate multiple models with hyperparameter tuning"""
    results = {}
    n_classes = len(np.unique(y_train))

    # Adjusted parameters to prevent validation size issues
    mlp_params = {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu'],
        'learning_rate_init': [0.001],
        'alpha': [0.0001, 0.001],
        'early_stopping': [True],
        'validation_fraction': [0.3],  # Increased to ensure enough samples per class
        'max_iter': [1000]
    }

    slp_params = {
        'max_iter': [1000],
        'eta0': [0.1, 0.01],
        'penalty': ['l2'],
        'alpha': [0.0001, 0.001]
    }

    knn_params = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    models = {
        'SLP': (Perceptron(), slp_params),
        'MLP': (MLPClassifier(), mlp_params),
        'KNN': (KNeighborsClassifier(), knn_params)
    }

    # Train and evaluate each model
    for name, (model, params) in models.items():
        print(f"Training and evaluating {name} for {dataset_title}...")

        # Grid search with stratified CV
        grid = GridSearchCV(
            model, params, cv=3, scoring='balanced_accuracy',
            n_jobs=-1, verbose=1
        )

        try:
            grid.fit(X_train, y_train)

            # Store hyperparameter tuning results
            results[name] = {
                'best_params': grid.best_params_,
                'best_score': grid.best_score_,
                'model': grid.best_estimator_
            }

            # Evaluate on test set
            y_pred = grid.predict(X_test)

            # Calculate metrics with proper handling of edge cases
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )

            results[name].update({
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'classification_report': classification_report(
                    y_test, y_pred, zero_division=0, output_dict=True
                )
            })

        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            results[name] = {'error': str(e)}
            continue

    return results


def plot_learning_curves(models, X, y, dataset_title):
    """Plot learning curves with error handling and improved visualization"""
    plt.figure(figsize=(15, 5))

    for i, (name, model_data) in enumerate(models.items(), 1):
        if 'error' in model_data:
            continue

        model = model_data['model']
        train_sizes = np.linspace(0.1, 1.0, 5)

        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes,
                cv=3, n_jobs=-1, scoring='balanced_accuracy'
            )

            plt.subplot(1, len(models), i)

            # Plot mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            plt.plot(train_sizes, train_mean, label='Training score', color='blue')
            plt.fill_between(
                train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.1, color='blue'
            )

            plt.plot(train_sizes, val_mean, label='Cross-validation score', color='green')
            plt.fill_between(
                train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.1, color='green'
            )

            plt.title(f'{name} Learning Curve\n{dataset_title}')
            plt.xlabel('Training Examples')
            plt.ylabel('Balanced Accuracy')
            plt.legend(loc='lower right')
            plt.grid(True)

        except Exception as e:
            print(f"Error plotting learning curve for {name}: {str(e)}")
            continue

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'learning_curves_{dataset_title}_{timestamp}.png')
    plt.close()


def save_results(results, dataset_title):
    """Save results to Excel with improved error handling and formatting"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'ml_results_{dataset_title}_{timestamp}.xlsx'

    try:
        with pd.ExcelWriter(filename) as writer:
            # Overall metrics
            metrics_data = []
            for model_name, model_results in results.items():
                if 'error' in model_results:
                    continue

                metrics_data.append({
                    'Model': model_name,
                    'Best Parameters': str(model_results['best_params']),
                    'CV Score': model_results['best_score'],
                    'Test Accuracy': model_results['test_accuracy'],
                    'Test Precision': model_results['test_precision'],
                    'Test Recall': model_results['test_recall'],
                    'Test F1': model_results['test_f1']
                })

            pd.DataFrame(metrics_data).to_excel(
                writer, sheet_name='Overall_Metrics', index=False
            )

            # Detailed class metrics
            for model_name, model_results in results.items():
                if 'error' in model_results:
                    continue

                class_metrics = model_results['classification_report']
                metrics_df = pd.DataFrame(class_metrics).transpose()
                metrics_df.to_excel(
                    writer, sheet_name=f'{model_name}_Class_Metrics'
                )

    except Exception as e:
        print(f"Error saving results: {str(e)}")


def main():
    """Main execution function with error handling"""
    datasets = {
        'breast-cancer.csv': 'Breast Cancer',
        'iris.csv': 'Iris'
    }

    for dataset_path, dataset_name in datasets.items():
        try:
            print(f"\nProcessing {dataset_name} dataset...")

            # Load and prepare data
            X_train, X_test, y_train, y_test, title = load_and_prepare_data(dataset_path)

            # Train and evaluate models
            results = train_and_evaluate_models(
                X_train, X_test, y_train, y_test, title
            )

            # Plot learning curves
            plot_learning_curves(results, X_train, y_train, title)

            # Save results
            save_results(results, title)

        except Exception as e:
            print(f"Error processing {dataset_name} dataset: {str(e)}")
            continue


if __name__ == "__main__":
    main()