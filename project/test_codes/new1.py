import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing
def preprocess_data(data):
    # Create copy of data
    df = data.copy()
    
    # Handle missing values
    df['Value'] = df['Value'].fillna(df.groupby(['Measure', 'Country', 'Citizenship'])['Value'].transform('mean'))
    
    # Create label encoders for categorical columns
    encoders = {}
    for col in ['Measure', 'Country', 'Citizenship']:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    
    # Create features and target
    X = df[['Measure', 'Country', 'Citizenship', 'Year']]
    y = df['Value']
    
    # Split data into train, validation and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, 
            encoders, scaler)

# Model Training and Evaluation
def train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test):
    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Train R2': r2_score(y_train, y_pred_train),
        'Validation R2': r2_score(y_val, y_pred_val),
        'Test R2': r2_score(y_test, y_pred_test),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Validation RMSE': np.sqrt(mean_squared_error(y_val, y_pred_val)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Train MAE': mean_absolute_error(y_train, y_pred_train),
        'Validation MAE': mean_absolute_error(y_val, y_pred_val),
        'Test MAE': mean_absolute_error(y_test, y_pred_test)
    }
    
    return model, metrics, y_pred_test

# Visualization Functions
def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# Prediction Function for New Data
def predict_for_new_input(model, encoders, scaler, measure, country, citizenship, year):
    # Encode categorical inputs
    encoded_measure = encoders['Measure'].transform([measure])[0]
    encoded_country = encoders['Country'].transform([country])[0]
    encoded_citizenship = encoders['Citizenship'].transform([citizenship])[0]
    
    # Create input array
    X_new = np.array([[encoded_measure, encoded_country, encoded_citizenship, year]])
    
    # Scale input
    X_new_scaled = scaler.transform(X_new)
    
    # Make prediction
    prediction = model.predict(X_new_scaled)[0]
    
    return prediction

def plot_migration_trends(data):
    """Create comprehensive visualizations of migration trends"""
    plt.style.use('seaborn')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Overall yearly trends by measure
    plt.subplot(2, 2, 1)
    yearly_measure = data.groupby(['Year', 'Measure'])['Value'].sum().reset_index()
    for measure in yearly_measure['Measure'].unique():
        measure_data = yearly_measure[yearly_measure['Measure'] == measure]
        plt.plot(measure_data['Year'], measure_data['Value'], label=measure, marker='o')
    plt.title('Overall Migration Trends by Measure', fontsize=12)
    plt.xlabel('Year')
    plt.ylabel('Total Value')
    plt.legend()
    plt.grid(True)
    
    # 2. Top 10 countries by total migration volume
    plt.subplot(2, 2, 2)
    top_countries = (data[data['Measure'] == 'Arrivals']
                    .groupby('Country')['Value']
                    .sum()
                    .sort_values(ascending=False)
                    .head(10))
    sns.barplot(x=top_countries.values, y=top_countries.index)
    plt.title('Top 10 Countries by Total Arrivals', fontsize=12)
    plt.xlabel('Total Arrivals')
    
    # 3. Citizenship distribution over time
    plt.subplot(2, 2, 3)
    citizenship_yearly = (data[data['Measure'] == 'Arrivals']
                         .groupby(['Year', 'Citizenship'])['Value']
                         .sum()
                         .reset_index())
    for citizenship in citizenship_yearly['Citizenship'].unique():
        citizenship_data = citizenship_yearly[citizenship_yearly['Citizenship'] == citizenship]
        plt.plot(citizenship_data['Year'], citizenship_data['Value'], 
                label=citizenship, marker='o')
    plt.title('Yearly Trends by Citizenship', fontsize=12)
    plt.xlabel('Year')
    plt.ylabel('Total Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # 4. Net migration heatmap for top countries
    plt.subplot(2, 2, 4)
    net_migration = data[data['Measure'] == 'Net']
    top_countries_net = (net_migration.groupby('Country')['Value']
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                        .index)
    
    pivot_data = net_migration[net_migration['Country'].isin(top_countries_net)].pivot_table(
        index='Country',
        columns='Year',
        values='Value',
        aggfunc='sum'
    )
    
    sns.heatmap(pivot_data, cmap='RdYlBu', center=0)
    plt.title('Net Migration Heatmap for Top Countries', fontsize=12)
    plt.xlabel('Year')
    plt.ylabel('Country')
    
    plt.tight_layout()
    plt.show()

def plot_regional_analysis(data):
    """Create visualizations for regional migration patterns"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Regional distribution of arrivals
    plt.subplot(1, 2, 1)
    regions = ['Oceania', 'Asia', 'Europe', 'Americas', 'Africa and the Middle East']
    regional_data = (data[data['Measure'] == 'Arrivals']
                    .groupby('Country')['Value']
                    .sum()
                    .reset_index())
    
    regional_totals = []
    for region in regions:
        region_countries = [country for country in data['Country'].unique() 
                          if region in country or country == region]
        regional_total = regional_data[regional_data['Country'].isin(region_countries)]['Value'].sum()
        regional_totals.append(regional_total)
    
    plt.pie(regional_totals, labels=regions, autopct='%1.1f%%')
    plt.title('Distribution of Total Arrivals by Region', fontsize=12)
    
    # 2. Yearly trends by region
    plt.subplot(1, 2, 2)
    yearly_regional = []
    for region in regions:
        region_countries = [country for country in data['Country'].unique() 
                          if region in country or country == region]
        region_data = (data[(data['Measure'] == 'Arrivals') & 
                           (data['Country'].isin(region_countries))]
                      .groupby('Year')['Value']
                      .sum()
                      .reset_index())
        region_data['Region'] = region
        yearly_regional.append(region_data)
    
    yearly_regional_df = pd.concat(yearly_regional)
    for region in regions:
        region_data = yearly_regional_df[yearly_regional_df['Region'] == region]
        plt.plot(region_data['Year'], region_data['Value'], label=region, marker='o')
    
    plt.title('Yearly Migration Trends by Region', fontsize=12)
    plt.xlabel('Year')
    plt.ylabel('Total Arrivals')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_seasonal_patterns(data):
    """Analyze and visualize seasonal patterns in migration"""
    plt.figure(figsize=(15, 6))
    
    # Calculate average values by year
    yearly_avg = (data[data['Measure'] == 'Arrivals']
                 .groupby('Year')['Value']
                 .mean()
                 .reset_index())
    
    # Plot trend with rolling average
    plt.plot(yearly_avg['Year'], yearly_avg['Value'], label='Yearly Average', alpha=0.5)
    plt.plot(yearly_avg['Year'], 
            yearly_avg['Value'].rolling(window=5).mean(), 
            label='5-Year Moving Average',
            linewidth=2)
    
    plt.title('Long-term Migration Trends with Moving Average', fontsize=12)
    plt.xlabel('Year')
    plt.ylabel('Average Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution function
def run_migration_analysis(data):
    # Preprocess data
    print("Preprocessing data...")
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, encoders, scaler = preprocess_data(data)
    
    # Train and evaluate model
    print("\nTraining and evaluating model...")
    model, metrics, y_pred_test = train_and_evaluate_model(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test
    )
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create visualizations
    print("\nGenerating plots...")
    plot_actual_vs_predicted(y_test, y_pred_test)
    plot_residuals(y_test, y_pred_test)
    plot_feature_importance(model, ['Measure', 'Country', 'Citizenship', 'Year'])
    print("Generating migration trend visualizations...")
    plot_migration_trends(data)
    
    print("\nGenerating regional analysis...")
    plot_regional_analysis(data)
    
    print("\nAnalyzing seasonal patterns...")
    analyze_seasonal_patterns(data)
    
    return model, encoders, scaler

# Example usage
if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('migration_nz.csv')
    
    # Run analysis
    model, encoders, scaler = run_migration_analysis(data)
    
    # Example of making a prediction for new input
    prediction = predict_for_new_input(
        model, encoders, scaler,
        measure='Net',
        country='Asia',
        citizenship='New Zealand Citizen',
        year=2014
    )
    print(f"\nPredicted value for new input: {prediction:.2f}")