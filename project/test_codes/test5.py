import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class MigrationAnalysisSystem:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.data = None
        
    def load_data(self, file_path):
        """Load and validate data"""
        self.data = pd.read_csv(file_path)
        print("\nData loaded successfully!")
        print("\nDataset Overview:")
        print(self.data.info())
        print("\nSample data:")
        print(self.data.head())
        
    def preprocess_data(self):
        """Preprocess the data with enhanced handling of categories"""
        df = self.data.copy()
        
        # Handle missing values
        for col in ['Measure', 'Country', 'Citizenship']:
            df[col] = df[col].fillna('Unknown')
        
        df['Value'] = df['Value'].fillna(df.groupby(['Measure', 'Country', 'Citizenship'])['Value'].transform('mean'))
        
        # Create label encoders for categorical columns
        categorical_columns = ['Measure', 'Country', 'Citizenship']
        for col in categorical_columns:
            self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col])
        
        # Create features and target
        X = df[['Measure', 'Country', 'Citizenship', 'Year']]
        y = df['Value']
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test)

    def train_model(self, X_train, y_train):
        """Train the RandomForest model"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        print("\nModel trained successfully!")

    def evaluate_model(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Evaluate model performance"""
        predictions = {
            'train': self.model.predict(X_train),
            'val': self.model.predict(X_val),
            'test': self.model.predict(X_test)
        }
        
        actuals = {
            'train': y_train,
            'val': y_val,
            'test': y_test
        }
        
        metrics = {}
        for dataset in ['train', 'val', 'test']:
            metrics[f'{dataset}_r2'] = r2_score(actuals[dataset], predictions[dataset])
            metrics[f'{dataset}_rmse'] = np.sqrt(mean_squared_error(actuals[dataset], predictions[dataset]))
            metrics[f'{dataset}_mae'] = mean_absolute_error(actuals[dataset], predictions[dataset])
        
        return metrics, predictions['test'], actuals['test']

    def plot_prediction_results(self, y_test, y_pred):
        """Plot prediction analysis results"""
        plt.figure(figsize=(15, 5))
        
        # Actual vs Predicted
        plt.subplot(131)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted')
        
        # Residuals
        plt.subplot(132)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Feature Importance
        plt.subplot(133)
        importance = pd.DataFrame({
            'feature': ['Measure', 'Country', 'Citizenship', 'Year'],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title('Feature Importance')
        
        plt.tight_layout()
        plt.show()

    def plot_migration_trends(self):
        """Plot migration trends analysis"""
        plt.style.use('seaborn')
        plt.figure(figsize=(20, 15))
        
        # Overall yearly trends by measure
        plt.subplot(2, 2, 1)
        yearly_measure = self.data.groupby(['Year', 'Measure'])['Value'].sum().reset_index()
        for measure in yearly_measure['Measure'].unique():
            measure_data = yearly_measure[yearly_measure['Measure'] == measure]
            plt.plot(measure_data['Year'], measure_data['Value'], label=measure, marker='o')
        plt.title('Overall Migration Trends by Measure')
        plt.xlabel('Year')
        plt.ylabel('Total Value')
        plt.legend()
        
        # Top 10 countries
        plt.subplot(2, 2, 2)
        top_countries = (self.data[self.data['Measure'] == 'Arrivals']
                        .groupby('Country')['Value']
                        .sum()
                        .sort_values(ascending=False)
                        .head(10))
        sns.barplot(x=top_countries.values, y=top_countries.index)
        plt.title('Top 10 Countries by Total Arrivals')
        
        # Citizenship trends
        plt.subplot(2, 2, 3)
        citizenship_yearly = (self.data[self.data['Measure'] == 'Arrivals']
                            .groupby(['Year', 'Citizenship'])['Value']
                            .sum()
                            .reset_index())
        for citizenship in citizenship_yearly['Citizenship'].unique():
            citizenship_data = citizenship_yearly[citizenship_yearly['Citizenship'] == citizenship]
            plt.plot(citizenship_data['Year'], citizenship_data['Value'], label=citizenship, marker='o')
        plt.title('Yearly Trends by Citizenship')
        plt.xlabel('Year')
        plt.ylabel('Total Value')
        plt.legend()
        
        # Regional analysis
        plt.subplot(2, 2, 4)
        regions = ['Oceania', 'Asia', 'Europe', 'Americas', 'Africa and the Middle East']
        regional_data = []
        for region in regions:
            region_total = self.data[
                (self.data['Measure'] == 'Arrivals') & 
                (self.data['Country'].str.contains(region, na=False))
            ]['Value'].sum()
            regional_data.append((region, region_total))
        
        regional_df = pd.DataFrame(regional_data, columns=['Region', 'Total'])
        sns.barplot(data=regional_df, x='Total', y='Region')
        plt.title('Total Arrivals by Region')
        
        plt.tight_layout()
        plt.show()

    def make_prediction(self):
        """Interactive prediction function"""
        # Get available options
        measures = self.data['Measure'].unique()
        countries = self.data['Country'].unique()
        citizenships = self.data['Citizenship'].unique()
        
        print("\nAvailable options:")
        print("\nMeasures:", measures)
        print("\nCitizenships:", citizenships)
        
        # Get user input
        while True:
            try:
                measure = input("\nEnter measure (Arrivals/Departures/Net): ")
                if measure not in measures:
                    raise ValueError("Invalid measure")
                
                country = input("Enter country: ")
                if country not in countries:
                    raise ValueError("Invalid country")
                
                citizenship = input("Enter citizenship: ")
                if citizenship not in citizenships:
                    raise ValueError("Invalid citizenship")
                
                year = int(input("Enter year (1979-2030): "))
                if year < 1979 or year > 2030:
                    raise ValueError("Year out of range")
                
                # Encode and scale input
                encoded_measure = self.encoders['Measure'].transform([measure])[0]
                encoded_country = self.encoders['Country'].transform([country])[0]
                encoded_citizenship = self.encoders['Citizenship'].transform([citizenship])[0]
                
                X_new = np.array([[encoded_measure, encoded_country, encoded_citizenship, year]])
                X_new_scaled = self.scaler.transform(X_new)
                
                # Make prediction
                prediction = self.model.predict(X_new_scaled)[0]
                prediction = max(0, prediction)  # Ensure non-negative
                
                print(f"\nPredicted {measure} for {citizenship} in {country} for {year}: {prediction:.2f}")
                break
                
            except ValueError as e:
                print(f"Error: {e}")
                if input("\nTry again? (y/n): ").lower() != 'y':
                    break

def main():
    """Main function to run the analysis system"""
    system = MigrationAnalysisSystem()
    
    # Get file path from user
    while True:
        file_path = input("Enter the path to your CSV file: ")
        try:
            system.load_data(file_path)
            break
        except Exception as e:
            print(f"Error loading file: {e}")
            if input("Try again? (y/n): ").lower() != 'y':
                return
    
    # Train model
    print("\nPreprocessing data and training model...")
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test = system.preprocess_data()
    system.train_model(X_train_scaled, y_train)
    
    # Evaluate model
    metrics, y_pred, y_test_actual = system.evaluate_model(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test
    )
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    print("\nGenerating visualizations...")
    system.plot_prediction_results(y_test_actual, y_pred)
    system.plot_migration_trends()
    
    # Interactive predictions
    while True:
        system.make_prediction()
        if input("\nMake another prediction? (y/n): ").lower() != 'y':
            break
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()