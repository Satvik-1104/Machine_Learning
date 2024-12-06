import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import StandardScaler
import tkinter.scrolledtext as scrolledtext
from datetime import datetime

class MigrationPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Migration Prediction Tool")
        self.root.geometry("800x800")
        self.root.configure(bg="#f0f8ff")
        
        # Initialize metrics storage
        self.model_metrics = {}
        self.feature_importance = None
        
        # Load and prepare data
        self.load_data()
        self.setup_model()
        self.create_gui()

    def load_data(self):
        try:
            self.data = pd.read_csv('migration_nz.csv')
            
            # Efficient data cleaning
            self.data['Value'] = pd.to_numeric(self.data['Value'], errors='coerce')
            
            # Fill missing values more efficiently
            categorical_cols = ['Country', 'Measure', 'Citizenship']
            self.data[categorical_cols] = self.data[categorical_cols].fillna(self.data[categorical_cols].mode().iloc[0])
            
            # Handle missing values in Value column using ffill/bfill
            self.data['Value'] = self.data.groupby(['Country', 'Measure', 'Citizenship'])['Value'].transform(
                lambda x: x.ffill().bfill()  # Fixed deprecated fillna method
            )
            
            # Simplified mappings
            # In load_data() method, update these lines:
            self.data['Measure'] = self.data['Measure'].replace({
                "Arrivals": 0, "Departures": 1, "Net": 2
            }).astype(int)  # Add explicit type conversion

            self.data['Citizenship'] = self.data['Citizenship'].replace({
                "New Zealand Citizen": 0, 
                "Australian Citizen": 1, 
                "Total All Citizenships": 2
            }).astype(int)  # Add explicit type conversion
            
            self.data['Year'] = pd.to_numeric(self.data['Year'], errors='coerce')
            self.data = self.data.sort_values('Year')
            
            # Validate country data
            self.validate_country_data()
            
        except Exception as e:
            messagebox.showerror("Data Loading Error", f"Error loading data: {str(e)}")
            raise

    def validate_country_data(self):
        """Validate and prepare country data"""
        # Ensure 'All countries' exists and handle duplicates
        if 'All countries' not in self.data['Country'].unique():
            # Create 'All countries' aggregation if it doesn't exist
            all_countries_data = self.data.groupby(['Year', 'Measure', 'Citizenship'])['Value'].sum().reset_index()
            all_countries_data['Country'] = 'All countries'
            self.data = pd.concat([self.data, all_countries_data], ignore_index=True)
        
        # Create and validate country mapping
        self.country_mapping = pd.Series(
            self.data['Country'].factorize()[0], 
            index=self.data['Country']
        ).to_dict()
        self.reverse_country_mapping = {v: k for k, v in self.country_mapping.items()}
        self.data['Country'] = self.data['Country'].map(self.country_mapping)
        
        # Get unique countries for dropdown
        self.countries = sorted(self.country_mapping.keys())

    def create_gui(self):
        """Create the graphical user interface"""
        # Style configuration
        style = ttk.Style()
        style.configure('Custom.TFrame', background='#f0f8ff')
        style.configure('Custom.TLabel', 
                    background='#f0f8ff',
                    font=('Helvetica', 12))
        style.configure('Title.TLabel',
                    background='#f0f8ff',
                    font=('Helvetica', 24, 'bold'),
                    foreground='#2c3e50')
        style.configure('Custom.TButton',
                    font=('Helvetica', 12),
                    padding=10)

        # Main container
        main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title with model performance
        title_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        title_frame.pack(fill='x', pady=20)
        
        title = ttk.Label(
            title_frame,
            text="Migration Flow Predictor",
            style='Title.TLabel'
        )
        title.pack()
        
        if self.model_metrics:
            metrics_text = f"Model Performance: R² = {self.model_metrics['r2']:.3f}"
            metrics_label = ttk.Label(
                title_frame,
                text=metrics_text,
                style='Custom.TLabel'
            )
            metrics_label.pack()

        # Input container
        input_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        input_frame.pack(fill='x', padx=50)

        # Country Selection
        ttk.Label(
            input_frame,
            text="Select Country:",
            style='Custom.TLabel'
        ).pack(anchor='w', pady=(10,0))
        
        self.country_var = tk.StringVar()
        country_combo = ttk.Combobox(
            input_frame,
            textvariable=self.country_var,
            values=self.countries,
            state='readonly',
            width=30
        )
        country_combo.pack(fill='x', pady=(5,10))
        country_combo.set(self.countries[0])

        # Measure Selection
        ttk.Label(
            input_frame,
            text="Select Measure:",
            style='Custom.TLabel'
        ).pack(anchor='w', pady=(10,0))
        
        measures = ["Arrivals (0)", "Departures (1)", "Net (2)"]
        self.measure_var = tk.StringVar()
        measure_combo = ttk.Combobox(
            input_frame,
            textvariable=self.measure_var,
            values=measures,
            state='readonly',
            width=30
        )
        measure_combo.pack(fill='x', pady=(5,10))
        measure_combo.set(measures[0])

        # Citizenship Selection
        ttk.Label(
            input_frame,
            text="Select Citizenship:",
            style='Custom.TLabel'
        ).pack(anchor='w', pady=(10,0))
        
        citizenships = ["New Zealand Citizen (0)", "Australian Citizen (1)", "All Citizenships (2)"]
        self.citizenship_var = tk.StringVar()
        citizenship_combo = ttk.Combobox(
            input_frame,
            textvariable=self.citizenship_var,
            values=citizenships,
            state='readonly',
            width=30
        )
        citizenship_combo.pack(fill='x', pady=(5,10))
        citizenship_combo.set(citizenships[0])

        # Year Selection
        ttk.Label(
            input_frame,
            text="Enter Year (1900-2050):",
            style='Custom.TLabel'
        ).pack(anchor='w', pady=(10,0))
        
        self.year_var = tk.StringVar()
        self.year_entry = ttk.Entry(
            input_frame,
            textvariable=self.year_var,
            width=32
        )
        self.year_entry.pack(fill='x', pady=(5,10))
        self.year_var.set(str(datetime.now().year))

        # Predict Button
        predict_btn = ttk.Button(
            input_frame,
            text="Predict Migration",
            command=self.predict,
            style='Custom.TButton'
        )
        predict_btn.pack(pady=20)

        # Results Display
        self.result_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        self.result_frame.pack(fill='both', expand=True, pady=20)

        self.result_text = scrolledtext.ScrolledText(
            self.result_frame,
            height=15,
            font=('Helvetica', 12),
            wrap=tk.WORD,
            bg='#ffffff'
        )
        self.result_text.pack(fill='both', expand=True)

    def add_enhanced_features(self, X):
        # Vectorized calculations for enhanced features
        X['year_squared'] = X['Year'] ** 2
        X['year_cubed'] = X['Year'] ** 3
        X['years_from_present'] = X['Year'] - datetime.now().year
        X['years_from_present_squared'] = X['years_from_present'] ** 2
        X['year_sin'] = np.sin(2 * np.pi * X['Year'] / 10)
        X['year_cos'] = np.cos(2 * np.pi * X['Year'] / 10)
        
        # Efficient period indicators
        X['pre_1979_factor'] = 1 / (1 + np.exp((X['Year'] - 1979) / 2))
        X['post_2016_factor'] = 1 / (1 + np.exp((2016 - X['Year']) / 2))
        
        return X

    def setup_model(self):
        # Prepare features more efficiently
        X = self.data[['Country', 'Measure', 'Year', 'Citizenship']]
        y = self.data['Value']
        
        # Add enhanced features
        X = self.add_enhanced_features(X)
        
        # Scale features
        self.scaler = StandardScaler()
        numeric_cols = [col for col in X.columns if col not in ['Country', 'Measure', 'Citizenship']]
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        
        # Optimized model configuration
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )
        
        # Split and fit
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
        # Simple exponential weights
        weights = np.exp(np.linspace(-1, 0, len(y_train)))
        self.model.fit(X_train, y_train, sample_weight=weights)
        
        # Calculate metrics
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        y_pred_test = self.model.predict(X_test)
        self.model_metrics = {
            'r2': r2_score(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'oob_score': self.model.oob_score_
        }

    def predict(self):
        try:
            country = self.country_var.get()
            measure = int(self.measure_var.get().split('(')[1].split(')')[0])
            citizenship = int(self.citizenship_var.get().split('(')[1].split(')')[0])
            year = int(self.year_var.get())
            
            if year < 1900 or year > 2050:
                raise ValueError("Year must be between 1900 and 2050")
            
            # Prepare prediction data efficiently
            pred_data = pd.DataFrame([[
                self.country_mapping[country], measure, year, citizenship
            ]], columns=['Country', 'Measure', 'Year', 'Citizenship'])
            
            # Add features and scale
            pred_data = self.add_enhanced_features(pred_data)
            numeric_cols = [col for col in pred_data.columns if col not in ['Country', 'Measure', 'Citizenship']]
            pred_data[numeric_cols] = self.scaler.transform(pred_data[numeric_cols])
            
            # Make prediction
            prediction = self.model.predict(pred_data)[0]
            
            # Simplified results display
            result_text = f"""
Prediction Results:
------------------
Country: {country}
Measure: {["Arrivals", "Departures", "Net"][measure]}
Year: {year}
Citizenship: {["New Zealand", "Australian", "All"][citizenship]} Citizen{'s' if citizenship == 2 else ''}

Predicted migrants: {int(prediction):,}

Model Performance:
R² Score: {self.model_metrics['r2']:.3f}
RMSE: {self.model_metrics['rmse']:.0f}
"""
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            
        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = MigrationPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()