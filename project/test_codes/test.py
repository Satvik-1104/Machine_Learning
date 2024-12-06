import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk
from tkinter import ttk, messagebox, font
from sklearn.preprocessing import StandardScaler
import tkinter.scrolledtext as scrolledtext

class MigrationPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Migration Prediction Tool")
        self.root.geometry("800x800")  # Increased height for additional info
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
            
            # Data cleaning and preprocessing
            self.data['Value'] = pd.to_numeric(self.data['Value'], errors='coerce')
            self.data.fillna({
                "Value": self.data.groupby(['Country', 'Measure', 'Citizenship'])['Value'].transform('median')
            }, inplace=True)
            
            # Convert categorical variables
            self.data['Measure'] = self.data['Measure'].replace({
                "Arrivals": 0, "Departures": 1, "Net": 2
            })
            self.data['Citizenship'] = self.data['Citizenship'].replace({
                "New Zealand Citizen": 0, 
                "Australian Citizen": 1, 
                "Total All Citizenships": 2
            })
            
            # Create country mapping with validation
            self.validate_country_data()
            
        except Exception as e:
            messagebox.showerror("Data Loading Error", f"Error loading data: {str(e)}")
            raise

    def validate_country_data(self):
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

    def setup_model(self):
        # Prepare features
        X = self.data[['Country', 'Measure', 'Year', 'Citizenship']]
        y = self.data['Value']

        # Calculate trends properly by group
        recent_trends = []
        for idx, row in X.iterrows():
            mask = (
                (self.data['Country'] == row['Country']) &
                (self.data['Measure'] == row['Measure']) &
                (self.data['Citizenship'] == row['Citizenship']) &
                (self.data['Year'] < row['Year'])
            )
            recent_values = self.data[mask]['Value'].tail(5).mean()
            recent_trends.append(recent_values if not pd.isna(recent_values) else 0)

        X['recent_trend'] = recent_trends

        # Add year-based features
        X['year_squared'] = X['Year'] ** 2
        X['years_from_present'] = X['Year'] - X['Year'].max()

        # Handle 'All countries' differently
        X['is_total'] = (X['Country'] == self.country_mapping.get('All countries', -1)).astype(int)

        # Scale features independently
        self.scaler = StandardScaler()
        numeric_cols = ['Year', 'recent_trend', 'year_squared', 'years_from_present']
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.20, 
            random_state=42,
            stratify=X[['Measure', 'Citizenship']]
        )

        # Use a more robust model configuration
        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Fit model and store feature importances
        self.model.fit(X_train, y_train)
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Store validation metrics
        y_pred_test = self.model.predict(X_test)
        self.model_metrics = {
            'r2': r2_score(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mean_absolute_error': np.mean(np.abs(y_test - y_pred_test))
        }

    def create_gui(self):
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

        # Input container with validation
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
        country_combo.set(self.countries[0])  # Set default value

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
        measure_combo.set(measures[0])  # Set default value

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
        citizenship_combo.set(citizenships[0])  # Set default value

        # Year Selection with validation
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
        current_year = pd.Timestamp.now().year
        self.year_var.set(str(current_year))  # Set default to current year

        # Predict Button
        predict_btn = ttk.Button(
            input_frame,
            text="Predict Migration",
            command=self.predict,
            style='Custom.TButton'
        )
        predict_btn.pack(pady=20)

        # Results Display with additional context
        self.result_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        self.result_frame.pack(fill='both', expand=True, pady=20)

        self.result_text = scrolledtext.ScrolledText(
            self.result_frame,
            height=15,  # Increased height for more information
            font=('Helvetica', 12),
            wrap=tk.WORD,
            bg='#ffffff'
        )
        self.result_text.pack(fill='both', expand=True)

    def predict(self):
        try:
            # Get and validate input values
            country = self.country_var.get()
            measure = int(self.measure_var.get().split('(')[1].split(')')[0])
            citizenship = int(self.citizenship_var.get().split('(')[1].split(')')[0])
            year = int(self.year_var.get())
            
            # Validate year
            if year < 1900 or year > 2050:
                raise ValueError("Year must be between 1900 and 2050")
            
            country_encoded = self.country_mapping[country]
            
            # Get historical data with proper filtering
            historical = self.data[
                (self.data['Country'] == country_encoded) & 
                (self.data['Measure'] == measure) & 
                (self.data['Citizenship'] == citizenship)
            ].sort_values('Year')
            
            # Calculate recent trend more accurately
            recent_avg = historical['Value'].tail(5).mean() if not historical.empty else 0
            
            # Prepare input data with all features
            input_data = pd.DataFrame([[
                country_encoded,
                measure,
                year,
                citizenship,
                recent_avg,
                year ** 2,
                year - self.data['Year'].max(),
                1 if country == 'All countries' else 0
            ]], columns=['Country', 'Measure', 'Year', 'Citizenship', 'recent_trend', 
                        'year_squared', 'years_from_present', 'is_total'])
            
            # Scale numeric features
            numeric_cols = ['Year', 'recent_trend', 'year_squared', 'years_from_present']
            input_data[numeric_cols] = self.scaler.transform(input_data[numeric_cols])
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            # Validate prediction
            if prediction < 0:
                prediction = 0
                warning = "\nWarning: Negative prediction adjusted to 0"
            else:
                warning = ""
            
            # Calculate prediction deviation from historical average
            if not historical.empty:
                deviation = abs(prediction - recent_avg) / recent_avg * 100
                if deviation > 50:  # More than 50% deviation
                    warning += f"\nWarning: Prediction deviates {deviation:.1f}% from recent average"
            
            # Format results with additional context
            result_text = f"""
Prediction Results:
------------------
Country: {country}
Measure: {['Arrivals', 'Departures', 'Net'][measure]}
Year: {year}
Citizenship: {['New Zealand', 'Australian', 'All'][citizenship]}

Predicted migrants: {int(prediction):,}{warning}

Model Metrics:
R² Score: {self.model_metrics['r2']:.3f}
RMSE: {self.model_metrics['rmse']:.0f}
Mean Absolute Error: {self.model_metrics['mean_absolute_error']:.0f}
"""
            
            # Add historical context
            if not historical.empty:
                last_5_years = historical.tail(5)
                avg_5year = last_5_years['Value'].mean()
                result_text += f"\nHistorical Context:"
                result_text += f"\n- Last 5 years average: {int(avg_5year):,}"
                
                max_year = last_5_years['Year'].max()
                last_value = last_5_years.loc[last_5_years['Year'] == max_year, 'Value'].iloc[0]
                result_text += f"\n- Last recorded value ({int(max_year)}): {int(last_value):,}"
                
                # Add trend information
                trend = last_5_years['Value'].pct_change().mean() * 100
                trend_direction = '↑' if trend > 0 else '↓'
                result_text += f"\n- Recent trend: {trend_direction} {abs(trend):.1f}% per year"
                
                # Add prediction context
                if year > max_year:
                    result_text += f"\n- Forecasting {year - max_year} years into the future"
                elif year < max_year:
                    result_text += f"\n- Backcasting {max_year - year} years into the past"
            
            # Add feature importance information
            result_text += "\n\nTop influencing factors:"
            top_features = self.feature_importance.head(3)
            for _, row in top_features.iterrows():
                result_text += f"\n- {row['feature']}: {row['importance']:.3f}"
            
            # Update result display
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            
        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def validate_inputs(self, country, measure, citizenship, year):
        """Validate all input parameters"""
        if country not in self.countries:
            raise ValueError("Invalid country selection")
        
        if measure not in [0, 1, 2]:
            raise ValueError("Invalid measure selection")
            
        if citizenship not in [0, 1, 2]:
            raise ValueError("Invalid citizenship selection")
            
        if not isinstance(year, int):
            raise ValueError("Year must be an integer")
            
        if year < 1900 or year > 2050:
            raise ValueError("Year must be between 1900 and 2050")
            
        return True

    def plot_historical_trend(self):
        """Create a plot of historical trends for the selected parameters"""
        try:
            country = self.country_var.get()
            measure = int(self.measure_var.get().split('(')[1].split(')')[0])
            citizenship = int(self.citizenship_var.get().split('(')[1].split(')')[0])
            
            historical = self.data[
                (self.data['Country'] == self.country_mapping[country]) & 
                (self.data['Measure'] == measure) & 
                (self.data['Citizenship'] == citizenship)
            ].sort_values('Year')
            
            if historical.empty:
                messagebox.showwarning("No Data", "No historical data available for selected parameters")
                return
                
            plt.figure(figsize=(10, 6))
            plt.plot(historical['Year'], historical['Value'], marker='o')
            plt.title(f"Historical Migration Trend - {country}")
            plt.xlabel("Year")
            plt.ylabel("Number of Migrants")
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error creating plot: {str(e)}")

    def export_results(self):
        """Export prediction results to CSV"""
        try:
            results = self.result_text.get(1.0, tk.END)
            if not results.strip():
                messagebox.showwarning("No Data", "No results to export")
                return
                    
            filename = f"migration_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:  # Explicitly set encoding to utf-8
                f.write(results)
                    
            messagebox.showinfo("Success", f"Results exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting results: {str(e)}")


def main():
    try:
        root = tk.Tk()
        app = MigrationPredictorApp(root)
        
        # Add menu bar
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Results", command=app.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Plot Historical Trend", command=app.plot_historical_trend)
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        raise

if __name__ == "__main__":
    main()