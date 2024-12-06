# Only linear regression - 50.7% accuracy
# Only Random forest - incorrect prediction for future values
# Combination of Random forest and Linear regression - more that 90% accuracy



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from scipy.stats import linregress
import threading
import queue


class MigrationAnalysisSystem:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.data = None
        self.trend_models = {}
        self.last_historical_year = 2016
        self.progress_callback = None
        self.cancel_flag = False

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def load_data(self, file_path):
        """Load and validate data with progress updates"""
        try:
            self.data = pd.read_csv(file_path)
            if self.progress_callback:
                self.progress_callback("Data loaded from CSV", 30)

            # Calculate trend models for future predictions
            self._calculate_trend_models()
            return self.data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def _calculate_trend_models(self):
        """Calculate trend models with progress updates"""
        if self.progress_callback:
            self.progress_callback("Calculating trend models...", 50)

        historical_data = self.data[self.data['Year'] <= self.last_historical_year].copy()
        total_combinations = len(historical_data['Measure'].unique()) * \
                             len(historical_data['Country'].unique()) * \
                             len(historical_data['Citizenship'].unique())
        processed = 0

        for measure in historical_data['Measure'].unique():
            if self.cancel_flag:
                return

            for country in historical_data['Country'].unique():
                for citizenship in historical_data['Citizenship'].unique():
                    mask = (historical_data['Measure'] == measure) & \
                           (historical_data['Country'] == country) & \
                           (historical_data['Citizenship'] == citizenship)

                    subset = historical_data[mask]

                    if len(subset) > 1:
                        years = subset['Year'].values
                        values = subset['Value'].values
                        slope, intercept, r_value, p_value, std_err = linregress(years, values)
                        key = (measure, country, citizenship)
                        self.trend_models[key] = {
                            'slope': slope,
                            'intercept': intercept,
                            'last_value': values[-1] if len(values) > 0 else 0,
                            'last_year': years[-1] if len(years) > 0 else self.last_historical_year
                        }

                    processed += 1
                    if self.progress_callback and processed % 100 == 0:
                        progress = 50 + (processed / total_combinations * 50)
                        self.progress_callback(f"Processing trends... ({processed}/{total_combinations})", progress)

        if self.progress_callback:
            self.progress_callback("Trend calculations complete", 100)

    def preprocess_data(self):
        """Preprocess the data with enhanced handling of categories"""
        df = self.data.copy()

        # Filter for historical data only for training
        df_historical = df[df['Year'] <= self.last_historical_year].copy()

        # Handle missing values
        for col in ['Measure', 'Country', 'Citizenship']:
            df_historical[col] = df_historical[col].fillna('Unknown')

        df_historical['Value'] = df_historical['Value'].fillna(
            df_historical.groupby(['Measure', 'Country', 'Citizenship'])['Value'].transform('mean'))

        # Create label encoders for categorical columns
        categorical_columns = ['Measure', 'Country', 'Citizenship']
        for col in categorical_columns:
            self.encoders[col] = LabelEncoder()
            df_historical[col] = self.encoders[col].fit_transform(df_historical[col])

        # Create features and target
        X = df_historical[['Measure', 'Country', 'Citizenship', 'Year']]
        y = df_historical['Value']

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

    def predict_value(self, measure, country, citizenship, year):
        """Make prediction with trend-based forecasting for future years"""
        if year <= self.last_historical_year:
            # Use the trained model for historical predictions
            encoded_measure = self.encoders['Measure'].transform([measure])[0]
            encoded_country = self.encoders['Country'].transform([country])[0]
            encoded_citizenship = self.encoders['Citizenship'].transform([citizenship])[0]

            X_new = np.array([[encoded_measure, encoded_country, encoded_citizenship, year]])
            X_new_scaled = self.scaler.transform(X_new)
            prediction = self.model.predict(X_new_scaled)[0]
        else:
            # Use trend-based forecasting for future predictions
            key = (measure, country, citizenship)
            if key in self.trend_models:
                trend_model = self.trend_models[key]
                years_ahead = year - trend_model['last_year']

                # Calculate base prediction using trend
                base_prediction = trend_model['slope'] * year + trend_model['intercept']

                # Apply dampening factor for long-term predictions
                dampening_factor = 1 / (1 + 0.1 * years_ahead)  # Adjust the 0.1 to control dampening strength

                # Combine trend with dampening
                prediction = trend_model['last_value'] + \
                             (base_prediction - trend_model['last_value']) * dampening_factor
            else:
                # If no trend model exists, use the average value from historical data
                mask = (self.data['Measure'] == measure) & \
                       (self.data['Country'] == country) & \
                       (self.data['Citizenship'] == citizenship)
                prediction = self.data[mask]['Value'].mean()

        # Ensure prediction is non-negative
        prediction = max(0, prediction)

        # Add random variation for future predictions to make them more realistic
        if year > self.last_historical_year:
            variation = np.random.normal(0, prediction * 0.05)  # 5% random variation
            prediction = max(0, prediction + variation)

        return prediction


class MigrationAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Migration Analysis System")
        self.root.geometry("1200x800")

        # Initialize the analysis system
        self.system = MigrationAnalysisSystem()
        self.system.set_progress_callback(self.update_progress)

        # Create main containers
        self.create_gui_elements()

        # Add progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.control_frame,
                                      variable=self.progress_var,
                                      maximum=100,
                                      length=200,
                                      mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=5)
        self.progress.pack_forget()  # Hide initially

    def create_gui_elements(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.root, padding="5")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.prediction_plots_tab = ttk.Frame(self.notebook)
        self.trends_plots_tab = ttk.Frame(self.notebook)
        self.metrics_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.prediction_plots_tab, text="Prediction Analysis")
        self.notebook.add(self.trends_plots_tab, text="Migration Trends")
        self.notebook.add(self.metrics_tab, text="Model Metrics")
        self.notebook.add(self.prediction_tab, text="Make Predictions")

        # Add control elements
        self.create_controls()
        self.create_prediction_plots_tab()
        self.create_trends_plots_tab()
        self.create_metrics_tab()
        self.create_prediction_tab()

    def create_controls(self):
        # Load Data Button
        self.load_btn = ttk.Button(self.control_frame, text="Load Data", command=self.load_data)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Train Model Button
        self.train_btn = ttk.Button(self.control_frame, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        # Status Label
        self.status_var = tk.StringVar(value="Please load data to begin")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)

    def create_prediction_plots_tab(self):
        # Create frame for prediction analysis plots
        self.prediction_plot_frame = ttk.Frame(self.prediction_plots_tab)
        self.prediction_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_trends_plots_tab(self):
        # Create frame for migration trends plots
        self.trends_plot_frame = ttk.Frame(self.trends_plots_tab)
        self.trends_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_metrics_tab(self):
        # Create text widget for metrics
        self.metrics_text = ScrolledText(self.metrics_tab, height=20)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_prediction_tab(self):
        # Create prediction form
        form_frame = ttk.Frame(self.prediction_tab)
        form_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # Input fields
        self.measure_var = tk.StringVar()
        self.country_var = tk.StringVar()
        self.citizenship_var = tk.StringVar()
        self.year_var = tk.StringVar()

        # Labels and entries
        ttk.Label(form_frame, text="Measure:").grid(row=0, column=0, padx=5, pady=5)
        self.measure_combo = ttk.Combobox(form_frame, textvariable=self.measure_var, state="readonly")
        self.measure_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Country:").grid(row=1, column=0, padx=5, pady=5)
        self.country_combo = ttk.Combobox(form_frame, textvariable=self.country_var, state="readonly")
        self.country_combo.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Citizenship:").grid(row=2, column=0, padx=5, pady=5)
        self.citizenship_combo = ttk.Combobox(form_frame, textvariable=self.citizenship_var, state="readonly")
        self.citizenship_combo.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Year (1979-2030):").grid(row=3, column=0, padx=5, pady=5)
        self.year_entry = ttk.Entry(form_frame, textvariable=self.year_var)
        self.year_entry.grid(row=3, column=1, padx=5, pady=5)

        # Predict button
        self.predict_btn = ttk.Button(form_frame, text="Make Prediction", command=self.make_prediction,
                                    state=tk.DISABLED)
        self.predict_btn.grid(row=4, column=0, columnspan=2, pady=20)

        # Results
        self.prediction_result = ttk.Label(form_frame, text="")
        self.prediction_result.grid(row=5, column=0, columnspan=2, pady=10)

        # Add note about predictions
        note_text = "Note: Predictions after 2016 are based on historical trends and include uncertainty factors"
        self.prediction_note = ttk.Label(form_frame, text=note_text, font=('Helvetica', 8, 'italic'))
        self.prediction_note.grid(row=6, column=0, columnspan=2, pady=5)

    def update_progress(self, message, value):
        """Update progress bar and status message"""
        self.progress_var.set(value)
        self.status_var.set(message)
        self.root.update_idletasks()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            # Disable controls during loading
            self.load_btn.config(state=tk.DISABLED)
            self.train_btn.config(state=tk.DISABLED)
            self.progress.pack(side=tk.LEFT, padx=5)  # Show progress bar

            # Create and start loading thread
            def loading_thread():
                try:
                    self.system.load_data(file_path)
                    self.root.after(0, self.loading_complete)
                except Exception as e:
                    self.root.after(0, lambda: self.loading_error(str(e)))

            thread = threading.Thread(target=loading_thread)
            thread.daemon = True
            thread.start()

    def loading_complete(self):
        """Called when loading is complete"""
        self.status_var.set("Data loaded successfully!")
        self.load_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.NORMAL)
        self.progress.pack_forget()  # Hide progress bar
        self.update_prediction_options()

    def loading_error(self, error_message):
        """Called when loading encounters an error"""
        messagebox.showerror("Error", f"Error loading file: {error_message}")
        self.load_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.DISABLED)
        self.progress.pack_forget()
        self.status_var.set("Error loading data")

    def update_prediction_options(self):
        # Update combobox options
        self.measure_combo['values'] = list(self.system.data['Measure'].unique())
        self.country_combo['values'] = list(self.system.data['Country'].unique())
        self.citizenship_combo['values'] = list(self.system.data['Citizenship'].unique())

    def train_model(self):
        try:
            self.status_var.set("Training model...")
            self.root.update()

            # Preprocess and train
            X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test = self.system.preprocess_data()
            self.system.train_model(X_train_scaled, y_train)

            # Evaluate model
            metrics, y_pred, y_test_actual = self.system.evaluate_model(
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test
            )

            # Update metrics display
            self.display_metrics(metrics)

            # Update plots
            self.update_plots(y_test_actual, y_pred)

            self.status_var.set("Model trained successfully!")
            self.predict_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {e}")

    def display_metrics(self, metrics):
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "Model Performance Metrics:\n\n")
        for metric, value in metrics.items():
            self.metrics_text.insert(tk.END, f"{metric}: {value:.4f}\n")

    def update_plots(self, y_test, y_pred):
        # Clear previous plots
        for widget in self.prediction_plot_frame.winfo_children():
            widget.destroy()
        for widget in self.trends_plot_frame.winfo_children():
            widget.destroy()

        # Create prediction analysis plots (in Prediction Analysis tab)
        fig1 = plt.Figure(figsize=(12, 6))  # Increased height for better visibility

        # Actual vs Predicted
        ax1 = fig1.add_subplot(131)
        ax1.scatter(y_test, y_pred, alpha=0.5)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted')

        # Residuals
        ax2 = fig1.add_subplot(132)
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')

        # Feature Importance
        ax3 = fig1.add_subplot(133)
        importance = pd.DataFrame({
            'feature': ['Measure', 'Country', 'Citizenship', 'Year'],
            'importance': self.system.model.feature_importances_
        }).sort_values('importance', ascending=False)
        sns.barplot(data=importance, x='importance', y='feature', ax=ax3)
        ax3.set_title('Feature Importance')

        fig1.tight_layout(pad=3.0)  # Increased padding

        # Create canvas for prediction plots
        canvas1 = FigureCanvasTkAgg(fig1, self.prediction_plot_frame)
        canvas1.draw()
        toolbar1 = NavigationToolbar2Tk(canvas1, self.prediction_plot_frame)
        toolbar1.update()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create migration trends plots (in Migration Trends tab)
        fig2 = plt.Figure(figsize=(12, 12))

        # Overall yearly trends with future projections
        ax4 = fig2.add_subplot(221)
        yearly_measure = self.system.data.groupby(['Year', 'Measure'])['Value'].sum().reset_index()

        for measure in yearly_measure['Measure'].unique():
            measure_data = yearly_measure[yearly_measure['Measure'] == measure]
            historical_data = measure_data[measure_data['Year'] <= self.system.last_historical_year]
            future_data = measure_data[measure_data['Year'] > self.system.last_historical_year]

            ax4.plot(historical_data['Year'], historical_data['Value'],
                    label=f'{measure} (Historical)', marker='o')
            if not future_data.empty:
                ax4.plot(future_data['Year'], future_data['Value'],
                        label=f'{measure} (Projected)', linestyle='--', marker='x')

        ax4.axvline(x=self.system.last_historical_year, color='gray', linestyle=':',
                   label='Historical Data Cutoff')
        ax4.set_title('Migration Trends by Measure (with Projections)')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Total Value')
        ax4.legend()

        # Top 10 countries
        ax5 = fig2.add_subplot(222)
        historical_data = self.system.data[
            (self.system.data['Measure'] == 'Arrivals') &
            (self.system.data['Year'] <= self.system.last_historical_year)
        ]
        top_countries = (historical_data.groupby('Country')['Value']
                        .sum()
                        .sort_values(ascending=False)
                        .head(10))
        sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax5)
        ax5.set_title('Top 10 Countries by Historical Arrivals')

        # Citizenship trends
        ax6 = fig2.add_subplot(223)
        citizenship_yearly = (self.system.data[self.system.data['Measure'] == 'Arrivals']
                            .groupby(['Year', 'Citizenship'])['Value']
                            .sum()
                            .reset_index())

        for citizenship in citizenship_yearly['Citizenship'].unique():
            citizenship_data = citizenship_yearly[citizenship_yearly['Citizenship'] == citizenship]
            historical = citizenship_data[citizenship_data['Year'] <= self.system.last_historical_year]
            future = citizenship_data[citizenship_data['Year'] > self.system.last_historical_year]

            ax6.plot(historical['Year'], historical['Value'],
                    label=citizenship, marker='o')
            if not future.empty:
                ax6.plot(future['Year'], future['Value'],
                        linestyle='--', marker='x')

        ax6.axvline(x=self.system.last_historical_year, color='gray', linestyle=':')
        ax6.set_title('Yearly Trends by Citizenship (with Projections)')
        ax6.set_xlabel('Year')
        ax6.set_ylabel('Total Value')
        ax6.legend()

        # Regional analysis
        ax7 = fig2.add_subplot(224)
        regions = ['Oceania', 'Asia', 'Europe', 'Americas', 'Africa and the Middle East']
        regional_data = []
        historical_data = self.system.data[
            (self.system.data['Measure'] == 'Arrivals') &
            (self.system.data['Year'] <= self.system.last_historical_year)
        ]

        for region in regions:
            region_total = historical_data[
                historical_data['Country'].str.contains(region, na=False)
            ]['Value'].sum()
            regional_data.append((region, region_total))

        regional_df = pd.DataFrame(regional_data, columns=['Region', 'Total'])
        sns.barplot(data=regional_df, x='Total', y='Region', ax=ax7)
        ax7.set_title('Total Historical Arrivals by Region')

        fig2.tight_layout(pad=3.0)  # Increased padding

        # Create canvas for trends plots
        canvas2 = FigureCanvasTkAgg(fig2, self.trends_plot_frame)
        canvas2.draw()
        toolbar2 = NavigationToolbar2Tk(canvas2, self.trends_plot_frame)
        toolbar2.update()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def make_prediction(self):
        try:
            measure = self.measure_var.get()
            country = self.country_var.get()
            citizenship = self.citizenship_var.get()
            year = int(self.year_var.get())

            if not all([measure, country, citizenship, year]):
                raise ValueError("Please fill in all fields")

            if year < 1979 or year > 2030:
                raise ValueError("Year must be between 1979 and 2030")

            # Make prediction using the enhanced predict_value method
            prediction = self.system.predict_value(measure, country, citizenship, year)

            # Update result label with additional context
            result_text = f"Predicted {measure} for {citizenship} in {country} for {year}: {prediction:.2f}"
            if year > self.system.last_historical_year:
                result_text += "\n(Based on historical trends with uncertainty factors)"

            self.prediction_result.config(text=result_text)

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {e}")


def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    root.style = ttk.Style()

    # Try to set a modern theme if available
    try:
        root.style.theme_use('clam')
    except tk.TclError:
        pass

    app = MigrationAnalysisGUI(root)

    # Configure grid weight
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()