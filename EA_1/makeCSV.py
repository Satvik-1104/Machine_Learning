import numpy as np
import pandas as pd

# Given data
X = np.array([[1500, 3, 20, 10],
              [1800, 4, 15, 8],
              [2000, 4, 5, 12],
              [1200, 2, 30, 15],
              [1700, 3, 10, 7],
              [2500, 5, 8, 5],
              [1300, 2, 25, 18],
              [1900, 4, 12, 16],
              [1600, 3, 20, 9],
              [2100, 4, 5, 11]])

y = np.array([300, 400, 500, 200, 350, 600, 220, 320, 330, 480])

# Create a DataFrame
columns = ['Size(sq.ft)', 'Bedrooms', 'Age', 'Distance(miles)', 'Price(1000$)']
dataset = np.column_stack((X, y))
df = pd.DataFrame(dataset, columns=columns)

# Save to CSV
df.to_csv('house_prices.csv', index=False)

print("Data saved to house_prices.csv")
