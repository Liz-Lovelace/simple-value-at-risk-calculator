import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# Load data from Excel file
file_path = "testvar.xlsx"  # Replace with your actual file path
data = pd.read_excel(file_path)

# Ensure columns are correctly named
data.columns = ['Ticker', 'Date', 'Price']

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Convert Price column to numeric
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

# Drop rows with invalid (NaN) prices
data = data.dropna(subset=['Price'])

# Sort data by Date and Ticker
data = data.sort_values(by=['Date', 'Ticker'])

# Step 1: Ask for Investment Amount and Percentages
investment_amount = float(input("Enter the total amount of money to invest: "))
num_stocks = len(data['Ticker'].unique())
print(f"There are {num_stocks} stocks in your portfolio.")
print("Enter the percentage of your total investment for each stock:")

# Get investment percentages for each stock
investment_percentages = {}
for ticker in data['Ticker'].unique():
    percentage = float(input(f"Percentage for {ticker} (e.g., 20 for 20%): "))
    investment_percentages[ticker] = percentage / 100

# Check if percentages add up to 100%
if not np.isclose(sum(investment_percentages.values()), 1.0):
    raise ValueError("The percentages do not add up to 100%. Please re-enter.")

# Step 2: Calculate Portfolio Weights
# Calculate number of shares for each stock
latest_prices = data.groupby('Ticker')['Price'].last()
num_shares = {
    ticker: (investment_amount * investment_percentages[ticker]) / latest_prices[ticker]
    for ticker in data['Ticker'].unique()
}

# Step 3: Calculate Daily Returns
# Calculate percentage change in price for each stock
data['Return'] = data.groupby('Ticker')['Price'].pct_change()

# Merge returns into a pivot table
returns_pivot = data.pivot(index='Date', columns='Ticker', values='Return').dropna()

# Step 4: Compute Portfolio Returns
# Calculate portfolio return as weighted sum of individual returns
weights = pd.Series(investment_percentages)
portfolio_returns = returns_pivot.dot(weights)

# Step 5: Fit a Distribution to Historical Returns
# Option 1: Fit a normal distribution
mu = portfolio_returns.mean()
sigma = portfolio_returns.std()

# Option 2: Fit a t-distribution (optional)
params = t.fit(portfolio_returns)
df_t, loc_t, scale_t = params

# Step 6: Monte Carlo Simulation
num_simulations = int(input("Enter the number of simulations (e.g., 10000): "))
time_horizon = int(input("Enter the time horizon for VaR (in days): "))

# Option 2: Using t-distribution (uncomment to use)
simulated_returns = t.rvs(df_t, loc=loc_t, scale=scale_t, size=(time_horizon, num_simulations))

# Calculate cumulative returns over the time horizon
cumulative_returns = np.cumprod(1 + simulated_returns, axis=0)[-1] - 1

# Step 7: Calculate Simulated Portfolio Values
# Initial portfolio value
portfolio_value = investment_amount

# Simulated portfolio values at the end of the time horizon
simulated_portfolio_values = portfolio_value * (1 + cumulative_returns)

# Simulated portfolio losses
simulated_losses = portfolio_value - simulated_portfolio_values

# Step 8: Calculate VaR
confidence_level = float(input("Enter the confidence level (e.g., 0.99 for 99% VaR): "))

# Corrected percentile calculation
VaR = np.percentile(simulated_losses, (1 - confidence_level) * 100)

print(f"The {confidence_level * 100}% VaR over {time_horizon} days is: ${VaR:.2f}")


# Step 9: Plot the Distribution of Simulated Losses
plt.figure(figsize=(10, 6))
plt.hist(simulated_losses, bins=50, edgecolor='k')
plt.axvline(VaR, color='r', linestyle='dashed', linewidth=2)
plt.title('Distribution of Simulated Portfolio Losses')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.show()
