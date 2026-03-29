import yfinance as yf 
import pandas as pd
import numpy as np 

tickers = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'RELIANCE.NS', 'NTPC.NS',
    'LT.NS', 'TATASTEEL.NS', 'BHARTIARTL.NS', 'INFY.NS', 'ULTRACEMCO.NS',
    'BAJFINANCE.NS', 'M&M.NS', 'MCX.NS', 'OIL.NS', 'SHRIRAMFIN.NS',
    'TITAN.NS', 'FORTIS.NS', 'KOTAKBANK.NS', 'DIVISLAB.NS', 'MARICO.NS',
    'LTF.NS', 'AUBANK.NS', 'KEI.NS', 'BEL.NS', 'MUTHOOTFIN.NS']

weights_raw = [
    6.08, 4.85, 3.48, 3.47, 3.41,
    3.23, 2.63, 2.21, 2.17, 2.13,
    1.97, 1.85, 1.84, 1.78, 1.75,
    1.71, 1.53, 1.52, 1.49, 1.34,
    1.23, 1.20, 1.18, 1.15, 1.15]

weights = np.array(weights_raw)
weights = weights / weights.sum()

prices = yf.download(
    tickers,
    start='2022-01-01',
    end='2025-01-01',
    auto_adjust=True
)['Close']

returns = prices.pct_change().dropna()

portfolio_returns = returns.dot(weights)

print(f"Average daily return: {portfolio_returns.mean():.4f}")
print(f"Daily std deviation:  {portfolio_returns.std():.4f}")
print(f"Worst single day:     {portfolio_returns.min():.4f}")
print(f"Best single day:      {portfolio_returns.max():.4f}")

worst_date = portfolio_returns.idxmin()
print(f"Worst day was: {worst_date}")
print(f"Loss on that day: {portfolio_returns.min():.4f}")

#VAR Method 1: Historical Simulation
import matplotlib.pyplot as plt

portfolio_value = 2000000
confidence = 0.95

VaR_pct = np.percentile(portfolio_returns, (1 - confidence) * 100)
VaR_inr = portfolio_value * abs(VaR_pct)

print(f"Historical VaR (95%, 1-day)")
print(f"As % of portfolio : {VaR_pct:.4f} ({VaR_pct*100:.2f}%)")
print(f"In Rupees         : ₹{VaR_inr:,.0f}")

plt.figure(figsize=(10, 5))
plt.hist(portfolio_returns, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
plt.axvline(VaR_pct, color='red', linewidth=2, label=f'95% VaR = {VaR_pct*100:.2f}%')
plt.title('Edelweiss Flexi Cap Fund — Daily Return Distribution')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

#VAR Method 2: Parametric Simulation:
from scipy.stats import norm

portfolio_value = 2_000_000
confidence = 0.95

mean = portfolio_returns.mean()
std  = portfolio_returns.std()

VaR_parametric_pct = norm.ppf(1 - confidence, mean, std)
VaR_parametric_inr = portfolio_value * abs(VaR_parametric_pct)

print(f"Parametric VaR (95%, 1-day)")
print(f"As % of portfolio : {VaR_parametric_pct:.4f} ({VaR_parametric_pct*100:.2f}%)")
print(f"In Rupees         : ₹{VaR_parametric_inr:,.0f}")


#Normal Distribution Curve
x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
pdf = norm.pdf(x, mean, std)  # the theoretical bell curve

fig, ax = plt.subplots(figsize=(10, 5))


ax.hist(portfolio_returns, bins=50, density=True, 
        color='steelblue', edgecolor='white', alpha=0.6, label='Actual Returns')

ax.plot(x, pdf, color='black', linewidth=2, label='Normal Distribution (assumed)')

x_fill = np.linspace(portfolio_returns.min(), VaR_parametric_pct, 500)
ax.fill_between(x_fill, norm.pdf(x_fill, mean, std), 
                color='red', alpha=0.4, label='5% Loss Tail')

ax.axvline(VaR_parametric_pct, color='red', linewidth=2, 
           linestyle='--', label=f'95% Parametric VaR = {VaR_parametric_pct*100:.2f}%')

ax.set_title('Edelweiss Flexi Cap Fund — Parametric VaR (Normal Distribution)')
ax.set_xlabel('Daily Return')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()

#VAR Method 3: Monte Carlo Simulation

np.random.seed(42)

mc_mean = portfolio_returns.mean()
mc_std  = portfolio_returns.std()
n_simulations = 10_000

# ── CHART 1: Distribution of 10,000 Single-Day Returns ───────────────

simulated_returns = np.random.normal(mc_mean, mc_std, n_simulations)

VaR_mc_pct = np.percentile(simulated_returns, (1 - confidence) * 100)
VaR_mc_inr = portfolio_value * abs(VaR_mc_pct)

print(f"Monte Carlo VaR (95%, 1-day)")
print(f"As % of portfolio : {VaR_mc_pct:.4f} ({VaR_mc_pct*100:.2f}%)")
print(f"In Rupees         : ₹{VaR_mc_inr:,.0f}")

fig1, ax1 = plt.subplots(figsize=(10, 5))

# Plot full distribution
ax1.hist(simulated_returns, bins=100,
         color='mediumseagreen', edgecolor='white', alpha=0.8,
         label='10,000 Simulated Returns')

# Overlay loss tail — use same bins, no density
ax1.hist(simulated_returns[simulated_returns <= VaR_mc_pct],
         bins=100,
         color='red', alpha=0.6, label='5% Loss Tail')

ax1.axvline(VaR_mc_pct, color='darkred', linewidth=2,
            linestyle='--', label=f'95% Monte Carlo VaR = {VaR_mc_pct*100:.2f}%')

ax1.set_title('Edelweiss Flexi Cap Fund — Monte Carlo VaR (10,000 Simulations)')
ax1.set_xlabel('Simulated Daily Return')
ax1.set_ylabel('Frequency')
ax1.legend()
plt.tight_layout()
plt.show()

# ── CHART 2: Path Simulation — 1,000 Scenarios Over 252 Days ─────────

np.random.seed(42)

n_paths = 1000
n_days  = 252

simulated_paths = np.random.normal(mc_mean, mc_std, (n_days, n_paths))

# Build portfolio value paths starting from ₹20L
price_paths = np.zeros((n_days + 1, n_paths))
price_paths[0] = portfolio_value

for t in range(1, n_days + 1):
    price_paths[t] = price_paths[t-1] * (1 + simulated_paths[t-1])


fig2, ax2 = plt.subplots(figsize=(12, 6))

# All 1000 paths in faint blue
ax2.plot(price_paths, color='steelblue', alpha=0.05, linewidth=0.8)

# Key percentile lines
ax2.plot(np.percentile(price_paths, 5, axis=1),
         color='red', linewidth=2, linestyle='--', label='5th Percentile (Worst 5%)')
ax2.plot(np.percentile(price_paths, 50, axis=1),
         color='black', linewidth=2, label='Median Scenario')
ax2.plot(np.percentile(price_paths, 95, axis=1),
         color='green', linewidth=2, linestyle='--', label='95th Percentile (Best 5%)')

# Starting value
ax2.axhline(portfolio_value, color='orange', linewidth=1.5,
            linestyle=':', label='Starting Value ₹20L')

ax2.set_title('Edelweiss Flexi Cap Fund — Monte Carlo Path Simulation (1,000 Scenarios, 252 Days)')
ax2.set_xlabel('Trading Days')
ax2.set_ylabel('Portfolio Value (₹)')
ax2.legend()
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L'))
plt.tight_layout()
plt.show()

# 1 Year Summary
final_values = price_paths[-1]
VaR_1year    = portfolio_value - np.percentile(final_values, 5)

print(f"\n── 1-Year Simulation Summary ──")
print(f"Starting portfolio value      : ₹{portfolio_value:,.0f}")
print(f"Median value after 1 year     : ₹{np.median(final_values):,.0f}")
print(f"Best case (95th pct) 1 year   : ₹{np.percentile(final_values, 95):,.0f}")
print(f"Worst case (5th pct) 1 year   : ₹{np.percentile(final_values, 5):,.0f}")
print(f"1-Year 95% VaR                : ₹{VaR_1year:,.0f}")

#Comparing the three methods:

fig, ax = plt.subplots(figsize=(12, 6))

# Historical distribution
ax.hist(portfolio_returns, bins=50, density=True,
        color='steelblue', edgecolor='white', alpha=0.5,
        label='Actual Returns (Historical)')

# Monte Carlo distribution
ax.hist(simulated_returns, bins=100, density=True,
        color='mediumseagreen', edgecolor='white', alpha=0.4,
        label='Simulated Returns (Monte Carlo)')

x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
ax.plot(x, norm.pdf(x, mean, std),
        color='black', linewidth=2, label='Normal Distribution (Parametric)')

ax.axvline(VaR_pct, color='blue', linewidth=2,
           linestyle='--', label=f'Historical VaR = {VaR_pct*100:.2f}%')
ax.axvline(VaR_parametric_pct, color='black', linewidth=2,
           linestyle='--', label=f'Parametric VaR = {VaR_parametric_pct*100:.2f}%')
ax.axvline(VaR_mc_pct, color='green', linewidth=2,
           linestyle='--', label=f'Monte Carlo VaR = {VaR_mc_pct*100:.2f}%')

ax.set_title('Edelweiss Flexi Cap Fund — VaR Method Comparison (95%, 1-Day)')
ax.set_xlabel('Daily Return')
ax.set_ylabel('Density')
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

#A Broad Reality about Markets
fig, ax = plt.subplots(figsize=(12, 6))

# Historical distribution in blue
ax.hist(portfolio_returns, bins=50, density=True,
        color='steelblue', edgecolor='white', alpha=0.5,
        label='Actual Returns (Historical)')

# Monte Carlo distribution in green
ax.hist(simulated_returns, bins=100, density=True,
        color='mediumseagreen', edgecolor='white', alpha=0.4,
        label='Simulated Returns (Monte Carlo)')

# Normal curve for Parametric
x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
ax.plot(x, norm.pdf(x, mean, std),
        color='black', linewidth=2, label='Normal Distribution (Parametric)')

# Three VaR lines
ax.axvline(VaR_pct, color='blue', linewidth=2,
           linestyle='--', label=f'Historical VaR = {VaR_pct*100:.2f}%')
ax.axvline(VaR_parametric_pct, color='black', linewidth=2,
           linestyle='--', label=f'Parametric VaR = {VaR_parametric_pct*100:.2f}%')
ax.axvline(VaR_mc_pct, color='green', linewidth=2,
           linestyle='--', label=f'Monte Carlo VaR = {VaR_mc_pct*100:.2f}%')

# Plotting the Fat Tail

ax.axvspan(-0.07, -0.025, alpha=0.15, color='red', label='Fat Tail Zone')

ax.annotate('FAT TAIL\nActual losses here\nare more frequent\nthan normal curve predicts',
            xy=(-0.05, 0.5),
            xytext=(-0.055, 8),
            fontsize=9,
            color='red',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            ha='center')

ax.annotate('Normal curve\npredicts near zero\nprobability here',
            xy=(-0.04, 0.1),
            xytext=(-0.035, 5),
            fontsize=8,
            color='black',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            ha='center')

ax.set_title('Edelweiss Flexi Cap Fund — VaR Method Comparison (95%, 1-Day)\nRed Zone = Fat Tail Region')
ax.set_xlabel('Daily Return')
ax.set_ylabel('Density')
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

#BACKTESTING DATA (using training and testing)

train = portfolio_returns['2022-01-01':'2023-12-31']
test  = portfolio_returns['2024-01-01':'2024-12-31']

print(f"Training period : {train.index[0].date()} to {train.index[-1].date()}")
print(f"Training days   : {len(train)}")
print(f"Testing period  : {test.index[0].date()} to {test.index[-1].date()}")
print(f"Testing days    : {len(test)}")

VaR_backtest = np.percentile(train, (1 - confidence) * 100)
print(f"\nVaR estimated from training data: {VaR_backtest*100:.2f}%")

# Exception = day where actual loss exceeded VaR threshold
exceptions = test[test < VaR_backtest]
n_exceptions = len(exceptions)
exception_rate = n_exceptions / len(test) * 100

print(f"\n── Backtesting Results ──────────────────────────────")
print(f"Testing days              : {len(test)}")
print(f"Number of exceptions      : {n_exceptions}")
print(f"Exception rate            : {exception_rate:.2f}%")
print(f"Expected rate (95% VaR)   : 5.00%")

# Step 4: Basel Traffic Light
if n_exceptions <= 4:
    zone = "🟢 GREEN — Model is accurate"
elif n_exceptions <= 9:
    zone = "🟡 YELLOW — Model needs review"
else:
    zone = "🔴 RED — Model underestimates risk"

print(f"Basel Traffic Light       : {zone}")

# Plotting the Backtesting
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(test.index, test.values,
        color='steelblue', linewidth=1, label='2024 Daily Returns')

# VaR threshold line
ax.axhline(VaR_backtest, color='red', linewidth=2,
           linestyle='--', label=f'95% VaR Threshold = {VaR_backtest*100:.2f}%')

ax.scatter(exceptions.index, exceptions.values,
           color='red', zorder=5, s=50,
           label=f'Exceptions ({n_exceptions} days)')

ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

ax.set_title(f'Edelweiss Flexi Cap Fund — Backtesting 2024\n{n_exceptions} Exceptions out of {len(test)} days ({exception_rate:.1f}%)')
ax.set_xlabel('Date')
ax.set_ylabel('Daily Return')
ax.legend()
plt.tight_layout()
plt.show()

# STRESS TESTING (based on real historical Nifty 50 daily returns)
scenarios = {
    'COVID Crash (Mar 23, 2020)'       : -0.1313,  # Nifty fell 13.13% in one day
    'Election Shock (Jun 4, 2024)'     : -0.0629,  # Your actual worst day
    '2008 Financial Crisis (Oct 2008)' : -0.1196,  # Nifty fell ~12% in one day
}

portfolio_value = 2000000

print("Stress Test Results ")
print(f"Portfolio Value         : ₹{portfolio_value:,.0f}")
print(f"95% Historical VaR      : ₹{portfolio_value * abs(VaR_backtest):,.0f} ({VaR_backtest*100:.2f}%)")


stress_results = {}
for scenario, shock in scenarios.items():
    loss_inr = portfolio_value * abs(shock)
    times_var = abs(shock) / abs(VaR_backtest)
    stress_results[scenario] = {
        'shock'     : shock,
        'loss_inr'  : loss_inr,
        'times_var' : times_var
    }
    print(f"\nScenario : {scenario}")
    print(f"Shock    : {shock*100:.2f}%")
    print(f"Loss     : ₹{loss_inr:,.0f}")
    print(f"Multiple : {times_var:.1f}x VaR")

# Plotting the Stress Test
fig, ax = plt.subplots(figsize=(10, 6))

scenario_names = list(stress_results.keys())
losses         = [stress_results[s]['loss_inr'] for s in scenario_names]
var_line       = portfolio_value * abs(VaR_backtest)

# Horizontal bars for each scenario
bars = ax.barh(scenario_names, losses, 
               color=['crimson', 'darkorange', 'darkred'],
               edgecolor='white', height=0.5)

# VaR reference line
ax.axvline(var_line, color='blue', linewidth=2,
           linestyle='--', label=f'95% VaR = ₹{var_line:,.0f}')

for bar, loss in zip(bars, losses):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
            f'₹{loss:,.0f}', va='center', fontsize=10, fontweight='bold')

ax.set_title('Edelweiss Flexi Cap Fund — Stress Test Scenarios\nvs 95% Historical VaR Threshold')
ax.set_xlabel('Portfolio Loss (₹)')
ax.legend()
plt.tight_layout()
plt.show()

# CVAR (Conditional Value at Risk) - How much will I lose in a day, given I definitely lose
# ── Historical CVaR
tail_returns      = portfolio_returns[portfolio_returns <= VaR_pct]
CVaR_hist_pct     = tail_returns.mean()
CVaR_hist_inr     = portfolio_value * abs(CVaR_hist_pct)

# ── Parametric CVaR 
# Formula: CVaR = mean - std * (pdf(z) / (1 - confidence))
# where z is the VaR z-score
from scipy.stats import norm
z                 = norm.ppf(1 - confidence)
CVaR_param_pct    = mean - std * (norm.pdf(z) / (1 - confidence))
CVaR_param_inr    = portfolio_value * abs(CVaR_param_pct)

# ── Monte Carlo CVaR 
mc_tail           = simulated_returns[simulated_returns <= VaR_mc_pct]
CVaR_mc_pct       = mc_tail.mean()
CVaR_mc_inr       = portfolio_value * abs(CVaR_mc_pct)


print("── VaR vs CVaR Comparison (95%, 1-Day) ─────────────────────")
print(f"{'Method':<25} {'VaR %':>8} {'VaR ₹':>12} {'CVaR %':>8} {'CVaR ₹':>12}")
print("─────────────────────────────────────────────────────────────")
print(f"{'Historical':<25} {VaR_pct*100:>8.2f}% {portfolio_value*abs(VaR_pct):>11,.0f} {CVaR_hist_pct*100:>8.2f}% {CVaR_hist_inr:>11,.0f}")
print(f"{'Parametric':<25} {VaR_parametric_pct*100:>8.2f}% {portfolio_value*abs(VaR_parametric_pct):>11,.0f} {CVaR_param_pct*100:>8.2f}% {CVaR_param_inr:>11,.0f}")
print(f"{'Monte Carlo':<25} {VaR_mc_pct*100:>8.2f}% {portfolio_value*abs(VaR_mc_pct):>11,.0f} {CVaR_mc_pct*100:>8.2f}% {CVaR_mc_inr:>11,.0f}")

# Plotting the CVAR Simulations
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(portfolio_returns, bins=50, density=True,
        color='steelblue', edgecolor='white', alpha=0.6,
        label='Actual Returns')

x_var = portfolio_returns[(portfolio_returns <= VaR_pct)]
ax.hist(x_var, bins=20, density=True,
        color='orange', alpha=0.7, label='VaR Tail (worst 5%)')

# VaR line
ax.axvline(VaR_pct, color='orange', linewidth=2,
           linestyle='--', label=f'95% VaR = {VaR_pct*100:.2f}%')

# CVaR line
ax.axvline(CVaR_hist_pct, color='red', linewidth=2,
           linestyle='--', label=f'CVaR = {CVaR_hist_pct*100:.2f}%')

ax.set_title('Edelweiss Flexi Cap Fund — VaR vs CVaR\nVaR marks where the tail starts, CVaR is the average loss within it')
ax.set_xlabel('Daily Return')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()