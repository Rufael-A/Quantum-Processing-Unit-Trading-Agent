#!/usr/bin/env python3
"""
Quantum Trading Agent - Financial Model with yfinance
This module implements the financial data processing, backtesting, and optimization
framework using yfinance for real market data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from ibm_quantum_hardware_implementation import QuantumTradingAgent, IBMQuantumHardwareOptimizer

class FinancialDataProcessor:
    """
    Financial Data Processor for the Quantum Trading Agent.
    This class handles data acquisition, preprocessing, and feature engineering
    for financial time series data.
    """
    
    def __init__(self, data_dir='../data'):
        """
        Initialize the Financial Data Processor.
        
        Args:
            data_dir (str): Directory to store financial data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_market_data(self, tickers, start_date, end_date, interval='1d', save=True):
        """
        Download historical market data for the specified tickers.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval ('1d', '1wk', '1mo', etc.)
            save (bool): Whether to save data to disk
            
        Returns:
            dict: Dictionary of pandas DataFrames with historical data
        """
        print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
        
        # Download data for each ticker
        data = {}
        for ticker in tickers:
            try:
                # Download data
                ticker_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
                
                # Check if data was successfully downloaded
                if ticker_data.empty:
                    print(f"Warning: No data found for {ticker}")
                    continue
                
                # Store data
                data[ticker] = ticker_data
                
                # Save data if requested
                if save:
                    file_path = os.path.join(self.data_dir, f"{ticker}_{start_date}_{end_date}.csv")
                    ticker_data.to_csv(file_path)
                    print(f"Saved data for {ticker} to {file_path}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error downloading data for {ticker}: {e}")
        
        print(f"Downloaded data for {len(data)} tickers")
        return data
    
    def load_market_data(self, ticker, start_date, end_date):
        """
        Load market data from disk if available, otherwise download.
        
        Args:
            ticker (str): Ticker symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: DataFrame with historical data
        """
        file_path = os.path.join(self.data_dir, f"{ticker}_{start_date}_{end_date}.csv")
        
        # Check if file exists
        if os.path.exists(file_path):
            print(f"Loading data for {ticker} from {file_path}")
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            print(f"Data file not found for {ticker}, downloading...")
            data = self.download_market_data([ticker], start_date, end_date)
            return data.get(ticker)
    
    def calculate_technical_indicators(self, data):
        """
        Calculate technical indicators for the given data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Moving Average Convergence Divergence (MACD)
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Price Rate of Change (ROC)
        df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
        df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
        df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
        
        # Momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Volume indicators
        df['Volume_ROC'] = df['Volume'].pct_change(periods=1) * 100
        df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_5']
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def create_features_and_targets(self, data, lookback_window=10, prediction_horizon=5, target_type='binary'):
        """
        Create features and targets for machine learning.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV and technical indicators
            lookback_window (int): Number of past days to use as features
            prediction_horizon (int): Number of days ahead to predict
            target_type (str): Type of target ('binary', 'return', or 'direction')
            
        Returns:
            tuple: (features, targets) arrays
        """
        # Select features
        feature_columns = [
            'Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20',
            'EMA_5', 'EMA_10', 'EMA_20', 'MACD', 'MACD_Signal',
            'RSI', 'BB_Upper', 'BB_Lower', 'ATR', 'ROC_5',
            'ROC_10', 'Momentum_5', '%K', '%D', 'Volume_Ratio'
        ]
        
        # Normalize features
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data[feature_columns])
        normalized_df = pd.DataFrame(normalized_data, index=data.index, columns=feature_columns)
        
        # Create sequences for lookback window
        X = []
        y = []
        
        for i in range(len(normalized_df) - lookback_window - prediction_horizon):
            # Features: lookback window of data
            X.append(normalized_df.iloc[i:i+lookback_window].values)
            
            # Target: future price movement
            current_price = data['Close'].iloc[i+lookback_window-1]
            future_price = data['Close'].iloc[i+lookback_window+prediction_horizon-1]
            
            if target_type == 'binary':
                # Binary classification: 1 if price goes up, 0 if down
                target = 1 if future_price > current_price else 0
            elif target_type == 'return':
                # Regression: percentage return
                target = (future_price - current_price) / current_price
            elif target_type == 'direction':
                # Multi-class: -1 for down, 0 for sideways, 1 for up
                pct_change = (future_price - current_price) / current_price
                if pct_change > 0.01:  # Up more than 1%
                    target = 1
                elif pct_change < -0.01:  # Down more than 1%
                    target = -1
                else:  # Sideways
                    target = 0
            else:
                raise ValueError(f"Unknown target_type: {target_type}")
            
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def prepare_data_for_quantum(self, features, max_qubits=10):
        """
        Prepare data for quantum processing by reducing dimensionality.
        
        Args:
            features (np.ndarray): Feature array
            max_qubits (int): Maximum number of qubits to use
            
        Returns:
            np.ndarray: Reduced feature array suitable for quantum processing
        """
        # Get shape of features
        n_samples, n_timesteps, n_features = features.shape
        
        # If the number of features is already small enough, return as is
        if n_features <= max_qubits:
            return features
        
        # Otherwise, reduce dimensionality
        # Method 1: Select most important features (simplified approach)
        important_indices = [0, 1, 2, 3, 6, 8, 10, 12, 14, 16]  # Example indices
        reduced_features = features[:, :, important_indices[:max_qubits]]
        
        # Method 2: Average across time steps if needed
        if n_timesteps > 1 and n_features * n_timesteps > max_qubits:
            # Average across time steps to get a single vector per sample
            reduced_features = np.mean(reduced_features, axis=1)
        
        return reduced_features


class QuantumTradingStrategy:
    """
    Quantum Trading Strategy implementation.
    This class integrates the quantum trading agent with financial data
    to create and optimize trading strategies.
    """
    
    def __init__(self, backend_name='ibm_sherbrooke', use_real_hardware=False):
        """
        Initialize the Quantum Trading Strategy.
        
        Args:
            backend_name (str): Name of the IBM quantum backend
            use_real_hardware (bool): Whether to use real quantum hardware
        """
        self.backend_name = backend_name
        self.use_real_hardware = use_real_hardware
        
        # Initialize quantum trading agent
        self.quantum_agent = QuantumTradingAgent(backend_name, use_real_hardware)
        
        # Initialize data processor
        self.data_processor = FinancialDataProcessor()
        
        # Strategy parameters
        self.lookback_window = 10
        self.prediction_horizon = 5
        self.rebalance_frequency = 5
        self.risk_factor = 0.5
        
        # Performance tracking
        self.performance_history = []
    
    def optimize_strategy_parameters(self, historical_data, param_grid=None):
        """
        Optimize strategy parameters using grid search.
        
        Args:
            historical_data (dict): Historical price data for multiple assets
            param_grid (dict): Parameter grid to search
            
        Returns:
            dict: Optimal parameters and performance metrics
        """
        if param_grid is None:
            param_grid = {
                'lookback_window': [5, 10, 15],
                'prediction_horizon': [1, 3, 5],
                'rebalance_frequency': [1, 5, 10],
                'risk_factor': [0.3, 0.5, 0.7]
            }
        
        print("Optimizing strategy parameters...")
        
        # Track best parameters and performance
        best_sharpe = 0
        best_params = {}
        all_results = []
        
        # Perform grid search
        for lookback in param_grid['lookback_window']:
            for horizon in param_grid['prediction_horizon']:
                for rebalance in param_grid['rebalance_frequency']:
                    for risk in param_grid['risk_factor']:
                        print(f"Testing parameters: lookback={lookback}, horizon={horizon}, rebalance={rebalance}, risk={risk}")
                        
                        # Update strategy parameters
                        self.lookback_window = lookback
                        self.prediction_horizon = horizon
                        self.rebalance_frequency = rebalance
                        self.risk_factor = risk
                        
                        # Run backtest with current parameters
                        backtest_results = self.quantum_agent.backtest_strategy(
                            historical_data,
                            lookback_window=lookback,
                            rebalance_frequency=rebalance,
                            risk_factor=risk
                        )
                        
                        # Extract performance metrics
                        sharpe_ratio = backtest_results['sharpe_ratio']
                        
                        # Record results
                        result = {
                            'lookback_window': lookback,
                            'prediction_horizon': horizon,
                            'rebalance_frequency': rebalance,
                            'risk_factor': risk,
                            'sharpe_ratio': sharpe_ratio,
                            'cumulative_returns': backtest_results['cumulative_returns'],
                            'max_drawdown': backtest_results['max_drawdown']
                        }
                        
                        all_results.append(result)
                        
                        # Update best parameters if better performance found
                        if sharpe_ratio > best_sharpe:
                            best_sharpe = sharpe_ratio
                            best_params = {
                                'lookback_window': lookback,
                                'prediction_horizon': horizon,
                                'rebalance_frequency': rebalance,
                                'risk_factor': risk
                            }
                            print(f"New best parameters found! Sharpe ratio: {best_sharpe:.4f}")
        
        # Update strategy with best parameters
        self.lookback_window = best_params['lookback_window']
        self.prediction_horizon = best_params['prediction_horizon']
        self.rebalance_frequency = best_params['rebalance_frequency']
        self.risk_factor = best_params['risk_factor']
        
        print(f"Optimization complete. Best parameters: {best_params}")
        print(f"Best Sharpe ratio: {best_sharpe:.4f}")
        
        # Return optimization results
        optimization_results = {
            'best_params': best_params,
            'best_sharpe_ratio': best_sharpe,
            'all_results': all_results
        }
        
        return optimization_results
    
    def backtest(self, tickers, start_date, end_date, optimize_params=False):
        """
        Backtest the quantum trading strategy on historical data.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            optimize_params (bool): Whether to optimize strategy parameters
            
        Returns:
            dict: Backtest results including performance metrics
        """
        print(f"Starting backtest for {tickers} from {start_date} to {end_date}...")
        
        # Download historical data
        historical_data = {}
        for ticker in tickers:
            data = self.data_processor.load_market_data(ticker, start_date, end_date)
            if data is not None and not data.empty:
                # Calculate technical indicators
                data_with_indicators = self.data_processor.calculate_technical_indicators(data)
                # Extract close prices for backtesting
                historical_data[ticker] = data_with_indicators['Close'].values
        
        # Optimize strategy parameters if requested
        if optimize_params:
            optimization_results = self.optimize_strategy_parameters(historical_data)
            print(f"Using optimized parameters: {optimization_results['best_params']}")
        
        # Run backtest with current parameters
        backtest_results = self.quantum_agent.backtest_strategy(
            historical_data,
            lookback_window=self.lookback_window,
            rebalance_frequency=self.rebalance_frequency,
            risk_factor=self.risk_factor
        )
        
        # Store performance history
        self.performance_history.append({
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'sharpe_ratio': backtest_results['sharpe_ratio'],
            'cumulative_returns': backtest_results['cumulative_returns'],
            'max_drawdown': backtest_results['max_drawdown']
        })
        
        return backtest_results
    
    def monte_carlo_forward_test(self, tickers, start_date, end_date, num_simulations=100, horizon=30):
        """
        Perform Monte Carlo forward testing to estimate future performance.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date for historical data in 'YYYY-MM-DD' format
            end_date (str): End date for historical data in 'YYYY-MM-DD' format
            num_simulations (int): Number of Monte Carlo simulations
            horizon (int): Forward testing horizon in days
            
        Returns:
            dict: Forward test results including performance metrics
        """
        print(f"Starting Monte Carlo forward test for {tickers}...")
        
        # Download historical data
        historical_data = {}
        for ticker in tickers:
            data = self.data_processor.load_market_data(ticker, start_date, end_date)
            if data is not None and not data.empty:
                # Calculate technical indicators
                data_with_indicators = self.data_processor.calculate_technical_indicators(data)
                # Extract close prices for forward testing
                historical_data[ticker] = data_with_indicators['Close'].values
        
        # Run Monte Carlo forward test
        forward_test_results = self.quantum_agent.monte_carlo_forward_test(
            historical_data,
            num_simulations=num_simulations,
            horizon=horizon,
            lookback_window=self.lookback_window,
            risk_factor=self.risk_factor
        )
        
        return forward_test_results
    
    def find_optimal_asset_combination(self, ticker_universe, start_date, end_date, max_assets=5):
        """
        Find the optimal combination of assets that maximizes Sharpe ratio.
        
        Args:
            ticker_universe (list): List of potential ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            max_assets (int): Maximum number of assets to include
            
        Returns:
            dict: Optimal asset combination and performance metrics
        """
        print(f"Finding optimal asset combination from {len(ticker_universe)} tickers...")
        
        # Download historical data for all tickers
        all_historical_data = {}
        for ticker in ticker_universe:
            data = self.data_processor.load_market_data(ticker, start_date, end_date)
            if data is not None and not data.empty:
                # Extract close prices
                all_historical_data[ticker] = data['Close'].values
        
        # Calculate individual asset performance
        asset_performance = {}
        for ticker, prices in all_historical_data.items():
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            # Calculate Sharpe ratio
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            asset_performance[ticker] = sharpe_ratio
        
        # Sort assets by individual Sharpe ratio
        sorted_assets = sorted(asset_performance.items(), key=lambda x: x[1], reverse=True)
        
        # Select top assets
        top_assets = [asset[0] for asset in sorted_assets[:max_assets]]
        
        print(f"Selected top {len(top_assets)} assets: {top_assets}")
        
        # Test different combinations of top assets
        best_sharpe = 0
        best_combination = []
        
        # Start with all top assets
        current_combination = top_assets.copy()
        historical_data = {ticker: all_historical_data[ticker] for ticker in current_combination}
        
        # Run backtest with current combination
        backtest_results = self.quantum_agent.backtest_strategy(
            historical_data,
            lookback_window=self.lookback_window,
            rebalance_frequency=self.rebalance_frequency,
            risk_factor=self.risk_factor
        )
        
        # Extract performance metrics
        sharpe_ratio = backtest_results['sharpe_ratio']
        
        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_combination = current_combination.copy()
        
        print(f"Best asset combination: {best_combination}")
        print(f"Best Sharpe ratio: {best_sharpe:.4f}")
        
        # Return results
        optimal_combination = {
            'assets': best_combination,
            'sharpe_ratio': best_sharpe,
            'backtest_results': backtest_results
        }
        
        return optimal_combination
    
    def plot_backtest_results(self, backtest_results, title="Quantum Trading Strategy Backtest"):
        """
        Plot backtest results.
        
        Args:
            backtest_results (dict): Backtest results from backtest method
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot portfolio value
        portfolio_value = backtest_results['portfolio_value']
        ax1.plot(portfolio_value, linewidth=2)
        ax1.set_title(title)
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True)
        
        # Add performance metrics as text
        sharpe_ratio = backtest_results['sharpe_ratio']
        cumulative_returns = backtest_results['cumulative_returns']
        max_drawdown = backtest_results['max_drawdown']
        
        metrics_text = f"Sharpe Ratio: {sharpe_ratio:.4f}\n"
        metrics_text += f"Cumulative Returns: {cumulative_returns:.2%}\n"
        metrics_text += f"Maximum Drawdown: {max_drawdown:.2%}"
        
        ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Plot drawdowns
        equity_curve = np.array(portfolio_value)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        
        ax2.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Trading Days')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        fig.savefig('../data/backtest_results.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_monte_carlo_results(self, forward_test_results, title="Monte Carlo Forward Test"):
        """
        Plot Monte Carlo forward test results.
        
        Args:
            forward_test_results (dict): Forward test results from monte_carlo_forward_test method
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot Sharpe ratio distribution
        all_sharpe_ratios = forward_test_results['all_sharpe_ratios']
        ax1.hist(all_sharpe_ratios, bins=20, alpha=0.7)
        ax1.axvline(forward_test_results['avg_sharpe_ratio'], color='red', linestyle='--', linewidth=2)
        ax1.axvline(forward_test_results['sharpe_ratio_95ci'][0], color='green', linestyle='--', linewidth=2)
        ax1.axvline(forward_test_results['sharpe_ratio_95ci'][1], color='green', linestyle='--', linewidth=2)
        ax1.set_title(f"{title} - Sharpe Ratio Distribution")
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_ylabel('Frequency')
        ax1.grid(True)
        
        # Add legend
        ax1.legend(['Average', '95% CI Lower', '95% CI Upper'])
        
        # Add performance metrics as text
        avg_sharpe = forward_test_results['avg_sharpe_ratio']
        ci_lower = forward_test_results['sharpe_ratio_95ci'][0]
        ci_upper = forward_test_results['sharpe_ratio_95ci'][1]
        
        metrics_text = f"Average Sharpe Ratio: {avg_sharpe:.4f}\n"
        metrics_text += f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
        metrics_text += f"Probability of Sharpe > 2: {np.mean(np.array(all_sharpe_ratios) > 2):.2%}"
        
        ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Plot cumulative probability
        sorted_sharpe = np.sort(all_sharpe_ratios)
        cumulative_prob = np.arange(1, len(sorted_sharpe) + 1) / len(sorted_sharpe)
        
        ax2.plot(sorted_sharpe, cumulative_prob, linewidth=2)
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=1)
        ax2.axvline(2.0, color='green', linestyle='--', linewidth=1)
        ax2.set_title("Cumulative Probability Distribution")
        ax2.set_xlabel('Sharpe Ratio')
        ax2.set_ylabel('Cumulative Probability')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        fig.savefig('../data/monte_carlo_results.png', dpi=300, bbox_inches='tight')
        
        return fig


class OptimalAssetSelector:
    """
    Optimal Asset Selector for the Quantum Trading Strategy.
    This class identifies the optimal asset combination for trading.
    """
    
    def __init__(self, data_dir='../data'):
        """
        Initialize the Optimal Asset Selector.
        
        Args:
            data_dir (str): Directory to store financial data
        """
        self.data_dir = data_dir
        self.data_processor = FinancialDataProcessor(data_dir)
    
    def get_market_indices(self):
        """
        Get a list of major market indices.
        
        Returns:
            list: List of index ticker symbols
        """
        indices = [
            '^GSPC',  # S&P 500
            '^DJI',   # Dow Jones Industrial Average
            '^IXIC',  # NASDAQ Composite
            '^RUT',   # Russell 2000
            '^FTSE',  # FTSE 100
            '^N225',  # Nikkei 225
            '^HSI',   # Hang Seng Index
            '^GDAXI', # DAX
            '^FCHI',  # CAC 40
            'GC=F',   # Gold Futures
            'SI=F',   # Silver Futures
            'CL=F',   # Crude Oil Futures
            'BTC-USD', # Bitcoin
            'ETH-USD'  # Ethereum
        ]
        
        return indices
    
    def get_sector_etfs(self):
        """
        Get a list of sector ETFs.
        
        Returns:
            list: List of ETF ticker symbols
        """
        etfs = [
            'XLK',  # Technology
            'XLF',  # Financial
            'XLE',  # Energy
            'XLV',  # Healthcare
            'XLY',  # Consumer Discretionary
            'XLP',  # Consumer Staples
            'XLI',  # Industrial
            'XLB',  # Materials
            'XLU',  # Utilities
            'XLRE', # Real Estate
            'GLD',  # Gold
            'SLV',  # Silver
            'USO',  # Oil
            'UNG',  # Natural Gas
            'QQQ',  # NASDAQ 100
            'SPY',  # S&P 500
            'DIA',  # Dow Jones
            'IWM',  # Russell 2000
            'EEM',  # Emerging Markets
            'EFA'   # Developed Markets
        ]
        
        return etfs
    
    def get_blue_chip_stocks(self):
        """
        Get a list of blue-chip stocks.
        
        Returns:
            list: List of stock ticker symbols
        """
        stocks = [
            'AAPL',  # Apple
            'MSFT',  # Microsoft
            'AMZN',  # Amazon
            'GOOGL', # Alphabet (Google)
            'META',  # Meta (Facebook)
            'TSLA',  # Tesla
            'NVDA',  # NVIDIA
            'JPM',   # JPMorgan Chase
            'V',     # Visa
            'JNJ',   # Johnson & Johnson
            'WMT',   # Walmart
            'PG',    # Procter & Gamble
            'MA',    # Mastercard
            'UNH',   # UnitedHealth
            'HD',    # Home Depot
            'BAC',   # Bank of America
            'XOM',   # Exxon Mobil
            'CSCO',  # Cisco
            'PFE',   # Pfizer
            'INTC'   # Intel
        ]
        
        return stocks
    
    def analyze_asset_correlations(self, tickers, start_date, end_date):
        """
        Analyze correlations between assets.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        print(f"Analyzing correlations for {len(tickers)} assets...")
        
        # Download data
        all_data = {}
        for ticker in tickers:
            data = self.data_processor.load_market_data(ticker, start_date, end_date)
            if data is not None and not data.empty:
                all_data[ticker] = data['Close']
        
        # Create DataFrame with all close prices
        close_prices = pd.DataFrame(all_data)
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        return correlation_matrix
    
    def calculate_asset_metrics(self, tickers, start_date, end_date):
        """
        Calculate performance metrics for assets.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: DataFrame with asset metrics
        """
        print(f"Calculating metrics for {len(tickers)} assets...")
        
        # Download data
        all_data = {}
        for ticker in tickers:
            data = self.data_processor.load_market_data(ticker, start_date, end_date)
            if data is not None and not data.empty:
                all_data[ticker] = data['Close']
        
        # Create DataFrame with all close prices
        close_prices = pd.DataFrame(all_data)
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        # Calculate metrics
        metrics = []
        
        for ticker in returns.columns:
            ticker_returns = returns[ticker]
            
            # Calculate metrics
            annualized_return = ticker_returns.mean() * 252
            annualized_volatility = ticker_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Calculate maximum drawdown
            cum_returns = (1 + ticker_returns).cumprod()
            peak = cum_returns.cummax()
            drawdown = (peak - cum_returns) / peak
            max_drawdown = drawdown.max()
            
            # Calculate Sortino ratio (downside risk only)
            negative_returns = ticker_returns[ticker_returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252)
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Store metrics
            metrics.append({
                'ticker': ticker,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown
            })
        
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        return metrics_df
    
    def select_optimal_assets(self, start_date, end_date, num_assets=5, min_sharpe=1.0):
        """
        Select optimal assets for trading based on performance metrics and correlations.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            num_assets (int): Number of assets to select
            min_sharpe (float): Minimum Sharpe ratio threshold
            
        Returns:
            list: List of selected ticker symbols
        """
        print("Selecting optimal assets for trading...")
        
        # Get universe of assets
        indices = self.get_market_indices()
        etfs = self.get_sector_etfs()
        stocks = self.get_blue_chip_stocks()
        
        all_tickers = indices + etfs + stocks
        
        # Calculate metrics for all assets
        metrics_df = self.calculate_asset_metrics(all_tickers, start_date, end_date)
        
        # Filter assets by minimum Sharpe ratio
        filtered_df = metrics_df[metrics_df['sharpe_ratio'] >= min_sharpe]
        
        if filtered_df.empty:
            print(f"No assets meet the minimum Sharpe ratio of {min_sharpe}. Lowering threshold...")
            # Sort by Sharpe ratio and take top assets
            filtered_df = metrics_df.sort_values('sharpe_ratio', ascending=False).head(num_assets * 2)
        
        # Get tickers of filtered assets
        filtered_tickers = filtered_df['ticker'].tolist()
        
        # Calculate correlation matrix for filtered assets
        correlation_matrix = self.analyze_asset_correlations(filtered_tickers, start_date, end_date)
        
        # Select diverse assets with low correlation
        selected_assets = []
        remaining_assets = filtered_df.sort_values('sharpe_ratio', ascending=False)['ticker'].tolist()
        
        # Start with the highest Sharpe ratio asset
        selected_assets.append(remaining_assets.pop(0))
        
        # Add assets with low correlation to already selected assets
        while len(selected_assets) < num_assets and remaining_assets:
            # Calculate average correlation with selected assets
            avg_correlations = []
            
            for asset in remaining_assets:
                corrs = [correlation_matrix.loc[asset, selected] for selected in selected_assets]
                avg_correlations.append((asset, np.mean(corrs)))
            
            # Sort by correlation (ascending)
            avg_correlations.sort(key=lambda x: x[1])
            
            # Add asset with lowest average correlation
            next_asset = avg_correlations[0][0]
            selected_assets.append(next_asset)
            remaining_assets.remove(next_asset)
        
        print(f"Selected {len(selected_assets)} assets: {selected_assets}")
        
        return selected_assets


# Main execution
if __name__ == "__main__":
    # Initialize asset selector
    asset_selector = OptimalAssetSelector()
    
    # Define date range for analysis
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    # Select optimal assets
    optimal_assets = asset_selector.select_optimal_assets(start_date, end_date, num_assets=5)
    
    # Initialize quantum trading strategy
    strategy = QuantumTradingStrategy(backend_name='ibm_sherbrooke', use_real_hardware=False)
    
    # Backtest strategy
    backtest_results = strategy.backtest(optimal_assets, start_date, end_date, optimize_params=True)
    
    # Plot backtest results
    strategy.plot_backtest_results(backtest_results)
    
    # Perform Monte Carlo forward test
    forward_test_results = strategy.monte_carlo_forward_test(optimal_assets, start_date, end_date, num_simulations=100)
    
    # Plot Monte Carlo results
    strategy.plot_monte_carlo_results(forward_test_results)
    
    print("Analysis complete. Results saved to data directory.")
