import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def download_crypto_data(symbol="BTC-USD", start_date=None, end_date=None, interval="1h", save_path=None):
    """
    Download cryptocurrency data from Yahoo Finance
    
    Args:
        symbol (str): Symbol of the cryptocurrency (e.g., "BTC-USD", "ETH-USD")
        start_date (str): Start date in "YYYY-MM-DD" format. If None, defaults to 6 months ago
        end_date (str): End date in "YYYY-MM-DD" format. If None, defaults to today
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        save_path (str): Path to save the downloaded data as CSV. If None, data won't be saved
    
    Returns:
        pandas.DataFrame: DataFrame containing price data
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if start_date is None:
        start = datetime.now() - timedelta(days=180)  # 6 months of data
        start_date = start.strftime("%Y-%m-%d")
    
    # Download data
    print(f"Downloading {symbol} data from {start_date} to {end_date} with interval {interval}...")
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    
    # Ensure all required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Clean data
    data = data.dropna()
    
    # Save data if path provided
    if save_path is not None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path)
        print(f"Data saved to {save_path}")
    
    return data

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
    
    Returns:
        pandas.DataFrame: DataFrame with added technical indicators
    """
    # Copy the dataframe to avoid modifying the original
    df_indicators = df.copy()
    
    # Validate data types
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if col in df_indicators.columns:
            if not pd.api.types.is_numeric_dtype(df_indicators[col]):
                # Try to convert to numeric
                df_indicators[col] = pd.to_numeric(df_indicators[col], errors='coerce')
                if not pd.api.types.is_numeric_dtype(df_indicators[col]):
                    raise ValueError(f"Column {col} must be numeric, but got {df_indicators[col].dtypes}")
    
    # 1. Moving Averages
    df_indicators['SMA_7'] = df_indicators['Close'].rolling(window=7).mean()
    df_indicators['SMA_20'] = df_indicators['Close'].rolling(window=20).mean()
    df_indicators['EMA_7'] = df_indicators['Close'].ewm(span=7, adjust=False).mean()
    df_indicators['EMA_20'] = df_indicators['Close'].ewm(span=20, adjust=False).mean()
    
    # 2. Relative Strength Index (RSI)
    delta = df_indicators['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df_indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. Moving Average Convergence Divergence (MACD)
    ema_12 = df_indicators['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_indicators['Close'].ewm(span=26, adjust=False).mean()
    df_indicators['MACD'] = ema_12 - ema_26
    df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. Bollinger Bands
    df_indicators['BB_Middle'] = df_indicators['Close'].rolling(window=20).mean()
    std_dev = df_indicators['Close'].rolling(window=20).std()
    df_indicators['BB_Upper'] = df_indicators['BB_Middle'] + (std_dev * 2.0)
    df_indicators['BB_Lower'] = df_indicators['BB_Middle'] - (std_dev * 2.0)
    
    # 5. Volatility
    df_indicators['Volatility'] = df_indicators['Close'].rolling(window=20).std()
    
    # 6. Price momentum (Rate of Change)
    df_indicators['ROC'] = df_indicators['Close'].pct_change(periods=10) * 100
    
    # 7. Average True Range (ATR)
    high_low = df_indicators['High'] - df_indicators['Low']
    high_close = (df_indicators['High'] - df_indicators['Close'].shift()).abs()
    low_close = (df_indicators['Low'] - df_indicators['Close'].shift()).abs()
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df_indicators['ATR'] = true_range.rolling(window=14).mean()
    
    # Drop NaN values
    df_indicators = df_indicators.dropna()
    
    return df_indicators


def normalize_data(df):
    """
    Normalize the data for better model performance
    
    Args:
        df (pandas.DataFrame): DataFrame with price and indicator data
    
    Returns:
        pandas.DataFrame: Normalized DataFrame
    """
    # Copy the dataframe to avoid modifying the original
    df_normalized = df.copy()
    
    # Normalize price columns by the first value in the dataset
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
        if col in df_normalized.columns:
            df_normalized[col] = df_normalized[col] / df_normalized[col].iloc[0]
    
    # Normalize volume by its average
    if 'Volume' in df_normalized.columns:
        df_normalized['Volume'] = df_normalized['Volume'] / df_normalized['Volume'].mean()
    
    # Scale technical indicators to a reasonable range
    # RSI is already scaled (0-100)
    # Scale MACD and other indicators
    for col in ['MACD', 'MACD_Signal']:
        if col in df_normalized.columns:
            df_normalized[col] = (df_normalized[col] - df_normalized[col].mean()) / df_normalized[col].std()
    
    # Volatility and ATR
    for col in ['Volatility', 'ATR']:
        if col in df_normalized.columns:
            df_normalized[col] = df_normalized[col] / df_normalized['Close'].mean()
    
    return df_normalized


def prepare_data_for_env(data_path=None, symbol="BTC-USD", start_date=None, end_date=None, 
                        interval="1h", add_indicators=True, normalize=True):
    """
    Prepare data for the trading environment
    
    Args:
        data_path (str): Path to CSV file with price data. If None, data will be downloaded
        symbol (str): Symbol to download if data_path is None
        start_date (str): Start date for download
        end_date (str): End date for download
        interval (str): Data interval
        add_indicators (bool): Whether to add technical indicators
        normalize (bool): Whether to normalize the data
    
    Returns:
        pandas.DataFrame: Processed DataFrame ready for the trading environment
    """
    # Load data from file or download
    if data_path is not None and os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Convert numeric columns to float
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        # Create directory for data if needed
        if data_path is not None:
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Download data
        df = download_crypto_data(symbol, start_date, end_date, interval, save_path=data_path)
    
    # Add technical indicators if requested
    if add_indicators:
        df = add_technical_indicators(df)
    
    # Normalize data if requested
    if normalize:
        df = normalize_data(df)
    
    # Final cleanup
    df = df.dropna()
    
    return df


def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into training, validation, and test sets
    
    Args:
        df (pandas.DataFrame): DataFrame to split
        train_ratio (float): Ratio of data for training
        val_ratio (float): Ratio of data for validation
        test_ratio (float): Ratio of data for testing
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Check that ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split the data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df 