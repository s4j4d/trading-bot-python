from data.loader import load_data_from_json
import json

# Look at your training data
df_crypto = load_data_from_json("market_data.json")

print("DataFrame columns:", df_crypto.columns.tolist())
print("DataFrame shape:", df_crypto.shape)

# Check if DataFrame is empty
if df_crypto.empty:
    print("❌ ERROR: DataFrame is empty!")
    print("This happens because the data loader skips first 1000 rows, but your JSON only has 500 data points.")
    print("The data loader has this line: 'return df[1000:]' which removes all your data.")
    print("\nTo fix this, you need to either:")
    print("1. Use a larger dataset with >1000 data points, or")
    print("2. Modify the data loader to not skip rows, or") 
    print("3. Use a different JSON file")
    
    # Let's try to load the raw data to see what we have
    with open("market_data.json", 'r') as f:
        raw_data = json.load(f)
    
    print(f"\nRaw JSON data info:")
    print(f"Number of timestamps: {len(raw_data.get('t', []))}")
    print(f"Number of close prices: {len(raw_data.get('c', []))}")
    
    if len(raw_data.get('c', [])) > 0:
        closes = raw_data['c']
        print(f"First price: {closes[0]}")
        print(f"Last price: {closes[-1]}")
        print(f"Overall trend: {(closes[-1] / closes[0] - 1) * 100:.2f}%")
        
        # Check if it's mostly declining
        if closes[-1] < closes[0]:
            print("🔴 DECLINING MARKET: This explains why your model learned to sell!")
        else:
            print("🟢 RISING MARKET: Model should have learned to buy/hold")
            
else:
    print("DataFrame head:")
    print(df_crypto.head())

    # The column is named 'close', not 'c'
    print(f"\nPrice at start: {df_crypto['close'].iloc[0]:.2f}")
    print(f"Price at end: {df_crypto['close'].iloc[-1]:.2f}")
    print(f"Overall trend: {(df_crypto['close'].iloc[-1] / df_crypto['close'].iloc[0] - 1) * 100:.2f}%")

    # Additional analysis
    print(f"\nData period: {df_crypto.index[0]} to {df_crypto.index[-1]}")
    print(f"Number of data points: {len(df_crypto)}")
    print(f"Min price: {df_crypto['close'].min():.2f}")
    print(f"Max price: {df_crypto['close'].max():.2f}")
    print(f"Price volatility (std): {df_crypto['close'].std():.2f}")

    # Check for trends in different periods
    total_days = (df_crypto.index[-1] - df_crypto.index[0]).days
    print(f"Total days: {total_days}")

    if len(df_crypto) > 100:
        # First quarter
        q1_end = len(df_crypto) // 4
        q1_trend = (df_crypto['close'].iloc[q1_end] / df_crypto['close'].iloc[0] - 1) * 100
        
        # Last quarter  
        q4_start = 3 * len(df_crypto) // 4
        q4_trend = (df_crypto['close'].iloc[-1] / df_crypto['close'].iloc[q4_start] - 1) * 100
        
        print(f"First quarter trend: {q1_trend:.2f}%")
        print(f"Last quarter trend: {q4_trend:.2f}%")
