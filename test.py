import pandas as pd

df = pd.read_csv('data_processed/prices/prices.csv')
symbol_count = df['symbol'].nunique()

print(f"Symbols: {symbol_count}/377")
print(f"Retention: {100*symbol_count/377:.1f}%")

# Should be >98%
assert symbol_count >= 370, "Too many symbols removed!"