import pandas as pd

df = pd.read_csv('./Dummy Data/BAR1History.csv')

df['Material'] =  "'" + df['Material'].astype(str) + "'"

df['Plant'] = 'BAR1'

df['Quantity'] = df['Quantity'].str.replace(',', '').replace('"','').astype(float)

df['EntryDate'] = pd.to_datetime(df['EntryDate'], format='%m/%d/%Y')

df = df.sort_values(by=['Material', 'EntryDate'])

# df['DayOfWeek'] = df['EntryDate'].dt.dayofweek
# df['Month'] = df['EntryDate'].dt.month
# df['DayOfYear'] = df['EntryDate'].dt.dayofyear
# df['WeekOfYear'] = df['EntryDate'].dt.isocalendar().week
#df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
#df['DailyChange'] = df.groupby('Material')['Quantity'].diff()
#rolling_window = 3
#df['AvgIncreaseOrDecrease'] = df.groupby('Material')['DailyChange'].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())

df.to_csv('Sanitized/BAR1Sanitized.csv', index=False)

print(df.isnull().sum())
