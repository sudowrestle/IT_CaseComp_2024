import os
import pandas
import datetime
from fastai.tabular.all import *
from fastai.text.all import *
from fastai.collab import *


local_path = os.path.dirname(__file__)

df = pd.read_csv('Sanitized/BAR1Sanitized.csv', parse_dates=['EntryDate'])

cont_names = ['Material', 'Quantity', 'DayOfWeek', 'Month', 'DayOfYear', 'WeekOfYear', 'DailyChange', 'AvgIncreaseOrDecrease']
cat_names = ['Plant', 'PurchaseOrder', 'UnitofMeasure', 'EntryDate']
y_names = 'Quantity'

procs = [Categorify, FillMissing, Normalize]

splits = RandomSplitter(valid_pct=0.2)(range_of(df))
to = TabularPandas(df, procs, cat_names, cont_names, y_names, splits=splits)

dls = to.dataloaders(bs=64)

learn = tabular_learner(dls, metrics=rmse)

learn.fit_one_cycle(5, 1e-2)

tomorrow_sample = pd.DataFrame({
    'Material': [1010003],
    'Plant': ['BAR1'],
    'EntryDate': [pd.to_datetime('2024-11-28')],
    'DayOfWeek': [pd.to_datetime('2024-11-28').dayofweek],
    'Month': [11],
    'DayOfYear': [332],
    'WeekOfYear': [48],
    'IsWeekend': [0],  
    'DailyChange': [0],
    'AvgIncreaseOrDecrease': [0] 
})

transformed_sample = to.transform(tomorrow_sample)

prediction = learn.predict(transformed_sample.iloc[0])
print(f"Predicted Quantity for tomorrow: {prediction[0]}")
