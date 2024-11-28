import os
import pandas as pd
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *

rel_path = os.path.dirname(__file__) + '/Sanitized/BAR1Sanitized.csv'

df = pd.read_csv(rel_path)

print(df.dtypes)
print(df.info())
print(df.isnull().sum())

date = pd.to_datetime("2024-05-28")

day_of_week = date.dayofweek  # Monday=0, Sunday=6
month = date.month
day_of_year = date.dayofyear
week_of_year = date.isocalendar().week
is_weekend = 1 if day_of_week >= 5 else 0

prediction_df = pd.DataFrame(
    columns=['Material', 'Plant', 'UnitofMeasure', 'DebitCredit', 'DayOfWeek', 'Month', 'DayOfYear', 'WeekOfYear', 'IsWeekend'],
    data={
    'Material': ['1010003'],
    'Plant': ['BAR1'],
    'UnitofMeasure': ['KG'],
    'DebitCredit': ['S'],
    'DayOfWeek': [day_of_week],
    'Month': [month],
    'DayOfYear': [day_of_year],
    'WeekOfYear': [week_of_year],
    'IsWeekend': [is_weekend]
})

columns_to_keep = ['Material', 'Plant', 'UnitofMeasure', 'DebitCredit', 'Quantity', 'DayOfWeek', 'Month', 'DayOfYear', 'WeekOfYear', 'IsWeekend']
df = df[columns_to_keep]

dls = TabularDataLoaders.from_df(df, path=rel_path, y_names="Quantity", 
    cat_names = ['Material', 'Plant', 'UnitofMeasure', 'DebitCredit', 'DayOfWeek', 'Month', 'DayOfYear', 'WeekOfYear', 'IsWeekend'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(5)

test_dl = learn.dls.test_dl(prediction_df)
learn.get_preds(dl = test_dl)

learn.show_results(max_n=15)
row, clas, probs = learn.predict(prediction_df.iloc[0])
print(f"Prediction quantity is: {clas.item()}")



