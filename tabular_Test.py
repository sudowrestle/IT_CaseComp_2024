import os
import pandas as pd
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *

rel_path = os.path.dirname(__file__)

df = pd.read_csv(rel_path + '/Sanitized/BAR1Sanitized.csv')

print(df.dtypes)
print(df.info())
print(df.isnull().sum())

date = pd.to_datetime("2022-05-25")

day_of_week = date.dayofweek
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
    'DebitCredit': ['H'],
    'DayOfWeek': [day_of_week],
    'Month': [month],
    'DayOfYear': [day_of_year],
    'WeekOfYear': [week_of_year],
    'IsWeekend': [is_weekend]
})

columns_to_keep = ['Material', 'Plant', 'UnitofMeasure', 'DebitCredit', 'Quantity', 'DayOfWeek', 'Month', 'DayOfYear', 'WeekOfYear', 'IsWeekend']
df = df[columns_to_keep]

dls = TabularDataLoaders.from_df(df, path='./', y_names="Quantity", 
    cat_names = ['Material', 'Plant', 'UnitofMeasure', 'DebitCredit', 'DayOfWeek', 'Month', 'DayOfYear', 'WeekOfYear', 'IsWeekend'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=rmse)

learn = load_learner(rel_path+'/PermaModel/TrainedExport.pkl')

learn.dls = dls


learn.fit_one_cycle(20)

learn.show_results(max_n=15)
row, tensorVal, probs = learn.predict(prediction_df.iloc[0])
print(f"Prediction quantity is: {tensorVal.item()}")



