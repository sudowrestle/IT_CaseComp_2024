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

prediction_df = pd.DataFrame(
    columns=['Material','Plant','UnitofMeasure','DebitCredit'],
    data={
    'Material': ['1010003'],
    'Plant': ['BAR1'],
    'UnitofMeasure': ['KG'],
    'EntryDate': ['2023-09-09'],
    'DebitCredit': ['S'],
})

columns_to_keep = ['Material', 'Plant', 'UnitofMeasure', 'DebitCredit', 'Quantity']
df = df[columns_to_keep]

dls = TabularDataLoaders.from_df(df, path=rel_path, y_names="Quantity", 
    cat_names = ['Material', 'Plant','UnitofMeasure', 'DebitCredit'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=[rmse,mae],loss_func=MSELossFlat())
learn.fit_one_cycle(2)

learn.show_results(max_n=15)
row, clas, probs = learn.get_preds(prediction_df.iloc[0])
print(f"Prediction quantity is: {clas.item()}")



