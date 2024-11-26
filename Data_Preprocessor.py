import pandas as pd
import fastai as fast

data = pd.read_csv('WILL INSERT CSV FROM KEVO HERE')

dls = TabularDataLoaders.from_df(
    data,
    path='.',
    procs[]
)


