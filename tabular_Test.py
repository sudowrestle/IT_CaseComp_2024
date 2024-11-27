from fastai.tabular.all import *
from fastai.text.all import *
from fastai.collab import *


cfg = fastai_cfg()
cfg.data, cfg.path('./Dummy Data'), fastai_path('./Dummy Data')

path = untar_data(URLs.ADULT_SAMPLE, data='./Dummy Data')
path.ls()

df = pd.read_csv(path/'adult.csv')
df.head()

dls = TabularDataLoaders.from_csv(path/'adult.csv')
df.head()

dls = TabularDataLoaders.from_csv(
    path/'adult.csv', 
    path=path, 
    y_names="salary",
    cat_names = ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize]
)

splits = RandomSplitter(valid_pct=0.2)(range_of(df))

to = TabularPandas(
    df,
    procs=[Categorify, FillMissing, Normalize],
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    y_names = 'salary',
    splits=splits
)

to.xs.iloc[:2]

dls = to.dataloaders(bs=64)

dls.show_batch()

learn = tabular_learner(dls, metrics=accuracy)

learn.fit_one_cycle(1)

learn.show_results()

row, clas, probs = learn.predict(df.iloc[0])
row.show()
clas, probs

test_df = df.copy()
test_df.drop(['salary'], axis=1, inplace=True)
dl = learn.dls.test_dl(test_df)

learn.get_preds(dl=dl)
