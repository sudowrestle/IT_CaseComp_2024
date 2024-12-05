import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

rel_path = os.path.dirname(__file__)

data = pd.read_csv(rel_path+'\\ManyEpochs.csv')

df = pd.DataFrame(data)

sns.set_theme()

sns.relplot(data=df, x="Quantity", y="Quantity_pred", hue="Prediction")

plt.show()