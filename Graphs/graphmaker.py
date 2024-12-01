import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


data = pd.read_csv('Graphs/ManyEpochs.csv')

df = pd.DataFrame(data)

sns.set_theme()

sns.relplot(
    data=df,
    x="Quantity", y="Quantity_pred", hue="Prediction"
)

plt.show()