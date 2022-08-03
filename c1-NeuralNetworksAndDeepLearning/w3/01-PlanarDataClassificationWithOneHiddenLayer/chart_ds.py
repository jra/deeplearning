import plotly.express as px
import numpy as np
import pandas as pd

df = pd.read_csv('dataset.csv')
y = df['y'].astype(str)
fig = px.scatter(df, x='x0', y='x1', color=y)
fig.write_image('ds.png', engine='kaleido')
