import plotly.express as px
import numpy as np
import pandas as pd

df = pd.read_csv('costs_3lrs.csv')
#print(df)
#print(df.dtypes)
#print(df.info())
fig = px.line(df, x='iters', y=['0.01', '0.001', '0.0001'], title='Costs')
fig.write_image('costs_3lrs.png', engine='kaleido')
