import plotly.express as px
import numpy as np
import pandas as pd

df = pd.read_csv('costs.csv')
fig = px.line(df, x='iters', y='cost', title='Costs')
fi.g1write_image('costs.png', engine='kaleido')
