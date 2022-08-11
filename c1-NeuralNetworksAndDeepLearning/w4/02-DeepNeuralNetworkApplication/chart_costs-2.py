import plotly.express as px
import numpy as np
import pandas as pd

df = pd.read_csv('costs-2.csv')
fig = px.line(df, x='iters', y='cost', title='Costs')
fig.write_image('costs-2.png', engine='kaleido')
