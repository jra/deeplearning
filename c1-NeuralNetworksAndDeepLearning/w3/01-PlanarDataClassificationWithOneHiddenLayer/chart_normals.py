import math

import plotly.express as px
import numpy as np
import pandas as pd

df = pd.read_csv('normals.csv')
print(df)
print('df')
print(df['vals'].size)
print(df['vals'].dtype)
print(df['vals'].min())
print(df['vals'].max())
print(df['vals'].mean())
print(df['vals'].var())
fig = px.histogram(
    df,
    x='vals',
    title='Normals',
    #nbins=20,
    #histfunc="count",
)
fig.write_image('normals.png', engine='kaleido')


rng = np.random.default_rng()
vals2 = rng.standard_normal(1000)
print('vals2')
print(vals2.size)
print(vals2.dtype)
print(vals2.min())
print(vals2.max())
print(vals2.mean())
print(vals2.var())
fig2 = px.histogram(
    vals2,
)
fig2.write_image('normals_vals2.png', engine='kaleido')


def normal(rng, mu, sig):
    q, u, v = 0, 0, 0
    while True:
        u = rng.uniform()
        v = 1.7156 * (rng.uniform() - 0.5)
        x = u - 0.449871
        y = abs(v) + 0.386595
        q = math.pow(x, 2) + y * (y*0.19600-x*0.25472)
        if not ((q>0.27597) and
            ((q>0.27846) or
             (math.pow(v, 2) > (-4.0 * math.log(u) * math.pow(u, 2))))):
            break
    #print(mu, sig, v, u)
    return mu + sig * v / u


n = 1000
vals3 = np.fromiter((normal(rng, 0.0, 1.0) for x in range(n)), dtype=float)
print('vals3')
print(vals3.size)
print(vals3.dtype)
print(vals3.min())
print(vals3.max())
print(vals3.mean())
print(vals3.var())
fig3 = px.histogram(
    vals3,
)
fig3.write_image('normals_vals3.png', engine='kaleido')
