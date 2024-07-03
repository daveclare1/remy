import numpy as np
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import plotly.graph_objects as go

marker_styles = {
    'drink': dict(symbol='line-ns', size=10, line_width=2, color='lightskyblue'),
    'food': dict(symbol='line-ns', size=20, line_width=2, color='black'),
    'poo': dict(symbol='triangle-up', color='brown', opacity=0.7),
    'wee': dict(symbol='circle', color='lightskyblue', opacity=0.7),
}

conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read(
    usecols=range(5),
    ttl=60,
    )
df = df.convert_dtypes()

df['timestamp'] = df['date'] + " " + df['time']
df.timestamp = pd.to_datetime(df.timestamp, dayfirst=True)
df.date = pd.to_datetime(df.date, dayfirst=True)
df.time = pd.to_datetime(df.time, format="%H:%M:%S")

# Now we have a clean dataframe

df['input'] = df.action.isin(['food', 'drink'])

# Compute time since last food or drink
col_last_food = []
col_last_water = []
col_wee_num = []
col_poo_num = []
last_food = None
last_water = None
wee_num = 0
poo_num = 0
for idx, row in df.iterrows():
    if row.action == 'food': 
        last_food = row.timestamp
        wee_num = 0
        poo_num = 0
    elif row.action == 'drink': 
        last_water = row.timestamp
        wee_num = 0

    col_last_food.append(last_food)
    col_last_water.append(last_water)
    col_wee_num.append(wee_num)
    col_poo_num.append(poo_num)
    
    if row.action == 'wee':
        wee_num = wee_num + 1
    elif row.action == 'poo':
        poo_num = poo_num + 1


df['last_food'] = col_last_food
df['last_water'] = col_last_water
df['wee_num'] = col_wee_num
df['poo_num'] = col_poo_num

df['day_idx'] = df.groupby('date').ngroup()
df['hit'] = (df['where'] == 'grass') | (df['where'] == None)

df['y_scatter'] = np.random.randint(-3600, 3600, df.shape[0])
df['y_scatter'] = df['y_scatter'].apply(lambda s:pd.Timedelta(seconds=s))
df['y_scatter'] = df['y_scatter'].mask(df.action.isin(['food','drink']), pd.Timedelta(0))

# plot timelines

fig = go.Figure()

for action, df_action in df.groupby('action'):
    fig.add_scatter(
        x=df_action.time,
        y=df_action.date + df_action.y_scatter,
        mode='markers',
        showlegend=True,
        name=action,
        marker=marker_styles[action],
    )

fig.update_layout(xaxis_tickformat='%H:%M')
fig.update_xaxes(nticks=10, showgrid=True)
fig.update_yaxes(nticks=int(df.day_idx.max())+1, showgrid=True)
st.plotly_chart(fig)

st.dataframe(df)