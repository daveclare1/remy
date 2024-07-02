import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import plotly.express as px

conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read(
    usecols=range(5),
    ttl=60,
    )
df = df.convert_dtypes()

df['timestamp'] = df['date'] + " " + df['time']
df.timestamp = pd.to_datetime(df.timestamp, dayfirst=True)
# df = df.drop(['date', 'time'], axis='columns')
df.date = pd.to_datetime(df.date, dayfirst=True)
df.time = pd.to_datetime(df.time)

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

# plot timelines

fig = px.scatter(
    df,
    x='time',
    y='date',
    color='action',
    symbol = 'action',
    symbol_sequence= ['circle', 'square', 'diamond-tall', 'hourglass'],
)
fig.update_traces(marker=dict(size=12, opacity=0.7))
fig.update_layout(xaxis_tickformat='%H:%M')
fig.update_xaxes(nticks=10, showgrid=True)
fig.update_yaxes(nticks=int(df.day_idx.max())+1, showgrid=True)
st.plotly_chart(fig)

for date, df_date in df.groupby('date'):
    print(date)


st.dataframe(df)