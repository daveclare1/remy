import numpy as np
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import plotly.graph_objects as go

marker_styles = {
    'drink': dict(symbol='line-ns', size=10, line_width=2, line_color='darkblue', opacity=0.7),
    'food': dict(symbol='line-ns', size=20, line_width=2, color='black'),
    'poo (good)': dict(symbol='triangle-up', size=12, color='brown', opacity=0.7),
    'poo (bad)': dict(symbol='triangle-up-open', size=12, color='brown', opacity=0.7),
    'wee (good)': dict(symbol='circle', size=12, color='lightskyblue', opacity=0.7),
    'wee (bad)': dict(symbol='circle-open', size=12, color='lightskyblue', opacity=0.7),
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
col_last_wee = []
col_wee_num = []
col_poo_num = []
last_food = None
last_water = None
last_wee = None
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
    col_last_wee.append(last_wee)
    col_wee_num.append(wee_num)
    col_poo_num.append(poo_num)
    
    if row.action == 'wee':
        wee_num = wee_num + 1
        last_wee = row.timestamp
    elif row.action == 'poo':
        poo_num = poo_num + 1


df['last_food'] = col_last_food
df['last_water'] = col_last_water
df['last_wee'] = col_last_wee
df['wee_num'] = col_wee_num
df['poo_num'] = col_poo_num

df['day_idx'] = df.groupby('date').ngroup()
df['hit'] = ((df['where'] == 'grass') | (df['where'] == None) | (df['where'] == 'drive')).fillna(False)
df['action_detail'] = df.action + np.where((~df.input)&(df.hit), ' (good)', '')
df['action_detail'] = df.action_detail + np.where((~df.input)&(~df.hit), ' (bad)', '')
        
df['y_scatter'] = np.random.randint(-7200, 7200, df.shape[0])
df['y_scatter'] = df['y_scatter'].apply(lambda s:pd.Timedelta(seconds=s))
df['y_scatter'] = df['y_scatter'].mask(df.action.isin(['food','drink']), pd.Timedelta(0))


# title
st.title("Remy's 'schedule'")

# plot timelines

fig1 = go.Figure()

for action, df_action in df.groupby('action_detail'):
    fig1.add_scatter(
        x=df_action.time,
        y=df_action.date + df_action.y_scatter,
        mode='markers',
        showlegend=True,
        name=action,
        marker=marker_styles[action],
        hovertemplate ='%{x}'
    )

fig1.update_xaxes(nticks=10, showgrid=True, tickformat='%H:%M' )
fig1.update_yaxes(nticks=int(df.day_idx.max())+1, 
                  showgrid=True, 
                  tickformat='%a %d %b')
st.plotly_chart(fig1)

# wee timings
def add_timing_plot(fig:go.Figure, df:pd.DataFrame, since_col:str, title:str):
    timedelta_ms = ((df_wee0.timestamp - df_wee0[since_col]) / 1e6).astype('int64').astype(np.int32)
    plot_time = pd.to_timedelta(timedelta_ms, unit='ms') + pd.Timestamp("1970/01/01")
    fig2.add_box(
        x=plot_time,
        boxpoints='all',
        name=title,
    )

fig2 = go.Figure()
df_wee0 = df.query("(action=='wee') & (wee_num==0)")
df_wee = df.query("(action=='wee')")

add_timing_plot(fig2, df_wee0, 'last_water', 'From drinking')
add_timing_plot(fig2, df_wee0, 'last_food', 'From eating')
add_timing_plot(fig2, df_wee, 'last_wee', 'From last wee')

fig2.update_layout(title='Time before wee', showlegend=False)
fig2.update_xaxes(nticks=10, showgrid=True, tickformat='%H:%M')
st.plotly_chart(fig2)

st.dataframe(df)