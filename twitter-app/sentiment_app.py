# Import dependencies
## DASH
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

## PLOTLY
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

## OTHERS
import json
from textwrap import dedent as d
import random
from collections import deque as de
import sqlite3
import pandas as pd
import time
from datetime import datetime as dt

# Set up Dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
    }
}

app.layout = html.Div(
    [   html.H2('Twitter Sentiment Analysis'),
   
    # animate=True will update graph in real time
    # animate=False will not update graph
        dcc.Graph(id='sentiment-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1*100
            ),
    ]
)

# Pull in data
@app.callback(
    Output('sentiment-graph', 'figure'),
    inputs=[Input('graph-update', 'interval')])
def update_graph_scatter(self):
    
    # Use pandas to convert database data to dataframe
    # Load database into dataframe by passing the SQL query and the connection object
    # Include search subject for analysis
    # Limit number of tweets return, sort to get newest tweets
    try:
        conn = sqlite3.connect("_Projects\Project-03\PoliticsPredicted6.db")
        c = conn.cursor()
        
        df = pd.read_sql("SELECT * FROM TwitterDB WHERE tweet LIKE '%Trump%' ORDER BY unix DESC LIMIT 500", conn)

        # Use moving average to assess sentiment rating
        # # Order data chronologically before analyzing
        
        # # if you encounter a "year is out of range" error the timestamp
        # # may be in milliseconds, try `ts /= 1000` in that case
        # print('DATETIME STRING------------')
        # ts = df['unix']
        # df['unix'] = dt.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        df.sort_values(by=['unix'], inplace=True)
        print('ONE---------------------')
        print(df.unix)

        # Convert linux timestamp to more readable date format
        # ms stands for miliseconds
        df['unix'] = pd.to_datetime(df['unix'], unit='ms')

        # df['unix'] = dt.strptime(df['unix'], "%Y-%m-%d %H:%M:%S")
        # df.set_index('unix', inplace=True)
        print('TWO---------------------')
        print(df.unix)

        # Use ? in query and build out param with '%' + sentiment + '%'
        # This helps against SQL injection (per YouTube video)
        
        # df['sentiment_moving_avg'] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df['sentiment_moving_avg'] = df.sentiment.rolling(5).mean()
        print('THREE---------------------')
        print(df.unix)

        # df.dropna(inplace=True)
        print('FOUR---------------------')
        print(df.unix)


        # Resample index (in case we want to analyze 500 data points instead of thousands)
        # df = df.resample('6000ms').mean # 60 seconds, perform mean average
        
   
        X = df.unix[-100:]
        Y = df.sentiment_moving_avg[-100:]

        print('X-------------------')
        print(X)

        print('Y-------------------')
        print(Y)
    
        # Create traces
        data = plotly.graph_objs.Scatter(
                # x=datetime.utcfromtimestamp(X).strftime('%Y-%m-%d %H:%M:%S'),
                x=X,
                y=Y,
                name='Scatter',
                mode= 'lines+markers'
                )
        print('DATA------------------------')
        print(data)

        return {'data': [data]} 
    
   # Access the attributes of the exception object 
   # Use 'Exception' to only accept exceptions that you mean to catch
    except Exception as e:
        print('ERROR--------------------:')
        print(e)
        with open('errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')


    
    # py.iplot(data, filename='scatter-mode')


if __name__ == "__main__":
    app.run_server(debug=True)
