import datetime
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

#Preprocess
df = pd.read_csv("./dataset_road.csv", low_memory=False)
df.replace(["?", "--", "-1"], np.nan, inplace=True)
#df.dropna(how='all')

df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df['month'] = df.Date.dt.month

#replace the keys with actual meaning
df['Accident_Level']=df.Accident_Severity
df['Accident_Level'].replace({1:'Fatal', 2:'Serious', 3:'Slight'}, inplace=True)

df['Week']=df.Day_of_Week
df['Week'].replace({1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
                    5: "Thursday", 6: "Friday", 7: "Saturday"}, inplace=True)

#clean out other and unknown
df=df[~df['Weather_Conditions'].isin([8,9])]
df['weather_condition']=df.Weather_Conditions
df['weather_condition'].replace({1: "Fine no high winds", 2: "Raining no high winds", 3: "Snowing no high winds",
                        4: "Fine + high winds", 5: "Raining + high winds", 6: "Snowing + high winds",
                        7: "Fog or mist", 8: "Other", 9: "Unknown", -1: "Data missing or out of range"}, inplace=True)

df['light_condition']=df.Light_Conditions
df['light_condition'].replace({1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit",
                               6: "Darkness - no lighting", 7: "Darkness - lighting unknown",
                              -1: "Data missing or out of range"}, inplace=True)

df['Age_band']=df.Age_Band_of_Driver
df['Age_band'].replace({'1':"Under 16",'2':"Under 16",'3':"Under 16",'4':"16-25",'5':"16-25",'6':"26-35",
                        '7':"36-45",'8':"46-55",'9':"56-65",'10':"66-75",'11':"Over 75"}, inplace=True)

df['Vehicle_Manoeuvre'].replace({'1': "Reversing", '2': "Parked", '3': "Waiting to go - held up",
                         '4': "Slowing or stopping", '5': "Moving off", '6':"U-turn",
                         '7': "Turning left", '8': "Waiting to turn left", '9':"Turning right", '10': "Waiting to turn right",
                        '11': "Changing lane to left", '12': "Changing lane to right",
                        '13': "Overtaking moving vehicle - offside", '14': "Overtaking static vehicle - offside",
                        '15': "Overtaking - nearside", '16':"Going ahead left-hand bend",
                        '17': "Going ahead right-hand bend", '18': "Going ahead other", '99': "unknown (self reported)"
                            }, inplace=True)

df['Vehicle_Type'].replace({'1': "Pedal cycle", '2': "Motorcycle 50cc and under", '3': "Motorcycle 125cc and under",
                            '4': "Motorcycle over 125cc and up to 500cc", '5': "Motorcycle over 500cc",
                            '8': "Taxi/Private hire car", '9': "Car", '10': "Minibus (8 - 16 passenger seats)",
                            '11': "Bus or coach (17 or more pass seats)", '16': "Ridden horse",
                            '17': "Agricultural vehicle", '18': "Tram", '19': "Van / Goods 3.5 tonnes mgw or under",
                            '20': "Goods over 3.5t. and under 7.5t", '21': "Goods 7.5 tonnes mgw and over",
                            '22': "Mobility scooter", '23': "Electric motorcycle", '90': "Other vehicle",
                            '97': "Motorcycle - unknown cc", '98': "Goods vehicle - unknown weight",
                            '99': "Unknown vehicle type (self rep only)"}, inplace=True)

#manually set colormap for accident severity
acc_level_color = {'Fatal':'red', 'Serious':'orange', 'Slight':'lightseagreen'}

def weather_acc():
    df1 = df.groupby(['weather_condition', 'Accident_Level'])["Accident_Index"].nunique().reset_index(name='Amount_of_Accident')
    fig = px.bar(df1, x="weather_condition", y="Amount_of_Accident", color="Accident_Level",
                 title="Accident severity in different weather conditions")
    return fig

def line_acc():
    df1 = df.groupby(['Date','Accident_Level'])["Accident_Index"].nunique().reset_index(name='Amount_of_Accident')
    fig = px.line(df1, x='Date', y='Amount_of_Accident', color='Accident_Level',
                  color_discrete_map=acc_level_color)
    return fig


df['Police_Force'].replace(62, 60, inplace=True)
#merge North and South Wales
df['Police_Force'].replace(48, 1, inplace=True)
#merge metropolitan police and London city

city_select = [
	{'label': 'London', 'value': 1},
    {'label': 'Manchester', 'value': 6},
	{'label': 'Wales', 'value': 60}
	]

pie_select =[
    {'label': 'Age distribution of drivers', 'value': 'Age_band'},
	{'label': 'Vehicle Manoeuvre distribution', 'value': 'Vehicle_Manoeuvre'},
    {'label': 'Vehicle type distribution', 'value': 'Vehicle_Type'},
    {'label': 'Days of Week distribution', 'value': 'Week'},
]

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Great Britain Road Safety Data Dashboard"),
    html.Div(children=[
        html.Div(style={'float': 'left','width':'30vw'}, children=[
            dcc.Dropdown(
                id='pie_selection',
                options=pie_select, value='Week',
                multi=False),
            dcc.Graph(id='pie_figure'),
        ]),
        html.Div(style={'float': 'right','width':'66vw'}, children=[
            html.Br(),html.Br(),
            dcc.Graph(id='line_figure', figure=line_acc())
        ]),
    ]),

        html.Div(
            style={'float': 'left', 'width':'48vw', 'height':'100vh'},
            children=[
                html.H2("Environmental factors in general"),
                html.Br(),html.Br(),
                dcc.Checklist(['Exclude fine weather?'],
                              id='exclude_weather_left'),
                dcc.Graph(id='weather_general'),

                dcc.Checklist(['Exclude daylight?'],
                              id='exclude_day_left'),
                dcc.Graph(id='light_general')
            ]),

        html.Div(style={'float': 'right', 'width':'48vw', 'height':'100vh'},
            children=[
                html.H2("Environmental factors in specific area"),
                dcc.Dropdown(
                    id='city_selection',
                    options=city_select, value=1,
                    multi=False),
            html.Div(children=[
                dcc.Checklist(['Exclude fine weather?'],
                              id='exclude_weather_right'),
                dcc.Graph(id='weather_right'),

                dcc.Checklist(['Exclude daylight?'],
                              id='exclude_day_right'),
                dcc.Graph(id='light_right'),
            ])
        ])
    ])


@app.callback(
	Output('pie_figure', 'figure'),
	Input('pie_selection', 'value'))
def diff_pie(cat):
    df1 = df.groupby(cat)["Accident_Index"].nunique().reset_index(name='Amount_of_Accident')
    if cat == 'Week':
        fig=px.histogram(df1, x=cat, y='Amount_of_Accident', labels={'Amount_of_Accident': "accidents"},
                         category_orders={cat:["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]})
    else:
        fig = px.pie(df1, values='Amount_of_Accident', names=cat,
                     title="{} distribution of accidents".format(cat))
    return fig

@app.callback(
	Output('weather_general', 'figure'),
	Input('exclude_weather_left', 'value'))
def weather_general(exclude):
    df1=df
    if exclude:
        df1=df1[~df1['weather_condition'].isin(['Fine no high winds'])]
    df1=df1.groupby(['weather_condition', 'Accident_Level'])["Accident_Index"].nunique().reset_index(name='Amount_of_Accident')
    fig = px.bar(df1, x="weather_condition", y="Amount_of_Accident", color="Accident_Level",
                 color_discrete_map=acc_level_color,
                 title="Accident severity in different weather conditions")
    return fig


@app.callback(
	Output('weather_right', 'figure'),
	Input('city_selection', 'value'),
    Input('exclude_weather_right', 'value'))
def weather_right(city, exclude):
    if city==None:
        df1=df
    else:
        df1=df[df['Police_Force']==city]
    if exclude:
        df1=df1[~df1['weather_condition'].isin(['Fine no high winds'])]
    df1=df1.groupby(['weather_condition', 'Accident_Level'])["Accident_Index"].nunique().reset_index(name='Amount_of_Accident')
    fig = px.bar(df1, x="weather_condition", y="Amount_of_Accident", color="Accident_Level",
                 color_discrete_map=acc_level_color,
                 title="Accident severity in different weather conditions")
    return fig

@app.callback(
	Output('light_general', 'figure'),
    Input('exclude_day_left', 'value'))
def light_general(exclude):
    df1=df
    if exclude:
        df1=df1[~df1['light_condition'].isin(['Daylight'])]
    df1=df1.groupby(['light_condition', 'Accident_Level'])["Accident_Index"].nunique().reset_index(name='Amount_of_Accident')
    fig = px.bar(df1, x="light_condition", y="Amount_of_Accident", color="Accident_Level",
                 color_discrete_map=acc_level_color,
                 title="Accident severity in different light conditions")
    return fig

@app.callback(
	Output('light_right', 'figure'),
	Input('city_selection', 'value'),
    Input('exclude_day_right', 'value'))
def light_acc(city, exclude):
    if city==None:
        df1=df
    else:
        df1=df[df['Police_Force']==city]
    if exclude:
        df1=df1[~df1['light_condition'].isin(['Daylight'])]
    df1=df1.groupby(['light_condition', 'Accident_Level'])["Accident_Index"].nunique().reset_index(name='Amount_of_Accident')
    fig = px.bar(df1, x="light_condition", y="Amount_of_Accident", color="Accident_Level",
                 color_discrete_map=acc_level_color,
                 title="Accident severity in different light conditions")
    return fig


if __name__ == '__main__':
	app.run_server(debug=False)



'''
def month2seasons(x):
    if x in [9, 10, 11]:
        season = 'Autumn'
    elif x in [12, 1, 2]:
        season = 'Winter'
    elif x in [3, 4, 5]:
        season = 'Spring'
    elif x in [6, 7, 8]:
        season = 'Summer'
    return season

df['Season'] = df['month'].apply(month2seasons)
df.info()

def season_acc():
    df1 = df.groupby(['Season', 'Accident_Level'])["Accident_Index"].nunique().reset_index(name='Amount_of_Accident')
    fig = px.bar(df1, x="Season", y="Amount_of_Accident", color="Accident_Level",
                 title="Accident Severity in different seasons")
    return fig
'''