import dash
import datetime
from copy import copy

import pandas as pd
import seaborn as sns
import numpy as np

import dash
import dash_cytoscape as cyto
from dash import dcc
from dash.dependencies import Input, Output
from dash import html
from dash import dash_table

from sklearn.ensemble import IsolationForest

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

stylesheet = [
    {
        'selector': '.entrance',
        'style': {
            'width': 4,
            'height': 4,
            'background-color': '#4CFF00'
        }
    },
    {
        'selector': '.ranger-stop',
        'style': {
            'width': 4,
            'height': 4,
            'background-color': '#FFD800'
        }
    },
    {
        'selector': '.gate',
        'style': {
            'width': 4,
            'height': 4,
            'background-color': '#FF0000'
        }
    },
    {
        'selector': '.general-gate',
        'style': {
            'width': 4,
            'height': 4,
            'background-color': '#00FFFF'
        }
    },
    {
        # Because this doesn't have a number, slicing breaks a bit
        'selector': '.ranger-bas',
        'style': {
            'width': 4,
            'height': 4,
            'background-color': '#FF00DC'
        }
    },
    {
        'selector': '.camping',
        'style': {
            'width': 4,
            'height': 4,
            'background-color': '#FF6A00'
        }
    },
    {
        'selector': '.waypoint',
        'style': {
            'width': 0.1,
            'height': 0.1,
            'background-color': 'white'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'line-color': 'white',
            'width': 1
        }
    }]
NR_COLORS_IN_PALETTE = 256

plot_colors = {
    'background': '#C1C1C1',
    'text': '#000000'
}

locations = pd.read_csv('data/locations.csv')
waypoints = pd.read_csv('data/waypoints.csv')
aug_paths = pd.read_csv('data/augmented_paths.csv')
trace_data = pd.read_csv('data/trace_data.csv', parse_dates=['initial-timestamp'])

# Store all initial timestamps as initial dates as well, simplifies later interpretation
#   and guarantees avoidance of unintended time of day cutoff moments.
# Used by the map.
trace_data['initial-date'] = pd.DatetimeIndex(trace_data['initial-timestamp']).date

# Load activity data, and forcibly interpret the dates as base Python dates
# to simplify later interpretation and avoid accidental unintended cutoffs.
# Also change the trace_id lists from their string form to actual lists.
# Used by the line graph.
activity_data = pd.read_csv('data/activity_per_date_per_sensor_pair.csv',
                            parse_dates=['date'])
activity_data['date'] = [timestamp.date() for timestamp in activity_data['date']]
activity_data['traces-detected'] = [eval(f'list({trace_id_list})') \
                                    for trace_id_list in activity_data['traces-detected']]

# Dictionary mapping each trace's ID to its vehicle type
traces_to_types = dict(zip(trace_data['car-id'], trace_data['car-type']))

first_date = activity_data['date'].min()
last_date = activity_data['date'].max()
nr_days_in_data = (last_date - first_date).days

df_sensor = pd.read_csv("data/Lekagul Sensor Data.csv", parse_dates=['Timestamp'])
df_sensor["gate_type"] = df_sensor["gate-name"]
df_sensor["gate_type"].replace(
    ['ranger-stop0', 'ranger-stop1', 'ranger-stop2', 'ranger-stop3', 'ranger-stop4', 'ranger-stop5', 'ranger-stop6',
     'ranger-stop7', 'ranger-base'], "ranger-stop", inplace=True)
df_sensor["gate_type"].replace(['gate0', 'gate1', 'gate2', 'gate3', 'gate4', 'gate5', 'gate6', 'gate7', 'gate8'],
                               "gate", inplace=True)
df_sensor["gate_type"].replace(
    ['camping0', 'camping1', 'camping2', 'camping3', 'camping4', 'camping5', 'camping6', 'camping7', 'camping8'],
    "camping", inplace=True)
df_sensor["gate_type"].replace(
    ['general-gate0', 'general-gate1', 'general-gate2', 'general-gate3', 'general-gate4', 'general-gate5',
     'general-gate6', 'general-gate7'], "general-gate", inplace=True)
df_sensor["gate_type"].replace(['entrance0', 'entrance1', 'entrance2', 'entrance3', 'entrance4'], "entrance",
                               inplace=True)

df_sensor["Vehicle_type"] = df_sensor["car-type"]
df_sensor["Vehicle_type"].replace({"1": "2 axle car (or motorcycle)", "2": "2 axle truck", "3": "3 axle truck",
                                   "4": "4 axle (and above) truck", "5": "2 axle bus", "6": "3 axle bus",
                                   "2P": "Park service vehicles"}, inplace=True)

df_sensor['initial-date'] = pd.DatetimeIndex(df_sensor.Timestamp).date
first = df_sensor['Timestamp'].min().date()

gate_type_option = [
    {'label': 'All gates', 'value': 'all'},
    {'label': 'Ranger stop', 'value': 'ranger-stop'},
    {'label': 'Gate', 'value': 'gate'},
    {'label': 'Camping', 'value': 'camping'},
    {'label': 'General gate', 'value': 'general-gate'},
    {'label': 'Entrance', 'value': 'entrance'},
]

# Create a mapping from car_type in the dataset to easily understandable labels
#   as they should be shown in the visualisation.
car_type_to_description = [
    {'label': '2 axle car (or motorcycle)', 'value': '1'},
    {'label': '2 axle truck', 'value': '2'},
    {'label': '3 axle truck', 'value': '3'},
    {'label': '4 axle (and above) truck', 'value': '4'},
    {'label': '2 axle bus', 'value': '5'},
    {'label': '3 axle bus', 'value': '6'},
    {'label': 'Preserve ranger vehicles', 'value': '2P'}
]

bar_chart_legend_colors = {
    # Color scheme used is Plotly's default, we set it manually
    #   here to force consistent colors for each type.
    '2 axle car (or motorcycle)': '#666FF8',
    '2 axle truck': '#EF553D',
    '3 axle truck': '#01CB91',
    '4 axle (and above) truck': '#FF6692',
    '2 axle bus': '#FEA05A',
    '3 axle bus': '#15D1F6',
    'Park service vehicles':'#AC63FB'
}

aug_paths_numpy = aug_paths.to_numpy()
hex_palette = sns.color_palette("Blues_r", n_colors=NR_COLORS_IN_PALETTE).as_hex()


def agg_lists(series_of_lists):
    """
        Helper function for line graph when showing one graph per
        vehicle type. Received a series of "lists" passed as strings,
        converts them to actual lists and concatenates them.
    """
    full_list = []
    for row in series_of_lists:
        full_list.extend(row)
    return list(np.unique(full_list))


def check_existence(parts, pair):
    """Helper function to check if a tuple of elements 'pair' is present in the list 'parts'."""
    for check_pair in parts:
        if check_pair[0] == pair[0] and check_pair[1] == pair[1]:
            return True
    return False


def get_path_parts(full_path):
    """
        Function that takes in a list of sensors crossed in a trace, and
        puts each unique pair of subsequent sensors in that list into
        a list of (from, to) sensor pairs that describes all paths that 
        this particular trace covered that was not already present in
        all_unique_paths from earlier traces driving that path.
        Note that (from, to) always has the two sensors in alphabetical order.
    """
    parts = []
    for i in range(0, len(full_path) - 1):
        if full_path[i] < full_path[i + 1] and \
                not check_existence(all_unique_paths, (full_path[i], full_path[i + 1])):
            parts.append((full_path[i], full_path[i + 1]))

        elif full_path[i] >= full_path[i + 1] and \
                not check_existence(all_unique_paths, (full_path[i + 1], full_path[i])):
            parts.append((full_path[i + 1], full_path[i]))
    return parts


def calculate_edge_activity(trace_dataframe):
    """
        Function that makes use of the previously determined list containing
        all unique paths, and given a dataframe containing a collection of traces
        iterates over every trace and over each component of the route driven by
        that trace. Each of those components is tracked as a single occurence of
        'activity on that road', and all of those occurences are collected together
        in an edge_activity_dictionary describigin for every (from, to) sensor pair
        how many times a vehicle drove over the path between those sensors.
        Note that (from, to) always has the two sensors in alphabetical order.
    """
    # Finding the column number is needed because of using 
    #   numpy instead of pandas (done purely for speed)
    route_taken_column = list(trace_dataframe.columns).index('route-taken')

    trace_data_numpy = trace_dataframe.to_numpy()
    edge_activity_dictionary = {tuple(route_section): 0 for route_section in all_unique_paths}

    for index in range(0, trace_data_numpy.shape[0]):
        route_taken = trace_data_numpy[index][route_taken_column].split('/')
        for i in range(0, len(route_taken) - 1):
            if route_taken[i] < route_taken[i + 1]:
                first_sensor = route_taken[i]
                second_sensor = route_taken[i + 1]
            else:
                first_sensor = route_taken[i + 1]
                second_sensor = route_taken[i]

            sub_aug_paths = aug_paths_numpy[aug_paths_numpy[:, 1] == first_sensor]
            sub_aug_paths = sub_aug_paths[sub_aug_paths[:, 2] == second_sensor]
            if not sub_aug_paths.shape[0]:
                # This is a self loop (the same sensor is triggered twice in a row).
                # There is no road activity to track here.
                continue

            augmented_route_section = sub_aug_paths[0][0]
            augmented_route_section = augmented_route_section.split('/')

            for j in range(0, len(augmented_route_section) - 1):
                sensor_1 = augmented_route_section[j]
                sensor_2 = augmented_route_section[j + 1]
                if sensor_1 < sensor_2:
                    edge_activity_dictionary[(augmented_route_section[j],
                                              augmented_route_section[j + 1])] += 1
                else:
                    edge_activity_dictionary[(augmented_route_section[j + 1],
                                              augmented_route_section[j])] += 1

    return edge_activity_dictionary


all_unique_paths = []
for _, row in aug_paths.iterrows():
    all_unique_paths.extend(get_path_parts(row['paths'].split('/')))


def iso_forest(data, cont):
    """
        Function that takes in a dataset and a contamination parameter,
        and uses these arguments to build an isolation forest on all data
        which is subsequently used to predict which of these data points
        are outliers.
    """
    all_vars = ['time-taken-hours', 'distance-travelled-km', 'avg-kmh',
                'sensor-count', 'number-campings', 'unique-sensors']
    df_iso = data[all_vars]
    df_new = copy(data)
    
    #Make Isolation Forest Model
    model=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(cont),
                          max_features=1.0, random_state=42)
    model.fit(df_iso.values)

    #Find anomalies
    df_new['scores']=model.decision_function(df_iso.values)
    df_new['anomaly']=model.predict(df_iso.values)

    # Select and return anomalies
    anomaly=df_new.loc[df_new['anomaly']==-1]
    return anomaly


# These are used as global variables in the dashboards, modified
#   by map updates and then used in the tooltips of each one.
edge_activity_dictionary = calculate_edge_activity(trace_data)
anomaly_edge_activity    = calculate_edge_activity(trace_data)

# Scaling the edge activities to color range values (0-NR_COLORS_IN_PALETTE)
# No backup for 0 activity is needed here, never happens on the default settings.
max_value = max(edge_activity_dictionary.values())
for key in edge_activity_dictionary:
    edge_activity_dictionary[key] = int(round(edge_activity_dictionary[key]
                                              / max_value
                                              * NR_COLORS_IN_PALETTE - 1, 0))

locations_vertices = [{'data': {'id': row['location']},
                       'position': {'x': row['x'], 'y': row['y']},
                       'classes': row['location'][:-1]} \
                      for _, row in locations.iterrows()]

waypoints_vertices = [{'data': {'id': row['location']},
                       'position': {'x': row['x'], 'y': row['y']},
                       'classes': 'waypoint',
                       'selectable': False} \
                      for _, row in waypoints.iterrows()]

all_edges = [{'data': {'source': pair[0], 'target': pair[1]},
              'selectable': False,
              'style': {'line-color': hex_palette[edge_activity_dictionary[pair]]}} \
                  for pair in edge_activity_dictionary]

all_elements = copy(locations_vertices)
all_elements.extend(waypoints_vertices)
all_elements.extend(all_edges)

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='edge_activity_dictionary'),
    dcc.Store(id='anomaly_edge_activity'),
    dcc.Store(id='df_iso'),
    dcc.Tabs(className='tabs_container', children=[
        dcc.Tab(label='General Graphs', className='tab', selected_style={'backgroundColor': '#C1C1C1'}, 
            children=[
            html.Div(
                id='left_side_div',
                children=[
                    html.H1(className='textbox', children=["Boonsong Lekagul Nature Preserve"]),
                    cyto.Cytoscape(
                        id='cytoscape_map',
                        className='cytoscape_map',
                        layout={'name': 'preset'},
                        # Cytoscape styling for specifically size
                        #   can't be done via CSS for some reason.
                        style={'width': '42vw', 'height': '67vh'},
                        elements=all_elements, stylesheet=stylesheet,
                        userPanningEnabled=False, userZoomingEnabled=False,
                        boxSelectionEnabled=True, autolock=True
                    ),
                    html.Div(id='map_bottom_spacing'),
                    html.Div(
                        id='sensor_legend',
                        children=[html.Img(src='https://i.ibb.co/d5ptH3X/Sensor-legend.png')]
                    ),
                    html.Div(
                        id='legend_div',
                        children=[
                            html.Img(id='color_bar', src='https://i.ibb.co/tpYqbdd/colors.png'),
                            html.Div(children='0'),
                            html.Div(id='legend_max_val', children=str(max_value))
                        ]
                    ),
                    html.Div(
                        id='sensor_list_div',
                        children=[
                            html.P(id='map_selectedSensor'),
                            html.P(id='map_selectedRoad')])
                ]),
            html.Div(id='right_side_div',
                children=[
                    html.Div(id="right_top_space_div"),

                    # Maybe there is a cleaner way to do this, but this way works to have
                    #   a full range where the user can select any range from the very first day
                    #   in the dataset to the very last day.
                    dcc.RangeSlider(
                        0,nr_days_in_data-1,1,
                        id='date_slider',
                        marks={
                             0: 'May 2015',
                            31: 'June 2015',
                            61: 'July 2015',
                            92: 'Aug 2015',
                            123: 'Sept 2015',
                            153: 'Oct 2015',
                            184: 'Nov 2015',
                            214: 'Dec 2015',
                            245: 'Jan 2016',
                            276: 'Feb 2016',
                            305: 'March 2016',
                            336: 'April 2016',
                            366: 'May 2016'
                        }
                    ),

                    # This is the dropdown where the user can select which individual
                    #   car_types they want to have visualised.
                    dcc.Dropdown(
                        id='type_selection',
                        className='menu',
                        placeholder='Select individual car types...',
                        options=car_type_to_description,
                        multi=True),

                    dcc.Checklist(['Equal Y-axes between vehicle types?',
                                    'Show average KM travelled per day?'],
                                    id='equal_axes', className='menu'),

                    dcc.Graph(className='general_graph', id='line_graphs_figure'),

                    dcc.Dropdown(
                        id='gate_selection',
                        className='menu',
                        placeholder='Select a gate type...',
                        options=gate_type_option,
                        multi=False),

                    dcc.Graph(className='general_graph test', id='bar_graphs_figure')

                ])

        ]),

        dcc.Tab(label='Check the anomalies', className='tab', selected_style={'backgroundColor': '#C1C1C1'},
            children=[
            html.Div(id='anomaly_left_div', 
                children=[
                    html.H1(className='textbox', children=["Detected anomalies"]),
                    dcc.Dropdown(
                        options = car_type_to_description,
                        value = '1',
                        id = 'car_selector',
                        className = 'menu',
                        clearable=False
                    ),

                    html.Div(
                        id = 'map_and_legends',
                        children=[
                            html.H3(className='textbox', id='anomaly_map_title',
                                    children=["Roads driven most often by anomalies"]),
                            cyto.Cytoscape(
                                id='cytoscape_map_2',
                                className='cytoscape_map',
                                layout={'name': 'preset'},
                                # Cytoscape styling for specifically size
                                #   can't be done via CSS for some reason.
                                style={'width': '25vw', 'height': '45vh', 'float':'left'},
                                elements=all_elements, stylesheet=stylesheet,
                                userPanningEnabled=False, userZoomingEnabled=False,
                                boxSelectionEnabled=True, autolock=True
                            ),
                            html.Div(id='anomaly_legends',
                                children=[
                                    html.Div(
                                        className='legend_left_float',
                                        children=[html.Img(src='https://i.ibb.co/d5ptH3X/Sensor-legend.png')]
                                    ),
                                    html.Div(
                                        className='legend_left_float',
                                        children=[
                                            html.Img(
                                                id='anomaly_color',
                                                src='https://i.ibb.co/tpYqbdd/colors.png'
                                            ),
                                            html.Div(children='0'),
                                            html.Div(id='legend_max_val_2', children=str(max_value))
                                    ]),
                                    html.Div(
                                        id='anomaly_tooltips',
                                        children=[
                                            html.P(id='anomaly_map_selectedSensor'),
                                            html.P(id='anomaly_map_selectedRoad')])
                                                ])
                        ]),

                    html.Div(
                        children=[
                            dcc.Graph(
                                id = 'radar_graph_1')
                    ]),

                    html.Div(
                        children=[
                            dcc.Graph(
                                id = 'radar_graph_2'
                            )
                    ])
            ]),

            html.Div(id='anomaly_right_div', children=[
                html.H3(className='textbox',
                        children=['Anomalies found, sorted descending by degree of abnormality:']),

                html.Div(id = 'anomaly_table')
            ])

            
        ])
    ])
])


@app.callback(
    Output('line_graphs_figure', 'figure'),
    Input('type_selection', 'value'),
    Input('date_slider', 'value'),
    Input('equal_axes', 'value'),
    Input('cytoscape_map', 'selectedNodeData'))
def update_line_graph(vehicle_types, date_range, equal_axes, node_list):
    # Make a copy of the dataset, so that filters and stuff do not
    #   affect the original loaded data. No memory issues, as the dataset
    #   we are using here (processed data) is relatively small.
    graph_data = copy(activity_data)

    if equal_axes and ('Show average KM travelled per day?' in equal_axes):
        average_requested = True
    else:
        average_requested = False
    
    # Note that ALL menu options are None, instead of True or False,
    #   until the user interactions with them for the first time. Using
    #   an if statement like this catches both the None and == False options.
    if date_range:
        range_start = first_date + datetime.timedelta(date_range[0])
        range_end = first_date + datetime.timedelta(date_range[1])
        graph_data = graph_data[graph_data['date'] >= range_start]
        graph_data = graph_data[graph_data['date'] <= range_end]

    if node_list:
        node_subset_used = True
        sensor_list = [data['id'] for data in node_list]
        graph_data = graph_data[graph_data["from-sensor"].isin(sensor_list) \
                                | graph_data["to-sensor"].isin(sensor_list)]
    else:
        node_subset_used = False

    aggregated_by_date = graph_data.groupby(by=['date'], as_index=False) \
                                   .agg({'distance-travelled-km':sum, 'traces-detected':agg_lists})
    
    if vehicle_types:
        nr_types = len(vehicle_types)
        
        # This check has to be done this way, because the checkbox is None
        #   until it is first clicked. None is neither True nor False, so
        #   cannot be immediately used as a parameter for make_subplots.
        if equal_axes and ('Equal Y-axes between vehicle types?' in equal_axes):
            fig = make_subplots(1, nr_types, shared_yaxes=True)
        else:
            fig = make_subplots(1, nr_types)

        # Small indicator title to show which visualisation mode we're in.
        # Mainly for testing purposes.
        fig.update_layout(title_text='Multiple graph mode')

        agg_by_date_numpy = aggregated_by_date.to_numpy()
        new_df_rows = []
        # Using numpy for speed purposes, do a manual splitting of aggregated rows
        #   by separating each type and splitting the distance travelled between types.
        for row in agg_by_date_numpy:
            date = row[0]
            distance_travelled = row[1]
            nr_traces = len(row[2])
            occurences_per_type = {'1': [], '2': [], '2P': [], '3': [], '4': [], '5': [], '6': []}
            
            # Each (aggregated) activity data row contains a list of all traces that
            #   were seen there. Here we split that list into one list per type.
            for trace_id in row[2]:
                occurences_per_type[traces_to_types[trace_id]].append(trace_id)
            
            # Then after splitting up the occurrences, determine the distance covered
            #   per vehicle type and store that as a row in the new dataframe-to-be.
            for car_type in occurences_per_type:
                fraction_of_total_on_date = len(occurences_per_type[car_type])/nr_traces
                new_df_rows.append([date, car_type, distance_travelled*fraction_of_total_on_date,
                                    occurences_per_type[car_type]])
                
        agg_by_date_and_type = pd.DataFrame(new_df_rows,
                                            columns=['date', 'car-type',
                                                     'distance-travelled-km', 'traces-detected'])

        for type_index in range(len(vehicle_types)):
            # First get the specific type of vehicle for this subplot, then
            #   use an annoyingly complex way to select the corresponding label
            #   from the dictionary that should be shown in the legend.
            specific_type = vehicle_types[type_index]
            type_description = next(item for item in car_type_to_description \
                                    if item["value"] == specific_type)['label']

            # Another, smaller copy to further modify for single types.
            data_subset = copy(agg_by_date_and_type)
            data_subset = data_subset[data_subset['car-type'] == specific_type]
            if average_requested:
                # Use a quick lambda function to calculate each row's average.
                calculate_avg = lambda dist, nr_traces: dist/nr_traces if nr_traces > 0 else 0
                data_subset['distance-travelled-km'] = [calculate_avg(dist, len(traces)) \
                                                       for dist, traces \
                                                       in zip(data_subset['distance-travelled-km'],
                                                              data_subset['traces-detected'])]

            # Note how easy it is to convert it to KM here instead of meters.
            fig.add_trace(trace=go.Scatter(x=data_subset["date"],
                                           y=data_subset['distance-travelled-km'],
                                           name=type_description),
                          row=1, col=type_index+1)
    else:
        # No types were selected, so just show the total sum/average as a single graph.
        
        if average_requested:
            # Use a quick lambda function to calculate each row's average.
            calculate_avg = lambda dist, nr_traces: dist/nr_traces if nr_traces > 0 else 0
            aggregated_by_date['distance-travelled-km'] = [calculate_avg(dist, len(traces)) \
                                                          for dist, traces \
                                                          in zip(aggregated_by_date['distance-travelled-km'],
                                                                 aggregated_by_date['traces-detected'])]
        
        # Probably slightly unnecessary to make a 1x1 subplot, but oh well.
        fig = make_subplots(1,1)
        fig.update_layout(title_text='This is a single graph')
        
        fig.add_trace(trace=go.Scatter(x=aggregated_by_date["date"],
                                       y=aggregated_by_date['distance-travelled-km']),
                      row=1, col=1)
    
    fig.update_xaxes(title='Date')

    # For both options: Only put y-axis label at the left end of the graph,
    #  otherwise it repeats for every subplot.
    if average_requested:
        fig.update_yaxes(title_text="Kilometers traveled in day", row = 1, col = 1)
    else:
        fig.update_yaxes(title_text="Kilometers traveled in day", row = 1, col = 1)

    if average_requested:
        graph_title = 'Average kilometers traveled per day'
    else:
        graph_title = 'Total kilometers traveled per day'

    if node_subset_used:
        graph_title += ' in selected area'

    if vehicle_types:
        graph_title += ' per selected vehicle type'

    fig.update_layout(
        paper_bgcolor=plot_colors['background'],
        font_color=plot_colors['text'],
        title_text=graph_title
    )

    return fig


@app.callback(
    Output('gate_selection', 'disabled'),
    Input('cytoscape_map', 'selectedNodeData'))
def enable_disable_gate_selection(node_list):
    if node_list:
        # Some nodes on the map are selected, disable the menu.
        return True
    else:
        # No nodes on the map are selected, (re-) enable the menu.
        return False


@app.callback(
    Output('bar_graphs_figure', 'figure'),
    Input('date_slider', 'value'),
    Input('gate_selection', 'value'),
    Input('cytoscape_map', 'selectedNodeData'))
def car_distribution(date_range, gatetype, node_list):
    dft = copy(df_sensor)

    #Filter for date-range selected via slider
    if date_range:
        range_start = first_date + datetime.timedelta(date_range[0])
        range_end = first_date + datetime.timedelta(date_range[1])
        dft = dft[dft['initial-date'] >= range_start]
        dft = dft[dft['initial-date'] <= range_end]

    if node_list:
        sensor_list = [data['id'] for data in node_list]
        df2 = dft[dft['gate-name'].isin(sensor_list)]
        df2 = df2.groupby(['gate-name', 'Vehicle_type'])["car-id"].nunique().reset_index(name='car-id')
        fig = px.bar(df2, x="gate-name", y="car-id", color="Vehicle_type",
                     labels={'Vehicle_type': "Vehicle type"},
                     color_discrete_map=bar_chart_legend_colors, 
                     category_orders={'Vehicle_type': bar_chart_legend_colors.keys()},
                     title="Vehicle Distribution")
        fig.update_xaxes(title='Sensor name')
    else:
        if gatetype == "all" or gatetype == None:
            df1 = copy(dft)
            df1 = df1.groupby(['gate_type', 'Vehicle_type'])["car-id"].nunique().reset_index(name='car-id')
            fig = px.bar(df1, x="gate_type", y="car-id", color="Vehicle_type",
                         labels={'Vehicle_type': "Vehicle type"},
                         color_discrete_map=bar_chart_legend_colors,
                         category_orders={'Vehicle_type': bar_chart_legend_colors.keys()},
                         title="Vehicle Distribution in general")
            fig.update_xaxes(title='Sensor type')
        else:
            df1 = dft[dft.gate_type == gatetype]
            df1 = df1.groupby(['gate-name', 'Vehicle_type'])["car-id"].nunique().reset_index(name='car-id')
            fig = px.bar(df1, x="gate-name", y="car-id", color="Vehicle_type",
                         labels={'Vehicle_type': "Vehicle type"},
                         color_discrete_map=bar_chart_legend_colors,
                         category_orders={'Vehicle_type': bar_chart_legend_colors.keys()},
                         title="Vehicle Distribution at {}".format(gatetype))
            fig.update_xaxes(title='Sensor name')

    fig.update_yaxes(title_text="Number of unique vehicles", row = 1, col = 1)

    fig.update_layout(
        paper_bgcolor=plot_colors['background'],
        font_color=plot_colors['text']
    )

    return fig


@app.callback(Output('cytoscape_map', 'elements'),
              Output('legend_max_val', 'children'),
              Input('type_selection', 'value'),
              Input('date_slider', 'value'))
def update_normal_map_elements(vehicle_types, date_range):
    # Make a copy of the dataset, so that filters and stuff do not
    #   affect the original loaded data. No memory issues, as the dataset
    #   we are using here (processed data) is relatively small.
    graph_data = copy(trace_data)

    # Note that ALL menu options are None, instead of True or False,
    #   until the user interactions with them for the first time. Using
    #   an if statement like this catches both the None and == False options.
    if date_range:
        range_start = first_date + datetime.timedelta(date_range[0])
        range_end = first_date + datetime.timedelta(date_range[1])
        graph_data = graph_data[graph_data['initial-date'] >= range_start]
        graph_data = graph_data[graph_data['initial-date'] <= range_end]

    if vehicle_types:
        # This constructs a boolean series describing for each row
        #   whether it contains one of the selected types.
        code_string = f"(graph_data['car-type'] == '{vehicle_types[0]}')"
        for index in range(1, len(vehicle_types)):
            code_string += f" | (graph_data['car-type'] == '{vehicle_types[index]}')"
        boolean_series = eval(code_string)
        graph_data = graph_data[boolean_series]

    # Editing the global version of these variables, so that tooltips work.
    global edge_activity_dictionary
    edge_activity_dictionary = calculate_edge_activity(graph_data)

    scaled_edge_activity = {}

    max_value = max(edge_activity_dictionary.values())
    for key in edge_activity_dictionary:
        if edge_activity_dictionary[key] / max_value * NR_COLORS_IN_PALETTE < 1:
            # Scaling too small values results in a scaled value < 0.
            scaled_edge_activity[key] = 0
        else:
            # Scaling the edge activities to color range values (0-NR_COLORS_IN_PALETTE)
            scaled_edge_activity[key] = int(round(edge_activity_dictionary[key]
                                                  / max_value
                                                  * NR_COLORS_IN_PALETTE - 1, 0))

    # Instantiate every "road" in the graph as an edge from point to point with its calculated colour.
    all_edges = [{'data': {'source': pair[0], 'target': pair[1]},
                  'selectable': False, 'style': {'line-color': hex_palette[scaled_edge_activity[pair]]}} \
                 for pair in edge_activity_dictionary]

    # It is easier and has a minimal speed difference to 
    #   rebuild all_elements this way instead of finding the right
    #   edges and updating each of them in the existing all_elements.
    all_elements = copy(locations_vertices)
    all_elements.extend(waypoints_vertices)
    all_elements.extend(all_edges)
    return all_elements, max_value


@app.callback(Output('map_selectedSensor', 'children'),
              Input('type_selection', 'value'),
              Input('date_slider', 'value'),
              Input('cytoscape_map', 'mouseoverNodeData'))
def displaySelectedNodeData(vehicle_types, date_range, data):
    if data is None:
        return "No sensor (-s) selected."
    else:
        sensor_name = data['id']

        tooltip_data = copy(df_sensor)
        tooltip_data = tooltip_data[tooltip_data['gate-name'] == sensor_name]

        # Note that ALL menu options are None, instead of a (possibly empty) list
        #   of values, until the user interactions with them for the first time.
        #   Using an if statement like this catches both the None and
        #   empty list (happens if nothing is selected) options.
        if vehicle_types:
            # This constructs a boolean series describing for each row
            #   whether it contains one of the selected types.
            code_string = f"(tooltip_data['car-type'] == '{vehicle_types[0]}')"
            for index in range(1, len(vehicle_types)):
                code_string += f" | (tooltip_data['car-type'] == '{vehicle_types[index]}')"
            boolean_series = eval(code_string)
            tooltip_data = tooltip_data[boolean_series]

        if date_range:
            range_start = first_date + datetime.timedelta(date_range[0])
            range_end = first_date + datetime.timedelta(date_range[1])
            tooltip_data = tooltip_data[tooltip_data['initial-date'] >= range_start]
            tooltip_data = tooltip_data[tooltip_data['initial-date'] <= range_end]

        total_visits = tooltip_data.shape[0]

        if vehicle_types:
            return f"The selected vehicle types visited {sensor_name} {total_visits} times."
        else:
            return f"Vehicles visited {sensor_name} {total_visits} times."


@app.callback(Output('map_selectedRoad', 'children'),
              Input('cytoscape_map', 'mouseoverEdgeData'),
              Input('type_selection', 'value'))
def displaySelectedEdgeData(data, vehicle_types):
    if data is None:
        return "No road has been selected."
    else:
        from_sensor = data['source']
        to_sensor = data['target']

        # Edge activity dictionary uses alphabetically ordered tuples for access,
        #   so we check like this to get the activity for this specific path.
        if from_sensor < to_sensor:
            edge_activity_number = edge_activity_dictionary[(from_sensor, to_sensor)]
        else:
            edge_activity_number = edge_activity_dictionary[(to_sensor, from_sensor)]

        if vehicle_types:
            return f"The selected vehicle types drove on this section of road {edge_activity_number} times."
        else:
            return f"Vehicles drove on this section of road {edge_activity_number} times."


@app.callback(
    Output('df_iso', 'data'),
    Input('car_selector', 'value'))
def update_car_selector(car_selector):
    df_iso = trace_data[(trace_data['car-type'] == car_selector)]
    return copy(iso_forest(df_iso, 0.01)).to_json()


@app.callback(
    Output('radar_graph_1', 'figure'),
    Input('car_selector', 'value')
)
def update_average_radar(car_selector):
    #Filter dataset for selected car-type. Get means of attributes.
    df_radar = trace_data[(trace_data['car-type'] == car_selector)]
    df = df_radar[['time-taken-hours', 'distance-travelled-km', 'avg-kmh', 'sensor-count', 'unique-sensors', 'number-campings']].mean().round(2)

    #Construct radar graph 1
    fig = px.line_polar(df, r=[df['time-taken-hours'], df['distance-travelled-km'], df['avg-kmh'], df['sensor-count'], df['unique-sensors'], df['number-campings']],
                        theta = df.index, line_close = True)

    fig.update_layout(title='Data Average',
                      modebar_add=['zoom'],
                      )

    fig.update_layout(
        paper_bgcolor=plot_colors['background'],
        font_color=plot_colors['text']
    )

    return fig


@app.callback(
    Output('radar_graph_2', 'figure'),
    Input('df_iso', 'data')
)
def update_anomaly_radar(df_iso):
    #Get dataset of anomalies for selected car type, calculate means
    df1 = pd.read_json(df_iso)
    df2 = df1[['time-taken-hours', 'distance-travelled-km', 'avg-kmh', 'sensor-count', 'unique-sensors', 'number-campings']].mean().round(2)

    #Construct radar graph 2 (for anomalies)
    fig = px.line_polar(df2, r=[df2['time-taken-hours'], df2['distance-travelled-km'], df2['avg-kmh'], df2['sensor-count'], df2['unique-sensors'], df2['number-campings']],
                        theta=df2.index, line_close=True)

    fig.update_layout(title='Anomaly Average')

    fig.update_layout(
        paper_bgcolor=plot_colors['background'],
        font_color=plot_colors['text']
    )

    return fig


@app.callback(
    Output('anomaly_table', 'children'),
    Input('df_iso', 'data')
)
def update_datatable(df_iso):
    #Get full dataset of anomalies of selected car-type
    df_table = pd.read_json(df_iso, convert_dates=['initial-timestamp'])
    df_show = copy(df_table[['car-id', 'initial-timestamp', 'time-taken-hours', 'distance-travelled-km', 'avg-kmh', 'sensor-count', 'unique-sensors',
                             'number-campings', 'route-taken', 'trace-complete', 'scores']].sort_values(by='scores').round(2))
    df_show = df_show.drop(columns='scores')

    data = df_show.to_dict('rows')

    columns = [{"name": i, "id": i,} for i in (df_show.columns)]
    return dash_table.DataTable(data=data, columns=columns,
                                style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0},

                                tooltip_data=[
                                    {
                                        column: {'value': str(value), 'type': 'markdown'}
                                        for column, value in row.items()
                                    } for row in data
                                ],
                                tooltip_duration=None)

@app.callback(
    Output('cytoscape_map_2', 'elements'),
    Output('legend_max_val_2', 'children'),
    Input('df_iso', 'data'))
def update_anomaly_map(df_iso):
    graph_data = pd.read_json(df_iso)

    global anomaly_edge_activity
    anomaly_edge_activity = calculate_edge_activity(graph_data)

    scaled_edge_activity = {}

    max_value = max(anomaly_edge_activity.values())
    for key in anomaly_edge_activity:
        if anomaly_edge_activity[key] / max_value * NR_COLORS_IN_PALETTE < 1:
            # Scaling too small values results in a scaled value < 0.
            scaled_edge_activity[key] = 0
        else:
            # Scaling the edge activities to color range values (0-NR_COLORS_IN_PALETTE)
            scaled_edge_activity[key] = int(round(anomaly_edge_activity[key]
                                                  / max_value
                                                  * NR_COLORS_IN_PALETTE - 1, 0))

    all_edges = [{'data': {'source': pair[0], 'target': pair[1]},
                  'selectable': False, 'style': {'line-color': hex_palette[scaled_edge_activity[pair]]}} \
                 for pair in anomaly_edge_activity]

    all_elements = copy(locations_vertices)
    all_elements.extend(waypoints_vertices)
    all_elements.extend(all_edges)
    return all_elements, max_value


@app.callback(Output('anomaly_map_selectedSensor', 'children'),
              Input('cytoscape_map_2', 'mouseoverNodeData'),
              Input('df_iso', 'data'))
def displaySelectedNodeData_anomalies(data, df_iso):
    if data is None:
        return "No sensor (-s) selected."
    else:
        anomaly_car_ids = list(pd.read_json(df_iso)['car-id'])

        sensor_name = data['id']

        tooltip_data = copy(df_sensor)
        tooltip_data = tooltip_data[tooltip_data['gate-name'] == sensor_name]
        
        # This constructs a boolean series describing for each row
        #   whether it contains one of the selected types.
        code_string = f"(tooltip_data['car-id'] == '{anomaly_car_ids[0]}')"
        for index in range(1, len(anomaly_car_ids)):
            code_string += f" | (tooltip_data['car-id'] == '{anomaly_car_ids[index]}')"
        boolean_series = eval(code_string)
        tooltip_data = tooltip_data[boolean_series]

        total_visits = tooltip_data.shape[0]

        return f"Detected anomalies of this type visited {sensor_name} {total_visits} times."


@app.callback(Output('anomaly_map_selectedRoad', 'children'),
              Input('cytoscape_map_2', 'mouseoverEdgeData'))
def displaySelectedEdgeData_anomalies(data):
    if data is None:
        return "No road has been selected."
    else:
        from_sensor = data['source']
        to_sensor = data['target']

        # Edge activity dictionary uses alphabetically ordered tuples for access,
        #   so we check like this to get the activity for this specific path.
        if from_sensor < to_sensor:
            edge_activity_number = anomaly_edge_activity[(from_sensor, to_sensor)]
        else:
            edge_activity_number = anomaly_edge_activity[(to_sensor, from_sensor)]

        return f"Detected anomalies of this type drove on this section of road {edge_activity_number} times."


if __name__ == '__main__':
    app.run_server(debug=False)
