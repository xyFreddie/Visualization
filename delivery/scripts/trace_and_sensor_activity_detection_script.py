import pandas as pd
from datetime import datetime
from collections import defaultdict
from copy import copy
import warnings

timestamp_format = '%Y-%m-%d %H:%M:%S'

# If a gate-name includes one of these strings, it is recognised as the end of a trace.
end_point_contains = ('entrance', 'ranger-base')

# Need to specify which character to separate routes with. Comma would not
#   work well with default csv (COMMA separated values) settings, and dash
#   is used in the gate names themselves.
route_separator = '/'

main_data = pd.read_csv('data/Lekagul Sensor Data.csv')
distance_data = pd.read_csv('data/distances_v2.csv')

# I don't know yet whether route_taken or trace_completed are going to be useful,
#   but might as well track it if we're already iterating like this just in case.
# Note that the ongoing_traces dict already tracks car-id, so trace_data doesn't need it.
# Time taken is used to track seconds at first, but is converted to hours at the end.
trace_data_template = {'initial-timestamp': None, 'car-type': None,
                       'last-seen-at-sensor': None, 'distance-travelled-km': 0,
                       'time-taken-sec': 0, 'route-taken': None,
                       'sensor-count': 0, 'number-campings': 0}

def dict_per_date():
    """
    Used by date_activity dictionary to have desired structure.
    Defaultdict is used to have the behaviour of "create new"
    as a response to accessing a non-existent key instead of crashing.
    """
    return defaultdict(dict_per_sensor_combo)

def dict_per_sensor_combo():
    """
    Used by dict_per_date defaultdict to have desired structure
    on accessing non-existent keys instead of crashing.
    """
    return {'distance-travelled-km': 0, 'traces-detected': []}

# Using the above functions, create a defaultdict where you could access
#   any key, existent or not, and either store or update information
# Standard access format example:
#   date_activity['YYYY-MM-DD']['from_sensor/to_sensor']['traces_detected']
date_activity = defaultdict(dict_per_date)

ongoing_traces = {}
completed_traces = []


# Helper functions for the core algorithm
def find_distance(sensor_1, sensor_2, row, trace):
    """
    The distance between two sensors is stored in a dataset with one of the
    sensors being the 'from' sensor, and one being the 'to' sensor. Which one
    of the two sensors comes first varies from pair to pair, and so we may need
    to check both options to find the distance.
    """
    if sensor_1 == sensor_2:
        # This occurs for example when a vehicle entered and is now leaving a campsite.
        return 0
    
    distance_row = distance_data.loc[distance_data['from'] == sensor_1] \
                       .loc[distance_data['to'] == sensor_2]
    if distance_row.shape[0] == 0:
        # That order was not the right one, so the opposite should be the right one.
        distance_row = distance_data.loc[distance_data['from'] == sensor_2] \
                           .loc[distance_data['to'] == sensor_1]
        if distance_row.shape[0] == 0:
            # Neither option exists. Something is wrong here.
            print('============================================')
            print(f'ERROR description: specific data row:')
            print(row)
            print(f'Related car ID: {row["car-id"]}')
            print(f'Respective trace thus far: {trace}')
            warnings.warn('Warning: invalid action detected in trace.'
                          'Returning distance of 0. Check information above!')
            print('============================================')
            return 0
    # The found row is guaranteed to be unique, so we can return the found distance.
    return distance_row['dist-in-km'].item()


def convert_dict_entry_to_df_row(car_id, value_dict, completed_trace):
    """
    Helper function used to convert a dictionary entry from ongoing_traces to the format
    of a single row in the future analysis_result DataFrame. Also converts time taken to
    hours, and creates derived variables for average speed and number of unique sensors.
    """
    # These variables are just to keep the code more readable.
    init_time   = value_dict['initial-timestamp']
    car_type    = value_dict['car-type']
    time_taken  = value_dict['time-taken-sec'] / 3600
    distance    = value_dict['distance-travelled-km']
    avg_speed   = distance/time_taken
    route       = value_dict['route-taken']
    nr_sensors  = value_dict['sensor-count']
    nr_uniques  = len(set(route.split(route_separator)))
    nr_campings = value_dict['number-campings']
    return [car_id, init_time, car_type, time_taken, distance, avg_speed,
            route, nr_sensors, nr_uniques, nr_campings, completed_trace]


def convert_date_activities_to_dataframe(date_activities):
    """
    Helper function used to convert the entire dictionary showing activities per day
    per route into a DataFrame storing this activity in row format for future analysis
    and visualisation in our application.
    """
    list_of_rows = []
    for date in date_activities:
        for sensor_combo in date_activities[date]:
            from_sensor, to_sensor = sensor_combo.split('/')
            distance_sum = date_activities[date][sensor_combo]['distance-travelled-km']
            trace_list = date_activities[date][sensor_combo]['traces-detected']
            list_of_rows.append([date, from_sensor, to_sensor, distance_sum, trace_list])
    
    # Standard approach is to only create a DataFrame once, as appending to a df is a lot
    #   slower than appending to a list row by row.
    dataframe = pd.DataFrame(list_of_rows, columns=['date', 'from-sensor', 'to-sensor',
                                                    'distance-travelled-km', 'traces-detected'])
    return dataframe


def construct_activity_key(from_sensor, to_sensor):
    """
    Small function to put two sensor names into a single string in
    alphabetical order with a separator between them.
    """
    if from_sensor <= to_sensor:
        return from_sensor + route_separator + to_sensor
    else:
        return to_sensor + route_separator + from_sensor


# The actual main trace-tracking algorithm
for index, row in main_data.iterrows():
    # Parse the timestamp from string to base Python datetime, this allows
    #   easy calculation of time delta.
    timestamp = datetime.strptime(row['Timestamp'], timestamp_format)
    car_id = row['car-id']
    if car_id not in ongoing_traces:
        # New trace spotted. Create new instance of template, and fill in initial data.
        ongoing_traces[car_id] = copy(trace_data_template)
        ongoing_traces[car_id]['initial-timestamp'] = timestamp
        ongoing_traces[car_id]['car-type'] = row['car-type']
        ongoing_traces[car_id]['last-seen-at-sensor'] = row['gate-name']
        ongoing_traces[car_id]['route-taken'] = row['gate-name']
        ongoing_traces[car_id]['sensor-count'] += 1
        if 'camping' in row['gate-name']:
            ongoing_traces[car_id]['number-campings'] += 1
    else:
        # This car was already seen driving around, update tracking info.
        # Distance and time taken must be updated at every step, as termination of
        #   traces is not guaranteed so 'do this at the end' would not always work.
        previous_detection = ongoing_traces[car_id]['last-seen-at-sensor']
        initial_time = ongoing_traces[car_id]['initial-timestamp']
        travelled_distance = find_distance(previous_detection, row['gate-name'], row, ongoing_traces[car_id])
        
        ongoing_traces[car_id]['distance-travelled-km'] += travelled_distance
        ongoing_traces[car_id]['time-taken-sec'] = (timestamp - initial_time).total_seconds()
        ongoing_traces[car_id]['route-taken'] += route_separator + row['gate-name']
        ongoing_traces[car_id]['sensor-count'] += 1
        if 'camping' in row['gate-name']:
            ongoing_traces[car_id]['number-campings'] += 1

        # This vehicle performed some kind of activity in the preserve.
        # Track that distance and link it to the specific path it drove.
        activity_date = str(timestamp.date())
        activity_sensor_key = construct_activity_key(previous_detection, row['gate-name'])
        date_activity[activity_date][activity_sensor_key]['distance-travelled-km'] += travelled_distance
        date_activity[activity_date][activity_sensor_key]['traces-detected'].append(car_id)

        if any(end_gate_substring in row['gate-name'] \
               for end_gate_substring in end_point_contains):
            # This vehicle was in the preserve, but has now left.
            completed_traces.append(convert_dict_entry_to_df_row(car_id, ongoing_traces[car_id], True))
            ongoing_traces.pop(car_id)
        else:
            # This vehicle is currently still in the preserve.
            ongoing_traces[car_id]['last-seen-at-sensor'] = row['gate-name']

        
# Any trace remaining in ongoing_traces after algorithm completion did not complete properly.
for incomplete_trace_key in ongoing_traces:
    completed_traces.append(convert_dict_entry_to_df_row(incomplete_trace_key,
                                                         ongoing_traces[incomplete_trace_key],
                                                         False))

sensor_activity_df = convert_date_activities_to_dataframe(date_activity)

analysis_result = pd.DataFrame(completed_traces, columns=['car-id', 'initial-timestamp', 'car-type',
                                                          'time-taken-hours', 'distance-travelled-km',
                                                          'avg-kmh', 'route-taken', 'sensor-count',
                                                          'unique-sensors', 'number-campings',
                                                          'trace-complete'])

sensor_activity_df.to_csv('data/activity_per_date_per_sensor_pair.csv', index=False)
analysis_result.to_csv('data/trace_data.csv', index=False)
