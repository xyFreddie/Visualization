import pandas as pd
import numpy as np

# Source: https://en.wikipedia.org/wiki/Mile
mile_in_km = 1.609344

# Source: Data Descriptions for MC1 v2
map_length_miles = 12
map_length_pixels = 200

pixel_in_km = (map_length_miles * mile_in_km) / map_length_pixels

dataset = pd.read_csv('data/distances.csv')

dataset['dist-in-km'] = np.nan

# Iterate over the rows, and for each row calculate the distance in meters based on the given number of pixels (on-road)
for index, row in dataset.iterrows():
    dataset.loc[index, 'dist-in-km'] = row['distance'] * pixel_in_km

dataset.to_csv('data/distances_v2.csv', index=False)