#!/usr/bin/env python
# coding: utf-8

# # task 1

# In[ ]:





# In[6]:


import pandas as pd
import networkx as nx

def calculate_distance_matrix(dataset_path):
   
    df = pd.read_csv(dataset_path)


    G = nx.Graph()

   
    for index, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], distance=row['distance'])
    
    
    distance_matrix = nx.floyd_warshall_numpy(G, weight='distance')

   
    for i in range(distance_matrix.shape[0]):
        distance_matrix[i, i] = 0

    
    distance_df = pd.DataFrame(distance_matrix, index=G.nodes, columns=G.nodes)

    return distance_df


dataset_path = 'dataset-3.csv'
resulting_distance_matrix = calculate_distance_matrix(dataset_path)
print(resulting_distance_matrix)


# # task 2

# In[7]:


import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Extract column names as lists
    id_start = distance_matrix.columns.tolist()
    id_end = distance_matrix.index.tolist()

    # Initialize lists to store unrolled data
    id_start_unrolled = []
    id_end_unrolled = []
    distance_unrolled = []

    # Iterate through the distance matrix and unroll the data
    for start in id_start:
        for end in id_end:
            if start != end:
                id_start_unrolled.append(start)
                id_end_unrolled.append(end)
                distance_unrolled.append(distance_matrix.loc[start, end])

    # Create a new DataFrame from the unrolled data
    unrolled_df = pd.DataFrame({'id_start': id_start_unrolled, 'id_end': id_end_unrolled, 'distance': distance_unrolled})

    return unrolled_df

# Example usage:
# Replace 'result_distance_matrix' with the actual DataFrame from Question 1
result_distance_matrix = pd.DataFrame(...)  # Replace ... with the actual data
result_unrolled = unroll_distance_matrix(result_distance_matrix)

# Display the resulting unrolled DataFrame
print(result_unrolled)


# # task 3

# In[9]:


import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_value):
   
    subset_df = df[df['id_start'] == reference_value]

    
    avg_distance = subset_df['distance'].mean()

   
    lower_threshold = avg_distance - (avg_distance * 0.1)
    upper_threshold = avg_distance + (avg_distance * 0.1)

   
    within_threshold_df = df[(df['id_start'] != reference_value) & (df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]

    
    result_list = sorted(within_threshold_df['id_start'].unique())

    return result_list


unrolled_df = pd.DataFrame({
    'id_start': ['A', 'A', 'B', 'B', 'C'],
    'id_end': ['B', 'C', 'A', 'C', 'A'],
    'distance': [10, 15, 10, 20, 15]
})

reference_value = 'A'
result = find_ids_within_ten_percentage_threshold(unrolled_df, reference_value)
print(result)


# # task 4

# In[11]:


import pandas as pd

def calculate_toll_rate(df):
    
    rate_coefficients = {
        'motorcycle': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    
    for vehicle_type in rate_coefficients.keys():
        df[vehicle_type] = 0.0


    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df


df = pd.DataFrame({
    
    'id_start': ['A', 'A', 'B', 'B', 'C'],
    'id_end': ['B', 'C', 'A', 'C', 'A'],
    'distance': [10, 15, 10, 20, 15],
    'vehicle_type': ['car', 'car', 'truck', 'truck', 'motorcycle']
})
    

result_df = calculate_toll_rate(df)
print(result_df)


# # task 5

# In[12]:


import pandas as pd
from datetime import datetime, time, timedelta

def calculate_time_based_toll_rates(df):
    # Define time intervals and discount factors
    time_intervals = [
        {'start': time(0, 0), 'end': time(10, 0), 'weekday_factor': 0.8, 'weekend_factor': 0.7},
        {'start': time(10, 0), 'end': time(18, 0), 'weekday_factor': 1.2, 'weekend_factor': 0.7},
        {'start': time(18, 0), 'end': time(23, 59, 59), 'weekday_factor': 0.8, 'weekend_factor': 0.7}
    ]

    # Create new columns for start_day, start_time, end_day, and end_time
    df['start_day'] = df['start_time'].dt.day_name()
    df['end_day'] = df['end_time'].dt.day_name()

    # Convert start_time and end_time to datetime.time type
    df['start_time'] = df['start_time'].dt.time
    df['end_time'] = df['end_time'].dt.time

    # Calculate toll rates based on time intervals and discount factors
    for interval in time_intervals:
        weekday_mask = (df['start_day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))
        weekend_mask = (df['start_day'].isin(['Saturday', 'Sunday']))

        # Apply discount factors based on time intervals
        df.loc[weekday_mask & (df['start_time'] >= interval['start']) & (df['start_time'] < interval['end']), 'car'] *= interval['weekday_factor']
        df.loc[weekend_mask & (df['start_time'] >= interval['start']) & (df['start_time'] < interval['end']), 'car'] *= interval['weekend_factor']

    return df

# Example usage:
# Replace 'result_with_toll_rates' with the actual DataFrame from Question 4
result_with_toll_rates = pd.DataFrame(...)  # Replace ... with the actual data

result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_with_toll_rates)

# Display the resulting DataFrame with time-based toll rates
print(result_with_time_based_toll_rates)

