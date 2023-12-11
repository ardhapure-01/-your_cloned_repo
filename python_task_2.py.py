#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

def generate_car_matrix(dataset):
    df = pd.read_csv(dataset)
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    
    for i in range(min(car_matrix.shape[0], car_matrix.shape[1])):
        car_matrix.iloc[i, i] = 0

    return car_matrix

dataset_path = 'dataset-1.csv'
result_matrix = generate_car_matrix(dataset_path)


print(result_matrix)


# # task 2

# In[2]:


import pandas as pd

def get_type_count(dataset):
    
    df = pd.read_csv(dataset)

    
    df['car_type'] = pd.cut(df['car'], bins=[float('-inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

   
    type_counts = df['car_type'].value_counts().to_dict()

    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts


dataset_path = 'dataset-1.csv'
result_type_counts = get_type_count(dataset_path)


print(result_type_counts)


# # task 3

# In[3]:


import pandas as pd

def get_bus_indexes(dataset):
    
    df = pd.read_csv(dataset)

    bus_mean = df['bus'].mean()

   
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    bus_indexes.sort()

    return bus_indexes

dataset_path = 'dataset-1.csv'
result_bus_indexes = get_bus_indexes(dataset_path)

print(result_bus_indexes)


# # task 4

# In[6]:


import pandas as pd

def filter_routes(dataset_path):
    df = pd.read_csv(dataset_path)

    route_avg_truck = df.groupby('route')['truck'].mean()

    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    filtered_routes.sort()

    return filtered_routes

dataset_path = 'dataset-1.csv'  
result_filtered_routes = filter_routes(dataset_path)

print(result_filtered_routes)


# # task 5

# In[1]:


import pandas as pd

def multiply_matrix(result_matrix):
   
    modified_matrix ='dataset-1.csv'

   
    modified_matrix = modified_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    modified_matrix = modified_matrix.round(1)

    return modified_matrix


result_matrix = pd.DataFrame(result_filtered_routes) 
modified_result = multiply_matrix(result_matrix)

print(modified_result)


# # task 6

# In[9]:


import pandas as pd
from datetime import datetime, timedelta

def verify_timestamps(dataset):
   
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset['day'] = dataset['timestamp'].dt.day_name()
    dataset['time'] = dataset['timestamp'].dt.time
    
    def is_full_24_hours(group):
        start_time = min(group['time'])
        end_time = max(group['time'])
        return start_time == datetime.strptime('00:00:00', '%H:%M:%S').time() and end_time == datetime.strptime('23:59:59', '%H:%M:%S').time()


    correct_timestamps = dataset.groupby(['id', 'id_2']).apply(is_full_24_hours)

    return correct_timestamps


dataset_path = 'dataset-2.csv'
dataset = pd.read_csv(dataset_path)


result = verify_timestamps(dataset)


print(result)


# In[ ]:




