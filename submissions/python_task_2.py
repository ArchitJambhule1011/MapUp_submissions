import pandas as pd 
import numpy as np
from datetime import time, timedelta

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    distances = {}
    for i, row in df.iterrows():
      start = row["id_start"]
      end = row["id_end"]
      distance = row["distance"]

      distances[(start, end)] = distance
      distances[(end, start)] = distance

    distance_matrix = pd.DataFrame(index=df["id_start"].unique(), columns=df["id_start"].unique())
    distance_matrix.fillna(0, inplace=True)

    for i in range(len(distance_matrix.index)):
      for j in range(len(distance_matrix.columns)):
          start = distance_matrix.index[i]
          end = distance_matrix.columns[j]

          if start == end:
            distance_matrix.loc[start, end] = 0
          else:
            if (start, end) in distances:
                distance_matrix.loc[start, end] = distances[(start, end)]
            else:
                distance_matrix.loc[start, end] = distance_matrix.loc[end, start]
    

    return distance_matrix

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    upper_triangle = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
    unrolled_df = upper_triangle.reset_index().melt(id_vars='index', var_name='id_end', value_name='distance')
    unrolled_df.columns = ['id_start', 'id_end', 'distance']
    unrolled_df = unrolled_df.dropna()
    unrolled_df['id_start'] = unrolled_df['id_start'].astype(int)
    unrolled_df['id_end'] = unrolled_df['id_end'].astype(int)

    return unrolled_df

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_rows = df[df['id_start'] == reference_id]
    if reference_rows.empty:
       return []
    average_distance = reference_rows['distance'].mean()
    threshold_min = average_distance - (average_distance * 0.1)
    threshold_max = average_distance + (average_distance * 0.1)
    filtered_ids = df[(df['distance'] >= threshold_min) & (df['distance'] <= threshold_max)]['id_start'].unique()
    sorted_filtered_ids = sorted(filtered_ids)
    return sorted_filtered_ids

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle_type, rate_coefficient in rate_coefficients.items():
       column_name = f'{vehicle_type}_toll'
       df[column_name] = df['distance'] * rate_coefficient

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    time_ranges_weekdays = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]
    time_ranges_weekends = [
        (time(0, 0, 0), time(23, 59, 59), 0.7)
    ]

    df['start_day'] = df['end_day'] = df['start_time'] = df['end_time'] = None
    for _, group in df.groupby(['id_start', 'id_end']):
       for day in range(7):
          for start_time, end_time, discount_factor in (time_ranges_weekdays if day < 5 else time_ranges_weekends):
             start_datetime = pd.to_datetime(f"{day} {start_time}", format='%w %H:%M:%S')
             end_datetime = pd.to_datetime(f"{day} {end_time}", format='%w %H:%M:%S')

             mask = (group['start_time'] <= end_datetime) & (group['end_time'] >= start_datetime)
             df.loc[mask, 'start_day'] = df.loc[mask, 'end_day'] = start_datetime.strftime('%A')
             df.loc[mask, 'start_time'] = start_time
             df.loc[mask, 'end_time'] = end_time
             for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                toll_column = f'{vehicle_type}_toll'
                df.loc[mask, toll_column] *= discount_factor
                

    return df