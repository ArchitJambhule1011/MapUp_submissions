import pandas as pd

def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    df_subset = df[['id_1', 'id_2', 'car']].copy()
    df = df_subset.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    for i in df.index:
      if i in df.columns:
        df.loc[i, i] = 0

    return df

def get_type_count(df) -> dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    type_counts = df['car_type'].value_counts().to_dict()

    type_counts = dict(sorted(type_counts.items()))

    return type_counts

def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    mean_value = df['bus'].mean()
    high_limit = 2 * mean_value
    bus_index = [index for index, value in df['bus'].iteritems() if value > high_limit]
    bus_index.sort()
    return bus_index


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    filtered_routes = df.groupby('route')['truck'].mean().loc[lambda x : x > 7].index.tolist()
    filtered_routes.sort()
    return filtered_routes

def multiply_matrix(matrix) -> pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame): Input matrix.

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')
    result_df = pd.DataFrame(index=df.set_index(['id', 'id_2']).index)
    for (id_val, id_2_val), group in df.groupby(['id', 'id_2']):
        full_24_hours = (group['end_datetime'].max() - group['start_datetime'].min()) >= pd.Timedelta(hours=24)
        span_all_days = set(group['start_datetime'].dt.day_name()) == set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        result_df.loc[(id_val, id_2_val), 'timestamp_complete'] = full_24_hours and span_all_days

    result_series = result_df['timestamp_complete'].astype(bool)

    return result_series

