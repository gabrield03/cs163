import pandas as pd
import numpy as np
import requests
from io import StringIO
import os
import pickle

def fetch_historical_data(file_url, pickle_filename, pickle_filename_clean):

    # Check if the data already exists (is pickled)
    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            df = pickle.load(f)
            clean_data(df, pickle_filename_clean)

    else:
        response = requests.get(file_url)
            
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            df.to_pickle(pickle_filename)
            clean_data(df, pickle_filename_clean)
    
        else:
            print(f"Failed to fetch {file_url}")


#### NEED TO CLEAN THE DATA NEXT ####
def clean_data(df, pickle_filename_clean):
    # Month mapping for month-numeric column
    month_dict = {
        1: 'Jan', 2: 'Feb', 3: 'Mar',
        4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep',
        10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

    ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    sj_energy_df = ''
    sf_energy_df = ''
    sj_weather_df = ''
    sf_weather_df = ''

    if len(pickle_filename_clean) == 2 and os.path.exists(pickle_filename_clean[0]) and os.path.exists(pickle_filename_clean[1]):
        with open(pickle_filename_clean[0], 'rb') as f:
            sj_energy_df = pickle.load(f)
        with open(pickle_filename_clean[1], 'rb') as f:
            sf_energy_df = pickle.load(f)
    
    # Energy df - pkl not created - split into sj and sf
    elif len(pickle_filename_clean) == 2:
        # Add year-month column YYYY-MM
        df['year-month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)

        # Convert month column to month-numeric
        df.rename(columns = {'month': 'month-numeric'}, inplace = True)

        # Add month column that has the string name (Jan - Dec)
        df['month'] = df['month-numeric'].map(month_dict)

        # Convert month to categorical
        df['month'] = pd.Categorical(df['month'], categories=ordered_months, ordered=True)

        # Convert totalcustomers, totalkwh, and averagekwh to integers
        df['totalkwh'] = df['totalkwh'].fillna(0).astype(str).str.replace(',', '').astype(int)
        df['averagekwh'] = df['averagekwh'].fillna(0).astype(str).str.replace(',', '').astype(int)
        df['totalcustomers'] = df['totalcustomers'].fillna(0).astype(str).str.replace(',', '').astype(int)


        # Split the data into sj - 95110 and sf - 94102
        sj_energy_df = df[df.zipcode == 95110]
        sf_energy_df = df[df.zipcode == 94102]


        # Drop duplicates in each data frame -- might add this to weather df as well
        sj_energy_df = sj_energy_df.drop_duplicates(subset='year-month', keep='first')
        sf_energy_df = sf_energy_df.drop_duplicates(subset='year-month', keep='first')

        sj_energy_df['region'] = 'San Jose'
        sf_energy_df['region'] = 'San Francisco'

    
        # We have NaN values from 2024-07 to 2024-12
        # May need to make them 0

        sj_energy_df.to_pickle('sj_energy_df.pkl')
        sj_energy_df.to_pickle('sf_energy_df.pkl')


    elif len(pickle_filename_clean) == 1 and os.path.exists(pickle_filename_clean):
        if 'sj' in pickle_filename_clean:
            with open(pickle_filename_clean, 'rb') as f:
                sj_weather_df = pickle.load(f)
        else:
            with open(pickle_filename_clean, 'rb') as f:
                sf_weather_df = pickle.load(f)

    # Weather data - pkl not created
    else: 
        pickle_filename = ''
        # Change col names to lowercase for consistency
        df.columns = df.columns.str.lower()

        # Add region
        if 'USW00023293' == df['station'].iloc[0]:  # SJ Station
            df['region'] = 'San Jose'
            pickle_filename = 'sj_weather_df.pkl'
        else:                               # SF Station
            df['region'] = 'San Francisco'
            pickle_filename = 'sf_weather_df.pkl'

        # Drop station and name
        if 'station' in df.columns:
            df.drop(columns = ['station'], inplace = True)
        if 'name' in df.columns:
            df.drop(columns = ['name'], inplace = True)

        # Covert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month

        # Remove columns with a lot of NaN values --> drop cols (30% threshhold)
        threshold = 0.3

        for col in df.columns:
            # Count the number of NaN values in the column
            nan_count = df[col].isna().sum()
            
            # Calculate the percentage of NaN values in the column
            nan_percentage = nan_count / len(df[col])
            
            # Drop the column if the percentage exceeds the threshold
            if nan_percentage > threshold:
                df.drop(columns=[col], inplace=True)

        # Find the mean of the monthly values (bc weather is daily records)
        # 1. Find the columns to agg by, excluding year and month
        columns_to_agg = [col for col in df.columns if col not in ['year', 'month'] and pd.api.types.is_numeric_dtype(df[col])]

        # Create a dictionary to specify aggregation as 'mean' for each numerical column
        agg_dict = {col: 'mean' for col in columns_to_agg}

        # Apply groupby with the dynamically created aggregation dictionary
        df_monthly_weather = df.groupby(['year', 'month'], as_index=False, observed=False).agg(agg_dict)


        # Add year-month YYYY-MM column
        df_monthly_weather['year-month'] = df_monthly_weather['year'].astype(str) + '-' + df_monthly_weather['month'].astype(str).str.zfill(2)

        # Convert month column to month-numeric
        df_monthly_weather.rename(columns = {'month': 'month-numeric'}, inplace = True)

        # Add month column that has the string name (Jan - Dec)
        df_monthly_weather['month'] = df_monthly_weather['month-numeric'].map(month_dict)

        # Convert month to categorical
        df_monthly_weather['month'] = pd.Categorical(df_monthly_weather['month'], categories=ordered_months, ordered=True)

        df_monthly_weather.to_pickle(pickle_filename)

def combine_historical_data(df1, df2, df3, df4):
    # do something
    return 1

repo_urls = {
    f'https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/Energy/Combined_Energy_Data.csv': ['energy_data.pkl', ['sj_energy_df.pkl', 'sf_energy_df.pkl']],
    f'https://raw.githubusercontent.com/gabrield03/cs163/main/src_sample/interactiveVisualizations/Data/Weather/SJ_95110_SJAirport.csv': ['sj_weather_data.pkl', 'sj_weather_df.pkl'],
    f'https://raw.githubusercontent.com/gabrield03/cs163/main/src_sample/interactiveVisualizations/Data/Weather/SF_94102_DowntownSF.csv': ['sf_weather_data.pkl', 'sf_weather_df.pkl']
}

# Fetch and clean the historical data
for url, pickle_filenames in repo_urls.items():
    fetch_historical_data(url, pickle_filenames[0], pickle_filenames[1])


# Combine the historical data
if os.path.exists('sj_energy_df.pkl') and os.path.exists('sf_energy_df.pkl') and os.path.exists('sj_weather_df.pkl') and os.path.exists('sf_weather_df.pkl'):
    df1 = ''
    df2 = ''
    df3 = ''
    df4 = ''

    with open('sj_energy_df.pkl', 'rb') as f:
            df1 = pickle.load(f)
    with open('sf_energy_df.pkl', 'rb') as f:
            df2 = pickle.load(f)
    with open('sj_weather_df.pkl', 'rb') as f:
            df3 = pickle.load(f)
    with open('sf_weather_df.pkl', 'rb') as f:
            df4 = pickle.load(f)

    if len(df1) != 0 and len(df2) != 0 and len(df3) != 0 and len(df4) != 0:
        combine_historical_data(df1, df2, df3, df4)
    else:
        print('Some error occurred while loading clean data')
    


#### THIS WILL REPLACE LOAD_AND_PREPROCESS_DATA ####
def load_cleaned_data():
    return 1

def load_and_preprocess_data():
    # Load the data from github
    sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
    sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')

    # Drop missing and NA values
    sj_dff = sj_df.dropna()
    sf_dff = sf_df.dropna()

    # Add the region columns
    sj_dff['region'] = 'San Jose'
    sf_dff['region'] = 'San Francisco'


    # Add month-numeric to each dataset
    sj_dff['month_numeric'] = sj_dff['month'].map({
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 
        'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 
        'Nov': 11, 'Dec': 12
    })

    sf_dff['month_numeric'] = sf_dff['month'].map({
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 
        'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 
        'Nov': 11, 'Dec': 12
    })

    return sj_dff, sf_dff

def find_regional_diff(sj_df, sf_df, diffCol, newCol):
    dff = sj_df.copy()

    # Initialize the 'averagekwhdiff' column with NaN
    dff[newCol] = np.nan

    # Loop through each date in sj_df, check if the date exists in sf df, get the values and subtract
    for i, date in enumerate(sj_df['year-month']):
        if date in sf_df['year-month'].values:
            sf_value = sf_df[sf_df['year-month'] == date][diffCol].values[0]
            
            dff.loc[i, newCol] = sj_df.loc[i, diffCol] - sf_value
        else:
            # If data missing in sf df, just put NaN
            dff.loc[i, newCol] = np.nan 

    dff = dff.reset_index(drop=True)

    return dff


def combine_regions(df1, df2):
    # Reshape sj_df to include temperature and energy columns
    df1_melted_temp = pd.melt(
        df1,
        id_vars = [
            'zipcode', 'month', 'year', 'customerclass', 
            'combined', 'totalcustomers', 'totalkwh', 
            'averagekwh', 'year-month',
            'awnd', 'prcp', 'wdf2', 'wdf5', 'wsf2', 'wsf5', 'region'
        ],
        value_vars = ['tmax', 'tmin'], 
        var_name = 'temp_type', 
        value_name = 'temp'
    )

    df1_melted_energy = pd.melt(
        df1_melted_temp,
        id_vars = [
            'zipcode', 'month', 'year', 'customerclass', 
            'combined', 'totalcustomers', 'year-month', 
            'awnd', 'prcp', 'wdf2', 'wdf5',
            'wsf2', 'wsf5', 'region', 'temp_type', 'temp'
        ],
        value_vars = ['totalkwh', 'averagekwh'], 
        var_name = 'energy_type', 
        value_name = 'energy'
    )

    # Reshape sf_df to include temperature and energy columns
    df2_melted_temp = pd.melt(
        df2,
        id_vars = [
            'zipcode', 'month', 'year', 'customerclass', 
            'combined', 'totalcustomers', 'totalkwh',
            'averagekwh', 'year-month', 'prcp', 'region'
        ],
        value_vars = ['tmax', 'tmin'], 
        var_name = 'temp_type', 
        value_name = 'temp'
    )

    df2_melted_energy = pd.melt(
        df2_melted_temp,
        id_vars = [
            'zipcode', 'month', 'year', 'customerclass', 
            'combined', 'totalcustomers', 'year-month',
            'prcp', 'region', 'temp_type', 'temp'
        ],
        value_vars = ['totalkwh', 'averagekwh'], 
        var_name = 'energy_type', 
        value_name = 'energy'
    )

    # Combine both dataframes
    combined_df = pd.concat([df1_melted_energy, df2_melted_energy], ignore_index = True)

    combined_df['month_numeric'] = combined_df['month'].map(
        {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 
        'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 
        'Nov': 11, 'Dec': 12
        }
    )

    return combined_df