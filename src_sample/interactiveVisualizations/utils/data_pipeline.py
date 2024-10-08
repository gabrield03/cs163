import pandas as pd
import numpy as np
import requests
from io import StringIO
import os
import pickle

def fetch_historical_data():
    pickled_energy_file = 'energy_data.pkl'
    pickled_weather_sj_file = 'sj_data.pkl'
    pickled_weather_sf_file = 'sf_data.pkl'

    energy_df = ''
    sj_df = ''
    sf_df = ''

    # Energy Data
    if os.path.exists(pickled_energy_file):
        with open(pickled_energy_file, 'rb') as f:
            energy_df = pickle.load(f)
    else:
        
        energy_repo_url = 'https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/Energy/Combined_Energy_Data.csv'
    
        #for file in energy_files:
        #    file_url = energy_repo_URL + file
        response = requests.get(energy_repo_url)
            
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            energy_df = pd.read_csv(csv_data)
            energy_df.to_pickle('energy_data.pkl')
    
        else:
            print(f"Failed to fetch {energy_repo_url}")

        
    
    # SJ Weather Data
    if os.path.exists(pickled_weather_sj_file):
        with open(pickled_weather_sj_file, 'rb') as f:
            sj_df = pickle.load(f)
    else:
        sj_weather_repo_url = f'https://raw.githubusercontent.com/gabrield03/cs163/main/src_sample/interactiveVisualizations/Data/Weather/SJ_95110_SJAirport.csv'

        response = requests.get(sj_weather_repo_url)
        
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            sj_df = pd.read_csv(csv_data)
            sj_df.to_pickle('sj_data.pkl')

        else:
            print(f"Failed to fetch {sj_weather_repo_url}")



    # SF Weather Data
    if os.path.exists(pickled_weather_sf_file):
        with open(pickled_weather_sf_file, 'rb') as f:
            sf_df = pickle.load(f)
    else:
        sf_weather_repo_url = f'https://raw.githubusercontent.com/gabrield03/cs163/main/src_sample/interactiveVisualizations/Data/Weather/SF_94102_DowntownSF.csv'

        response = requests.get(sf_weather_repo_url)
        
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            sf_df = pd.read_csv(csv_data)
            sf_df.to_pickle('sf_data.pkl')

        else:
            print(f"Failed to fetch {sf_weather_repo_url}")

        



fetch_historical_data()

#### NEED TO CLEAN THE DATA NEXT ####
def clean_data(df):
    return 1

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