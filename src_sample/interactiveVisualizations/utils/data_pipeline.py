import pandas as pd
import numpy as np

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

# def fill_sf(df):
#     dff = df.copy()
#     months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#     for month in months:
#         # Filter data for the current month and the year 2014
#         sf_month_2014 = dff[(dff['month'] == month) & (dff['year'] == 2014)].copy()
        
#         # Modify the year to 2013 for this data
#         sf_month_2014['year'] = 2013
        
#         # Append this "filler" data to the dataframe
#         dff = pd.concat([dff, sf_month_2014], ignore_index=True)

#     # Sort the dataframe by year to maintain the chronological order
#     dff.sort_values(by='year', inplace=True)

#     return dff

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