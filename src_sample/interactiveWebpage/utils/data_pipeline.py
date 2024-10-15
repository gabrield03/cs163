import pandas as pd
import numpy as np
import requests
from io import StringIO
import os
import pickle
from dash.dash_table.Format import Format, Scheme
from dash import html
import plotly.graph_objects as go
import base64
from io import BytesIO

# Fetch the historical data
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
            print(f'Failed to fetch {file_url}')

# Clean and format the historical data
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

        # Pickle the data
        sj_energy_df.to_pickle('sj_energy_df.pkl')
        sf_energy_df.to_pickle('sf_energy_df.pkl')


    elif len(pickle_filename_clean) == 1 and os.path.exists(pickle_filename_clean[0]):
        if 'sj' in pickle_filename_clean:
            with open(pickle_filename_clean[0], 'rb') as f:
                sj_weather_df = pickle.load(f)
        else:
            with open(pickle_filename_clean[0], 'rb') as f:
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

        # Pickle the data
        df_monthly_weather.to_pickle(pickle_filename)

# Combine the historical dfs
def combine_historical_data(df1, df2, df3, df4):
    # Sort each df by year and month (month is categorical)

    # sj_energy_df: 2013-01 to 2024-06 - 138 records
    df1.sort_values(by = ['year', 'month'], inplace = True)
    df1.reset_index(drop = True, inplace = True)

    # sf_energy_df: 2013-01 to 2024-06 - 138 records
    df2.sort_values(by = ['year', 'month'], inplace = True)
    df2.reset_index(drop = True, inplace = True)

    # sj_weather_df: 2013-01 to 2024-09 - 144 records
    df3.sort_values(by = ['year', 'month'], inplace = True)
    df3.reset_index(drop = True, inplace = True)

    # sf_weather_df: 2013-08 to 2024-09 - 134 records
    df4.sort_values(by = ['year', 'month'], inplace = True)
    df4.reset_index(drop = True, inplace = True)

    
    # Combine sj energy and weather dfs - drop redundant columns and rename
    sj_combined = pd.merge(df1, df3, on = ['year', 'month'], how = 'inner')
    sj_combined.drop(columns = ['month-numeric_x', 'year-month_x'], inplace = True)
    sj_combined.rename(columns = {'month-numeric_y': 'month-numeric', 'year-month_y': 'year-month'}, inplace = True)

    # Combine sf energy and weather dfs - drop redundant columns and rename
    sf_combined = pd.merge(df2, df4, on = ['year', 'month'], how = 'inner')
    sf_combined.drop(columns = ['month-numeric_x', 'year-month_x'], inplace = True)
    sf_combined.rename(columns = {'month-numeric_y': 'month-numeric', 'year-month_y': 'year-month'}, inplace = True)

    sj_combined.to_pickle('sj_combined.pkl')
    sf_combined.to_pickle('sf_combined.pkl')

# Find the differences between a column between dataframes
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

# Format numerical columns
def format_columns(df):
    exclude_columns = ['zipcode', 'year', 'totalcustomers', 'averagekwh', 'month-numeric', 'totalkwh']
    columns = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_columns:
            columns.append({
                'name': col,
                'id': col,
                'type': 'numeric',
                'format': Format(precision = 3, scheme = Scheme.fixed)
            })
        else:
            columns.append({'name': col, 'id': col})

    return columns


# Create table headers
def create_table_header(df):
    headers = [html.Th('INDEX')]

    for col in df.columns:
        headers.append(html.Th(col))
    
    return html.Tr(headers)

# Function to create mini histograms for each numeric column
def create_histogram_image(df, column):
    fig = go.Figure()

    # Create a histogram for the column
    fig.add_trace(
        go.Histogram(
            x = df[column],
            marker = dict(
                color = '#154c79',
                line = dict(
                    color = '#ffffff',
                    width = 1.5,
                )
            )
        )
    )

    # Remove unnecessary plot elements
    fig.update_layout(
        xaxis = dict(
            showgrid = False,
            zeroline = False,
            visible = False,
        ),
        yaxis = dict(
            showgrid = False,
            zeroline = False,
            visible = False,
        ),
        margin = dict(
            l = 0,
            r = 0,
            t = 0,
            b = 0
        ),
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
    )

    # Convert the plot to an image that can be embedded into the table
    buffer = BytesIO()
    fig.write_image(
        buffer,
        format = 'png')
    buffer.seek(0)
    img_data = base64.b64encode(buffer.read()).decode('utf-8')

    return f'data:image/png;base64,{img_data}'

# Function to create summary statistics row (with mini histograms)
def create_table_summary_statistics(df):
    stats_row = [html.Td('Summary Stats')]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Generate mini histogram image for each numeric column
            hist_img = create_histogram_image(df, col)

            stats_row.append(
                html.Td(
                    html.Img(
                        src = hist_img,
                        style = {'height': '50px'}
                    )
                )
            ) 
        
        # For non-numeric columns
        else:  
            # summary = create_nonnumerical_summary(df, col) 
            # summary_html = html.Ul([html.Li(item) for item in summary])
            stats_row.append(html.Td('N/A', style = {'textAlign': 'center'}))

    return html.Tr(stats_row, className = 'summary-stats-row')

# Create table row data
def create_table_rows(df):
    rows = []

    for idx, row in df.iterrows():
        row_data = [html.Th(idx)] 
        for col in df.columns:
            row_data.append(html.Td(row[col]))
        rows.append(html.Tr(row_data))

    return rows




### Code to fetch, process, and save the historical data ###
repo_urls = {
    f'https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/Energy/Combined_Energy_Data.csv': ['energy_data.pkl', ['sj_energy_df.pkl', 'sf_energy_df.pkl']],
    f'https://raw.githubusercontent.com/gabrield03/cs163/main/src_sample/interactiveVisualizations/Data/Weather/SJ_95110_SJAirport.csv': ['sj_weather_data.pkl', ['sj_weather_df.pkl']],
    f'https://raw.githubusercontent.com/gabrield03/cs163/main/src_sample/interactiveVisualizations/Data/Weather/SF_94102_DowntownSF.csv': ['sf_weather_data.pkl', ['sf_weather_df.pkl']]
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