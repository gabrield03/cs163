import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dash.dash_table.Format import Format, Scheme
from dash import html
import plotly.graph_objects as go
import base64
import requests
from io import BytesIO
import shap
from joblib import dump, load

# ML libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from keras import Sequential

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError

from keras.layers import Dense, LSTM

import warnings
warnings.filterwarnings('ignore')

# SARIMA libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product


# Fetch the historical data
def fetch_historical_data(file_url, joblib_filename, joblib_filename_clean):

    # Check if the data already exists (is stored)
    if os.path.exists(joblib_filename):
        df = load(joblib_filename)
        clean_data(df, joblib_filename_clean)

    else:
        response = requests.get(file_url)
            
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            dump(df, joblib_filename)
            clean_data(df, joblib_filename_clean)
    
        else:
            print(f'Failed to fetch {file_url}')

# Clean and format the historical data
def clean_data(df, joblib_filename_clean):
    # Month mapping for month-numeric column
    month_dict = {
        1: 'Jan', 2: 'Feb', 3: 'Mar',
        4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep',
        10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

    ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    sj_energy_df = pd.DataFrame()
    sf_energy_df = pd.DataFrame()
    sj_weather_df = pd.DataFrame()
    sf_weather_df = pd.DataFrame()

    if len(joblib_filename_clean) == 2 and os.path.exists(joblib_filename_clean[0]) and os.path.exists(joblib_filename_clean[1]):
        sj_energy_df = load(joblib_filename_clean[0])
        sf_energy_df = load(joblib_filename_clean[1])
    
    # Energy df - joblib not created - split into sj and sf
    elif len(joblib_filename_clean) == 2:
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

        # Add columns - region
        sj_energy_df['region'] = 'SJ'
        sf_energy_df['region'] = 'SF'

        # Add columns - season
        conditions = [
            (sj_energy_df['month-numeric'].between(3, 5)),
            (sj_energy_df['month-numeric'].between(6, 8)),
            (sj_energy_df['month-numeric'].between(9, 11)),
            ((sj_energy_df['month-numeric'].between(1, 2)) | (sj_energy_df['month-numeric'] == 12))
        ]
        values = ['Spring', 'Summer', 'Fall', 'Winter']

        sj_energy_df['season'] = np.select(conditions, values)
        sf_energy_df['season'] = np.select(conditions, values)

        # joblib - dump the data
        dump(sj_energy_df, 'joblib_files/base_data/sj_energy_df.joblib')
        dump(sf_energy_df, 'joblib_files/base_data/sf_energy_df.joblib')


    elif len(joblib_filename_clean) == 1 and os.path.exists(joblib_filename_clean[0]):
        if 'sj' in joblib_filename_clean:
            sj_weather_df = load(joblib_filename_clean[0])
        else:
            sf_weather_df = load(joblib_filename_clean[0])

    # Weather data - joblib not created
    else: 
        joblib_filename = None
        # Change col names to lowercase for consistency
        df.columns = df.columns.str.lower()

        # Add region
        if 'USW00023293' == df['station'].iloc[0]:  # SJ Station
            df['region'] = 'SJ'
            joblib_filename = 'joblib_files/base_data/sj_weather_df.joblib'
        else:                               # SF Station
            df['region'] = 'SF'
            joblib_filename = 'joblib_files/base_data/sf_weather_df.joblib'

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

        # joblib - dump the data
        dump(df_monthly_weather, joblib_filename)

# Combine the historical dfs
def combine_historical_data(df1, df2, df3, df4):
    # joblib filenames
    joblib_filename_sj_combined = 'joblib_files/base_data/sj_combined.joblib'
    joblib_filename_sf_combined = 'joblib_files/base_data/sf_combined.joblib'

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

    if not os.path.exists(joblib_filename_sj_combined) or not os.path.exists(joblib_filename_sf_combined):
        dump(sj_combined, joblib_filename_sj_combined)
        dump(sf_combined, joblib_filename_sf_combined)

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


#### START functions for data.py tables ####
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
    exclude_columns = ['zipcode', 'year', 'totalcustomers', 'averagekwh', 'month-numeric', 'totalkwh']

    rows = []

    for idx, row in df.iterrows():
        row_data = [html.Th(idx)] 
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_columns:
                row_data.append(html.Td(f'{row[col]:.3f}'))
            else:
                row_data.append(html.Td(row[col]))
        rows.append(html.Tr(row_data, style = {'height': '25px'}))

    return rows

#### END functions for data.py tables ####


# Processing pipeline
def processing_pipeline(df, loc):
    # Transformer class for pipeline
    class ReshapeTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y = None):
            return self

        def transform(self, X, y = None):
            return X.reshape(-1, 3)
    
    joblib_filename_model = f'joblib_files/processed_data/{loc}_rf.joblib'
    joblib_filename_X_train = f'joblib_files/processed_data/{loc}_X_train.joblib'
    joblib_filename_X_test = f'joblib_files/processed_data/{loc}_X_test.joblib'
    joblib_filename_y_train = f'joblib_files/processed_data/{loc}_y_train.joblib'
    joblib_filename_y_test = f'joblib_files/processed_data/{loc}_y_test.joblib'
    joblib_filename_X_train_processed_df = f'joblib_files/processed_data/{loc}_X_train_processed_df.joblib'
    joblib_filename_X_test_processed_df = f'joblib_files/processed_data/{loc}_X_test_processed_df.joblib'
    joblib_filename_preprocessor = f'joblib_files/processed_data/{loc}_preprocessor.joblib'
    joblib_filename_importances = f'joblib_files/processed_data/{loc}_importances_df.joblib'
    joblib_filename_X_test_unscaled_df = f'joblib_files/processed_data/{loc}_X_test_unscaled_df.joblib'

    drop_list = ['zipcode', 'totalkwh', 'customerclass', 'combined', 'region', 'month-numeric', 'year-month', 'wdf5', 'wsf5', 'awnd', 'wdf2', 'wsf2']

    for col in drop_list:
        if col in df.columns:
            df.drop(columns = [col], inplace = True)

    X = df.drop(columns = ['averagekwh'])
    y = df['averagekwh']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # Categorical and numerical features for pipeline
    cat_col_list = ['year', 'month', 'season']
    num_col_list = [col for col in X.columns if col not in cat_col_list]

    # Create pipeline for numerical and categorical preprocessing
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(), num_col_list),
            ('cat', Pipeline(steps=[('encode', OrdinalEncoder()), ('reshape', ReshapeTransformer())]), cat_col_list)
        ]
    )
    
    # Fit the preprocessor on training data and transform it
    X_train_processed = preprocessor.fit_transform(X_train)

    ####using this for shap! another joblib file####
    num_col_names = num_col_list  # Names for numerical features
    cat_col_names = preprocessor.named_transformers_['cat'].named_steps['encode'].get_feature_names_out(cat_col_list)
    all_col_names = list(num_col_names) + list(cat_col_names)

    X_train_processed_df = pd.DataFrame(X_train_processed, columns = all_col_names)

    X_test_processed = preprocessor.transform(X_test)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns = all_col_names)

    # Combine the feature names
    all_features = num_col_list + cat_col_list

    # Fit the RandomForest model on the preprocessed training data
    rf = RandomForestRegressor().fit(X_train_processed, y_train)

    # Get feature importances
    rf_importances = rf.feature_importances_

    # Create a dataframe for feature importances
    importances_df = pd.DataFrame(data = rf_importances, index = all_features, columns = ['importances'])

    importances_df.reset_index(inplace = True)
    importances_df.columns = ['feature', 'importances']
    
    # joblib - dump all the data
    # Processed training and test data
    if not os.path.exists(joblib_filename_X_train) or not os.path.exists(joblib_filename_X_test) or not os.path.exists(joblib_filename_y_train) or not os.path.exists(joblib_filename_y_test):
        dump(X_train_processed, joblib_filename_X_train)
        dump(X_test_processed, joblib_filename_X_test)
        dump(y_train, joblib_filename_y_train)
        dump(y_test, joblib_filename_y_test)

    if not os.path.exists(joblib_filename_X_train_processed_df):
        dump(X_train_processed_df, joblib_filename_X_train_processed_df)

    if not os.path.exists(joblib_filename_X_test_processed_df):
        dump(X_test_processed_df, joblib_filename_X_test_processed_df)

    if not os.path.exists(joblib_filename_X_test_unscaled_df):
        dump(X_test, joblib_filename_X_test_unscaled_df)

    # rf model
    if not os.path.exists(joblib_filename_model):
        dump(rf, joblib_filename_model)

    # rf importances
    if not os.path.exists(joblib_filename_importances):
        dump(importances_df, joblib_filename_importances)

    return importances_df

# Process the data for LSTM
def lstm_data_processing(loc):
    df_url = f"https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/base_data/{loc}_combined.joblib"
    df = load_joblib_from_github(df_url)

    #### Data preprocessing for lstm ####
    # Drop unnecessary columns
    drop_list = ['zipcode', 'totalkwh', 'customerclass', 'combined', 'region', 'month-numeric']
    df.drop(drop_list, axis = 1, inplace = True)

    # Change year-month col to sin+cos for cyclical nature and lstm usability
    df['year-month'] = pd.to_datetime(df['year-month'], format = '%Y-%m')
    yr_mo = df['year-month'].dt.month

    mos_in_yr = 12

    # Calculate the cyclic features based on a 12-month cycle
    df['month_sin'] = np.sin(2 * np.pi * yr_mo / mos_in_yr)
    df['month_cos'] = np.cos(2 * np.pi * yr_mo / mos_in_yr)

    # Drop the original 'year-month' column - no longer needed
    df.drop(['year-month'], axis = 1, inplace = True)

    # Splitting the data
    test_size = 24
    val_size = 24
    train_size = len(df) - test_size - val_size

    train_df = df[:train_size]
    val_df = df[train_size: train_size + val_size]
    test_df = df[train_size + val_size:]

    # Separate cat, num, and cyclical features
    cat_columns = ['year', 'month', 'season']
    num_columns = train_df.select_dtypes(include=['float64', 'int32']).columns.difference(['month_sin', 'month_cos', 'year'])
    cyc_columns = ['month_sin', 'month_cos']

    # Ordinal encode categorical data
    encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)

    train_df_cat_encoded = pd.DataFrame(encoder.fit_transform(train_df[cat_columns]), columns = cat_columns)
    val_df_cat_encoded = pd.DataFrame(encoder.transform(val_df[cat_columns]), columns = cat_columns)
    test_df_cat_encoded = pd.DataFrame(encoder.transform(test_df[cat_columns]), columns = cat_columns)

    # Scale numerical data
    scaler = StandardScaler()

    train_df_num_scaled = pd.DataFrame(scaler.fit_transform(train_df[num_columns]), columns = num_columns)
    val_df_num_scaled = pd.DataFrame(scaler.transform(val_df[num_columns]), columns = num_columns)
    test_df_num_scaled = pd.DataFrame(scaler.transform(test_df[num_columns]), columns = num_columns)

    # Combine all processed features
    train_df_processed = pd.concat([train_df_num_scaled, train_df_cat_encoded, train_df[cyc_columns].reset_index(drop = True)], axis = 1)
    val_df_processed = pd.concat([val_df_num_scaled, val_df_cat_encoded, val_df[cyc_columns].reset_index(drop = True)], axis = 1)
    test_df_processed = pd.concat([test_df_num_scaled, test_df_cat_encoded, test_df[cyc_columns].reset_index(drop = True)], axis = 1)

    return train_df_processed, val_df_processed, test_df_processed, train_df, val_df, test_df, scaler, encoder

class DataWindow():
    def __init__(self, 
                 input_width, label_width, shift, 
                 train_df_proc, val_df_proc, test_df_proc, 
                 label_columns = None, scaler = None):
        self.train_df = train_df_proc
        self.val_df = val_df_proc
        self.test_df = test_df_proc
        self.scaler = scaler

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df_proc.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis = -1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def plot(self, model = None, plot_col = 'averagekwh', max_subplots = 1, loc = 'sj'):
        inputs, labels = self.sample_batch

        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        # Inverse-transform inputs for plotting
        averagekwh_index = self.column_indices[plot_col]
        input_values_flat = inputs[:, :, averagekwh_index].numpy().flatten()

        # Prepare for inverse transformation
        batch_size, time_steps, _ = inputs.shape
        n_features = self.scaler.scale_.shape[0]

        # Full input array for inverse transformation
        full_input_array = np.zeros((input_values_flat.shape[0], n_features))
        full_input_array[:, averagekwh_index] = input_values_flat

        # Inverse scale
        original_inputs_flat = self.scaler.inverse_transform(full_input_array)[:, averagekwh_index]
        original_inputs = original_inputs_flat.reshape(batch_size, time_steps, 1)

        # Inverse transform labels similarly
        original_labels = inverse_transform_predictions(labels.numpy(), self.scaler).reshape(batch_size, time_steps, 1)

        plot_title = 'San Jose' if loc == 'sj' else 'San Francisco'

        fig = go.Figure()

        for n in range(max_n):
            # Add inputs line plot
            fig.add_trace(go.Scatter(
                x=self.input_indices,
                y=original_inputs[n, :, plot_col_index],
                mode='lines+markers',
                name='Inputs',
                line=dict(color='blue'),
                marker=dict(symbol='circle')
            ))

            # Determine label column index for scatter plots
            label_col_index = self.label_columns_indices.get(plot_col, plot_col_index)

            # Add original labels as scatter plot
            fig.add_trace(go.Scatter(
                x=self.label_indices,
                y=original_labels[n, :, label_col_index],
                mode='markers',
                name='Labels',
                marker=dict(symbol='square', color='green', size=8, line=dict(color='black', width=1))
            ))

            # If model is provided, generate predictions and plot them
            if model is not None:
                predictions = model(inputs)
                original_predictions = inverse_transform_predictions(predictions.numpy(), self.scaler).reshape(batch_size, time_steps, 1)

                fig.add_trace(go.Scatter(
                    x=self.label_indices,
                    y=original_predictions[n, :, label_col_index],
                    mode='markers',
                    name='Predictions',
                    marker=dict(symbol='x', color='red', size=8, line=dict(color='black', width=1))
                ))

        fig.update_layout(
            title = f'LSTM Predictions for {plot_title}',
            xaxis_title = 'Time Steps (Months)',
            yaxis_title = 'Average Energy Usage (kWh)',
            legend_title="Legend",
            # legend = dict(x = 0.8, y = 1.3),
            height=500
        )

        return fig

    def make_dataset(self, data):
        data = np.array(data, dtype = np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data = data,
            targets = None,
            sequence_length = self.total_window_size,
            sequence_stride = 1,
            shuffle = True,
            batch_size = 32
        )
        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result
    
def compile_and_fit(model, window, patience = 10, max_epochs = 100):
    early_stopping = EarlyStopping(monitor = 'val_loss',
                                    patience = patience,
                                    mode = 'min')
    
    model.compile(loss = MeanSquaredError(),
                    optimizer = Adam(),
                    metrics = [MeanAbsoluteError()])
    
    history = model.fit(window.train,
                        epochs = max_epochs,
                        validation_data = window.val,
                        callbacks = [early_stopping])
    
    return history

# Inverse transformation functions
def inverse_transform_predictions(data, scaler):
    # Reshape data to match the scaler's input shape
    data_reshaped = data.reshape(-1, 1)

    # Create an array for inverse_transform
    n_features = scaler.scale_.shape[0] 

    # Create full input arrays for predictions and labels
    full_predictions = np.zeros((data_reshaped.shape[0], n_features))

    # averagekwh - first feature
    full_predictions[:, 0] = data_reshaped.flatten()

    # Inverse transformation
    original_data = scaler.inverse_transform(full_predictions)[:, 0]

    return original_data

def inverse_transform_categorical(encoded_data, encoder, cat_columns):
    # Inverse encoding
    original_data = encoder.inverse_transform(encoded_data)

    return pd.DataFrame(original_data, columns=cat_columns)

# LSTM Single-Step Predictions on past data
def pred_lstm_single_step(loc, file_specifier, shift):
    joblib_filename_lstm_res = f'joblib_files/lstm/{loc}_lstm_single_step_{file_specifier}.joblib'

    train_df_processed, val_df_processed, test_df_processed, train_df, val_df, test_df, scaler, encoder = lstm_data_processing(loc)

    # Initialize data windows
    wide_window = DataWindow(
        input_width = 12, label_width = 12, shift = shift,
        train_df_proc = train_df_processed, val_df_proc = val_df_processed, test_df_proc = test_df_processed,
        label_columns = ['averagekwh'],
        scaler = scaler
    )

    # Build and train the LSTM model
    lstm_model = Sequential([
        LSTM(32, return_sequences = True),
        Dense(units=1)
    ])

    history = compile_and_fit(lstm_model, wide_window)

    # Gather predictions
    inputs, labels = wide_window.sample_batch
    predictions = lstm_model(inputs)
    
    # Get the plot figure
    fig = wide_window.plot(model = lstm_model, plot_col = 'averagekwh', max_subplots = 1, loc = loc)

    res = {
        'train_score': lstm_model.evaluate(wide_window.train, verbose = 0),
        'val_score': lstm_model.evaluate(wide_window.val, verbose = 0),
        'test_score': lstm_model.evaluate(wide_window.test, verbose = 0),
        'inputs': inputs.numpy(),
        'labels': labels.numpy(),
        'predictions': predictions.numpy(),
        'scaler': scaler,
        'encoder': encoder,
        'fig': fig,
    }

    if not os.path.exists(joblib_filename_lstm_res):
        dump(res, joblib_filename_lstm_res)

    return res

# LSTM Multi-Step Predictions on past data
def pred_lstm_multi_step(loc, file_specifier, shift):
    joblib_filename_lstm_res = f'joblib_files/lstm/{loc}_lstm_multi_step_{file_specifier}.joblib'

    train_df_processed, val_df_processed, test_df_processed, train_df, val_df, test_df, scaler, encoder = lstm_data_processing(loc)

    # Initialize data windows
    multi_window = DataWindow(
        input_width = 12, label_width = 12, shift = shift,
        train_df_proc = train_df_processed, val_df_proc = val_df_processed, test_df_proc = test_df_processed,
        label_columns = ['averagekwh'],
        scaler = scaler
    )
    # Build the LSTM model
    ms_lstm_model = Sequential([
        LSTM(32, return_sequences = True),
        Dense(units = 1, kernel_initializer = tf.initializers.zeros),
    ])

    # Compile and fit the model
    history = compile_and_fit(ms_lstm_model, multi_window)

    # Gather data for predictions
    inputs, labels = multi_window.sample_batch
    predictions = ms_lstm_model(inputs)

    # Get the plot figure
    fig = multi_window.plot(model = ms_lstm_model, plot_col = 'averagekwh', max_subplots = 1, loc = loc)

    # Return model performance and predictions for further processing
    res = {
        'train_score': ms_lstm_model.evaluate(multi_window.train, verbose = 0),
        'val_score': ms_lstm_model.evaluate(multi_window.val, verbose = 0),
        'test_score': ms_lstm_model.evaluate(multi_window.test, verbose = 0),
        'inputs': inputs.numpy(),
        'labels': labels.numpy(),
        'predictions': predictions.numpy(),
        'scaler': scaler,
        'encoder': encoder,
        'fig': fig,
    }

    if not os.path.exists(joblib_filename_lstm_res):
        dump(res, joblib_filename_lstm_res)

    return res


# Make predictions on user input
def make_predictions():
    return 1




# Predict with hypothetical data
def predict_with_hypothetial(model, hypothetical_df, data_window):
    # Process hypothetical data
    hypothetical_processed = data_window.process_hypothetical_input(hypothetical_df)

    # Convert to numpy array and reshape to (batch_size, time_steps, features)
    hypothetical_input = hypothetical_processed.to_numpy().reshape(1, data_window.input_width, -1)

    # Make predictions
    predictions = model.predict(hypothetical_input)

    # Inverse-transform predictions for interpretability
    original_predictions = inverse_transform_predictions(predictions, data_window.scaler)
    
    return original_predictions



# SARIMA predictions
#### What we're doing:  fitting a set of predefined functions of a certain order (p,d,q)(P,D,Q)m, ####
#### and finding out which order resulted in the best fit. ####
def pred_sarima(loc, file_specifier):
    joblib_filename_sarima_res = f'joblib_files/sarima/{loc}_sarima_{file_specifier}.joblib'

    # df = load(f'joblib_files/base_data/{loc}_combined.joblib')

    df_url = f"https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/base_data/{loc}_combined.joblib"
    df = load_joblib_from_github(df_url)

    ps = range(0, 4, 1)
    qs = range(0, 4, 1)
    Ps = range(0, 4, 1)
    Qs = range(0, 4, 1)

    SARIMA_order_list = list(product(ps, qs, Ps, Qs))

    train = df['averagekwh'][:-12]
    test = df.loc[-12:]

    d = 1
    D = 1
    s = 12
    
    '''
    best SARIMA values sj:
    SARIMA(2, 1, 1)(0, 1, 1)12 --> AIC: 1082.58

    best SARIMA values sj:
    SARIMA(1, 1, 1)(3, 1, 3)12 --> AIC: 876.93
    '''

    index_range = [126, 137]

    SARIMA_model = None
    if loc == 'sj':
        SARIMA_model = SARIMAX(train, order = (2, 1, 1), seasonal_order = (0, 1, 1, 12), simple_differencing = False)
    else:
        SARIMA_model = SARIMAX(train, order = (1, 1, 1), seasonal_order = (3, 1, 3, 12), simple_differencing = False)
        index_range = [119, 130]

    SARIMA_model_fit = SARIMA_model.fit(disp = False)

    # Forecast the number of monthly averagekwh for the year of 2024 to compare the predicted values to the observed values in the test set
    SARIMA_pred = SARIMA_model_fit.get_prediction(index_range[0], index_range[1]).predicted_mean

    test['SARIMA_pred'] = SARIMA_pred

    # Calculate the mean absolute percentage error (MAE)
    def calc_mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    mae_SARIMA = calc_mae(test['averagekwh'], test['SARIMA_pred'])

    res = {
        'test': test,
        'df': df,
        'mae_SARIMA': mae_SARIMA
    }

    if not os.path.exists(joblib_filename_sarima_res):
        dump(res, joblib_filename_sarima_res)
    
    return res


### Code to fetch, process, and save the historical data ###
repo_urls = {
    f'https://raw.githubusercontent.com/gabrield03/data_files/refs/heads/main/data/energy/Combined_Energy_Data.csv': ['joblib_files/base_data/energy_data.joblib', ['joblib_files/base_data/sj_energy_df.joblib', 'joblib_files/base_data/sf_energy_df.joblib']],
    f'https://raw.githubusercontent.com/gabrield03/data_files/refs/heads/main/data/weather/SJ_95110_SJAirport.csv': ['joblib_files/base_data/sj_weather_data.joblib', ['joblib_files/base_data/sj_weather_df.joblib']],
    f'https://raw.githubusercontent.com/gabrield03/data_files/refs/heads/main/data/weather/SF_94102_DowntownSF.csv': ['joblib_files/base_data/sf_weather_data.joblib', ['joblib_files/base_data/sf_weather_df.joblib']]
}

# Fetch and clean the historical data
for url, joblib_filenames in repo_urls.items():
    fetch_historical_data(url, joblib_filenames[0], joblib_filenames[1])


# Combine the historical data
if os.path.exists('joblib_files/base_data/sj_energy_df.joblib') and os.path.exists('joblib_files/base_data/sf_energy_df.joblib') and os.path.exists('joblib_files/base_data/sj_weather_df.joblib') and os.path.exists('joblib_files/base_data/sf_weather_df.joblib'):
    df1 = load('joblib_files/base_data/sj_energy_df.joblib')
    df2 = load('joblib_files/base_data/sf_energy_df.joblib')
    df3 = load('joblib_files/base_data/sj_weather_df.joblib')
    df4 = load('joblib_files/base_data/sf_weather_df.joblib')

    if len(df1) != 0 and len(df2) != 0 and len(df3) != 0 and len(df4) != 0:
        combine_historical_data(df1, df2, df3, df4)
    else:
        print('Some error occurred while loading clean data')


### Load Data ###
def load_joblib_from_github(raw_url: str):
    """Load a joblib file directly from GitHub RAW."""
    response = requests.get(raw_url)
    response.raise_for_status()
    return load(BytesIO(response.content))