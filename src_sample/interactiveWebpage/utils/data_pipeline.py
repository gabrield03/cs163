import pandas as pd
import numpy as np
import requests
from io import StringIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
from dash.dash_table.Format import Format, Scheme
from dash import html
import plotly.graph_objects as go
import base64
from io import BytesIO
import shap
from joblib import dump, load

# ML libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error

from keras import Sequential
from keras import layers


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
def processing_pipeline(df):
    # Transformer class for pipeline
    class ReshapeTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return X.reshape(-1, 3)
        
    loc = 'sf'
    if 'awnd' in df.columns:
        loc = 'sj'
    
    joblib_filename_model = f'joblib_files/processed_data/{loc}_rf.joblib'
    #joblib_filename_df = f'joblib_files/processed_data/{loc}_df_processed.joblib'
    joblib_filename_X_train = f'joblib_files/processed_data/{loc}_X_train.joblib'
    joblib_filename_X_test = f'joblib_files/processed_data/{loc}_X_test.joblib'
    joblib_filename_y_train = f'joblib_files/processed_data/{loc}_y_train.joblib'
    joblib_filename_y_test = f'joblib_files/processed_data/{loc}_y_test.joblib'
    joblib_filename_X_train_processed_df = f'joblib_files/processed_data/{loc}_X_train_processed_df.joblib'
    joblib_filename_preprocessor = f'joblib_files/processed_data/{loc}_preprocessor.joblib'
    joblib_filename_importances = f'joblib_files/processed_data/{loc}_importances_df.joblib'



    drop_list = ['zipcode', 'totalkwh', 'customerclass', 'combined', 'region', 'month-numeric', 'year-month']

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
        transformers=[
            ('num', StandardScaler(), num_col_list),
            ('cat', Pipeline(steps=[('encode', OrdinalEncoder()), ('reshape', ReshapeTransformer())]), cat_col_list)
        ]
    )
    #global preprocessor_gl
    #preprocessor_gl = preprocessor # fix later

    
    
    # Fit the preprocessor on training data and transform it
    X_train_processed = preprocessor.fit_transform(X_train)


    ####using this for shap! another joblib file####
    num_col_names = num_col_list  # Names for numerical features
    cat_col_names = preprocessor.named_transformers_['cat'].named_steps['encode'].get_feature_names_out(cat_col_list)  # Categorical feature names
    all_col_names = list(num_col_names) + list(cat_col_names)

    X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_col_names)
    X_test_processed = preprocessor.transform(X_test)

    # Combine the feature names
    all_features = num_col_list + cat_col_list

            # PIPELINE STUFF
            # # Create a pipeline that includes the preprocessor and a model (RandomForestRegressor in this case)
            # pipeline = Pipeline(steps=[
            #     ('preprocessor', preprocessor),
            #     ('model', RandomForestRegressor())
            # ])

            # # Fit the pipeline on the training data
            # pipeline.fit(X_train, y_train)

            ## Make predictions on the test data
            # y_pred = pipeline.predict(X_test)

    # Fit the RandomForest model on the preprocessed training data
    rf = RandomForestRegressor().fit(X_train_processed, y_train)

    # Get feature importances
    rf_importances = rf.feature_importances_

    # Create a dataframe for feature importances
    importances_df = pd.DataFrame(data=rf_importances, index = all_features, columns=['importances'])

    importances_df.reset_index(inplace=True)
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

    # rf model
    if not os.path.exists(joblib_filename_model):
        dump(rf, joblib_filename_model)

    # rf importances
    if not os.path.exists(joblib_filename_importances):
        dump(importances_df, joblib_filename_importances)

    if not os.path.exists(joblib_filename_preprocessor):
        dump(preprocessor, joblib_filename_preprocessor)


    return importances_df

def calc_shap(loc):
    joblib_filename_X_train = f'joblib_files/processed_data/{loc}_X_train_processed_df.joblib'
    joblib_filename_model = f'joblib_files/processed_data/{loc}_rf.joblib'

    X_train = None
    model = None

    X_train = load(joblib_filename_X_train)
    model = load(joblib_filename_model)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)

    shap_values_array = shap_values.values
    # Get feature names from the DataFrame
    feature_names = X_train.columns
    
    shap_df = pd.DataFrame(shap_values_array, columns=feature_names)

    # Compute the mean absolute SHAP values for each feature
    shap_mean_abs = shap_df.abs().mean().sort_values(ascending=False)

    # Convert to a DataFrame for plotting
    shap_plot_df = pd.DataFrame(shap_mean_abs).reset_index()
    shap_plot_df.columns = ['Feature', 'Mean SHAP Value']

    # Create the Plotly figure for visualization
    fig = {
        'data': [{
            'x': shap_plot_df['Feature'],
            'y': shap_plot_df['Mean SHAP Value'],
            'type': 'bar',
            'marker': {'color': 'blue'},
        }],
        'layout': {
            'title': f'Mean SHAP Values for {loc}',
            'xaxis': {'title': 'Features'},
            'yaxis': {'title': 'Mean SHAP Value'},
        }
    }

    return fig

def calc_lstm(loc, request_new_joblib, file_specifier):
    joblib_filename_lstm_res = f'joblib_files/lstm/{loc}_lstm_results_{file_specifier}.joblib'
    joblib_filename_lstm_model = f'joblib_files/lstm/{loc}_lstm_model_{file_specifier}.joblib'
    joblib_filename_X_test_step = f'joblib_files/lstm/{loc}_X_test_step_{file_specifier}.joblib'
    joblib_filename_X_train_step = f'joblib_files/lstm/{loc}_X_train_step_{file_specifier}.joblib'

    ## Note: X_train and X_test are numpy arrays
    ## Note: y_train and y_test are pandas series
    X_train = load(f'joblib_files/processed_data/{loc}_X_train.joblib')
    X_test = load(f'joblib_files/processed_data/{loc}_X_test.joblib')
    y_train = load(f'joblib_files/processed_data/{loc}_y_train.joblib')
    y_test = load(f'joblib_files/processed_data/{loc}_y_test.joblib')

    # Rescale target variables (y_train, y_test) using MinMaxScaler
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_numpy = y_test.values.reshape(-1, 1)

    # Reshape input data for LSTM (samples, time_steps, features)
    # Think im stuck with 12 for sj and 7 for sf
    time_steps = 1
    if loc == 'sj':
        time_steps = 12
    else:
        time_steps = 7

    X_train = X_train.reshape((X_train.shape[0], time_steps, X_train.shape[1] // time_steps))
    X_test = X_test.reshape((X_test.shape[0], time_steps, X_test.shape[1] // time_steps))

    # Build the LSTM model
    model = Sequential()
    model.add(layers.LSTM(50, return_sequences = True, input_shape = (time_steps, X_train.shape[2])))
    model.add(layers.LSTM(50))
    model.add(layers.Dense(1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Train the model
    model.fit(
        X_train,
        y_train_scaled,
        epochs = 10,
        batch_size = 32,
        verbose = 1
    )

    # Make predictions
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    # Calculate mse and mae scores
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    scores = {'mse': mse, 'mae': mae}
    
    actual_data = {'time': list(range(len(y_test_numpy))), 'values': y_test_numpy.flatten()}
    lstm_predictions = {'time': list(range(len(y_pred))), 'values': y_pred.flatten()}


    # joblib dump the files if it doesnt exist or we are requesting a new one
    if request_new_joblib:
        res = {
            'scores': scores,
            'actual_data': actual_data,
            'predictions': lstm_predictions,
        }

        dump(res, joblib_filename_lstm_res)
        dump(model, joblib_filename_lstm_model)
        dump(X_test, joblib_filename_X_test_step)
        dump(X_train, joblib_filename_X_train_step)

    # Return scores and predictions
    return scores, actual_data, lstm_predictions


# work in progress
# LSTM Future Predictions
def lstm_predict(model, last_known_data, future_steps=4): # BASICALLY PASS FOR NOW
    return 1
    future_predictions = []
    
    # Reshape the last known data for LSTM input
    input_data = last_known_data#.reshape((1, last_known_data.shape[0], last_known_data.shape[1]))
    
    for _ in range(future_steps):
        # Predict
        prediction = model.predict(input_data)
        
        # Store the predicted value
        future_predictions.append(prediction[0, 0])
        
        prediction_reshaped = prediction.reshape((1, 1, 1))  # Shape it to (1, 1, features)

        # Concatenate along the time axis
        input_data = np.append(input_data[:, 1:, :], prediction_reshaped, axis=1)

    future_predictions = np.array(future_predictions)

    # Inverse transform the predictions to get them back to the original scale
    #sc = preprocessor_gl.named_transformers_['num']
    predictions_original_scale = sc.inverse_transform(future_predictions.reshape(-1, 1).repeat(9, axis=1))

    return predictions_original_scale




### Code to fetch, process, and save the historical data ###
repo_urls = {
    f'https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveWebpage/data/energy/Combined_Energy_Data.csv': ['joblib_files/base_data/energy_data.joblib', ['joblib_files/base_data/sj_energy_df.joblib', 'joblib_files/base_data/sf_energy_df.joblib']],
    f'https://raw.githubusercontent.com/gabrield03/cs163/main/src_sample/interactiveWebpage/data/weather/SJ_95110_SJAirport.csv': ['joblib_files/base_data/sj_weather_data.joblib', ['joblib_files/base_data/sj_weather_df.joblib']],
    f'https://raw.githubusercontent.com/gabrield03/cs163/main/src_sample/interactiveWebpage/data/weather/SF_94102_DowntownSF.csv': ['joblib_files/base_data/sf_weather_data.joblib', ['joblib_files/base_data/sf_weather_df.joblib']]
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