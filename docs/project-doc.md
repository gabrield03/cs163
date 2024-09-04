# Proposal: Energy Consumption Forecasting

## Project Summary
<!--- Write a summary of your project including the project goals, broader impacts, and data sources -->

- This project will develop a predictive model for energy (electricity) consumption in California (northern CA/Bay Area/location in NA still undecided) by analyzing historical energy usage and weather data.

- The goal is to forecast energy demands which will be used to prepare for periods of high and low energy usage by optimizing energy distribution. This will reduce costs and system strain. Forecasting energy consumption impacts energy management agencies, environmental sustainability, and financial costs for both consumers and energy providers. Historical data will be sourced from government agencies such as the Department of Energy (DOE) and the National Oceanic and Atmospheric Administration (NOAA), ensuring reliable data. Other potential sources of data are the UCI Machine Learning Repository and Kaggle. Additionally, data.gov provides API for real-time data collection.


## Data Sources
<!--- List data sources, including the existing datasets and anything you are going to collect by yourself. It is expected to combine two or more data sources in your project. -->
<!--- Each dataset should be briefly explained: what kinds of data are available, who collected the dataset, how the data was collected. -->

- Energy Consumption Data:
	- This dataset will include historical energy consumption records for the [X] region, typically collected by local and national energy agencies. Data may include daily, monthly, and annual energy usage.

	- The Total Energy Data and Statistics dataset, provided by the DOE, offers monthly and annual records of US regional energy consumption, production, imports, exports, and prices dating back to 1973. It includes residential, industrial, and commercial end-use sectors.

- Weather Data:
	- Historical weather data will be obtained from the NOAA database, a federal meteorological agency that gathers information relating to the weather, climate, and environment.

	- A potential dataset is the Daily Weather Records, a collection of weather-related records of temperature, humidity, wind speed, and precipitation. The data was gathered from various weather stations across the US. The dataset has been collected for more than 30 years and records are current to the present day.
	- The San Francisco Weather Data dataset is from Kaggle and was provides weather-related data from 1973 to 2023. It contains daily weather observations in San Francisco, CA. The data was gathered from Meteostat. The dataset includes various temperature, wind, and precipitation records.


## Expected Major Findings
<!--- List and explain what information you want to obtain in this project. Explain how valuable this project could be based on the objective discussion. You may want to list main claims and questions you want to answer through the project. -->

- This project is expected to identify patterns and correlations between weather conditions and energy consumption in Northern CA (or somewhere else). The main findings are likely to include:
	- Weather events and factors that most significantly impact energy consumption and demand.
	- Seasonal trends in energy usage.
	- Forecasts of energy consumption that can be used to optimize energy distribution and reduce costs.

## Preprocessing Steps
<!--- List major preprocessing steps needed for the datasets and explain why. -->

- Data Aggregation:
	- Combine/align the energy consumption data with its corresponding weather data. This would require matching time periods due to differences in data collection intervals.

- Data Cleaning:
	- Identifying and removing inconsistent or missing data from energy and weather datasets.
	- Handle outliers or anomalies in the data.

- Feature Engineering and Dimensionality Reduction:
	- Create new features that better represent the relationship between energy consumption and weather. 
	- Identify the most impactful features in the data.

- Data Normalization:
	- Normalize the data considering the differences in scales between energy consumption and weather variables (not yet sure of actual scales, yet).


<!--- 
----------
The following sections should be used for the full proposal document. These are not required for the proposal draft discussion.
----------
-->




## Basic Data Properties and Analysis Techniques
<!--- Based on the lectures on "Exploratory Data Analysis" and "Data and Sampling", list and explain what types of basic statistical analysis you plan to provide to give the meta information and overall picture of the datasets. -->



## Automation, Scalability, and Portability
<!--- Assume that newer datasets will become available from the same source in future, or you need to ask your colleague to inherit this project. What will be major challenges? List and explain technical and implementational practices you will use to enhance automation, scalability, and portability aspects of the project. -->




<!--- 
----------
The following sections should be used for the analysis planning. These are not required for the proposal document submission.
----------
-->



## Data Analysis and Algorithms
<!--- List and describe what types of (advanced) analysis you plan to conduct. This section should be tied back to the expected major findings. (If needed, you can update the findings section.) When selecting algorithms to obtain the analysis results, provide a brief explanation of the algorithmic properties and logic. You should clearly define the inputs and outputs of each algorithm. -->




<!--- 
----------
The following sections should be used for the analysis outcome presentation. These are not required for the analysis plan submission.
----------
-->
# Analysis Outcomes
<!--- Explain the analysis you conducted and show the results. Discuss how the data, your analysis, and/or visualization can support the claims or findings. What will be the recommendations or suggestions you can make based on the results? Use bullet points, tables, and figures (if possible) to increase the readability of the document. -->



<!--- 
----------
The following sections should be used for the visualization planning. These are not required for the analysis outcome presentation.
----------
-->


# Visualization
## Visualization Plan
<!--- List and explain what types of plots you plan to provide and what you are trying to show through the diagrams. You should explore the best way to visualize the information and message based on the lectures on visualization and related topics. It is required to have at least two interactive graphing and five static plots. -->

## Web Page Plan
<!--- Explain how many pages you will have in total and what content will be shown in each page. (Each diagram discussed above should be given a proper location in this section. Also, it is required to have (1) "Project Objective" page, which explains the main goals and data sources, and (2) "Analytical Methods" page, where you explain the major techniques used in the project and provide further references. -->
