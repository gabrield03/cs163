# Proposal: Effects of Weather on Energy Consumption in the Bay Area

## Project Summary
<!--- Write a summary of your project including the project goals, broader impacts, and data sources -->

- This project will develop a predictive model for energy (electricity) consumption in the California Bay Area by analyzing historical weather trends and energy usage.
- The goal is to forecast energy demand (periods of high or low usage) which will be used to optimize energy distribution. Forecasting energy consumption impactboth energy provider and consumer costs, aid management agencies, and promote environmental sustainability. Data will be collected from governmental agencies such as the U.S. Energy Information Administation (EIA) and the National Oceanic and Atmospheric Administration (NOAA). Other sources of data include data.gov (real-time data) and PG&E electricity usage records. Data collected from these sources ensure accuracte and reliable data.



## Data Sources
<!--- List data sources, including the existing datasets and anything you are going to collect by yourself. It is expected to combine two or more data sources in your project. -->
<!--- Each dataset should be briefly explained: what kinds of data are available, who collected the dataset, how the data was collected. -->

- Energy Data:
	- This dataset was collected by PG&E, a company that provides electricity and natural gas to consumers in northern and central California. The data is publically accessible through PG&E's public data sets. Energy usage data is reported by ZIP code, per month, in kilowatt-hours (kWh). It contains usage by customer type (residential, commercial, agricultural, and industrial) and report total customers, total kWh, and average kWh. 
	- Another dataset is provided by the EIA, a federal agency that collects and shares data on energy supply and consumption in the United States. This dataset lists daily PG&E electricity consumption per day. It is recorded in megawatt-hours (mWh). 

- Weather Data:
	- Historical weather records will be sourced from the NOAA. The NOAA is a federal agency that records national weather, climate, environment, and ocean information. The NOAA gathers data from the Global Historical Climatology Network. It includes various station networks across the country. For this project specifically, the weather data will be sourced from various Bay Area weather stations. The datasets include daily measurements such as maximum and minimum temperature, precipitation, and wind speed. 



## Expected Major Findings
<!--- List and explain what information you want to obtain in this project. Explain how valuable this project could be based on the objective discussion. You may want to list main claims and questions you want to answer through the project. -->

- This project aims to uncover significant relationships between weather conditions and energy consumption patterns in California's Bay Area. The key findings are expected to include:
	- Identification of Critical Weather Factors:
		- An analysis of which weather variables (e.g., temperature, humidity, extreme events) have the greatest impact on energy consumption.

	- Seasonal Energy Consumption Trends:
		- Insights into how energy usage fluctuates throughout the year.

	- Predictive Model for Energy Consumption Forecasting:
		- Development of models to forecast energy consumption based on projected weather patterns.

	- Broader Implications:
		- The findings may provide valuable information for energy providers to improve energy efficiency.



## Preprocessing Steps
<!--- List major preprocessing steps needed for the datasets and explain why. -->

- Data Aggregation:
	- Combine the energy consumption data with corresponding weather data by aligning their time intervals.
		- Energy data is available monthly, while weather data is available daily. To integrate these, daily weather data will be aggregated into monthly averages (e.g., max temperature per month).
		- Since energy data is provided by ZIP code, each weather station will need to be mapped to the appropriate ZIP code or nearby areas to ensure proper alignment between datasets.

- Data Cleaning:
	- Address any inconsistencies, missing values, or anomalies in both datasets
		- In the energy data, some ZIP codes may have missing monthly records. Missing values may be interpolated, especially if the gaps are small, to maintain data consistency.
		- Outliers and anomalies in both energy and weather data will be reviewed. In some cases, extreme weather events or usage spikes may be relevant and kept for analysis, while others may need to be flagged.

- Feature Scaling:
	- Normalize the features to ensure they are on comparable scales.
		- Different variables like temperature, humidity, and energy consumption may have varying units and magnitudes. Standardization or min/max scaling will be applied, depending on data distributions, to ensure features contribute equally to analysis and modeling.

- Feature Selection:
	- Identify the most relevant features for analyzing the impact of weather on energy consumption.
		- Some features in the weather data (e.g., wind speed, humidity) may have minimal influence, while others like temperature might be more significant. Feature selection techniques will help improve the analysis by focusing on the most impactful variables.

- Data Pipelining:
	- Implement a data processing pipeline to streamline tasks such as feature scaling, selection, and model training.
		- Using a pipeline will allow efficient experimentation with parameters, such as through grid search, while automating the transformation of raw data into a suitable format for analysis and modeling.



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
