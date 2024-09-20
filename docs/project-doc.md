# Proposal: Effects of Weather on Energy Consumption in the Bay Area

## Project Summary
<!--- Write a summary of your project including the project goals, broader impacts, and data sources -->

- This project aims to predict electricity consumption in the California Bay Area by analyzing historical weather data and energy usage patterns. The primary goal is to forecast periods of high or low energy demand, which can assist in optimizing energy distribution. Identifying key weather factors that influence electricity usage can help energy providers reduce operational costs, enhance efficiency, and support environmental sustainability.
- Data will be sourced from the U.S. Energy Information Administration (EIA), PG&E, and the National Oceanic and Atmospheric Administration (NOAA). Data collected from these agencies will ensure accurate and reliable datasets.


## Data Sources
<!--- List data sources, including the existing datasets and anything you are going to collect by yourself. It is expected to combine two or more data sources in your project. -->
<!--- Each dataset should be briefly explained: what kinds of data are available, who collected the dataset, how the data was collected. -->

- Energy Data:
	- PG&E Electricity Usage:
		- This publicly accessible dataset from PG&E, includes monthly electricity consumption by ZIP code for northern and central California. It provides details on customer types (residential, commercial, etc.), total kilowatt-hours (kWh) consumed, and the number of customers per ZIP code. This dataset is useful for analyzing trends in energy usage over time at a local level.

	- EIA Daily Energy Data:
		- The U.S. Energy Information Administration (EIA) provides daily electricity consumption data for the PG&E service area, measured in megawatt-hours (mWh). This dataset offers a more granular view of energy consumption patterns over time.

- Weather Data:
	- NOAA Climate Data:
		- The weather data will be sourced from the National Oceanic and Atmospheric Administration (NOAA), specifically from the Global Historical Climatology Network. This dataset includes daily weather metrics like maximum/minimum temperatures, precipitation, and wind speed. Multiple Bay Area weather stations will be used to align weather patterns with energy consumption.


## Expected Major Findings
<!--- List and explain what information you want to obtain in this project. Explain how valuable this project could be based on the objective discussion. You may want to list main claims and questions you want to answer through the project. -->

- This project aims to identify relationships between weather conditions and energy consumption patterns in the California Bay Area. The key findings are expected to include:
	- Identification of Critical Weather Factors:
		- An analysis of the weather variables (e.g., temperature, humidity, extreme events) that have the greatest impact on energy consumption.

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
		- Since energy data is provided by ZIP code, each weather station will need to be mapped to the ZIP codes they service to properly align the datasets.

- Data Cleaning:
	- Address any inconsistencies, missing values, or anomalies in both datasets.
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
		- Using a pipeline will allow efficient use of grid search to identify the best parameters while automating the transformation of raw data into a suitable format for analysis and modeling.



<!--- 
----------
The following sections should be used for the full proposal document. These are not required for the proposal draft discussion.
----------
-->




## Basic Data Properties and Analysis Techniques
<!--- Based on the lectures on "Exploratory Data Analysis" and "Data and Sampling", list and explain what types of basic statistical analysis you plan to provide to give the meta information and overall picture of the datasets. -->

- Exploratory Data Analysis (EDA):
	- The EDA techniques I will use to provide the meta information and overall picture of the datasets will show the data's general distributions and variability.
		- Specifically, the distribution of monthly energy usage for specific regions, energy usage trends over time, and the variability in energy usage per month. The same will be conducted with weather data.
		- Additionally, the impact of temperature on energy usage and correlations between energy usage and temperature will be explored.

	- Energy
		- Distribution of Energy Usage:
			- For both San Jose (SJ) and San Francisco (SF), I will visualize the distribution of average energy usage using Histogram and Kernel Density Estimation (KDE) plots.
			- These provide insights into how energy usage varies across the regions. Understanding the central tendency, spread, and skewness of energy usage helps identify patterns or anomalies in the data.
			
			- [SJ - Hist/KDE: Average Energy Usage](/docs/assets/images/Hist_Avg_Energy_Usage_95110)
			- [SF - Hist/KDE: Average Energy Usage](/docs/assets/images/Hist_Avg_Energy_Usage_94102)

		- Heatmap of Energy Usage:
			- I will create a heatmap that visualizes energy usage over time, with years on the x-axis and months on the y-axis.
			- This will show patterns in energy consumption and any long-term trends that emerge over the years. Coupled with weather statistics, potentially providing evidence of climate-related shifts in energy usage.
			
			- [SJ - Heatmap: Average Energy Usage](/docs/assets/images/Heatmap_Avg_Energy_Usage_95110)
			- [SF - Heatmap: Average Energy Usage](/docs/assets/images/Heatmap_Avg_Energy_Usage_94102)

	- Weather
		- Distribution of Average Maximum and Minimum Temperatures:
			- 

		- Heatmap of Average Maximum and Minimum Temperatures:
			- 

	- Combined:
		- Average Monthly Energy Usage and Temperature Comparison:
			- 

- Statistical Testing:
	- I plan to use Analyis of Variance (ANOVA), one-way and multi-way, to evaluate whether the observed differences in energy usage across months and between regions (SJ and SF) are statistically significant.
		- One-way ANOVA will be used to compute the variance in energy usage by month and whether those differences are statistically significant.
		- Multi-way ANOVA will be used to compute the variance between groups (average energy usage, temperature, and region) and assess whether the results are statistically significant.

	- ANOVA Assumptions:
		- Normality:
			- [Scipy - normaltest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)  
			- The data within each group should be normally distributed.
			- I will use the normaltest to assess whether the distribution of energy usage follows a normal distribution for each group (month).

		- Homogeneity of Variance: 
			- [Scipy - Levene Test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html)  
			- The variance of the data within each group should be equal.
			- I will use Levene's test to determine whether the variance in energy usage is similar across each group.

		- Independence:
			- Observations within each group should be independent.
			- Assuming that energy usage in different months is independent of each other.

	- One-way ANOVA:
		- [Scipy - one-way ANOVA](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
		- 

	- Multi-way ANOVA:
		- 



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
