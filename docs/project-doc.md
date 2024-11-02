# Proposal: Effects of Weather on Energy Consumption in the Bay Area

## Project Summary
<!--- Write a summary of your project including the project goals, broader impacts, and data sources -->

- This project investigates the relationship between electricity consumption and weather patterns in the California Bay Area. By analyzing historical weather data and energy usage trends, the primary goal is to uncover key weather factors that influence electricity usage. After discovering influential weather factors, electricity demands predictions will be made. The broader goal is to explore how shifts in climate (e.g., increased frequency of extreme temperatures) affect energy usage. Identifying these factors not only helps energy providers reduce operational costs by increasing energy distribution efficiency but also helps the general public concerned with climate change and how it affects their local regions. This analysis could be expanded to other regional areas by providing a template for understanding climate-driven energy trends on a larger scale.

- Data will be sourced from the U.S. Energy Information Administration (EIA), PG&E, and the National Oceanic and Atmospheric Administration (NOAA). These are reliable data sources that ensures that any published data is accurate and trutworthy.


## Data Sources
<!--- List data sources, including the existing datasets and anything you are going to collect by yourself. It is expected to combine two or more data sources in your project. -->
<!--- Each dataset should be briefly explained: what kinds of data are available, who collected the dataset, how the data was collected. -->

- Energy Data:
	- PG&E Electricity Usage:
		- Includes publicly accessible data from PG&E that contain monthly electricity consumption by ZIP code for northern and central California. Data consists of customer types (residential, commercial, etc.), average kilowatt-hours per customer (kWh) consumed, total kilowatt-hours (kWh) consumed, and the number of customers per ZIP code.

- Weather Data:
	- NOAA Climate Data:
		- Includes data from the National Oceanic and Atmospheric Administration (NOAA), consisting of daily maximum/minimum temperatures, precipitation, and wind speed measurements from various Bay Area (San Jose and San Francisco) weather stations.


## Expected Major Findings
<!--- List and explain what information you want to obtain in this project. Explain how valuable this project could be based on the objective discussion. You may want to list main claims and questions you want to answer through the project. -->

- Climate Impact on Energy Demand:
	- Identification of weather variables (e.g., temperatures, precipitation, extreme weather events etc.) that significantly affect energy consumption. This analysis could reveal regions that are more susceptible or vulnerable to the effects of climate change. I expect to learn about the disproportionate affect that various weather conditions have on different regions.
	
- Predictive Models for Energy Consumption Forecasting:
	- I expect to develop predictive models for energy consumption that use weather data to forecast periods of high and low-energy demand influenced by weather patterns and seasonal changes.

- Generalizable Climate Models:
	- Ideally, this analysis will have a broader impact beyond just predicting energy consumption in the Bay Area. It can serve as a model for analyzing the energy-climate relationship in regions with different climate conditions.
	- For instance, regions less affected by temperature fluctuations may have more stable energy demands despite having broader climate variability. By providing a generalized model, this analysis could guide climate adaptation strategies for different regions.


## Preprocessing Steps
<!--- List major preprocessing steps needed for the datasets and explain why. -->

- Data Aggregation:
	- Combine the energy consumption data with corresponding weather data by aligning their time intervals.
		- Energy data is available monthly, while weather data is available daily. To integrate these, daily weather data will be aggregated into monthly averages (e.g., max average temperature per month).
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

- **Exploratory Data Analysis (EDA)**:
	- The EDA techniques I will use to provide the meta information and overall picture of the data will focus on showing the data's general distributions and variability.
		- Specifically, the distribution of monthly energy usage for specific regions, energy usage trends over time, and the variability in energy usage per month. The same will be conducted with weather data.
		- Additionally, the impact of temperature on energy usage and correlations between energy usage and temperature will be explored.

	- Energy
		- Distribution of Energy Usage:
			- For both San Jose (SJ) and San Francisco (SF), I created visualizations to show the distribution of average energy usage using Histogram and Kernel Density Estimation (KDE) plots. The visualizations explain how energy usage varies across regions. It depicts the central tendency, spread, and skew of energy usage - we can use these to identify patterns or trends in the data. Each regional plot is plotted on the same x-axis scale to display the difference in average energy usage between regions.

			![alt text](/docs/assets/plots/Hist_Avg_Energy_Usage_95110.png)

			![alt text](/docs/assets/plots/Hist_Avg_Energy_Usage_94102.png)


		- Heatmap of Energy Usage:
			- These heatmaps show energy usage over time (by month and year), with years on the x-axis and months on the y-axis. This shows patterns in energy consumption and any long-term trends that emerge over the years. Coupled with weather statistics, it can potentially provide evidence of climate-related shifts in energy usage.
			
			| Average Energy Usage Heatmap (SJ - 95110)                              | Average Energy Usage Heatmap (SF - 94102)                              |
			|:---------------------------------------------------------------------: | :---------------------------------------------------------------------:|
			| ![Energy Heatmap](/docs/assets/plots/Heatmap_Avg_Energy_Usage_95110.png) | ![Energy Heatmap](/docs/assets/plots/Heatmap_Avg_Energy_Usage_94102.png) |


	- Weather
		- Distribution of Average Maximum and Minimum Temperatures:
			- Similar to the energy visualizations, I these Histograms and KDE plots display the distributions of the average maximum and average minimum temperatures in each region. The plots combine the distribution for each of the temperature metrics to show the indvidual differences between minimum and maximum temperatures. Additionally, each plot is on the same x-axis scale to display the difference in average temperature between regions.

			![alt text](/docs/assets/plots/Hist_Avg_Max_Min_Temp_95110.png)

			![alt text](/docs/assets/plots/Hist_Avg_Max_Min_Temp_94102.png)


		- Heatmap of Average Maximum and Minimum Temperatures:
			- Like the energy visualizations of temperature, these show the temperature (min and max) data over time in each region (SJ and SF). Comparing the heatmaps of temperatures allows us to visualize the regional differences in temperature trends from 2013 to 2024.

	
			
			| Average Max Temperature Heatmap (SJ - 95110)                           | Average Min Temperature Heatmap (SJ - 95110)                           |
			|:---------------------------------------------------------------------: | :---------------------------------------------------------------------:|
			| ![Max Temp Heatmap](/docs/assets/plots/Heatmap_Avg_Max_Temp_95110.png) | ![Min Temp Heatmap](/docs/assets/plots/Heatmap_Avg_Min_Temp_95110.png) |


			| Average Max Temperature Heatmap (SF - 94102)                           | Average Min Temperature Heatmap (SF - 94102)                           |
			|:---------------------------------------------------------------------: | :---------------------------------------------------------------------:|
			| ![Max Temp Heatmap](/docs/assets/plots/Heatmap_Avg_Max_Temp_94102.png) | ![Min Temp Heatmap](/docs/assets/plots/Heatmap_Avg_Min_Temp_94102.png) |


	- Energy Usage and Weather Combined:
		- Average Monthly Energy Usage and Temperature Comparison:
			- These are dual-axis plot that compare the average energy usage per month (from 2013 to 2024) with the average maximum and minimum temperatures, by region. Visually, energy usage is plotted using a box plot and the min/max temperatures are plotted with line plots. With these plots, the goal is to identify patterns, relationships, or correlations between temperature and energy consumption as they change over time.

			![alt text](/docs/assets/plots/Dual_Ax_95110.png)

			![alt text](/docs/assets/plots/Dual_Ax_94102.png)



- **Statistical Testing**:
	- I used Analyis of Variance (ANOVA) (one-way and multi-way) to evaluate whether the variance observed in average energy usage across months, between regions (SJ and SF), and by temperature are statistically significant.
		- One-way ANOVA is used to compute the variance in average energy usage by month.
		- Multi-way ANOVA is used to compute the variance between groups (average energy usage, temperature, and region).

	- ANOVA Assumptions
		- Normality: The data within each group should be normally distributed.
			- [Scipy - normaltest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)  

			- I used the normaltest to test whether the average energy usage per month in a region is not normally distributed.
			- The Null Hypothesis is that the data follows a normal distribution. We reject the null hypothesis if the p-value < 0.05.
			It indicates that there is a low probability of sampling data from a normally distributed population that produces such an extreme value of the statistic.

			| | normal statistic |
			| --- | --- |
			| SJ (95110) | 5.004 |
			| SF (94102) | 18.809 |

			![alt text](/docs/assets/statistical-testing/normality_test_95110.png)

			![alt text](/docs/assets/statistical-testing/normality_test_94102.png)

			- Results:
				- SJ p-value = 0.081911
				- SF p-value = 0.000118

			- Interpretation:
				- SJ's p-value > 0.05, we fail to reject the null hypothesis.
				- SF's p-value < 0.05, we reject the null hypothesis. There is evidence that the sampled data is not normally distributed.


		- Homogeneity of Variance: The variance of the data within each group should be equal.
			- [Scipy - Levene Test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html)  

			- I used Levene's test to determine whether the variance in energy usage is similar across each group. It tests the null hypothesis
			that all input samples (average kWh) are from populations (months) with equal variances.
			- The p-value represents the proportion of values in the null distribution greater than or equal to the observed value of the statistic.

			| | Levene statistic |
			| --- | --- |
			| SJ (95110) | 1.019 |
			| SF (94102) | 1.324 |

			![alt text](/docs/assets/statistical-testing/homogeneity_of_variance_95110.png)

			![alt text](/docs/assets/statistical-testing/homogeneity_of_variance_94102.png)

			- Results:
				- SJ p-value = 0.364
				- SF p-value = 0.270

			- Interpretation:
				- SJ and SF p-values > 0.05, we fail to reject the null hypothesis.
				- *Note: Actual SF p-value = 0.434		&nbsp; &nbsp; &nbsp; &nbsp; 	Actual SF p-value = 0.219
					- At this momeny, I am unsure why the the incorrect p-value is shown in the plot, but it does not affect the interpreation of the results.

		- Independence: Observations within each group should be independent.
			- Assuming that energy usage in different months is independent of each other, but I do not think that observations can be considered independent because data at the end of a prior day rolls into the data of the next day (e.g., energy usage at 11:59 PM on day 1 is continous into 12:00 AM on day 2).

	- One-way ANOVA:
		- [Scipy - one-way ANOVA](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)

		- Null Hypothesis: The average energy usage across the months is the same.
		- Alternative Hypothesis: At least one of the month's mean is different.

		- Results:
			| | ANOVA for the average monthly energy usage  | |
			| --- 			| --- 			| --- 			|
			|				|	p-value		| 	F-statistic	|
			| SJ (95110)	| 1.846e-28 	|	27.56 		|
			| SF (94102)	| 1.190e-32		|	35.85		|

		- Interpretation:
			- The p-values for each region are extremely small. Because p < 0.05, we reject the null hypothesis. The observed differences between the groups (months) are very unlikely to have occurred by random chance. There is a significant difference between at least one pair of group means.
			- This result suggests that there might be a relationship between energy consumption and time of year. It can help direct us in understanding how energy usage is affected by external factors. Next to look at is how region and weather affect energy consumption.

	- Multi-way ANOVA:
		- [GeeksforGeeks two-way ANOVA](https://www.geeksforgeeks.org/how-to-perform-a-two-way-anova-in-python/)
		- Merged the two SJ and SF datasets
		- Null Hypothesis: The average energy usage across the months, region, and max temperature is the same.
		- Alternative Hypothesis: At least one of the month's mean is different.

		- Results:
			|     				| p-value 		| F-statistic 	|
			| --- 				| --- 			| --- 			| 
			| Month 			| 4.201e-57 	| 53.071 		|
			| Region 			| 2.566e-115 	| 1968.4 		|
			| Temp (max) 		| 6.242e-02 	| 3.5056 		|
			| Month & Region 	| 2.928e-38 	| 29.532 		|
			| Month & Temp 		| 4.255e-14		| 9.5045 		|
			| Region & Temp 	| 2.086e-3 		| 9.6892 		|

		- Interpretation:
			- All of the resulting p-values are extremely small except for Temp (max). In all other cases besides Temp (max), we reject the null hypothesis because p < 0.05. 
			- These results are highly suggestive that there are relationships between combinations of the month, region, and temperature. Using ANOVA has allowed us to statistically, see, some of the underlying relationships in the data that may exist.
			- There are other weather features that can and will be tested (e.g., Temp (min), precipitation, average daily wind speed, direction, etc.)



## Automation, Scalability, and Portability
<!--- Assume that newer datasets will become available from the same source in future, or you need to ask your colleague to inherit this project. What will be major challenges? List and explain technical and implementational practices you will use to enhance automation, scalability, and portability aspects of the project. -->
- Automation:
	- I will automate the functionality of data preprocessing so that new data can be incorporated into the existing functionality of the model/program.
		- Regularly (hourly) check for and pull data from data.gov APIs or another energy and weather database.
		- Use pipelines to ensure timely and efficient retrieval, processing, and fitting of new data.

- Scalability:
	- The size of the data must be considered as it grows on its own when new data is retrieved. Integrating with the Cloud like AWS will increase the ability to store data.
	- Processing the data quickly is also a factor if data is constantly being retrieved. However, this is also controllable if the interval that the data is that is retrieved is lengthened. For instance, pulling data every minute may require some type of parallel processing setup, whereas pulling it every hour will be less intensive.

- Portability:
	- To meet portability requirements, the project should be easily transferred from one person to another. This should be seamless even if they are on different systems.
		- Using a Docker container will allow the project to be easily reproduced in various system environments. Docker can be used to aggregate the required dependencies, libraries, and system configurations of the project.
		- Include a requirements file for other cases so that an aggregated list of necessary dependencies is available. 



<!--- 
----------
The following sections should be used for the analysis planning. These are not required for the proposal document submission.
----------
-->



## Data Analysis and Algorithms
<!--- List and describe what types of (advanced) analysis you plan to conduct. This section should be tied back to the expected major findings. (If needed, you can update the findings section.) When selecting algorithms to obtain the analysis results, provide a brief explanation of the algorithmic properties and logic. You should clearly define the inputs and outputs of each algorithm. -->

- The advanced analysis in this project will be approached using two primary methods:
	1. Feature Analysis:
		- This analysis will focus on understanding the influence of historical weather data on energy consumption in San Francisco and San Jose. The goal is to identify which weather variables (e.g., temperature, precipitation, wind speed) have a significant impact on energy consumption and whether these impacts differ between the two regions. This will help reveal any disproportionate effects of weather conditions on energy demand in each area.

	2. Time-Series Analysis:
		- To forecast future energy consumption, I will use time-series models such as Long-Short Term Memory (LSTM) networks and Autoregressive Integrated Moving Average (ARIMA). The aim is to create accurate predictions based on historical weather data, which will help energy providers and the public prepare for fluctuations in energy demand.


- Feature Analysis with Random Forests to Analyze Key Weather Variables:
	- Random Forest regressors are ensemble methods that train multiple decision trees and aggregate their results. By combining several base estimators within a given learning algorithm, Random Forests improve the generalizability of a single estimator. This method is particularly suitable for assessing the impact of climate variables (such as maximum and minimum temperature, precipitation, wind speed, etc.) as it captures non-linear relationships, enabling it to model the complex interactions inherent in weather patterns.
	
	- Interpretability with SHapley Additive exPlanations (SHAP):
		- While Random Forest models can be complex to interpret, SHAP will be used to provide explanations of individual features as they contribute to the model's predictions. SHAP values assign each feature an importance score based on its contribution to the prediction, offering a clear way to compare which weather variables are driving energy consumption in each region. I will implement SHAP on a trained Random Forest Regressor model. SHAP results for a single feature will consider all possible combinations of feature coalitions and average the feature's marginal contribution across the coalitions. Essentially, th SHAP value measures how much a feature's value contributes to the difference between the model's predicted output and the average model output.

    - Partial Dependence Plots (PDP):
		- PDPs may also be incorporated to visualize how changes in individual weather variables impact energy consumption, helping to clarify relationships that SHAP scores identify.

	- Regional Comparison:
		- Two models—one for San Francisco and one for San Jose—will be trained. By comparing feature importance between the models, I will analyze whether specific weather variables have a greater impact in one region than in the other. The preliminary focus will be on variables like maximum and minimum temperatures, precipitation, and seasonality. Results will guide further analysis on regional differences.

- Forecasting Energy Demand with Historical Data:
	- Time-Series Analysis:
		- Long-Short Term Memory (LSTM):
			- LSTM networks are designed to handle time-series data by retaining information over time, making them effective for capturing long-term dependencies in weather and energy usage data. I will use LSTM to predict future energy consumption based on historical weather data. The model will initially produce monthly energy predictions, with the possibility of transitioning to daily forecasts if daily energy records become available or if the current data (daily weather records and monthly energy data) is proves to be sufficient in predicting daily energy consumption.

		- Autoregressive Integrated Moving Average (ARIMA):
			- Seasonal ARIMA will be used to test time-series forecasting when dealing with smaller datasets or where simpler modeling is appropriate. ARIMA works by creating a linear equation that describes and forecasts the time-series data. It can provide some baseline predictions for energy consumption.



<!--- 
----------
The following sections should be used for the analysis outcome presentation. These are not required for the analysis plan submission.
----------
-->
# Analysis Outcomes
<!--- Explain the analysis you conducted and show the results. Discuss how the data, your analysis, and/or visualization can support the claims or findings. What will be the recommendations or suggestions you can make based on the results? Use bullet points, tables, and figures (if possible) to increase the readability of the document. -->
- This analysis demonstrates that energy consumption patterns in San Jose  and San Francisco are influenced by distinct weather-related factors, revealing how local climate characteristics can lead to differing energy demands between regions. Using a random forest model paired with SHAP (SHapley Additive exPlanations) - a statistical method that breaks down the impact of each feature on model predictions, I quantified the importance of various factors. SHAP is particularly valuable here because it assigns "importance" scores to features based on their average impact on model predictions which enables a clear assessment of each feature's role.


- In San Jose, seasonality emerged as the most significant predictor of energy consumption, with a mean SHAP value of 20, indicating a strong correlation between energy use and the time of year. Temperature variables followed, with maximum (Tmax) and minimum (Tmin) temperatures ranking second and third, suggesting that while temperature plays a role, seasonality's impact is notably higher. By contrast, San Francisco's energy consumption is most sensitive to temperature extremes. Tmax had the highest mean SHAP value (12), followed by Tmin (10) and the total number of customers (5), with seasonality
showing a relatively minor influence.



<!-- insert shap images> -->
<!-- Ex. ![alt text](/docs/assets/plots/Dual_Ax_95110.png) -->
		- San Jose's SHAP
![alt text](/src_sample/interactiveWebpage/assets/shap_plots/sj_shap.png | width = 75)

		- San Francisco's SHAP
![alt text](/src_sample/interactiveWebpage/assets/shap_plots/sf_shap.png | width = 75)


- To further explore these findings, SHAP decision plots were used to illustrate each feature's contribution to specific predictions. Partial Dependence Plots (PDP) then provided insight into how variations in the top three features impact energy consumption predictions for each region.

<!-- insert pdp images -->
<!-- Ex. ![alt text](/docs/assets/plots/Dual_Ax_95110.png) -->
		- San Jose's PDP's
			| Season PDP | Max Temperature PDP | Min Temperature PDP |
			|:---------------------------------------------------------------------: | :---------------------------------------------------------------------:|
			| ![alt text](/src_sample/interactiveWebpage/assets/pdp_plots/sj_pdp_season.png) | ![alt text](/src_sample/interactiveWebpage/assets/pdp_plots/sf_pdp_tmax.png) | ![alt text](/src_sample/interactiveWebpage/assets/pdp_plots/sf_pdp_tmax.png) |
  
		- San Francisco PDP's
			| Max Temperature PDP | Min Temperature PDP | Total Customers PDP |
			|:---------------------------------------------------------------------: | :---------------------------------------------------------------------:|
			| ![alt text](/src_sample/interactiveWebpage/assets/pdp_plots/sf_pdp_tmax.png) | ![alt text](/src_sample/interactiveWebpage/assets/pdp_plots/sf_pdp_tmax.png) | ![alt text](/src_sample/interactiveWebpage/assets/pdp_plots/sf_pdp_totalcustomers.png) |


- These insights suggest a fundamental difference in climate sensitivity between the two regions: San Francisco's energy demands are more heavily influenced by shifts in temperature, possibly indicating a higher sensitivity to climate variability. Conversely, San Jose's reliance on seasonality hints that while its energy consumption may be less responsive to incremental temperature changes, seasonal cycles play a dominant role in its demand pattern. However, because temperature and seasonality are interdependent [NEED REFERENCE TO BACK THIS CLAIM], it would be naive to conclude that San Jose is less vulnerable to climate changes; further analysis is necessary to determine the relationship between seasonality and global temperature shifts. This analysis suggests that regional energy planning could benefit from tailored approaches that account for these differing sensitivities.


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
