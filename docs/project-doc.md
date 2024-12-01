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

		- Seasonal Autoregressive Integrated Moving Average (SARIMA):
			- Seasonal ARIMA will be used to test time-series forecasting when dealing with smaller datasets or where simpler modeling is appropriate. ARIMA works by creating a linear equation that describes and forecasts the time-series data. It can provide some baseline predictions for energy consumption.



<!--- 
----------
The following sections should be used for the analysis outcome presentation. These are not required for the analysis plan submission.
----------
-->
# Analysis Outcomes
<!--- Explain the analysis you conducted and show the results. Discuss how the data, your analysis, and/or visualization can support the claims or findings. What will be the recommendations or suggestions you can make based on the results? Use bullet points, tables, and figures (if possible) to increase the readability of the document. -->
- Analyzing Key Weather Variables
	- This analysis demonstrates that energy consumption patterns in San Jose  and San Francisco are influenced by distinct weather-related factors, revealing how local climate characteristics can lead to differing energy demands between regions. Using a random forest model paired with SHAP (SHapley Additive exPlanations) - a statistical method that breaks down the impact of each feature on model predictions, I quantified the importance of various factors. SHAP is particularly valuable here because it assigns "importance" scores to features based on their average impact on model predictions which enables a clear assessment of each feature's role.


	- In San Jose, seasonality emerged as the most significant predictor of energy consumption, with a mean SHAP value of 20, indicating a strong correlation between energy use and the time of year. Temperature variables followed, with maximum (Tmax) and minimum (Tmin) temperatures ranking second and third, suggesting that while temperature plays a role, seasonality's impact is notably higher. By contrast, San Francisco's energy consumption is most sensitive to temperature extremes. Tmax had the highest mean SHAP value (12), followed by Tmin (10) and the total number of customers (5), with seasonality
	showing a relatively minor influence.

	- Regional SHAP

		<img src="/src_sample/interactiveWebpage/assets/shap_plots/shap_dot_plot.png" width = "75%">

	- To further explore these findings, SHAP decision plots were used to illustrate each feature's contribution to specific predictions. Partial Dependence Plots (PDP) then provided insight into how variations in the top three features impact energy consumption predictions for each region.
		- Regional SHAP Decision Plots

			| San Jose | San Francisco |
			|:---------------------------------------------------------------------: | :---------------------------------------------------------------------:|
			| <img src="/src_sample/interactiveWebpage/assets/shap_plots/sj_shap.png"> | <img src="/src_sample/interactiveWebpage/assets/shap_plots/sf_shap.png"> |


		- San Jose's PDP's - 3 Most Impactful Features
			| Season | Temperature (Max) | Temperature (Min) |
			|:---------------------------------------------------------------------: | :---------------------------------------------------------------------:| :---------------------------------------------------------------------:|
			| <img src="/src_sample/interactiveWebpage/assets/pdp_plots/sj_pdp_season.png"> | <img src="/src_sample/interactiveWebpage/assets/pdp_plots/sf_pdp_tmax.png"> | <img src="/src_sample/interactiveWebpage/assets/pdp_plots/sf_pdp_tmin.png"> |

		
		- San Francisco PDP's - 3 Most Impactful Features
			| Temperature (Max) | Temperature (Min) | Total Customers |
			|:---------------------------------------------------------------------: | :---------------------------------------------------------------------:| :---------------------------------------------------------------------:|
			| <img src="/src_sample/interactiveWebpage/assets/pdp_plots/sf_pdp_tmax.png"> | <img src="/src_sample/interactiveWebpage/assets/pdp_plots/sf_pdp_tmin.png"> | <img src="/src_sample/interactiveWebpage/assets/pdp_plots/sf_pdp_totalcustomers.png"> |

	- These insights tell us there is a fundamental difference in climate sensitivity between the two regions: San Francisco's energy demands are 
	more heavily influenced by shifts in temperature, possibly indicating a higher sensitivity to climate variability. 
	Conversely, San Jose's reliance on seasonality hints that while its energy consumption is less responsive to incremental temperature changes, 
	seasonal cycles play a dominant role in energy usage predictions. However, because temperature and seasonality are interdependent [NEED REFERENCE TO BACK THIS CLAIM], 
	it would be naive to conclude that San Jose is less vulnerable to shifts in climate change; further analysis is necessary to determine the 
	relationship between seasonality and global temperature shifts.

- Forecasting Energy Usage:
	- Two modeling approaches were used to predict energy consumption for each region: LSTM models and SARIMA models.
		
		- LSTM Models:
			- The LSTM models provided both "single-step" and "multi-step" predictions.
				- Single-Step Prediction:
					The single-step LSTM model forecasted energy consumption one month ahead for each region. It was trained using data 
					from the previous 12-month data window to account for seasonal trends. The performance was evaluated using Mean Absolute Error (MAE). 

				- Multi-Step Prediction:
					- The multi-step LSTM model was also trained on the preceding 12 months of data but it predicted energy consumption for the next 12 months.
					Similarly to the single-step LSTM model, performance was measured with MAE. San Jose's test MAE was 0.723, and San Francisco's test MAE was 0.750.

			- Single-Step Predictions
				| San Jose                                                                             | San Francisco                                                                        |
				|:-----------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------: |
				| <img src="/src_sample/interactiveWebpage/assets/lstm_plots/sj_lstm_single_step.png"> | <img src="/src_sample/interactiveWebpage/assets/lstm_plots/sf_lstm_single_step.png"> |
				
			- Evaluation Metrics:
				| Location      | Train MAE | Validation MAE | Test MAE |
				| :-----------: | :-------: | :------------: | :------: |
				| San Jose      | 10.081    | 25.707         | 35.506   |
				| San Francisco | 9.049     | 26.773         | 32.527   |

			- Multi-Step Predictions
				| San Jose                                                                            | San Francisco                                                                       |
				| :---------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------: |
				| <img src="/src_sample/interactiveWebpage/assets/lstm_plots/sj_lstm_multi_step.png"> | <img src="/src_sample/interactiveWebpage/assets/lstm_plots/sf_lstm_multi_step.png"> |

			- Evaluation Metrics:
				| Location      | Train MAE | Validation MAE | Test MAE |
				| :-----------: | :-------: | :------------: | :------: |
				| San Jose      | 11.094    | 29.911         | 34.985   |
				| San Francisco | 4.749     | 18.827         | 22.524   |

		- SARIMA Models:
			- SARIMA modeling was chosen because of the small size of the dataset. The models were trained on the entire dataset, excluding the final 12 months 
			for each region, which served as the 12-month prediction window. As with the LSTM models, MAE was used to assess performance. 
			The SARIMA model achieved an MAE of 22.895 for San Jose and 13.082 for San Francisco.

			- SARIMA 12-Month Predictions
				| San Jose                                                                     | San Francisco                                                                |
				| :--------------------------------------------------------------------------: | :--------------------------------------------------------------------------: |
				| <img src="/src_sample/interactiveWebpage/assets/sarima_plots/sj_sarima.png"> | <img src="/src_sample/interactiveWebpage/assets/sarima_plots/sf_sarima.png"> |

			- Evaluation Metrics
				| Location      | MAE    |
				| :-----------: | :----: |
				| San Jose      | 22.895 |
				| San Francisco | 13.082 |

	- Evaluating Model Performance
		- To interpret the mean absolute error (MAE) scores for each model and region, it’s important to consider that the models are based on energy usage data from two zip codes in the San Jose and San Francisco regions. The average monthly energy usage, recorded in kilowatt-hours (kWh), is specific to each household within those zip codes. The MAE indicates the average difference between the model's predicted energy usage and the actual value per household. For instance, a single-step LSTM model with a training MAE of 10.081 for San Jose means that the model’s predictions are, on average, off by 10 kWh per month per household.

		- To provide context, a kilowatt-hour represents the use of one kilowatt of power for one hour. For example, running a 100-watt light bulb for 10 hours or using a 50-watt LED TV for 20 hours both equate to one kWh of energy.

		- Overall, the LSTM models show significant overfitting, as evidenced by the lower training MAE scores compared to their test MAE scores. In both regions, the training MAEs for the single-step and multi-step LSTM models are much lower than the validation and test MAE. The validation and test MAE's, for both models and both regions, are in the 30 kWh range, except for San Francisco multi-step model, where the test MAE is in the low 20s.

		- Both regions have a dedicated LSTM and SARIMA model. When model performance is compared, we are comparing the SARIMA models with the multi-step LSTM models. The results are that the SARIMA models outperformed the LSTM multi-step models. Specifically, the SARIMA MAE scores were 22.895 for San Jose and 13.082 for San Francisco, while the LSTM multi-step models had test  MAE scores of 34.985 and 22.524, respectively.
		
		- These results align with expectations given the nature of each model. Neural network models, the LSTMs for this project, typically require large datasets to accurately learn and generalize complex, nonlinear relationships. The datasets in this project were fairly small, which may have limited the LSTM models’ performance. In contrast, SARIMA models have fewer parameters and are less complex which allow it to effectively learn underlying patterns with fewer data points.

<!--- 
----------
The following sections should be used for the visualization planning. These are not required for the analysis outcome presentation.
----------
-->


# Visualization
## Visualization Plan
<!--- List and explain what types of plots you plan to provide and what you are trying to show through the diagrams. You should explore the best way to visualize the information and message based on the lectures on visualization and related topics. It is required to have at least two interactive graphing and five static plots. -->

- The purpose of my visualizations is to communicate my project's main goals effectively. I aim for anyone visiting my webpage to quickly grasp the core ideas, outcomes, and methodology, regardless of their familiarity with the topic. I'll keep more detailed analysis and statistical methods on a separate page to avoid overwhelming visitors on the home page.

- The home page's primary goal is to capture visitors' attention, visually introduce the project, and spark interest. It opens with a scenic drone view of the San Francisco Bay Area, with a succinct title overlaid and key feature variables related to the project fading in and out below it. The video gradually fades to black at the bottom, hinting for visitors to scroll down. Here, a few initial plots summarize the project's objectives and findings. My goal is to keep these visuals clean and focused, letting the plots speak for themselves, with only essential information displayed.

- Since my project's two main objectives are (1) identifying weather features that most affect energy usage in San Jose and San Francisco and (2) exploring how climate shifts impact each region differently, my visuals will focus on these points.

    - First Plot: A simple horizontal bar graph displays each region's key features impacting energy consumption. A dropdown menu allows users to toggle between San Jose and San Francisco, updating the graph accordingly.

    - Second Plot: A line graph showing extreme weather occurrences in both regions by year, defined as temperatures in the top 90th percentile (tmax) or bottom 10th percentile (tmin). I'm refining this plot, as the current version appears a bit cluttered, making it challenging to interpret.

- Each plot has a concise title and a brief description emphasizing the main takeaway. For example, the bar graph's description notes that San Jose is most sensitive to seasonality, while San Francisco is more affected by temperature extremes. Words like "seasonality" and "temperature extremes" are highlighted to add visual impact and clarity. Similarly, the line graph highlights key phrases to focus visitors on the plot's significance.

- The two plots mentioned previously are aligned in the same row. Under them is another visualization/interactive section. With my project, I want to implore how changes in temperature (max and min) affect each region. The component to do this is still being worked on, but currently I have two dash sliders. One is max temperature and the other is min temperature. They each have 6 tick marks. The user can click on any combination of tick marks and a "prediction of average energy consumption" will be shown for each region. It is titled as "why should I care?" It has a short sentence that explains that changes in average temperatures have real affects on the energy usage. I am working on the best way to phrase this section to really grab the attention of any visitors.

	- The predictions displayed here are pre-calculated using two separate LSTM models trained on regional data. While both models used the same hypothetical input data, the San Jose LSTM also included additional features (e.g., awnd, wsf2, wsf5, wdf2, and wdf5) unavailable in the San Francisco data. Common features such as year, month, and total customers were standardized as 2025, January, and 4000, but I am considering making these values adjustable, though too many sliders may clutter the layout.

	- A more ambitious version of this visualization might use a geo-map for each region, with color-coded energy usage predictions based on the selected temperature values. Higher predictions would show as darker red, and lower ones as darker blue. I'm also debating whether to display model performance metrics here to validate the predictions, though this might fit better on a separate page with model details.

- One final home page visualization — a parallel coordinate plot showing feature interactions in each region — will likely be removed. Although it's interesting to see weather features interact, the plot is currently too complex to read and might distract or disengage visitors.

- Additional visualizations are located on other web pages:

    - The exploratory data analysis section includes bar graphs for each region, illustrating the distribution of significant features like temperature and energy usage, along with interactive heatmaps.

    - The analytical methods page includes bar plots, SHAP bubble plots, SHAP decision plots, partial dependence plots, and line graphs, all of which support the statistical analyses and methods used to investigate the project’s claims visually.

## Web Page Plan
<!--- Explain how many pages you will have in total and what content will be shown in each page. (Each diagram discussed above should be given a proper location in this section. Also, it is required to have (1) "Project Objective" page, which explains the main goals and data sources, and (2) "Analytical Methods" page, where you explain the major techniques used in the project and provide further references. -->

- My project website will have 4 pages and they are: a Home Page, Project Objective Page, Analytical Methods Page, and a Data Exploration Page. 
For this specific project, an About Me page will not be included, but future work will eventually implement an About Me page.
They will be structured and and contain the types of visuals as follows:

    - Home Page
        - The home page is designed to attract visitor attention and provide a visual summary of the project's goals in a clean, uncluttered layout.

        - This page will have 1 looping home page video and 3 key visuals:
			- The looping video is drone footage of the San Francisco city. It tells the web page visitor that the location is "SF or Bay Area".

			- Two bar plots that convey the main objectives and results of the project.
            	- The horizontal bar plot allows the visitor to understand, at a glance, which weather variables are the most important for each region.
				- The vertical bar plot shows the visitor the temperature trends over the past years and how they differ by region.
			
			- An interactive map with temperature sliders.
				- The visitor can change the minimum and maximum temperatures on each slider. The regional colors on the map will change 
				in relation to the difference in energy consumption from the lowest temperatures (Max: 60 °F, Min: 0 °F).
				- This allows the user to see how different temperatures are likely to affect energy usage in each area. Specifically, 
				that each area is affected differently.
				- Below is the actual energy consumption prediction for each region and a note informing visitors what 1 kWh is equivalent to.

    - Project Objective Page
        - This page will introduce the project's main goals, tables summarizing the regional datasets, and the data sources.

        - Visuals on this page will be minimal, but it will include two interactive tables showing regional data. 
		- Each numerical feature in the table will have a distribution plot to help visitors understand the data patterns.

		- The Data Sources section will provide a brief description of who the data was collected from, what the data contains, and a link to each source.

    - Analytical Methods Page
        - This page focuses on the primary analytical techniques in the project - feature analysis, extreme event analysis, and time-series analysis. 
		- The page will offer an in-depth explanation of the methods, with academic-level detail - not just an overview of results.

        - The page will contain various static and interactive visuals and in-depth descriptions of the visualizations and methods used:
            - Interactive: 
				- Feature importance bar plots
				- SHAP bubble plots
				- Extreme events bar plots
				- LSTM single-step and multi-step line graphs
				- SARIMA line graph

			- Static: 
				- SHAP decision plots
				- PDP plots

		- Summary of Analysis section:
			- This section summarizes the methodologies and findings of each of the analytical techniques (the objectives) of the project.


    - Data Exploration Page
        - This page focuses on exploratory data analysis (EDA) to provide a visual understanding of the data and highlight key features related to energy usage and temperature patterns.
        
		- All visuals on this page will be interactive:
			- There are two side-by-side regional energy and weather bar plots.
				- They depict average energy usage and average maximum and minimum temperatures for each region.
				- The user can select different drop down choices to see different regions.

            - Below that are two side-by-side heatmaps. They show similar information as the bar plots (average energy and (max/min) temperature), 
			but display the trends over time.
				- The heatmaps allow users to see how energy usage and temperature have changed over the past 10+ years. 
				Also, the monthly and yearly patterns and differences between regions can be clearly identified.