# Attrition-Analysis-and-Prediction
This project aims to provide insights into the factors influencing employee attrition and predict which employees are likely to leave the company. Let's refine the project to make it more closely aligned with real-time scenarios and address live problem statements within an organization.


# Python Data Cleaning Process:
Data Loading: I started by loading the company attribution dataset into Python using pandas. import pandas as pd

Load the Data Set data = pd.read_csv('company_attribution_data.csv')

Data Cleaning: I performed various data cleaning tasks to ensure the dataset is ready for analysis. This included check missing values, check duplicates, and check null values.

Check for missing values missing_values = data.isnull().sum() print("Missing Values:") print(missing_values)

Check for duplicate values duplicate_rows = data.duplicated().sum() print("\nDuplicate Rows:") print(duplicate_rows)

Data Transformation: I conducted data transformation tasks such as creating new columns, converting data types, and aggregating data as needed.

Create a new column for age group data['age_group'] = pd.cut(data['age'], bins=[20, 30, 40, 50, 60, 70], labels=['20-30', '31-40', '41-50', '51-60', '61-70'])

Convert categorical variables to appropriate data types data['gender'] = data['gender'].astype('category') data['department'] = data['department'].astype('category')

Data Analysis: I conducted exploratory data analysis to gain insights into the dataset and identify trends or patterns.

Calculate average age average_age = data['age'].mean()

Calculate percentage of attributed employees percentage_attributed = (data['attributed'].sum() / data.shape[0]) * 100


# Tableau Dashboard Creation:
Data Connection: I connected Tableau to the cleaned dataset and imported the necessary tables.

Chart Creation:

Total Employees Overview: Created a bar chart to visualize the total number of employees by gender, number of attributed employees, percentage of attributed employees, and average age.

Attribution Based on Gender: Developed a pie chart to show the distribution of attributed employees by gender.

Department-wise Attrition: Constructed a stacked bar chart to display attrition rates across different departments.

Age Group Analysis: Designed a histogram to illustrate the distribution of employees by age group.

Job Satisfaction by Job Role: Created a scatter plot to visualize job satisfaction ratings based on job roles.

Education Field-wise Attrition: Developed a grouped bar chart to illustrate attrition rates by education field.

Attribution Rate by Gender for Different Age Groups: Designed a heatmap to visualize attribution rates by gender across different age groups.

Dashboard Summary:

The interactive dashboard provides comprehensive insights into company attribution data. It allows users to explore various aspects such as gender distribution, department-wise attrition, age demographics, job satisfaction, and education field-wise attrition. Users can interact with the dashboard to gain deeper insights and identify trends or patterns. Overall, the dashboard facilitates data-driven decision-making processes for workforce management and strategic planning.
