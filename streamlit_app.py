import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()

st.title("🔮 Employee Attrition Prediction")
st.markdown("---")

# Create form columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    business_travel = st.selectbox("Business Travel", ['Rarely', 'Frequently', 'Non-Travel'])
    daily_rate = st.number_input("Daily Rate", min_value=0, max_value=2000, value=500)
    department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
    distance_from_home = st.number_input("Distance From Home (km)", min_value=0, max_value=50, value=5)
    education = st.selectbox("Education", [1, 2, 3, 4, 5])
    education_field = st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 2)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    hourly_rate = st.number_input("Hourly Rate", min_value=0, max_value=200, value=65)

with col2:
    job_involvement = st.slider("Job Involvement", 1, 4, 2)
    job_level = st.slider("Job Level", 1, 4, 1)
    job_role = st.selectbox("Job Role", ['Sales Representative', 'Laboratory Technician', 'Sales Manager', 'Research Scientist', 'Manager', 'Research Director', 'Healthcare Representative', 'Technician', 'Human Resources'])
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)
    marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
    monthly_income = st.number_input("Monthly Income", min_value=0, max_value=20000, value=5000)
    num_companies_worked = st.number_input("Number of Companies Worked in", min_value=0, max_value=10, value=2)
    over_time = st.selectbox("Over Time", ['Yes', 'No'])
    performance_rating = st.slider("Performance Rating", 1, 4, 3)
    relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 2)

col3, col4 = st.columns(2)

with col3:
    stock_option_level = st.slider("Stock Option Level", 0, 3, 0)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=5)
    training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
    work_life_balance = st.slider("Work Life Balance", 1, 4, 2)

with col4:
    years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=2)
    years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=40, value=1)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=40, value=0)
    years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=40, value=1)

# Create prediction data
if st.button("🚀 Predict Attrition", use_container_width=True):
    data_dict = {
        'Age': int(age),
        'BusinessTravel': str(business_travel),
        'DailyRate': int(daily_rate),
        'Department': department,
        'DistanceFromHome': int(distance_from_home),
        'Education': education,
        'EducationField': str(education_field),
        'EnvironmentSatisfaction': int(environment_satisfaction),
        'Gender': str(gender),
        'HourlyRate': int(hourly_rate),
        'JobInvolvement': int(job_involvement),
        'JobLevel': int(job_level),
        'JobRole': job_role,
        'JobSatisfaction': int(job_satisfaction),
        'MaritalStatus': str(marital_status),
        'MonthlyIncome': int(monthly_income),
        'NumCompaniesWorked': int(num_companies_worked),
        'OverTime': str(over_time),
        'PerformanceRating': int(performance_rating),
        'RelationshipSatisfaction': int(relationship_satisfaction),
        'StockOptionLevel': int(stock_option_level),
        'TotalWorkingYears': int(total_working_years),
        'TrainingTimesLastYear': int(training_times_last_year),
        'WorkLifeBalance': int(work_life_balance),
        'YearsAtCompany': int(years_at_company),
        'YearsInCurrentRole': int(years_in_current_role),
        'YearsSinceLastPromotion': int(years_since_last_promotion),
        'YearsWithCurrManager': int(years_with_curr_manager)
    }

    df = pd.DataFrame([data_dict])

    # Feature engineering (same as Flask app)
    df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] +
                                df['JobInvolvement'] +
                                df['JobSatisfaction'] +
                                df['RelationshipSatisfaction'] +
                                df['WorkLifeBalance']) / 5

    df.drop(['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'], axis=1, inplace=True)
    df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply(lambda x: 1 if x >= 2.8 else 0)
    df.drop('Total_Satisfaction', axis=1, inplace=True)

    df['Age_bool'] = df['Age'].apply(lambda x: 1 if x < 35 else 0)
    df.drop('Age', axis=1, inplace=True)

    df['DailyRate_bool'] = df['DailyRate'].apply(lambda x: 1 if x < 800 else 0)
    df.drop('DailyRate', axis=1, inplace=True)

    df['Department_bool'] = df['Department'].apply(lambda x: 1 if x == 'Research & Development' else 0)
    df.drop('Department', axis=1, inplace=True)

    df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x: 1 if x > 10 else 0)
    df.drop('DistanceFromHome', axis=1, inplace=True)

    df['JobRole_bool'] = df['JobRole'].apply(lambda x: 1 if x == 'Laboratory Technician' else 0)
    df.drop('JobRole', axis=1, inplace=True)

    df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x: 1 if x < 65 else 0)
    df.drop('HourlyRate', axis=1, inplace=True)

    df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply(lambda x: 1 if x < 4000 else 0)
    df.drop('MonthlyIncome', axis=1, inplace=True)

    df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply(lambda x: 1 if x > 3 else 0)
    df.drop('NumCompaniesWorked', axis=1, inplace=True)

    df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply(lambda x: 1 if x < 8 else 0)
    df.drop('TotalWorkingYears', axis=1, inplace=True)

    df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply(lambda x: 1 if x < 3 else 0)
    df.drop('YearsAtCompany', axis=1, inplace=True)

    df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply(lambda x: 1 if x < 3 else 0)
    df.drop('YearsInCurrentRole', axis=1, inplace=True)

    df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply(lambda x: 1 if x < 1 else 0)
    df.drop('YearsSinceLastPromotion', axis=1, inplace=True)

    df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply(lambda x: 1 if x < 1 else 0)
    df.drop('YearsWithCurrManager', axis=1, inplace=True)

    # Categorical conversions
    if business_travel == 'Rarely':
        df['BusinessTravel_Rarely'] = 1
        df['BusinessTravel_Frequently'] = 0
        df['BusinessTravel_No_Travel'] = 0
    elif business_travel == 'Frequently':
        df['BusinessTravel_Rarely'] = 0
        df['BusinessTravel_Frequently'] = 1
        df['BusinessTravel_No_Travel'] = 0
    else:
        df['BusinessTravel_Rarely'] = 0
        df['BusinessTravel_Frequently'] = 0
        df['BusinessTravel_No_Travel'] = 1
    df.drop('BusinessTravel', axis=1, inplace=True)

    if education_field == 'Life Sciences':
        df['EducationField_Life_Sciences'] = 1
        df['EducationField_Medical'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Technical_Degree'] = 0
        df['Education_Human_Resources'] = 0
        df['Education_Other'] = 0
    elif education_field == 'Medical':
        df['EducationField_Life_Sciences'] = 0
        df['EducationField_Medical'] = 1
        df['EducationField_Marketing'] = 0
        df['EducationField_Technical_Degree'] = 0
        df['Education_Human_Resources'] = 0
        df['Education_Other'] = 0
    elif education_field == 'Marketing':
        df['EducationField_Life_Sciences'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Marketing'] = 1
        df['EducationField_Technical_Degree'] = 0
        df['Education_Human_Resources'] = 0
        df['Education_Other'] = 0
    elif education_field == 'Technical Degree':
        df['EducationField_Life_Sciences'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Technical_Degree'] = 1
        df['Education_Human_Resources'] = 0
        df['Education_Other'] = 0
    elif education_field == 'Human Resources':
        df['EducationField_Life_Sciences'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Technical_Degree'] = 0
        df['Education_Human_Resources'] = 1
        df['Education_Other'] = 0
    else:
        df['EducationField_Life_Sciences'] = 0
        df['EducationField_Medical'] = 0
        df['EducationField_Marketing'] = 0
        df['EducationField_Technical_Degree'] = 0
        df['Education_Human_Resources'] = 1
        df['Education_Other'] = 1
    df.drop('EducationField', axis=1, inplace=True)

    if gender == 'Male':
        df['Gender_Male'] = 1
        df['Gender_Female'] = 0
    else:
        df['Gender_Male'] = 0
        df['Gender_Female'] = 1
    df.drop('Gender', axis=1, inplace=True)

    if marital_status == 'Married':
        df['MaritalStatus_Married'] = 1
        df['MaritalStatus_Single'] = 0
        df['MaritalStatus_Divorced'] = 0
    elif marital_status == 'Single':
        df['MaritalStatus_Married'] = 0
        df['MaritalStatus_Single'] = 1
        df['MaritalStatus_Divorced'] = 0
    else:
        df['MaritalStatus_Married'] = 0
        df['MaritalStatus_Single'] = 0
        df['MaritalStatus_Divorced'] = 1
    df.drop('MaritalStatus', axis=1, inplace=True)

    if over_time == 'Yes':
        df['OverTime_Yes'] = 1
        df['OverTime_No'] = 0
    else:
        df['OverTime_Yes'] = 0
        df['OverTime_No'] = 1
    df.drop('OverTime', axis=1, inplace=True)

    if stock_option_level == 0:
        df['StockOptionLevel_0'] = 1
        df['StockOptionLevel_1'] = 0
        df['StockOptionLevel_2'] = 0
        df['StockOptionLevel_3'] = 0
    elif stock_option_level == 1:
        df['StockOptionLevel_0'] = 0
        df['StockOptionLevel_1'] = 1
        df['StockOptionLevel_2'] = 0
        df['StockOptionLevel_3'] = 0
    elif stock_option_level == 2:
        df['StockOptionLevel_0'] = 0
        df['StockOptionLevel_1'] = 0
        df['StockOptionLevel_2'] = 1
        df['StockOptionLevel_3'] = 0
    else:
        df['StockOptionLevel_0'] = 0
        df['StockOptionLevel_1'] = 0
        df['StockOptionLevel_2'] = 0
        df['StockOptionLevel_3'] = 1
    df.drop('StockOptionLevel', axis=1, inplace=True)

    if training_times_last_year == 0:
        df['TrainingTimesLastYear_0'] = 1
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 0
    elif training_times_last_year == 1:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 1
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 0
    elif training_times_last_year == 2:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 1
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 0
    elif training_times_last_year == 3:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 1
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 0
    elif training_times_last_year == 4:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 1
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 0
    elif training_times_last_year == 5:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 1
        df['TrainingTimesLastYear_6'] = 0
    else:
        df['TrainingTimesLastYear_0'] = 0
        df['TrainingTimesLastYear_1'] = 0
        df['TrainingTimesLastYear_2'] = 0
        df['TrainingTimesLastYear_3'] = 0
        df['TrainingTimesLastYear_4'] = 0
        df['TrainingTimesLastYear_5'] = 0
        df['TrainingTimesLastYear_6'] = 1
    df.drop('TrainingTimesLastYear', axis=1, inplace=True)

    # Make prediction
    prediction = model.predict(df)

    st.markdown("---")
    if prediction == 0:
        st.success("✅ Employee Might NOT Leave The Job", icon="✅")
    else:
        st.error("⚠️ Employee Might LEAVE The Job", icon="⚠️")
