import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from prophet import Prophet

# Main title
st.title("Personal Finance Management System with LSTM and Interactive Features")

# Define sections/pages
sections = [
    "Introduction",
    "Data Visualization",
    "Forecasting",
    "Decision-Making Support",
    "Real-Time Updates",
    "Scenario Planning"
]
selected_section = st.selectbox("Navigate to:", sections)

# Load data only once
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load the data
try:
    data_cleaned = load_data("personal_finance_employees_V1.csv")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# Step 2: Data Visualization
if selected_section == "Data Visualization":
    st.subheader("Data Visualization")

    # Visualization options
    viz_options = [
        "Pie Chart (Expenses Breakdown)",
        "Correlation Matrix (Heatmap)",
        "KMeans Clustering Analysis"
    ]
    selected_viz = st.selectbox("Select Visualization Type:", viz_options)

    if selected_viz == "Pie Chart (Expenses Breakdown)":
        # Pie Chart Visualization
        employees = data_cleaned['Employee'].unique()
        selected_employee = st.selectbox("Select an Employee:", employees)

        try:
            # Calculate categorized expenses
            df_corrdata = {
                'Employee': data_cleaned['Employee'],
                'Bills': data_cleaned['Electricity Bill (£)'] + data_cleaned['Gas Bill (£)'] + data_cleaned['Water Bill (£)'],
                'Entertainment': data_cleaned['Amazon Prime (£)'] + data_cleaned['Netflix (£)'] + data_cleaned['Sky Sports (£)'],
                'Transport': data_cleaned['Transportation (£)'],
                'Savings': data_cleaned['Savings for Property (£)']
            }
            df_corrdata = pd.DataFrame(df_corrdata)

            # Filter for the selected employee
            subframe = df_corrdata[df_corrdata['Employee'] == selected_employee].drop(columns=['Employee'])

            # Prepare data for the pie chart
            expenses = subframe.iloc[0]  # Select the first (and only) row of the subframe
            labels = expenses.index
            sizes = expenses.values

            # Create the pie chart
            fig = go.Figure(
                data=[go.Pie(labels=labels, values=sizes, hole=0.3)],
                layout_title_text=f"Expenses Breakdown for {selected_employee}"
            )
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Pie chart error: {e}")

    elif selected_viz == "Correlation Matrix (Heatmap)":
        # Correlation Matrix Visualization
        try:
            df_corrdata = {
                'Income': data_cleaned['Monthly Income (£)'],
                'Bills': data_cleaned['Electricity Bill (£)'] + data_cleaned['Gas Bill (£)'] + data_cleaned['Water Bill (£)'],
                'Entertainment': data_cleaned['Amazon Prime (£)'] + data_cleaned['Netflix (£)'] + data_cleaned['Sky Sports (£)'],
                'Transport': data_cleaned['Transportation (£)'],
                'Savings': data_cleaned['Savings for Property (£)']
            }
            df_corrdata = pd.DataFrame(df_corrdata)
            df_corrdata.fillna(0, inplace=True)

            # Calculate correlation matrix
            corrMatrix = df_corrdata.corr()

            # Display heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corrMatrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Heatmap error: {e}")

    elif selected_viz == "KMeans Clustering Analysis":
        # KMeans Clustering Visualization
        try:
            # Prepare data for clustering
            data_cleaned.fillna(0, inplace=True)
            df_KmeansData = {
                'Income': data_cleaned['Monthly Income (£)'],
                'Spendings': data_cleaned['Electricity Bill (£)'] + data_cleaned['Gas Bill (£)'] +
                              data_cleaned['Netflix (£)'] + data_cleaned['Amazon Prime (£)'] +
                              data_cleaned['Groceries (£)'] + data_cleaned['Transportation (£)'] +
                              data_cleaned['Water Bill (£)'] + data_cleaned['Sky Sports (£)'] +
                              data_cleaned['Other Expenses (£)'] + data_cleaned['Savings for Property (£)'] +
                              data_cleaned['Monthly Outing (£)']
            }
            df_KmeansData = pd.DataFrame(df_KmeansData)

            # Elbow Method
            SumSquaresInClusters = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=11)
                kmeans.fit(df_KmeansData)
                SumSquaresInClusters.append(kmeans.inertia_)

            # Apply KMeans with 3 clusters
            kmeans = KMeans(n_clusters=3, init='k-means++', random_state=11)
            y_kmeans = kmeans.fit_predict(df_KmeansData)

            # Plot clustered data
            fig, ax = plt.subplots()
            ax.scatter(df_KmeansData.values[y_kmeans == 0, 0], df_KmeansData.values[y_kmeans == 0, 1], c='red', label='Cluster 1')
            ax.scatter(df_KmeansData.values[y_kmeans == 1, 0], df_KmeansData.values[y_kmeans == 1, 1], c='blue', label='Cluster 2')
            ax.scatter(df_KmeansData.values[y_kmeans == 2, 0], df_KmeansData.values[y_kmeans == 2, 1], c='yellow', label='Cluster 3')
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='grey', label='Centroids')
            ax.set_xlabel('Income')
            ax.set_ylabel('Spendings')
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"KMeans clustering error: {e}")

#step 3: Forecasting
elif selected_section == "Forecasting":
    st.subheader("Step 3: Forecasting with Facebook Prophet")

    try:
        # Select employee
        employees = data_cleaned['Employee'].unique()
        selected_employee = st.selectbox("Select an employee", employees)

        # Filter data for the selected employee
        employee_data = data_cleaned[data_cleaned['Employee'] == selected_employee][['Date', 'Monthly Income (£)']]
        employee_data.rename(columns={"Date": "ds", "Monthly Income (£)": "y"}, inplace=True)

        # Forecasting using Prophet
        model = Prophet()
        model.fit(employee_data)

        # Create future dataframe
        future = model.make_future_dataframe(periods=1, freq='M')

        # Forecast
        forecast = model.predict(future)

        # Plot forecast
        st.write(f"### Monthly Income Forecast for {selected_employee}")
        fig = model.plot(forecast)
        st.pyplot(fig)

        # Display forecast components
        st.write("### Forecast Components")
        fig_components = model.plot_components(forecast)
        st.pyplot(fig_components)
    except Exception as e:
        st.error(f"Prophet forecasting error: {e}")

# Step 4: Decision-Making Support
elif selected_section == "Decision-Making Support":
    st.subheader("Step 4: Decision-Making Support")
    try:
        current_savings = data_cleaned['Savings for Property (£)'].iloc[-1]
        interval = st.selectbox("Select Interval:", ["Daily", "Weekly", "Monthly"])
        savings_goal = st.number_input("Set Your Savings Target (£):", min_value=0.0, step=100.0)

        if current_savings < savings_goal:
            st.warning(f"You need to save an additional £{savings_goal - current_savings:.2f} to meet your target.")
        else:
            st.success("Congratulations! You have met your savings goal.")
    except Exception as e:
        st.error(f"Decision-making support error: {e}")

# Step 5: Real-Time Updates
elif selected_section == "Real-Time Updates":
    st.subheader("Step 5: Interactivity and Real-Time Updates")
    try:
        real_time_savings = st.slider("Adjust Current Savings (£):", min_value=0, max_value=int(current_savings + 5000), value=int(current_savings))
        updated_goal_status = "Met" if real_time_savings >= savings_goal else "Not Met"
        st.write(f"Updated Goal Status: {updated_goal_status}")
    except Exception as e:
        st.error(f"Real-time updates error: {e}")

# Step 6: Scenario Planning and Forecasting
elif selected_section == "Scenario Planning":
    st.subheader("Step 6: Scenario Planning and Forecasting")
    try:
        scenario_increase = st.number_input("Increase Savings by (%):", min_value=0, max_value=100, step=5)
        forecasted_savings = real_time_savings * (1 + scenario_increase / 100)
        st.write(f"If you increase savings by {scenario_increase}%, your forecasted savings will be £{forecasted_savings:.2f}.")
    except Exception as e:
        st.error(f"Scenario planning error: {e}")

# Default: Introduction
else:
    st.write("Welcome to the Personal Finance Management System. Use the dropdown menu to navigate through the application.")
