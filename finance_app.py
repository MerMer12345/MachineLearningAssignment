import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import xlabel, ylabel
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Main title
st.title("Personal Finance Management System with Forecasting and Interactive Features")

# Load data only once
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load the data
try:
    data_cleaned = load_data("personal_finance_employees_More_Months.csv")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Select employee at the beginning
employees = data_cleaned['Employee'].unique()
selected_employee = st.sidebar.selectbox("Select an Employee:", employees)

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

# Step 1: Introduction
if selected_section == "Introduction":
    st.write("Welcome to the Personal Finance Management System.")
    st.write(f"Currently analyzing data for {selected_employee}.")

# Step 2: Data Visualization
elif selected_section == "Data Visualization":
    st.subheader("Data Visualization")

    # Visualization options
    viz_options = [
        "Pie Chart (Expenses Breakdown)",
        "Correlation Matrix (Heatmap)",
        "KMeans Clustering Analysis"
    ]
    selected_viz = st.selectbox("Select Visualization Type:", viz_options)

    if selected_viz == "Pie Chart (Expenses Breakdown)":
        try:
            # Date Selection
            dates = data_cleaned[data_cleaned['Employee'] == selected_employee]['Date'].unique()
            selected_date = st.selectbox("Select a Date:", dates)

            # Calculate categorized expenses
            df_corrdata = {
                'Employee': data_cleaned['Employee'],
                'Date': data_cleaned['Date'],
                'Bills': data_cleaned['Electricity Bill (£)'] + data_cleaned['Gas Bill (£)'] + data_cleaned['Water Bill (£)'],
                'Entertainment': data_cleaned['Amazon Prime (£)'] + data_cleaned['Netflix (£)'] + data_cleaned['Sky Sports (£)'],
                'Transport': data_cleaned['Transportation (£)'],
                'Savings': data_cleaned['Savings for Property (£)']
            }
            df_corrdata = pd.DataFrame(df_corrdata)

            # Filter for the selected employee and date
            subframe = df_corrdata[
                (df_corrdata['Employee'] == selected_employee) &
                (df_corrdata['Date'] == selected_date)
                ].drop(columns=['Employee', 'Date'])

            # Check if the subframe has data
            if not subframe.empty:
                # Prepare data for the pie chart
                expenses = subframe.iloc[0]  # Select the first (and only) row of the subframe
                labels = expenses.index
                sizes = expenses.values

                # Create the pie chart
                fig = go.Figure(
                    data=[go.Pie(labels=labels, values=sizes, hole=0.3)],
                    layout_title_text=f"Expenses Breakdown for {selected_employee} on {selected_date}"
                )
                st.plotly_chart(fig)
            else:
                st.warning("No data available for the selected employee and date.")

        except Exception as e:
            st.error(f"Pie chart error: {e}")

    elif selected_viz == "Correlation Matrix (Heatmap)":
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

        try:

            # Prepare data for clustering

            data_cleaned.fillna(0, inplace=True)

            df_kmeans_data = pd.DataFrame({

                'Income': data_cleaned['Monthly Income (£)'],

                'Spendings': (

                        data_cleaned['Electricity Bill (£)'] +

                        data_cleaned['Gas Bill (£)'] +

                        data_cleaned['Netflix (£)'] +

                        data_cleaned['Amazon Prime (£)'] +

                        data_cleaned['Groceries (£)'] +

                        data_cleaned['Transportation (£)'] +

                        data_cleaned['Water Bill (£)'] +

                        data_cleaned['Sky Sports (£)'] +

                        data_cleaned['Other Expenses (£)'] +

                        data_cleaned['Savings for Property (£)'] +

                        data_cleaned['Monthly Outing (£)']

                )

            })


            # Elbow Method for optimal number of clusters

            def calculate_elbow(data, max_clusters=10):

                inertia_values = []

                for i in range(1, max_clusters + 1):
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=11)

                    kmeans.fit(data)

                    inertia_values.append(kmeans.inertia_)

                return inertia_values


            sum_squares_in_clusters = calculate_elbow(df_kmeans_data)

            # Plot Elbow Curve

            fig_elbow, ax_elbow = plt.subplots()

            ax_elbow.plot(range(1, 11), sum_squares_in_clusters, marker='o', linestyle='--')

            ax_elbow.set_title('Elbow Method for Optimal Clusters')

            ax_elbow.set_xlabel('Number of Clusters')

            ax_elbow.set_ylabel('Inertia')

            st.pyplot(fig_elbow)

            # Apply KMeans with 3 clusters

            kmeans = KMeans(n_clusters=3, init='k-means++', random_state=11)

            y_kmeans = kmeans.fit_predict(df_kmeans_data)

            # Plot clustered data

            fig_clusters, ax_clusters = plt.subplots()

            colors = ['red', 'blue', 'yellow']

            for cluster in range(3):
                ax_clusters.scatter(

                    df_kmeans_data.values[y_kmeans == cluster, 0],

                    df_kmeans_data.values[y_kmeans == cluster, 1],

                    c=colors[cluster],

                    label=f'Cluster {cluster + 1}'

                )

            ax_clusters.scatter(

                kmeans.cluster_centers_[:, 0],

                kmeans.cluster_centers_[:, 1],

                s=200, c='grey', marker='X', label='Centroids'

            )

            ax_clusters.set_xlabel('Income (£)')

            ax_clusters.set_ylabel('Spendings (£)')

            ax_clusters.set_title('KMeans Clustering Results')

            ax_clusters.legend()

            st.pyplot(fig_clusters)


        except KeyError as ke:

            st.error(f"Missing data columns for clustering: {ke}")

        except ValueError as ve:

            st.error(f"Value error in clustering process: {ve}")

        except Exception as e:

            st.error(f"An unexpected error occurred: {e}")

# Step 3: Forecasting
elif selected_section == "Forecasting":
    st.subheader("Forecasting with Multiple Models")

    try:
        # Filter data for the selected employee
        employee_data = data_cleaned[data_cleaned['Employee'] == selected_employee][['Date', 'Monthly Income (£)']]
        employee_data['Date'] = pd.to_datetime(employee_data['Date'])
        employee_data.rename(columns={"Date": "ds", "Monthly Income (£)": "y"}, inplace=True)

        # Model selection
        models = ["Facebook Prophet", "ARIMA", "SARIMAX"]
        selected_model = st.selectbox("Select a forecasting model", models)

        if selected_model == "Facebook Prophet":
            model = Prophet()
            model.fit(employee_data)

            future = model.make_future_dataframe(periods=1, freq='Y')  # Forecasting for the next year
            forecast = model.predict(future)

            st.write(f"### Monthly Income Forecast for {selected_employee} using Prophet")
            fig = model.plot(forecast)
            st.pyplot(fig)

        elif selected_model == "ARIMA":
            employee_data.set_index('ds', inplace=True)

            model = ARIMA(employee_data['y'], order=(1, 1, 1))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=12)
            st.write(f"### Monthly Income Forecast for {selected_employee} using ARIMA")
            st.line_chart(forecast)

        elif selected_model == "SARIMAX":
            employee_data.set_index('ds', inplace=True)

            model = SARIMAX(employee_data['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model_fit = model.fit()

            forecast = model_fit.get_forecast(steps=12)
            forecast_df = forecast.conf_int()
            forecast_df["Forecast"] = forecast.predicted_mean

            st.write(f"### Monthly Income Forecast for {selected_employee} using SARIMAX")
            st.line_chart(forecast_df["Forecast"])

    except Exception as e:
        st.error(f"Forecasting error: {e}")

# Step 4: Decision-Making Support
elif selected_section == "Decision-Making Support":
    st.subheader("Decision-Making Support")

    try:
        employee_data = data_cleaned[data_cleaned['Employee'] == selected_employee]

        if employee_data.empty:
            st.warning("No data available for the selected employee.")
        else:
            current_savings = employee_data['Savings for Property (£)'].loc[employee_data['Date'].idxmax()]
            st.info(f"You currently have £{current_savings} saved.")

            interval = st.selectbox("Select Interval:", ["Daily", "Weekly", "Monthly"])
            savings_goal = st.number_input("Set Your Savings Target (£):", min_value=0.0, step=100.0)

            if current_savings < savings_goal:
                shortfall = savings_goal - current_savings
                st.warning(f"You need to save an additional £{shortfall:.2f} to meet your target.")

                if interval == "Daily":
                    st.info(f"This means saving approximately £{shortfall / 30:.2f} per day.")
                elif interval == "Weekly":
                    st.info(f"This means saving approximately £{shortfall / 4:.2f} per week.")
                elif interval == "Monthly":
                    st.info(f"This means saving approximately £{shortfall:.2f} this month.")
            else:
                st.success("Congratulations! You have met your savings goal.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Step 5: Real-Time Updates
elif selected_section == "Real-Time Updates":
    st.subheader("Real-Time Updates")
    try:
        real_time_savings = st.slider("Adjust Current Savings (£):", min_value=0, max_value=int(current_savings + 5000), value=int(current_savings))
        updated_goal_status = "Met" if real_time_savings >= savings_goal else "Not Met"
        st.write(f"Updated Goal Status: {updated_goal_status}")
    except Exception as e:
        st.error(f"Real-time updates error: {e}")

# Step 6: Scenario Planning and Forecasting
elif selected_section == "Scenario Planning":
    st.subheader("Scenario Planning and Forecasting")
    try:
        scenario_increase = st.number_input("Increase Savings by (%):", min_value=0, max_value=100, step=5)
        forecasted_savings = real_time_savings * (1 + scenario_increase / 100)
        st.write(f"If you increase savings by {scenario_increase}%, your forecasted savings will be £{forecasted_savings:.2f}.")
    except Exception as e:
        st.error(f"Scenario planning error: {e}")
