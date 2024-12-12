from datetime import datetime
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import xlabel, ylabel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import plotly.express as px
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

# Select employee
employees = data_cleaned['Employee'].unique()
selected_employee = st.sidebar.selectbox("Select an Employee:", employees)

# Define sections/pages
sections = [
    "Introduction",
    "Input Expenses",
    "Data Visualization",
    "Forecasting",
    "Decision-Making Support",
    "Real-Time Updates and Scenario Planning"
]
selected_section = st.selectbox("Navigate to:", sections)

#Fill important variables
employee_data = data_cleaned[data_cleaned['Employee'] == selected_employee]
current_savings = employee_data['Savings for Property (£)'].loc[employee_data['Date'].idxmax()]

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
                'Bills': data_cleaned['Electricity Bill (£)'] + data_cleaned['Gas Bill (£)'] + data_cleaned[
                    'Water Bill (£)'],
                'Entertainment': data_cleaned['Amazon Prime (£)'] + data_cleaned['Netflix (£)'] + data_cleaned[
                    'Sky Sports (£)'],
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
                # Prepare data for the pie chart (individual breakdown)
                expenses = subframe.iloc[0]  # Select the first (and only) row of the subframe
                labels = expenses.index
                sizes = expenses.values

                # Create the pie chart for individual breakdown
                fig = go.Figure(
                    data=[go.Pie(labels=labels, values=sizes, hole=0.3)],
                    layout_title_text=f"Expenses Breakdown for {selected_employee} on {selected_date}"
                )
                st.plotly_chart(fig)
            else:
                st.warning("No data available for the selected employee and date.")

            # Calculate mean expenses for all employees
            mean_expenses = df_corrdata.drop(columns=['Employee', 'Date']).mean()

            # Prepare data for the mean pie chart
            mean_labels = mean_expenses.index
            mean_sizes = mean_expenses.values

            # Create the pie chart for mean expenses
            mean_fig = go.Figure(
                data=[go.Pie(labels=mean_labels, values=mean_sizes, hole=0.3)],
                layout_title_text="Mean Expenses Breakdown for All Employees"
            )
            st.plotly_chart(mean_fig)

        except Exception as e:
            st.error(f"Pie chart error: {e}")


    elif selected_viz == "Correlation Matrix (Heatmap)":
        try:
            # Date Selection
            dates = data_cleaned[data_cleaned['Employee'] == selected_employee]['Date'].unique()
            selected_date = st.selectbox("Select a Date:", dates)

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
            st.write(f"Heatmap for {selected_employee}")
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

            # Normalize and scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_kmeans_data)

            # Elbow Method for optimal number of clusters
            def calculate_elbow(data, max_clusters=10):
                inertia_values = []
                for i in range(1, max_clusters + 1):
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=11)
                    kmeans.fit(data)
                    inertia_values.append(kmeans.inertia_)
                return inertia_values

            sum_squares_in_clusters = calculate_elbow(scaled_data)

            # Plot Elbow Curve with Plotly
            optimal_clusters = 3  # This can be dynamically determined based on criteria if desired
            elbow_fig = go.Figure()
            elbow_fig.add_trace(go.Scatter(
                x=list(range(1, 11)),
                y=sum_squares_in_clusters,
                mode='lines+markers',
                name='Inertia'
            ))
            elbow_fig.add_vline(x=optimal_clusters, line_width=2, line_dash="dash", line_color="green")
            elbow_fig.update_layout(
                title="Elbow Method for Optimal Clusters",
                xaxis_title="Number of Clusters",
                yaxis_title="Inertia",
                template="plotly_white"
            )
            st.plotly_chart(elbow_fig)

            # Apply KMeans with the optimal number of clusters
            kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=11)
            y_kmeans = kmeans.fit_predict(scaled_data)

            # Prepare data for visualization
            df_kmeans_data['Cluster'] = y_kmeans
            df_kmeans_data['Cluster'] = df_kmeans_data['Cluster'].astype(str)  # For discrete coloring

            # Add cluster centers to the DataFrame
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            cluster_centers_df = pd.DataFrame(centers, columns=['Income', 'Spendings'])
            cluster_centers_df['Cluster'] = [f'Centroid {i+1}' for i in range(len(centers))]

            # Interactive Scatter Plot with Plotly
            scatter_fig = go.Figure()
            for cluster, color in zip(df_kmeans_data['Cluster'].unique(), px.colors.qualitative.Plotly):
                cluster_data = df_kmeans_data[df_kmeans_data['Cluster'] == cluster]
                scatter_fig.add_trace(go.Scatter(
                    x=cluster_data['Income'],
                    y=cluster_data['Spendings'],
                    mode='markers',
                    marker=dict(color=color),
                    name=f'Cluster {cluster}'
                ))

            # Add centroids to the scatter plot
            for i, row in cluster_centers_df.iterrows():
                scatter_fig.add_trace(go.Scatter(
                    x=[row['Income']],
                    y=[row['Spendings']],
                    mode='markers+text',
                    marker=dict(size=12, color='grey', symbol='x'),
                    name=row['Cluster'],
                    text=[row['Cluster']],
                    textposition='top center'
                ))

            scatter_fig.update_layout(
                title='KMeans Clustering Results',
                xaxis_title='Income',
                yaxis_title='Spendings',
                template='plotly_white'
            )

            st.plotly_chart(scatter_fig)

        except KeyError as ke:
            st.error(f"Missing data columns for clustering: {ke}")

        except ValueError as ve:
            st.error(f"Value error in clustering process: {ve}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")




elif selected_section == "Input Expenses":
    st.title("Employee Expenses Management")

    st.write("## Add New Entry")

    # Form to add a new row
    with st.form(key="add_row_form"):
        monthly_income = st.number_input("Monthly Income (£)", min_value=0.0, step=0.01)
        electricity_bill = st.number_input("Electricity Bill (£)", min_value=0.0, step=0.01)
        gas_bill = st.number_input("Gas Bill (£)", min_value=0.0, step=0.01)
        netflix = st.number_input("Netflix (£)", min_value=0.0, step=0.01)
        amazon_prime = st.number_input("Amazon Prime (£)", min_value=0.0, step=0.01)
        groceries = st.number_input("Groceries (£)", min_value=0.0, step=0.01)
        transportation = st.number_input("Transportation (£)", min_value=0.0, step=0.01)
        water_bill = st.number_input("Water Bill (£)", min_value=0.0, step=0.01)
        sky_sports = st.number_input("Sky Sports (£)", min_value=0.0, step=0.01)
        other_expenses = st.number_input("Other Expenses (£)", min_value=0.0, step=0.01)
        savings = st.number_input("Savings for Property (£)", min_value=0.0, step=0.01)
        monthly_outing = st.number_input("Monthly Outing (£)", min_value=0.0, step=0.01)
        date = st.date_input("Date", datetime.now())

        submit_button = st.form_submit_button(label="Add Row")

    if submit_button:
        new_row = {
                "Employee": selected_employee,
                "Monthly Income (£)": monthly_income,
                "Electricity Bill (£)": electricity_bill,
                "Gas Bill (£)": gas_bill,
                "Netflix (£)": netflix,
                "Amazon Prime (£)": amazon_prime,
                "Groceries (£)": groceries,
                "Transportation (£)": transportation,
                "Water Bill (£)": water_bill,
                "Sky Sports (£)": sky_sports,
                "Other Expenses (£)": other_expenses,
                "Savings for Property (£)": savings,
                "Monthly Outing (£)": monthly_outing,
                "Date": date.strftime("%Y-%m-%d")
            }

        # Add the new row to the DataFrame
        data_cleaned = data_cleaned.append(new_row, ignore_index=True)
        st.success("New entry added successfully!")

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

            # Savings goal and interval input
            interval = st.selectbox("Select Interval:", ["Daily", "Weekly", "Monthly"])
            savings_goal = st.number_input("Set Your Savings Target (£):", min_value=0.0, step=100.0)

            # Check for shortfall and calculate savings required per interval
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

            # Suggest areas for cost-cutting
            st.subheader("Expense Breakdown and Suggestions")

            # Select the latest available data for the employee
            latest_data = employee_data.loc[employee_data['Date'].idxmax()]

            # Define thresholds or benchmarks for expenses (these can be adjusted based on domain knowledge or data)
            thresholds = {
                'Netflix (£)': 10.0,
                'Amazon Prime (£)': 8.0,
                'Monthly Outing (£)': 50.0,
                'Sky Sports (£)': 20.0,
                'Groceries (£)': 200.0,
            }

            suggestions = []

            for column, threshold in thresholds.items():
                if latest_data[column] > threshold:
                    suggestions.append(f"Consider reducing your {column[:-4]} expense. Current: £{latest_data[column]:.2f}, Suggested: £{threshold:.2f}.")

            if suggestions:
                st.warning("Here are some areas where you could save money:")
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")
            else:
                st.success("Your expenses are within reasonable limits. Great job!")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


# Step 5: Real-Time Updates
elif selected_section == "Real-Time Updates and Scenario Planning":
    st.subheader("Real-Time Updates")
    try:
        st.info(f"You currently have £{current_savings} saved.")

        current_savings = employee_data['Savings for Property (£)'].loc[employee_data['Date'].idxmax()]
        current_bills = employee_data['Electricity Bill (£)'].loc[employee_data['Date'].idxmax()] + employee_data['Gas Bill (£)'].loc[employee_data['Date'].idxmax()] + employee_data['Water Bill (£)'].loc[employee_data['Date'].idxmax()]
        current_Income = employee_data['Monthly Income (£)'].loc[employee_data['Date'].idxmax()]
        current_Entertainment = employee_data['Netflix (£)'].loc[employee_data['Date'].idxmax()] + employee_data['Amazon Prime (£)'].loc[employee_data['Date'].idxmax()]+ employee_data['Sky Sports (£)'].loc[employee_data['Date'].idxmax()]+ employee_data['Monthly Outing (£)'].loc[employee_data['Date'].idxmax()]
        current_other = employee_data['Other Expenses (£)'].loc[employee_data['Date'].idxmax()] + employee_data['Transportation (£)'].loc[employee_data['Date'].idxmax()]

        savings_goal = st.number_input("Set Your Savings Target (£):", min_value=0.0, step=100.0)

        real_time_Income = st.slider("Adjust Monthly Income (£):", min_value=0, max_value=int(current_Income + 5000), value=int(current_Income))

        real_time_bills = st.slider("Adjust Bills (£):", min_value=0, max_value=int(current_bills + 1000), value=int(current_bills))

        real_time_entertainment = st.slider("Adjust Entertainment (£):", min_value=0, max_value=int(current_Entertainment + 1000), value=int(current_Entertainment))

        real_time_savings = st.slider("Adjust Current Savings (£):", min_value=0, max_value=int(current_savings + 5000), value=int(current_savings))

        real_time_other = st.slider("Adjust Other Expenses and Transportation (£):", min_value=0, max_value=int(current_savings + 5000), value=int(current_other))

        leftoverMoney = real_time_Income - real_time_bills - real_time_savings - real_time_other - real_time_entertainment

        updated_goal_status = "Met" if real_time_savings >= savings_goal else "not Met"
        st.info(f"You have {updated_goal_status} your Savings Goal")
        st.warning(f"You have {leftoverMoney} left over each month")

    except Exception as e:
        st.error(f"Real-time updates error: {e}")

