import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import base64
from pathlib import Path

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the data file path
data_file_path = Path(__file__).parent.parent / "data" / "Netflix Userbase.csv"

def load_data():
    """
    Loads the data from the CSV file and processes it.

    Returns:
        pd.DataFrame: The loaded and processed DataFrame.
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(data_file_path)

    # Convert 'Join Date' and 'Last Payment Date' columns to datetime format
    df['Join Date'] = pd.to_datetime(df['Join Date'], format='%d-%m-%y')
    df['Last Payment Date'] = pd.to_datetime(df['Last Payment Date'], format='%d-%m-%y')

    return df

def generate_subscription_visualizations(df):
    """
    Generates subscription visualizations.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Dictionary containing subscription visualizations.
    """
    # Generate visualizations for the subscription data
    subscription_visualizations = {}

    # Subscription performance visualizations
    # Revenue by subscription type
    revenue_by_subscription = df.groupby('Subscription Type')['Monthly Revenue'].sum().reset_index()
    fig_revenue_by_subscription = px.bar(revenue_by_subscription, x='Subscription Type', y='Monthly Revenue',
                                         title='Monthly Revenue by Subscription Type',
                                         labels={'Monthly Revenue': 'Monthly Revenue ($)'})

    # User acquisition over time
    user_acquisition = df.resample('M', on='Join Date').size().reset_index(name='New Users')
    fig_user_acquisition = px.line(user_acquisition, x='Join Date', y='New Users',
                                   title='User Acquisition Over Time', labels={'Join Date': 'Date', 'New Users': 'New Users'})

    subscription_visualizations['revenue_by_subscription'] = fig_revenue_by_subscription
    subscription_visualizations['user_acquisition'] = fig_user_acquisition

    # User demographics visualizations
    # Distribution of users by country
    country_distribution = df['Country'].value_counts().reset_index()
    country_distribution.columns = ['Country', 'Count']
    fig_country_distribution = px.choropleth(country_distribution, locations='Country', locationmode='country names', color='Count',
                                             title='Distribution of Users by Country')

    subscription_visualizations['country_distribution'] = fig_country_distribution

    # Average monthly revenue by gender
    gender_revenue = df.groupby('Gender')['Monthly Revenue'].mean().reset_index()
    fig_gender_revenue = px.bar(gender_revenue, x='Gender', y='Monthly Revenue',
                                 title='Average Monthly Revenue by Gender',
                                 labels={'Monthly Revenue': 'Average Monthly Revenue ($)'})

    subscription_visualizations['gender_revenue'] = fig_gender_revenue

    # Device usage visualizations
    device_usage = df['Device'].value_counts().reset_index()
    device_usage.columns = ['Device', 'Count']
    fig_device_usage = px.pie(device_usage, names='Device', values='Count', title='Device Usage')
    subscription_visualizations['device_usage'] = fig_device_usage

    # Additional visualizations
    age_to_revenue = px.scatter(df, x='Age', y='Monthly Revenue', trendline='ols', title='Age to Revenue')

    subscription_visualizations['age_to_revenue'] = age_to_revenue

    return subscription_visualizations


def calculate_key_metrics(df):
    """
    Calculates key metrics.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Dictionary containing key metrics.
    """
    # Calculate key metrics
    arpu = df['Monthly Revenue'].mean()
    churn_rate = 1 - (df[df['Last Payment Date'] >= pd.Timestamp.now()]['User ID'].nunique() / df['User ID'].nunique())

    # Calculate average monthly revenue by device type
    average_revenue_by_device = df.groupby('Device')['Monthly Revenue'].mean()

    # Calculate total revenue
    total_revenue = df['Monthly Revenue'].sum()

    # Calculate total number of users
    total_users = df['User ID'].nunique()

    # Calculate average age
    average_age = df['Age'].mean()

    # Calculate percentage of revenue by device type
    revenue_percentage_by_device = (average_revenue_by_device / total_revenue) * 100

    key_metrics = {
        'arpu': f"${arpu:.2f}",
        'churn_rate': f"{churn_rate:.2%}",
        'revenue_percentage_smartphone': revenue_percentage_by_device.get('Smartphone', 0),
        'revenue_percentage_tablet': revenue_percentage_by_device.get('Tablet', 0),
        'revenue_percentage_laptop': revenue_percentage_by_device.get('Laptop', 0),
        'revenue_percentage_smart_tv': revenue_percentage_by_device.get('Smart TV', 0),
        'average_revenue_by_device': average_revenue_by_device,
        'total_revenue': f"${total_revenue:,.2f}",
        'total_users': total_users,
        'average_age': f"{average_age:.1f}"
    }

    return key_metrics


# Load the data
df = load_data()

# Generate visualizations for the subscription data
subscription_visualizations = generate_subscription_visualizations(df)

# Calculate key metrics
key_metrics = calculate_key_metrics(df)

# Set custom color palette
custom_colors = ['pink', 'purple', 'lightgreen', 'lightblue']

# Define the homepage layout
homepage_layout = html.Div([
    html.H1("Welcome to the Subscription Service Dashboard of Hamitoly and Çayçoly",
            style={'color': 'red', 'font-family': 'Consolas', 'text-align': 'center'}),
    html.Div([
        dcc.Link('Go to Dashboard', href='/dashboard', style={'color': 'purple', 'font-family': 'Consolas', 'text-align': 'center'})
    ])
])

# Define the dashboard layout
dashboard_layout = html.Div([
    html.H1("Dashboard", style={'color': 'purple', 'font-family': 'Consolas', 'text-align': 'center'}),
    html.P("Made with love by MHC for AYC.",
           style={'color': 'black', 'font-family': 'Consolas', 'font-size': '16px', 'text-align': 'center'}),
    html.P("© 2024",
           style={'color': 'black', 'font-family': 'Consolas', 'font-size': '16px', 'text-align': 'center'}),
    html.Div([
        html.A(
            html.Button('Download CSV',
                        style={'color': 'white', 'background-color': 'purple', 'font-family': 'Consolas',
                               'font-size': '16px', 'text-align': 'center'}),
            id='download-link',
            download="Netflix Userbase.csv",
            href="",
            target="_blank",
        ),
    ], className='dashboard-section'),

    html.Div([
        html.H2("Subscription Performance", style={'color': 'purple', 'font-family': 'Consolas'}),
        dcc.Graph(figure=subscription_visualizations['revenue_by_subscription']),
        dcc.Graph(figure=subscription_visualizations['user_acquisition']),
    ], className='subscription-performance'),  # Separate division for Subscription Performance

    html.Div([
        html.H2("User Demographics", style={'color': 'purple', 'font-family': 'Consolas'}),
        dcc.Graph(figure=subscription_visualizations['country_distribution']),
        dcc.Graph(figure=subscription_visualizations['gender_revenue']),
    ], className='dashboard-section'),

    html.Div([
        html.H2("Device Usage", style={'color': 'purple', 'font-family': 'Consolas'}),
        dcc.Graph(figure=subscription_visualizations['device_usage']),
    ], className='dashboard-section'),

    html.Div([
        html.H2("Key Metrics", style={'color': 'purple', 'font-family': 'Consolas'}),
        html.Table([
            html.Tr([html.Td("Average Revenue Per User (ARPU)", style={'font-family': 'Consolas', 'font-size': '14px'}),
                     html.Td(key_metrics['arpu'], style={'font-family': 'Consolas', 'font-size': '14px'})]),
            html.Tr([html.Td("Churn Rate", style={'font-family': 'Consolas', 'font-size': '14px'}),
                     html.Td(key_metrics['churn_rate'], style={'font-family': 'Consolas', 'font-size': '14px'})]),
            html.Tr([html.Td("Average Monthly Revenue by Device Type",
                             style={'font-family': 'Consolas', 'font-size': '14px'})]),
            html.Tr([html.Td("Smartphone", style={'font-family': 'Consolas', 'font-size': '14px'}),
                     html.Td(key_metrics['revenue_percentage_smartphone'], style={'font-family': 'Consolas', 'font-size': '14px'})]),
            html.Tr([html.Td("Tablet", style={'font-family': 'Consolas', 'font-size': '14px'}),
                     html.Td(key_metrics['revenue_percentage_tablet'], style={'font-family': 'Consolas', 'font-size': '14px'})]),
            html.Tr([html.Td("Laptop", style={'font-family': 'Consolas', 'font-size': '14px'}),
                     html.Td(key_metrics['revenue_percentage_laptop'], style={'font-family': 'Consolas', 'font-size': '14px'})]),
            html.Tr([html.Td("Smart TV", style={'font-family': 'Consolas', 'font-size': '14px'}),
                     html.Td(key_metrics['revenue_percentage_smart_tv'], style={'font-family': 'Consolas', 'font-size': '14px'})]),
            html.Tr([html.Td("Total Revenue", style={'font-family': 'Consolas', 'font-size': '14px'}),
                     html.Td(key_metrics['total_revenue'], style={'font-family': 'Consolas', 'font-size': '14px'})]),
            html.Tr([html.Td("Total Users", style={'font-family': 'Consolas', 'font-size': '14px'}),
                     html.Td(key_metrics['total_users'], style={'font-family': 'Consolas', 'font-size': '14px'})]),
            html.Tr([html.Td("Average Age", style={'font-family': 'Consolas', 'font-size': '14px'}),
                     html.Td(key_metrics['average_age'], style={'font-family': 'Consolas', 'font-size': '14px'})]),
        ]),
    ], className='dashboard-section'),

], style={'background-color': 'pink'})


# Callback to update download link href
@app.callback(
    Output('download-link', 'href'),
    [Input('url', 'pathname')]
)
def update_download_link(pathname):
    if pathname == '/dashboard':
        csv_string = df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + base64.b64encode(csv_string.encode()).decode()
        return csv_string
    else:
        return ""


# Define the callback to switch between pages
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/dashboard':
        return dashboard_layout
    else:
        return homepage_layout


# Set the layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
