import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Function to apply custom CSS for mobile responsiveness
def set_mobile_css():
    st.markdown(
        """
        <style>
        @media only screen and (max-width: 600px) {
            .stApp {
                padding: 0 10px;
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                font-size: 1.2em !important;
            }
            .headline {
                font-size: 1.5em !important;
            }
            .stDataFrame th, .stDataFrame td {
                font-size: 0.8em !important;
            }
            .css-12w0qpk, .css-15tx938, .stSelectbox label, .stTable th, .stTable thead th, .dataframe th {
                font-size: 0.8em !important;
            }
        }
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted black;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 120px;
            background-color: black;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 0;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """, unsafe_allow_html=True
    )

# Load the dataset from Parquet
file_path = 'https://raw.githubusercontent.com/timurna/news-fcb/main/new.parquet'
data = pd.read_parquet(file_path)

# Calculate age from birthdate
data['Birthdate'] = pd.to_datetime(data['Birthdate'])
today = datetime.today()
data['Age'] = data['Birthdate'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

# Define position groups with potential overlaps
position_groups = {
    'IV': ['LCB', 'RCB', 'CB'],
    'AV': ['LB', 'RB'],
    'FLV': ['LWB', 'RWB'],
    'AVFLV': ['LB', 'RB', 'LWB', 'RWB'],
    'ZDM': ['DM', 'LDM', 'RDM'],
    'ZDMZM': ['DM', 'LDM', 'RDM', 'CM', 'RM', 'LM'],
    'ZM': ['CM', 'RM', 'LM'],
    'ZOM': ['CM', 'AM'],
    'FS': ['LW', 'RW'],
    'ST': ['CF', 'LF', 'RF']
}

# Assign positions to multiple groups
data['Position Groups'] = data['Position_y'].apply(lambda pos: [group for group, positions in position_groups.items() if pos in positions])

# Convert text-based numbers to numeric
physical_metrics = ['PSV-99', 'Distance', 'M/min', 'HSR Distance', 'HSR Count', 'Sprint Distance', 'Sprint Count',
                    'HI Distance', 'HI Count', 'Medium Acceleration Count', 'High Acceleration Count',
                    'Medium Deceleration Count', 'High Deceleration Count', 'Distance OTIP', 'M/min OTIP',
                    'HSR Distance OTIP', 'HSR Count OTIP', 'Sprint Distance OTIP', 'Sprint Count OTIP',
                    'HI Distance OTIP', 'HI Count OTIP', 'Medium Acceleration Count OTIP',
                    'High Acceleration Count OTIP', 'Medium Deceleration Count OTIP', 'High Deceleration Count OTIP'
                    ]

for metric in physical_metrics:
    data[metric] = pd.to_numeric(data[metric].astype(str).str.replace(',', '.'), errors='coerce')

# Calculate additional metrics
data['OnTarget%'] = (data['SOG'] / data['Shot']) * 100
data['TcklMade%'] = (data['Tckl'] / data['TcklAtt']) * 100
data['Pass%'] = (data['PsCmp'] / data['PsAtt']) * 100

# Normalize and calculate the scores
scaler = MinMaxScaler(feature_range=(0, 10))
quantile_transformer = QuantileTransformer(output_distribution='uniform')

def calculate_score(df, metrics):
    # Fill missing values with 0 for the specific metrics used
    df_subset = df[metrics].fillna(0)
    # Ensure all values are numeric
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
    # Apply quantile transformation and scaling
    return scaler.fit_transform(quantile_transformer.fit_transform(df_subset)).mean(axis=1)

# Calculate scores for each category
data['Physical Offensive Score'] = calculate_score(data, physical_metrics)
data['Physical Defensive Score'] = calculate_score(data, physical_metrics)
data['Offensive Score'] = calculate_score(data, [
    'PsAtt', 'PsCmp', 'Pass%', 'PsIntoA3rd', 'ProgPass', 'ThrghBalls', 'Touches', 'PsRec', 'ProgCarry', 'TakeOn', 'Success1v1'
])
data['Defensive Score'] = calculate_score(data, [
    'TcklMade%', 'TcklAtt', 'Tckl', 'AdjTckl', 'TcklA3', 'Blocks', 'Int', 'AdjInt', 'Clrnce'
])
data['Goal Threat Score'] = calculate_score(data, [
    'Goal', 'Shot/Goal', 'MinPerGoal', 'ExpG', 'xGOT', 'xG +/-', 
    'Shot', 'SOG', 'Shot conversion', 'OnTarget%'
])

# Now, add the 'Goal Threat Score' to the list of scores to be displayed
scores = [
    'Offensive Score',
    'Goal Threat Score',
    'Defensive Score', 
    'Physical Offensive Score', 
    'Physical Defensive Score'
]

# Define metrics list for other tables
metrics = physical_metrics + [
    'PsAtt', 'PsCmp', 'Pass%', 'PsIntoA3rd', 'ProgPass', 'ThrghBalls', 'Touches', 'PsRec', 'ProgCarry', 'TakeOn', 'Success1v1',
    'TcklMade%', 'TcklAtt', 'Tckl', 'AdjTckl', 'TcklA3', 'Blocks', 'Int', 'AdjInt', 'Clrnce',
    'OnTarget%', 'TcklMade%', 'Pass%'
]

# Mapping from League to the corresponding SkillCorner competition names
league_to_competition_display = {
    'Süper Lig (Türkiye)': 'TUR - Super Lig',
    'Allsvenskan (Sweden)': 'SWE - Allsvenskan',
    'Super League (Switzerland)': 'SUI - Super League',
    'Primeira Liga (Portugal)': 'POR - Primeira Liga',
    'Eliteserien (Norway)': 'NOR - Eliteserien',
    'Eredivisie (Netherlands)': 'NED - Eredivisie',
    'Serie A (Italy)': 'ITA - Serie A',
    'Bundesliga (Germany)': 'GER - Bundesliga',
    '2. Bundesliga (Germany)': 'GER - 2nd Bundesliga',
    'Ligue 2 (France)': 'FRA - Ligue 2',
    'Ligue 1 (France)': 'FRA - Ligue 1',
    'Primera División (Spain)': 'ESP - LaLiga',
    'Premier League (England)': 'ENG - Premier League',
    'Championship (England)': 'ENG - Championship',
    'Liga Pro (Ecuador)': 'ECU - Liga Pro',
    'Superliga (Denmark)': 'DEN - Superliga',
    'Serie A (Brazil)': 'BRA - Série A',
    'First Division A (Belgium)': 'BEL - Pro League',
    'Bundesliga (Austria)': 'AUT - Bundesliga',
    'Liga Profesional Argentina (Argentina)': 'ARG - Primera Division'
}

# User authentication (basic example)
def authenticate(username, password):
    return username == "fcbscouting24" and password == "fcbnews24"

def login():
    st.session_state.authenticated = False
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
                        st.session_state.authenticated = True
        else:
            st.error("Invalid username or password")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login()
else:
    set_mobile_css()

    # Display the logo at the top
    st.image('logo.png', use_column_width=False, width=800)

    # Create a single row for all the filters
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            leagues = sorted(data['League'].unique())  # Sort leagues alphabetically
            selected_league = st.selectbox("Select League", leagues, key="select_league")

        with col2:
            league_data = data[data['League'] == selected_league]

            # Week Summary and Matchday Filtering Logic
            week_summary = league_data.groupby(['League', 'Week']).agg({'Date.1': ['min', 'max']}).reset_index()
            week_summary.columns = ['League', 'Week', 'min', 'max']

            week_summary['min'] = pd.to_datetime(week_summary['min'])
            week_summary['max'] = pd.to_datetime(week_summary['max'])

            week_summary['Matchday'] = week_summary.apply(
                lambda row: f"{row['Week']} ({row['min'].strftime('%d.%m.%Y')} - {row['max'].strftime('%d.%m.%Y')})", axis=1
            )

            filtered_weeks = week_summary[week_summary['League'] == selected_league].sort_values(by='min').drop_duplicates(subset=['Week'])

            matchday_options = filtered_weeks['Matchday'].tolist()
            selected_matchday = st.selectbox("Select Matchday", matchday_options, key="select_matchday")

            selected_week = filtered_weeks[filtered_weeks['Matchday'] == selected_matchday]['Week'].values[0]

        with col3:
            position_group_options = list(position_groups.keys())
            selected_position_group = st.selectbox("Select Position Group", position_group_options, key="select_position_group")

    # Filter the data by the selected position group
    league_and_position_data = data[
        (data['League'] == selected_league) &
        (data['Week'] == selected_week) &
        (data['Position Groups'].apply(lambda groups: selected_position_group in groups))
    ]

    # Use a container to make the expandable sections span the full width
    with st.container():
        tooltip_headers = {metric: glossary.get(metric, '') for metric in scores + metrics}

        def display_metric_tables(metrics_list, title):
            with st.expander(title, expanded=False):  # Setting expanded=False to keep it closed by default
                for metric in metrics_list:
                    if metric not in league_and_position_data.columns:
                        st.write(f"Metric {metric} not found in the data")
                        continue

                    league_and_position_data[metric] = pd.to_numeric(league_and_position_data[metric], errors='coerce')

                    top10 = league_and_position_data[['Player_y', 'Age', 'Team_y', 'Position_y', metric]].dropna(subset=[metric]).sort_values(by=metric, ascending=False).head(10)

                    if top10.empty:
                        st.header(f"Top 10 Players in {metric}")
                        st.write("No data available")
                    else:
                        # Reset the index to create a rank column starting from 1
                        top10.reset_index(drop=True, inplace=True)
                        top10.index += 1
                        top10.index.name = 'Rank'

                        # Ensure the Rank column is part of the DataFrame before styling
                        top10 = top10.reset_index()

                        # Check if the League column exists before mapping
                        if 'League' in top10.columns:
                            top10['League'] = top10['League'].map(league_to_competition_display)

                        st.markdown(f"<h2>{metric}</h2>", unsafe_allow_html=True)
                        top10.rename(columns={'Player_y': 'Player', 'Team_y': 'Team', 'Position_y': 'Position'}, inplace=True)
                        top10[metric] = top10[metric].apply(lambda x: f"{x:.2f}")

                        def color_row(row):
                            return ['background-color: #d4edda' if row['Age'] < 24 else '' for _ in row]

                        top10_styled = top10.style.apply(color_row, axis=1)
                        top10_html = top10_styled.to_html()

                        for header, tooltip in tooltip_headers.items():
                            if tooltip:
                                top10_html = top10_html.replace(f'>{header}<', f'><span class="tooltip">{header}<span class="tooltiptext">{tooltip}</span></span><')

                        st.write(top10_html, unsafe_allow_html=True)

        display_metric_tables(scores, "Score Metrics")
        display_metric_tables(physical_metrics, "Physical Metrics")
        display_metric_tables(offensive_metrics, "Offensive Metrics")
        display_metric_tables(defensive_metrics, "Defensive Metrics")

    # Glossary section now placed below the metrics tables
    with st.expander("Glossary"):
        for metric, explanation in glossary.items():
            st.markdown(f"**{metric}:** {explanation}")

