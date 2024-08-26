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

# Glossary content
glossary = {
    # Score Metrics
    '**Score Metrics**': '',  # Empty string as value to remove the colon
    'Offensive Score': 'A score representing a player\'s overall offensive performance.',
    'Defensive Score': 'A score representing a player\'s overall defensive performance.',
    'Physical Offensive Score': 'A score representing a player\'s physical contributions to offensive play.',
    'Physical Defensive Score': 'A score representing a player\'s physical contributions to defensive play.',
    
    # Offensive Metrics
    '**Offensive Metrics**': '',  # Empty string as value to remove the colon
    'Take on into the Box': 'Number of successful dribbles into the penalty box.',
    'TouchOpBox': 'Number of touches in the opponent\'s penalty box.',
    'KeyPass': 'Passes that directly lead to a shot on goal.',
    '2ndAst': 'The pass that assists the assist leading to a goal.',
    'xA +/-': 'Expected Assists +/- difference.',
    'MinPerChnc': 'Minutes per chance created.',
    'PsAtt': 'Passes attempted.',
    'PsCmp': 'Passes completed.',
    'PsIntoA3rd': 'Passes into the attacking third.',
    'ProgPass': 'Progressive passes, advancing the ball significantly.',
    'ThrghBalls': 'Through balls successfully played.',
    'Touches': 'Total number of touches.',
    'PsRec': 'Passes received by the player.',
    'ProgCarry': 'Progressive carries, advancing the ball significantly.',
    'TakeOn': 'Attempted dribbles to beat an opponent.',
    'Success1v1': 'Successful 1v1 dribbles against an opponent.',
    'Goal': 'Goals scored.',
    'Shot/Goal': 'Shots per goal.',
    'MinPerGoal': 'Minutes per goal scored.',
    'GoalExPn': 'Goals excluding penalties.',
    'ExpG': 'Expected goals.',
    'xGOT': 'Expected goals on target.',
    'ExpGExPn': 'Expected goals excluding penalties.',
    'xG +/-': 'Expected goals +/- difference.',
    'Shot': 'Total shots taken.',
    'SOG': 'Shots on goal.',
    'Shot conversion': 'Percentage of shots converted to goals.',
    'Ast': 'Assists.',
    'xA': 'Expected assists.',
    
    # Additional Metrics
    '**Additional Metrics**': '',  # Empty string as value to remove the colon
    'OnTarget%': 'Percentage of shots on target out of total shots.',
    'TcklMade%': 'Percentage of tackles successfully made out of total tackle attempts.',
    'Pass%': 'Percentage of completed passes out of total passes attempted.',
    
    # Defensive Metrics
    '**Defensive Metrics**': '',  # Empty string as value to remove the colon
    'TcklAtt': 'Tackles attempted.',
    'Tckl': 'Tackles made.',
    'AdjTckl': 'Adjusted tackles, considering context.',
    'TcklA3': 'Tackles made in the attacking third.',
    'Blocks': 'Total blocks made.',
    'Int': 'Interceptions made.',
    'AdjInt': 'Adjusted interceptions, considering context.',
    'Clrnce': 'Clearances made.',
    
    # Physical Metrics
    '**Physical Metrics**': '',  # Empty string as value to remove the colon
    'Distance': 'Total distance covered by the player during the match.',
    'M/min': 'Meters covered per minute by the player.',
    'HSR Distance': 'High-speed running distance covered.',
    'HSR Count': 'Count of high-speed running actions.',
    'Sprint Distance': 'Total distance covered while sprinting.',
    'Sprint Count': 'Total sprints performed.',
    'HI Distance': 'High-intensity distance covered.',
    'HI Count': 'High-intensity actions performed.',
    'Medium Acceleration Count': 'Medium-intensity accelerations performed.',
    'High Acceleration Count': 'High-intensity accelerations performed.',
    'Medium Deceleration Count': 'Medium-intensity decelerations performed.',
    'High Deceleration Count': 'High-intensity decelerations performed.',
    'Distance OTIP': 'Distance covered off the ball in possession (OTIP).',
    'M/min OTIP': 'Meters per minute covered off the ball in possession (OTIP).',
    'HSR Distance OTIP': 'High-speed running distance covered off the ball in possession (OTIP).',
    'HSR Count OTIP': 'High-speed running actions performed off the ball in possession (OTIP).',
    'Sprint Distance OTIP': 'Sprint distance covered off the ball in possession (OTIP).',
    'Sprint Count OTIP': 'Sprint actions performed off the ball in possession (OTIP).',
    'HI Distance OTIP': 'High-intensity distance covered off the ball in possession (OTIP).',
    'HI Count OTIP': 'High-intensity actions performed off the ball in possession (OTIP).',
    'Medium Acceleration Count OTIP': 'Medium-intensity accelerations performed off the ball in possession (OTIP).',
    'High Acceleration Count OTIP': 'High-intensity accelerations performed off the ball in possession (OTIP).',
    'Medium Deceleration Count OTIP': 'Medium-intensity decelerations performed off the ball in possession (OTIP).',
    'High Deceleration Count OTIP': 'High-intensity decelerations performed off the ball in possession (OTIP).',
    'PSV-99': 'Custom metric PSV-99 (explanation needed).'
}

# Load the dataset from Parquet
file_path = 'https://raw.githubusercontent.com/timurna/news-fcb/main/test-neu-upd.parquet'
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
physical_metrics = ['Distance', 'M/min', 'HSR Distance', 'HSR Count', 'Sprint Distance', 'Sprint Count',
                    'HI Distance', 'HI Count', 'Medium Acceleration Count', 'High Acceleration Count',
                    'Medium Deceleration Count', 'High Deceleration Count', 'Distance OTIP', 'M/min OTIP',
                    'HSR Distance OTIP', 'HSR Count OTIP', 'Sprint Distance OTIP', 'Sprint Count OTIP',
                    'HI Distance OTIP', 'HI Count OTIP', 'Medium Acceleration Count OTIP',
                    'High Acceleration Count OTIP', 'Medium Deceleration Count OTIP', 'High Deceleration Count OTIP',
                    'PSV-99']

for metric in physical_metrics:
    data[metric] = pd.to_numeric(data[metric].astype(str).str.replace(',', '.'), errors='coerce')

# Calculate additional metrics
data['OnTarget%'] = (data['SOG'] / data['Shot']) * 100
data['TcklMade%'] = (data['Tckl'] / data['TcklAtt']) * 100
data['Pass%'] = (data['PsCmp'] / data['PsAtt']) * 100

# Normalize and calculate the scores
scaler = MinMaxScaler(feature_range=(0, 10))
quantile_transformer = QuantileTransformer(output_distribution='uniform')

# Calculate physical offensive score
data['Physical Offensive Score'] = scaler.fit_transform(
    quantile_transformer.fit_transform(data[physical_metrics].fillna(0))
).mean(axis=1)

# Calculate physical defensive score
data['Physical Defensive Score'] = scaler.fit_transform(
    quantile_transformer.fit_transform(data[physical_metrics].fillna(0))
).mean(axis=1)

# Calculate offensive score
offensive_metrics = [
    'PsAtt', 'PsCmp', 'Pass%', 'PsIntoA3rd', 'ProgPass', 'ThrghBalls', 'Touches', 'PsRec', 'ProgCarry', 'TakeOn', 'Success1v1'
]

data['Offensive Score'] = scaler.fit_transform(
    quantile_transformer.fit_transform(data[offensive_metrics].fillna(0))
).mean(axis=1)

# Calculate defensive score
defensive_metrics = [
    'TcklMade%', 'TcklAtt', 'Tckl', 'AdjTckl', 'TcklA3', 'Blocks', 'Int', 'AdjInt', 'Clrnce'
]

data['Defensive Score'] = scaler.fit_transform(
    quantile_transformer.fit_transform(data[defensive_metrics].fillna(0))
).mean(axis=1)

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
    st.image('FCBayern-Wortmarke-SF-ANSICHT.png', use_column_width=False, width=800)

    # Create a single row for all the filters
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            leagues = sorted(data['Competition'].unique())  # Sort leagues alphabetically
            selected_league = st.selectbox("Select League", leagues, key="select_league")

        with col2:
            league_data = data[data['Competition'] == selected_league]

            # Week Summary and Matchday Filtering Logic
            week_summary = league_data.groupby(['Competition', 'Week']).agg({'Date.1': ['min', 'max']}).reset_index()
            week_summary.columns = ['Competition', 'Week', 'min', 'max']

            week_summary['min'] = pd.to_datetime(week_summary['min'])
            week_summary['max'] = pd.to_datetime(week_summary['max'])

            week_summary['Matchday'] = week_summary.apply(
                lambda row: f"{row['Week']} ({row['min'].strftime('%d.%m.%Y')} - {row['max'].strftime('%d.%m.%Y')})", axis=1
            )

            filtered_weeks = week_summary[week_summary['Competition'] == selected_league].sort_values(by='min').drop_duplicates(subset=['Week'])

            matchday_options = filtered_weeks['Matchday'].tolist()
            selected_matchday = st.selectbox("Select Matchday", matchday_options, key="select_matchday")

            selected_week = filtered_weeks[filtered_weeks['Matchday'] == selected_matchday]['Week'].values[0]

        with col3:
            position_group_options = list(position_groups.keys())
            selected_position_group = st.selectbox("Select Position Group", position_group_options, key="select_position_group")

    # Filter the data by the selected position group
    league_and_position_data = data[
        (data['Competition'] == selected_league) &
        (data['Week'] == selected_week) &
        (data['Position Groups'].apply(lambda groups: selected_position_group in groups))
    ]

    # Use a container to make the expandable sections span the full width
    with st.container():
        scores = ['Offensive Score', 'Defensive Score', 'Physical Offensive Score', 'Physical Defensive Score']
        metrics = ['PSV-99'] + physical_metrics + ['Take on into the Box', 'TouchOpBox', 'KeyPass', '2ndAst', 'xA +/-', 'MinPerChnc', 
                                                   'PsAtt', 'PsCmp', 'PsIntoA3rd', 'ProgPass', 'ThrghBalls', 'Touches', 'PsRec', 
                                                   'ProgCarry', 'TakeOn', 'Success1v1', 
                                                   'TcklAtt', 'Tckl', 'AdjTckl', 'TcklA3', 
                                                   'Blocks', 'Int', 'AdjInt', 'Clrnce', 
                                                   'Goal', 'Shot/Goal', 'MinPerGoal', 'GoalExPn', 
                                                   'ExpG', 'xGOT', 'ExpGExPn', 'xG +/-', 
                                                   'Shot', 'SOG', 'Shot conversion', 'Ast', 'xA',
                                                   'OnTarget%', 'TcklMade%', 'Pass%']

        all_metrics = scores + metrics

        tooltip_headers = {metric: glossary.get(metric, '') for metric in all_metrics}

        def display_metric_tables(metrics_list, title):
            with st.expander(title, expanded=False):  # Setting expanded=True to make it open by default
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
