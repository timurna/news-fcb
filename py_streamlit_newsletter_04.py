import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image

# Function to apply custom CSS for mobile responsiveness and tooltips
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
    'Physical Offensive Score': 'A score representing a player\'s physical contributions to offensive play.',
    'Physical Defensive Score': 'A score representing a player\'s physical contributions to defensive play.',
    'Offensive Score': 'A score representing a player\'s overall offensive performance.',
    'Defensive Score': 'A score representing a player\'s overall defensive performance.',
    'Goal Threat': 'A score representing a player\'s goal-scoring potential based on key metrics like goals, shots, shots on goal, and accuracy.',
    'Distance': 'Total distance covered by the player during the match.',
    'M/min': 'Meters covered per minute by the player.',
    # Add explanations for other metrics...
}

# Define position groups globally so they can be used throughout the script
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

# Define offensive and defensive metrics globally
offensive_metrics = [
    'PsAtt', 'PsCmp', 'Pass%', 'PsIntoA3rd', 'ProgPass', 'ThrghBalls', 'Touches', 'PsRec', 'ProgCarry', 'TakeOn', 'Success1v1'
]

defensive_metrics = [
    'TcklMade%', 'TcklAtt', 'Tckl', 'AdjTckl', 'TcklA3', 'Blocks', 'Int', 'AdjInt', 'Clrnce'
]

@st.cache_data
def load_data(file_path):
    # Load the dataset
    data = pd.read_excel(file_path)
    
    # Calculate age from birthdate
    data['Birthdate'] = pd.to_datetime(data['Birthdate'])
    today = datetime.today()
    data['Age'] = data['Birthdate'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))
    
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
    
    return data, physical_metrics

# Load the data with caching
data, physical_metrics = load_data('data_newsletter.xlsx')

@st.cache_data
def calculate_scores(data, physical_metrics, offensive_metrics, defensive_metrics):
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
    data['Offensive Score'] = scaler.fit_transform(
        quantile_transformer.fit_transform(data[offensive_metrics].fillna(0))
    ).mean(axis=1)
    
    # Calculate defensive score
    data['Defensive Score'] = scaler.fit_transform(
        quantile_transformer.fit_transform(data[defensive_metrics].fillna(0))
    ).mean(axis=1)
    
    # Calculate Goal Threat score
    goal_threat_metrics = [
        'Goal', 'Shot', 'SOG', 'OnTarget%'
    ]
    
    weights = {
        'Goal': 2,        # Double weight for 'Goal'
        'Shot': 1,
        'SOG': 1,
        'OnTarget%': 1
    }
    
    # Normalize and calculate the weighted Goal Threat score
    normalized_goal = scaler.fit_transform(quantile_transformer.fit_transform(data[['Goal']].fillna(0))) * weights['Goal']
    normalized_shot = scaler.fit_transform(quantile_transformer.fit_transform(data[['Shot']].fillna(0))) * weights['Shot']
    normalized_sog = scaler.fit_transform(quantile_transformer.fit_transform(data[['SOG']].fillna(0))) * weights['SOG']
    normalized_ontarget = scaler.fit_transform(quantile_transformer.fit_transform(data[['OnTarget%']].fillna(0))) * weights['OnTarget%']
    
    data['Goal Threat'] = (
        normalized_goal +
        normalized_shot +
        normalized_sog +
        normalized_ontarget
    ).mean(axis=1) / sum(weights.values())
    
    scores = ['Offensive Score', 'Defensive Score', 'Goal Threat', 'Physical Offensive Score', 'Physical Defensive Score']
    
    return data, scores

# Calculate the scores with caching
data, scores = calculate_scores(data, physical_metrics, offensive_metrics, defensive_metrics)

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

    # Define the filters to get user input before displaying metrics
    col_filters1, col_filters2 = st.columns([1, 1])

    with col_filters1:
        leagues = data['newestLeague'].unique()
        selected_league = st.selectbox("Select League", leagues, key="select_league")
    
    with col_filters2:
        league_data = data[data['newestLeague'] == selected_league]

        # Week Summary and Matchday Filtering Logic
        week_summary = league_data.groupby(['newestLeague', 'Week']).agg({'Date.1': ['min', 'max']}).reset_index()
        week_summary.columns = ['newestLeague', 'Week', 'min', 'max']

        week_summary['min'] = pd.to_datetime(week_summary['min'])
        week_summary['max'] = pd.to_datetime(week_summary['max'])

        week_summary['Matchday'] = week_summary.apply(
            lambda row: f"{row['Week']} ({row['min'].strftime('%d.%m.%Y')} - {row['max'].strftime('%d.%m.%Y')})", axis=1
        )

        filtered_weeks = week_summary[week_summary['newestLeague'] == selected_league].sort_values(by='min').drop_duplicates(subset=['Week'])

        matchday_options = filtered_weeks['Matchday'].tolist()
        selected_matchday = st.selectbox("Select Matchday", matchday_options, key="select_matchday")

    selected_week = filtered_weeks[filtered_weeks['Matchday'] == selected_matchday]['Week'].values[0]
    league_and_position_data = data[(data['newestLeague'] == selected_league) & (data['Week'] == selected_week)]

    # Now define the layout with columns, starting with the filters
    col1, col2 = st.columns([1, 3])

    # Metrics tables in the second column
    with col2:
        position_group_options = list(position_groups.keys())
        selected_position_group = st.selectbox("Select Position Group", position_group_options, key="select_position_group")
        league_and_position_data = league_and_position_data[
            league_and_position_data['Position Groups'].apply(lambda groups: selected_position_group in groups)
        ]

        metrics = ['PSV-99'] + physical_metrics + offensive_metrics + defensive_metrics + ['Take on into the Box', 'TouchOpBox', 'KeyPass', '2ndAst', 'xA +/-', 'MinPerChnc', 
                                                   'PsAtt', 'PsCmp', 'PsIntoA3rd', 'ProgPass', 'ThrghBalls', 'Touches', 'PsRec', 
                                                   'ProgCarry', 'TakeOn', 'Success1v1', 
                                                   'TcklAtt', 'Tckl', 'AdjTckl', 'TcklA3', 
                                                   'Blocks', 'Int', 'AdjInt', 'Clrnce', 
                                                   'Goal', 'Shot/Goal', 'MinPerGoal', 'GoalExPn', 
                                                   'ExpG', 'xGOT', 'ExpGExPn', 'xG +/-', 
                                                   'Shot', 'SOG', 'Shot conversion', 'Ast', 'xA',
                                                   'OnTarget%', 'TcklMade%', 'Pass%']

        all_metrics = scores + metrics

        # Tooltip headers from the glossary
        tooltip_headers = {metric: glossary.get(metric, '') for metric in all_metrics}

        def display_metric_tables(metrics_list, title):
            with st.expander(title):
                for metric in metrics_list:
                    league_and_position_data[metric] = pd.to_numeric(league_and_position_data[metric], errors='coerce')

                    top10 = league_and_position_data[['Player_y', 'Age', 'Team_y', 'Position_y', metric]].dropna(subset=[metric]).sort_values(by=metric, ascending=False).head(10)

                    if top10.empty:
                        st.header(f"Top 10 Players in {metric}")
                        st.write("No data available")
                    else:
                        # Add tooltip to the header
                        st.markdown(f"<h2>{metric}</h2>", unsafe_allow_html=True)
                        top10.rename(columns={'Player_y': 'Player', 'Team_y': 'Team', 'Position_y': 'Position'}, inplace=True)
                        top10[metric] = top10[metric].apply(lambda x: f"{x:.2f}")

                        # Create HTML table with tooltips for headers
                        def color_row(row):
                            return ['background-color: #d4edda' if row['Age'] < 24 else '' for _ in row]

                        top10_styled = top10.style.apply(color_row, axis=1)
                        top10_html = top10_styled.to_html()

                        # Add tooltips to the headers
                        for header, tooltip in tooltip_headers.items():
                            if tooltip:
                                top10_html = top10_html.replace(f'>{header}<', f'><span class="tooltip">{header}<span class="tooltiptext">{tooltip}</span></span><')

                        st.write(top10_html, unsafe_allow_html=True)

                        # Add a download button for the table
                        buffer = BytesIO()
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.axis('tight')
                        ax.axis('off')
                        ax.table(cellText=top10.values, colLabels=top10.columns, cellLoc='center', loc='center')
                        plt.savefig(buffer, format='png')
                        buffer.seek(0)
                        st.download_button(
                            label="Download Table as PNG",
                            data=buffer,
                            file_name=f"{metric}_top10.png",
                            mime="image/png"
                        )
                        plt.close(fig)

        if selected_matchday and selected_position_group:
            display_metric_tables(scores, "Score Metrics")
            display_metric_tables(physical_metrics, "Physical Metrics")
            display_metric_tables(offensive_metrics, "Offensive Metrics")
            display_metric_tables(defensive_metrics, "Defensive Metrics")

    # Glossary section now placed below the metrics tables
    with col2:
        with st.expander("Glossary"):
            for metric, explanation in glossary.items():
                st.markdown(f"**{metric}:** {explanation}")
