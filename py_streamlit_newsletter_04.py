import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

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

# Load the dataset
file_path = 'data_newsletter.xlsx'  # Adjust this path if necessary
data = pd.read_excel(file_path)

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

# Calculate the timeframe from the Date column
data['Date'] = pd.to_datetime(data['Date'])
timeframe_start = data['Date'].min().strftime('%d-%m-%Y')
timeframe_end = data['Date'].max().strftime('%d-%m-%Y')
timeframe = f"{timeframe_start} - {timeframe_end}"

# Define the metrics for physical offensive score, physical defensive score, offensive score, defensive score, and goal threat score
physical_offensive_metrics = [
    'Distance', 'M/min', 'HSR Distance', 'HSR Count', 'Sprint Distance', 'Sprint Count',
    'HI Distance', 'HI Count', 'Medium Acceleration Count', 'High Acceleration Count',
    'Medium Deceleration Count', 'High Deceleration Count', 'PSV-99'
]

physical_defensive_metrics = [
    'Distance OTIP', 'M/min OTIP', 'HSR Distance OTIP', 'HSR Count OTIP', 'Sprint Distance OTIP',
    'Sprint Count OTIP', 'HI Distance OTIP', 'HI Count OTIP', 'Medium Acceleration Count OTIP',
    'High Acceleration Count OTIP', 'Medium Deceleration Count OTIP', 'High Deceleration Count OTIP'
]

offensive_metrics = [
    'PsAtt', 'PsCmp', 'Pass%', 'PsIntoA3rd', 'ProgPass', 'ThrghBalls', 'Touches', 'PsRec', 'ProgCarry', 'TakeOn', 'Success1v1'
]

defensive_metrics = [
    'TcklMade%', 'TcklAtt', 'Tckl', 'AdjTckl', 'TcklA3', 'Blocks', 'Int', 'AdjInt', 'Clrnce'
]

goal_threat_metrics = [
    'Goal', 'Shot', 'SOG', 'OnTarget%'
]

# Normalize and calculate the scores
scaler = MinMaxScaler(feature_range=(0, 10))
quantile_transformer = QuantileTransformer(output_distribution='uniform')

# Calculate physical offensive score
data['Physical Offensive Score'] = scaler.fit_transform(
    quantile_transformer.fit_transform(data[physical_offensive_metrics].fillna(0))
).mean(axis=1)

# Calculate physical defensive score
data['Physical Defensive Score'] = scaler.fit_transform(
    quantile_transformer.fit_transform(data[physical_defensive_metrics].fillna(0))
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
# Define the weights for each metric in Goal Threat Score
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

# Calculate the weighted average for the Goal Threat score
data['Goal Threat'] = (
    normalized_goal +
    normalized_shot +
    normalized_sog +
    normalized_ontarget
).mean(axis=1) / sum(weights.values())

# Score List
scores = ['Offensive Score', 'Defensive Score', 'Goal Threat', 'Physical Offensive Score', 'Physical Defensive Score']

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

    # Display the logo above the headline
    st.image('FCBayern-Wortmarke-SF-ANSICHT.png', use_column_width=False, width=800)
    
    # Main dashboard
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'FCBayernSans-CondSemiBold';
            src: url('/path/to/FCBayernSans-CondSemiBold.otf') format('opentype');
        }}
        .headline {{
            font-family: 'FCBayernSans-CondSemiBold', sans-serif;
            font-size: 3em;
            text-align: center;
        }}
        .tooltip {{
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted black;
        }}
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 120px;
            background-color: black;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 0;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position the tooltip above the text */
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        </style>
        <div class="headline">SCOUTING NEWSLETTER <br> {timeframe}</div>
        """,
        unsafe_allow_html=True
    )

    # Define columns for layout
    col1, col2 = st.columns([1, 3])

    # Glossary section in the first column
    with col1:
        with st.expander("Glossary"):
            for metric, explanation in glossary.items():
                st.markdown(f"**{metric}:** {explanation}")

    # Metrics tables in the second column
    with col2:
        # Filter for league
        leagues = data['Competition'].unique()
        selected_league = st.selectbox("Select League", leagues)
        league_data = data[data['Competition'] == selected_league]

        # Filter for position group
        position_group_options = list(position_groups.keys())
        selected_position_group = st.selectbox("Select Position Group", position_group_options)
        league_and_position_data = league_data[league_data['Position Groups'].apply(lambda groups: selected_position_group in groups)]

        # Metrics of interest
        metrics = ['PSV-99'] + physical_metrics + [
            'Take on into the Box', 'TouchOpBox', 'KeyPass', '2ndAst', 'xA +/-', 'MinPerChnc', 
            'PsAtt', 'PsCmp', 'PsIntoA3rd', 'ProgPass', 'ThrghBalls', 'Touches', 'PsRec', 
            'ProgCarry', 'TakeOn', 'Success1v1', 'TcklAtt', 'Tckl', 'AdjTckl', 'TcklA3', 
            'Blocks', 'Int', 'AdjInt', 'Clrnce', 'Goal', 'Shot/Goal', 'MinPerGoal', 'GoalExPn', 
            'ExpG', 'xGOT', 'ExpGExPn', 'xG +/-', 'Shot', 'SOG', 'Shot conversion', 'Ast', 'xA',
            'OnTarget%', 'TcklMade%', 'Pass%'
        ]

        # Combine scores and metrics
        all_metrics = scores + metrics

        # Add tooltip attributes to the table headers
        tooltip_headers = {metric: glossary.get(metric, '') for metric in all_metrics}

        def display_metric_tables(metrics_list, title):
            with st.expander(title):
                for metric in metrics_list:
                    # Ensure the metric column is numeric
                    league_and_position_data[metric] = pd.to_numeric(league_and_position_data[metric], errors='coerce')

                    # Check if the necessary columns exist
                    required_columns = ['Player_y', 'Age', 'Team_y', 'Position_y', metric]
                    missing_columns = [col for col in required_columns if col not in league_and_position_data.columns]
                    if missing_columns:
                        st.error(f"Missing columns in the dataset: {', '.join(missing_columns)}")
                        continue

                    # Drop rows with NaN values in the current metric
                    top10 = league_and_position_data[['Player_y', 'Age', 'Team_y', 'Position_y', metric]].dropna(subset=[metric]).sort_values(by=metric, ascending=False).head(10)

                    # Check if there are any rows after dropping NaNs
                    if top10.empty:
                        st.header(f"Top 10 Players in {metric}")
                        st.write("No data available")
                    else:
                        st.markdown(f"<h2>{metric}</h2>", unsafe_allow_html=True)
                        top10.rename(columns={'Player_y': 'Player', 'Team_y': 'Team', 'Position_y': 'Position'}, inplace=True)
                        top10[metric] = top10[metric].apply(lambda x: f"{x:.2f}")  # Format the values to two decimals

                        # Create HTML table with tooltips for headers and conditional formatting for U24
                        def color_row(row):
                            return ['background-color: #d4edda' if row['Age'] < 24 else '' for _ in row]

                        top10_styled = top10.style.apply(color_row, axis=1)
                        top10_html = top10_styled.to_html()

                        # Add tooltips to the headers using HTML and CSS
                        for header, tooltip in tooltip_headers.items():
                            if tooltip:
                                top10_html = top10_html.replace(f'>{header}<', f'><span class="tooltip">{header}<span class="tooltiptext">{tooltip}</span></span><')

                        # Display the styled DataFrame with tooltips
                        st.write(top10_html, unsafe_allow_html=True)

                        # Add a download button
                        buffer = BytesIO()
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.axis('tight')
                        ax.axis('off')
                        ax.table(cellText=top10.values, colLabels=top10.columns, cellLoc='center', loc='center')
                        plt.savefig(buffer, format='png')
                        buffer.seek(0)
                        image = Image.open(buffer)

                        st.download_button(
                            label="Download Table as PNG",
                            data=buffer,
                            file_name=f"{metric}_top10.png",
                            mime="image/png"
                        )
                        plt.close(fig)

        # Create collapsible sections for each metric category
        display_metric_tables(scores, "Score Metrics")
        display_metric_tables(physical_metrics, "Physical Metrics")
        display_metric_tables(offensive_metrics, "Offensive Metrics")
        display_metric_tables(defensive_metrics, "Defensive Metrics")
