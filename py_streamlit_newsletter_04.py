import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

# Load the dataset
file_path = 'data/cleaned_merged_newsletter_skillcorner_v2.xlsx'  # Adjust this path if necessary
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

# Define the metrics for physical offensive score, physical defensive score, offensive score, and defensive score
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
    # Display the logo above the headline
    st.image('/Users/tnakip-babucke/Documents/Vorlagen/Logos/Vereinslogo/Logo_FCB.png', use_column_width=False, width=200)
    
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
            color: black;
        }}
        </style>
        <div class="headline">SCOUTING NEWSLETTER <br> {timeframe}</div>
        """,
        unsafe_allow_html=True
    )

    # Filter for league
    leagues = data['Competition'].unique()
    selected_league = st.selectbox("Select League", leagues)
    league_data = data[data['Competition'] == selected_league]

    # Filter for position group
    position_group_options = list(position_groups.keys())
    selected_position_group = st.selectbox("Select Position Group", position_group_options)
    league_and_position_data = league_data[league_data['Position Groups'].apply(lambda groups: selected_position_group in groups)]

    # Metrics of interest
    scores = ['Physical Offensive Score', 'Physical Defensive Score', 'Offensive Score', 'Defensive Score']
    metrics = ['PSV-99'] + physical_metrics + ['Take on into the Box', 'TouchOpBox', 'KeyPass', '2ndAst', 'xA +/-', 'MinPerChnc', 
                                               'PsAtt', 'PsCmp', 'PsIntoA3rd', 'ProgPass', 'ThrghBalls', 'Touches', 'PsRec', 
                                               'ProgCarry', 'TakeOn', 'Success1v1', 'TcklAtt', 'Tckl', 'AdjTckl', 'TcklA3', 
                                               'Blocks', 'Int', 'AdjInt', 'Clrnce', 'Goal', 'Shot/Goal', 'MinPerGoal', 'GoalExPn', 
                                               'ExpG', 'xGOT', 'ExpGExPn', 'xG +/-', 'Shot', 'SOG', 'Shot conversion', 'Ast', 'xA',
                                               'OnTarget%', 'TcklMade%', 'Pass%']

    # Combine scores and metrics
    all_metrics = scores + metrics

    for metric in all_metrics:
        # Ensure the metric column is numeric
        league_and_position_data[metric] = pd.to_numeric(league_and_position_data[metric], errors='coerce')

        # Drop rows with NaN values in the current metric
        top10 = league_and_position_data[['Player_y', 'Age', 'Team_y', 'Position_y', metric]].dropna(subset=[metric]).sort_values(by=metric, ascending=False).head(10)

        # Check if there are any rows after dropping NaNs
        if top10.empty:
            st.header(f"Top 10 Players in {metric}")
            st.write("No data available")
        else:
            st.header(f"Top 10 Players in {metric}")
            top10.rename(columns={'Player_y': 'Player', 'Team_y': 'Team', 'Position_y': 'Position'}, inplace=True)
            top10[metric] = top10[metric].apply(lambda x: f"{x:.2f}")  # Format the values to two decimals

            # Create HTML table with conditional formatting
            def color_row(row):
                if row['Age'] < 24:
                    return [f'background-color: #d4edda']*len(row)
                return ['']*len(row)

            top10_styled = top10.style.apply(color_row, axis=1)

            # Display the styled DataFrame
            st.write(top10_styled.to_html(), unsafe_allow_html=True)
