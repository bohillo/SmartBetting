from utils import read_yaml
from parsing_utils import parse_games
from analysis_utils import prepare_games_df, analyze_games_df

import requests
from bs4 import BeautifulSoup
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

config = read_yaml('config.yml')

# Reading configuration
URL = config['APP']['ODDS_URL']
POINTS_PER_EXACT = config['RULES']['POINTS_PER_EXACT']
POINTS_PER_GOAL_DIFF = config['RULES']['POINTS_PER_GOAL_DIFF']
POINTS_PER_WINNER = config['RULES']['POINTS_PER_WINNER']

# Requesting the odds page
r = requests.get(URL)

# Parsing the odds
soup = BeautifulSoup(r.content, 'html5lib')

game_nodes = soup.findAll('div', attrs={'class': 'market-with-header'})
parsed_games = parse_games(game_nodes)

# Analyzing the odds
# Transforming dicts to dataframe, cleaning, odds to probs
games_df = prepare_games_df(parsed_games)

# Cross joining all possible exact result bets with all possible outcomes and calculating potential points gained
analysis_df = analyze_games_df(games_df=games_df,
                               points_per_exact=POINTS_PER_EXACT,
                               points_per_goal_diff=POINTS_PER_GOAL_DIFF,
                               points_per_winner=POINTS_PER_WINNER)

# Calculating the EV of every possible bet for each game
results_EV_df = analysis_df.\
    groupby(['home', 'away', 'game_datetime', 'home_goals_bet', 'away_goals_bet'], as_index=False).\
    agg({'points_EV': ['sum']}).droplevel(axis=1, level=1)

# Ranking bets by EV
results_EV_df['bet_rank'] = results_EV_df.\
                                sort_values(['points_EV'], ascending=[False]).\
                                groupby(['home', 'away', 'game_datetime']).\
                                cumcount() + 1

# Per each game picking top 3 bets
best_bets = results_EV_df[results_EV_df['bet_rank'] == 1].\
    merge(results_EV_df[results_EV_df['bet_rank'] == 2], on=['home', 'away', 'game_datetime'], suffixes=['_1st', '_2nd']).\
    merge(results_EV_df[results_EV_df['bet_rank'] == 3], on=['home', 'away', 'game_datetime']).\
    sort_values('game_datetime', ascending=True)

# Printing the results
for _, row in best_bets.iterrows():
    print(f"""{row['home']} - {row['away']} 
                1st bet: {row['home_goals_bet_1st']}:{row['away_goals_bet_1st']}, EV: {row['points_EV_1st']:.3}
                2nd bet: {row['home_goals_bet_2nd']}:{row['away_goals_bet_2nd']}, EV: {row['points_EV_2nd']:.3}
                3rd bet: {row['home_goals_bet']}:{row['away_goals_bet']}, EV: {row['points_EV']:.3}
                """)

