import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import shin #https://github.com/mberk/shin


class OddsProbsConverter(ABC):

    @abstractmethod
    def odds_to_probs(self, odds: list) -> list:
        pass


class NaiveOddsProbsConverter(OddsProbsConverter):

    def odds_to_probs(self, odds: list) -> list:
        inv_odds = [1/x for x in odds]
        sum_inv_odds = sum(inv_odds)
        return [x / sum_inv_odds for x in inv_odds]


class ShinOddsProbsConverter(OddsProbsConverter):

    def odds_to_probs(self, odds: list) -> list:
        odds_list = [x for x in odds] # if pd.Series is passed
        return shin.calculate_implied_probabilities(odds_list)['implied_probabilities']


def prepare_games_df(parsed_games: list, odds_converter: OddsProbsConverter = NaiveOddsProbsConverter()) -> pd.DataFrame:
    """
    Cleans the data frame with odds by removing negative odds values and adds additional columns for convenience
    e.g. difference in home/away goals scored, winner (i.e. home, away or draw), probability.

    :param parsed_games: list of parsed odds - one item = one exact result of one game (e.g. Poland-Mexico 1:0)
    :return: data frame with odds and additional columns - one row = one exact result of one game (e.g. Poland-Mexico 1:0)
    """
    df = pd.DataFrame(parsed_games)
    df = df[df['odds'] >= 0.0]
    df['goals_diff'] = df['home_goals'] - df['away_goals']
    df['winner'] = np.where(df['goals_diff'] > 0, 'home',
                            np.where(df['goals_diff'] < 0, 'away',
                                     'draw'))
    df['prob'] = df.groupby(['home', 'away']).odds.transform(odds_converter.odds_to_probs)
    return df


def analyze_games_df(games_df: pd.DataFrame,
                     points_per_exact: float = 4.0,
                     points_per_goal_diff: float = 2.0,
                     points_per_winner: float = 1.0) -> pd.DataFrame:
    """
    For each game cross joins each possible exact result bet with each possible real outcome
    and calculates potential points gained per bet

    :param games_df: Pandas dataframe with one row per each game & exact result pair
    :param points_per_exact: Points for hitting the exact game result
    :param points_per_goal_diff: Points for hitting the exact goals difference only (home - away)
    :param points_per_winner: Points for hitting the game winner only
    :return: Pandas dataframe with every combination of a bet and real exact result of each game
    with assigned potential # points gained and its contribution to EV (# points x probability)
    """
    cross_df = games_df.merge(games_df, on=['home', 'away', 'game_datetime'], suffixes=['_bet', '_real'])

    exact_result_hit = (cross_df['home_goals_bet'] == cross_df['home_goals_real']) & \
                       (cross_df['away_goals_bet'] == cross_df['away_goals_real'])
    goal_diff_hit = cross_df['goals_diff_bet'] == cross_df['goals_diff_real']
    winner_hit = cross_df['winner_bet'] == cross_df['winner_real']

    cross_df['points'] = np.where(exact_result_hit, points_per_exact,
                                  np.where(goal_diff_hit, points_per_goal_diff,
                                           np.where(winner_hit, points_per_winner,
                                                    0.0)))

    cross_df['points_EV'] = cross_df['points'] * cross_df['prob_real']
    return cross_df
