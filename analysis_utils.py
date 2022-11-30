import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import shin  # https://github.com/mberk/shin
from scipy.stats import poisson


class OddsProbsConverter(ABC):

    @abstractmethod
    def odds_to_probs(self, odds: list) -> list:
        pass


class NaiveOddsProbsConverter(OddsProbsConverter):

    def odds_to_probs(self, odds: list) -> list:
        inv_odds = [1 / x for x in odds]
        sum_inv_odds = sum(inv_odds)
        return [x / sum_inv_odds for x in inv_odds]


class ShinOddsProbsConverter(OddsProbsConverter):

    def odds_to_probs(self, odds: list) -> list:
        odds_list = [x for x in odds]  # if pd.Series is passed
        return shin.calculate_implied_probabilities(odds_list)['implied_probabilities']


def _add_basic_cols_to_games_df(df: pd.DataFrame):
    df['goals_diff'] = df['home_goals'] - df['away_goals']
    df['num_goals'] = df['home_goals'] + df['away_goals']
    df['winner'] = np.where(df['goals_diff'] > 0, 'home',
                            np.where(df['goals_diff'] < 0, 'away',
                                     np.where(df['home_goals'] >= 0, 'draw',
                                              None)))
    df['is_draw'] = np.where(df['winner'] == 'draw', True, False)
    return df

def prepare_games_df(parsed_games: list,
                     odds_converter: OddsProbsConverter = NaiveOddsProbsConverter(),
                     apply_ot_adj: bool = False) -> pd.DataFrame:
    """
    Cleans the data frame with odds by removing negative odds values and adds additional columns for convenience
    e.g. difference in home/away goals scored, winner (i.e. home, away or draw), probability.

    :param parsed_games: list of parsed odds - one item = one exact result of one game (e.g. Poland-Mexico 1:0)
    :param odds_converter: Converter from odds to probabilities (naive or Shin)
    :param apply_ot_adj: should probabilities be adjusted to reflect the exact result after potential overtime (30 minutes OT)

    :return: data frame with odds and additional columns - one row = one exact result of one game (e.g. Poland-Mexico 1:0)
    """
    df = pd.DataFrame(parsed_games)
    df = df[df['odds'] >= 0.0]
    df = _add_basic_cols_to_games_df(df)
    df['prob'] = df.groupby(['home', 'away']).odds.transform(odds_converter.odds_to_probs)

    if apply_ot_adj:
        # Distribution of the total number of goals after 90min full time
        df_num_goals = df.groupby(['home', 'away', 'num_goals'], as_index=False). \
            agg({'prob': ['sum']}). \
            droplevel(axis=1, level=1)
        # Maximum number of goals - to overwrite artificial negative # goals for 'other' final result with max + 1
        df_max_num_goals = df_num_goals.groupby(['home', 'away'], as_index=False). \
            agg({'num_goals': ['max']}). \
            droplevel(axis=1, level=1)
        df_num_goals = df_num_goals.merge(df_max_num_goals, on=['home', 'away'], suffixes=['', '_max'])
        df_num_goals['num_goals'] = np.where(df_num_goals['num_goals'] < 0,
                                             df_num_goals['num_goals_max'] + 1,
                                             df_num_goals['num_goals'])

        # Calculating avg total number of goals in 90 min FT per each game
        # i.e. lambda parameter in Poisson distribution fitted to actual total goals in 90mins distribution
        df_num_goals['ev_num_goals'] = df_num_goals['num_goals'] * df_num_goals['prob']
        df_ev_num_goals = df_num_goals.groupby(['home', 'away'], as_index=False). \
            agg({'ev_num_goals': ['sum']}). \
            droplevel(axis=1, level=1)
        df_num_goals = df_num_goals.merge(df_ev_num_goals, on=['home', 'away'], suffixes=['', '_ft'])

        # Modeling # of goals in overtime with Poisson distribution.
        # Lambda parameter for OT should be 1/3 of the lambda parameter for FT
        # (because 30min OT = 33% of 90min FT and we assume equal avg goals frequency in FT & OT)
        df_num_goals['ev_num_goals_ot'] = df_num_goals['ev_num_goals_ft'] / 3.0
        df_num_goals['poisson_prob_ft'] = poisson.pmf(df_num_goals['num_goals'], df_num_goals['ev_num_goals_ft'])
        df_num_goals['poisson_prob_ot'] = poisson.pmf(df_num_goals['num_goals'], df_num_goals['ev_num_goals_ot'])

        # Updating draw FT exact results with possible OT-only exact results.
        # (Duplication of rows since e.g. 1:0 might be reached after FT OR after 0:0 in FT and 1:0 in OT)
        df_ot = df.merge(df_num_goals, on=['home', 'away', 'num_goals'], how='left', suffixes=['', '_num_goals'])
        df_ot['prob'] = df_ot['prob'] * df_ot['poisson_prob_ot'] / df_ot['prob_num_goals']
        df_ot['is_draw'] = True
        df_ft_ot = df.merge(df_ot, on=['home', 'away', 'is_draw'], how='left', suffixes=['', '_ot'])
        df_ft_ot['home_goals_ft_ot'] = np.where(df_ft_ot['is_draw'],
                                                df_ft_ot['home_goals'] + df_ft_ot['home_goals_ot'],
                                                df_ft_ot['home_goals'])
        df_ft_ot['away_goals_ft_ot'] = np.where(df_ft_ot['is_draw'],
                                                df_ft_ot['away_goals'] + df_ft_ot['away_goals_ot'],
                                                df_ft_ot['away_goals'])

        df_ft_ot['prob_ft_ot'] = np.where(df_ft_ot['is_draw'],
                                          df_ft_ot['prob'] * df_ft_ot['prob_ot'],
                                          df_ft_ot['prob'])

        # Aggregating per exact game result after FT + potential OT (i.e. removing the above duplication)
        df_ft_ot_agg = df_ft_ot.groupby(['home', 'away', 'game_datetime', 'home_goals_ft_ot', 'away_goals_ft_ot'], as_index=False). \
            agg({'prob_ft_ot': ['sum']}). \
            droplevel(axis=1, level=1). \
            rename(columns={'home_goals_ft_ot': 'home_goals',
                            'away_goals_ft_ot': 'away_goals',
                            'prob_ft_ot': 'prob'})

        # Collecting the final games dataframe.
        # Note: more possible results will show up here than without OT adjustment (i.e. more rows)
        # e.g. 4:3 is not quoted by a bookie but it will appear as a consequence of 3:3 after 90 minutes + 1:0 in OT.
        df_res = df_ft_ot_agg.merge(df, on=['home', 'away', 'home_goals', 'away_goals'],
                                    how='left',
                                    suffixes=['', '_ft'])
        df_res = _add_basic_cols_to_games_df(df_res)[df.columns].astype({'home_goals': 'int32', 'away_goals': 'int32'})
    else:
        df_res = df

    return df_res


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
