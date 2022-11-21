from utils import safe_cast
import bs4
from datetime import datetime


def parse_single_odds(single_odds_node: bs4.element.Tag) -> dict:
    """
    Parses a html node with odds for a single result (e.g. 1:0 6.90)

    :param single_odds_node: A html node with one exact result and its odds
    :return: e.g. {'home_goals': 1, 'away_goals': 0, 'odds': 6.9}
    """
    odds_node = single_odds_node.find('span', attrs={'class': 'odds-value'})
    result_node = single_odds_node.find('span', attrs={'class': 'odds-name'})
    odds = float(odds_node.contents[0])
    home_away_goals = [safe_cast(x.strip(), int, -1) for x in result_node.contents[0].strip().split(':')]
    if len(home_away_goals) == 2:
        home_goals, away_goals = home_away_goals
    else:
        home_goals = -1
        away_goals = -1

    return {'home_goals': home_goals, 'away_goals': away_goals, 'odds': odds}


def parse_odds(odds_node: bs4.element.Tag) -> list:
    """
    Parses multiple exact results & odds for a single game
    :param odds_node: A html node with nested nodes with individual exact results and its odds
    :return: A list of dictionaries (single item = result of parse_single_odds)
    """
    return [parse_single_odds(single_odds_node) for single_odds_node in odds_node.findAll('a')]


def parse_game(game_node: bs4.element.Tag) -> list:
    """
    Extracts all relevant info about a single game: home&away teams, date&time, all quoted exact game results with odds
    :param game_node: A html node with all the data about a single game
    :return: A list of dictionaries with one item per each exact result quoted
    """
    game_name_tag = game_node.find('a', attrs={'class': 'names'})
    game_name = game_name_tag.text.strip()
    home, away = [x.strip() for x in game_name.split('-')]

    game_datetime_node = game_node.find('span', attrs={'class': 'datetime'})
    datetime_str = game_datetime_node.contents[0].strip()
    datetime_str_2 = datetime_str.replace('. ', '.'+str(datetime.today().year)+' ')
    game_datetime = datetime.strptime(datetime_str_2, '%d.%m.%Y %H:%M')

    odds_node = game_node.find('div', attrs={'class': 'odds'})
    return [{'home': home, 'away': away, 'game_datetime': game_datetime, **single_odds} \
            for single_odds in parse_odds(odds_node)]


def parse_games(game_nodes: bs4.element.ResultSet) -> list:
    """
    Extracts all relevant info about multiple games: home&away teams, date&time, all quoted exact game results with odds
    :param game_nodes: Set of html nodes, one per each game
    :return: Flattened list with one item per each game&exact result pair
    """
    return [l for gn in game_nodes for l in parse_game(gn)]


