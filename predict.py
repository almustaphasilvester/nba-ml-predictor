import requests
import json
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, commonteamroster, playergamelog
from nba_api.stats.static import teams
import datetime
from time import sleep
import tensorflow as tf
import sys

def get_teams():

    team_dict = teams.get_teams()
    df_teams = pd.DataFrame(team_dict)

    return df_teams

def get_team_roster(id, season=datetime.date.today().year-1):

    team_roster = commonteamroster.CommonTeamRoster(team_id=id, season=season)
    df_team_roster = team_roster.get_data_frames()[0]
    return df_team_roster

def get_team_game_data(id, season=datetime.date.today().year-1):

    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=id, season_nullable=season)
    games = gamefinder.get_data_frames()[0]
    return games

def get_player_game_data(player, season=datetime.date.today().year-1):

    player_game_stats = playergamelog.PlayerGameLog(player_id=player, season=season)
    player_game_stats = player_game_stats.get_data_frames()[0]
    return player_game_stats

def get_player_data(df_roster):

    player_data_list = []

    for player in df_roster.PLAYER_ID:
        player_data_list.append(get_player_game_data(player))
        sleep(0.8)
    
    df_player_data = pd.concat(player_data_list).reset_index(drop=True)
    
    return df_player_data

def get_player_averages(df):

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    use_cols = ['PTS', 'FGM', 'FGA', 'FG_PCT',
       'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
       'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

    for col in use_cols:
        df[f"AVG_{col}"] = df.sort_values(["GAME_DATE"]).groupby(["Player_ID"])[col].expanding().mean().reset_index(0,drop=True)
    
    df = df.sort_values(["GAME_DATE"], ascending=False).groupby(["Player_ID"]).first()
    
    return df

def get_game_averages(df):

    use_cols = ['PTS', 'FGM', 'FGA', 'FG_PCT',
       'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
       'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

    for col in use_cols:
        df[f"AVG_{col}"] = df.sort_values(["GAME_DATE"]).groupby(["TEAM_ID"])[col].expanding().mean().reset_index(0,drop=True)

    df = df.sort_values(["GAME_DATE"], ascending=False).groupby(["TEAM_ID"]).first()
    
    return df

def format_player_data(df):

    df = df.drop(['SEASON_ID', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL','PLUS_MINUS', 'VIDEO_AVAILABLE'], axis=1)
    df = df.sort_values("AVG_PTS", ascending=False).head(10)

    return df
# Input Team Names
def predict(model, home_team, away_team):

    df_team = get_teams()

    try:
        home_team_id = df_team[df_team["full_name"].values == home_team].values[0]
        away_team_id = df_team[df_team["full_name"].values == away_team].values[0]
    except Exception as e:
        print(e)
        print("One of these teams are not found. Please check your spelling!")

    # Get Roster
    df_home_roster = get_team_roster(home_team_id)
    df_away_roster = get_team_roster(away_team_id)

    # Get Team Data
    df_home_data = get_team_game_data(home_team_id)
    df_away_data = get_team_game_data(away_team_id)

    # Get Player Data
    df_home_players = get_player_data(df_home_roster)
    df_away_players = get_player_data(df_away_roster)

    # Get Averages of Data
    df_home_data = get_game_averages(df_home_data)
    df_away_data = get_game_averages(df_away_data)
    df_home_players = get_player_averages(df_home_players)
    df_away_players = get_player_averages(df_away_players)


    # Grab Top 10 Players
    df_home_players = df_home_players.sort_values("AVG_PTS", ascending=False).head(10)
    df_away_players = df_away_players.sort_values("AVG_PTS", ascending=False).head(10)

    # Columns to Drop
    col_players = ['SEASON_ID', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'FGM',
       'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
       'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
       'PLUS_MINUS', 'VIDEO_AVAILABLE']

    col_teams = ['SEASON_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID',
       'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
       'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
       'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']

    # Reformat Data for Model
    df_home_players = df_home_players.drop(col_players, axis=1)
    df_away_players = df_away_players.drop(col_players, axis=1)
    df_home_team = df_home_data.drop(col_teams, axis=1)
    df_away_team = df_away_data.drop(col_teams, axis=1)

    home_teams = np.array([df_home_team])
    away_teams = np.array([df_away_team])
    home_players = np.array([df_home_players])
    away_players = np.array([df_away_players])

    result = model.predict([home_players, home_teams, away_players, away_teams])

    if result[0][0]>0.585:
        print(f"Winning Team: {home_team}; Losing Team: {away_team}")
    else:
        print(f"Winning Team: {away_team}; Losing Team: {home_team}")

if __name__ == "__main__":
    model = tf.keras.models.load_model("Models/model")
    predict(model, sys.argv[1],sys.argv[2])