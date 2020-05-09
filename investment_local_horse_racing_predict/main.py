import os
from datetime import datetime
import pandas as pd
import numpy as np
import math
import pickle
import urllib.request
import json
import base64

from investment_local_horse_racing_predict.app_logging import get_logger
from investment_local_horse_racing_predict import flask


logger = get_logger(__name__)


def predict(race_id, asset, vote_cost_limit):
    logger.info(f"#predict: start: race_id={race_id}, asset={asset}, vote_cost_limit={vote_cost_limit}")

    df = join_crawled_data(race_id)
    df = calc_horse_jockey_trainer_score(df)
    df = merge_past_race(df)
    df, df_data, df_query, df_label = split_data_query_label(df, race_id)
    df_result, predict_algorithm = predict_result(df, df_data)
    horse_number = df_result.query("pred_result==1")["horse_number"].values[0]
    vote_cost, vote_parameters = calc_vote_cost(asset, vote_cost_limit, race_id, horse_number)
    odds_win = vote_parameters["parameters"]["odds_win"]
    vote_parameters["predict_algorithm"] = predict_algorithm

    result = {
        "race_id": race_id,
        "horse_number": int(horse_number),
        "vote_cost": vote_cost,
        "odds_win": odds_win,
        "parameters": vote_parameters}
    logger.debug(f"#predict: result={result}")

    return result


def join_crawled_data(race_id):
    logger.info(f"#join_crawled_data: start: race_id={race_id}")

    logger.debug("#join_crawled_data: read sql")

    with flask.get_crawler_db() as db_conn:
        sql = f"select start_datetime from race_info where race_id = '{race_id}'"

        df = pd.read_sql(sql=sql, con=db_conn)
        end_date = df["start_datetime"].values[0]
        start_date = end_date - np.timedelta64(365, "D")
        logger.debug(f"#join_crawled_data: start_date={start_date}, end_date={end_date}")

        sql = f"""select
            d.race_id,
            d.bracket_number,
            d.horse_number,
            d.horse_id,
            d.horse_weight,
            d.horse_weight_diff,
            d.trainer_id,
            d.jockey_id,
            d.jockey_weight,
            d.favorite,
            i.race_round,
            i.place_name,
            i.start_datetime,
            i.course_type,
            i.course_length,
            i.weather,
            i.moisture,
            r.result,
            r.arrival_time,
            o.odds_win,
            h.gender as gender_horse,
            h.birthday as birthday_horse,
            h.coat_color,
            j.birthday as birthday_jockey,
            j.gender as gender_jockey,
            j.first_licensing_year as first_licensing_year_jockey,
            t.birthday as birthday_trainer,
            t.gender as gender_trainer
        from
            race_denma as d
            inner join race_info as i on
                d.race_id = i.race_id
                and i.start_datetime >= '{start_date}'
                and i.start_datetime <= '{end_date}'
            left join race_result as r on
                d.race_id = r.race_id
                and d.horse_number = r.horse_number
            left join odds_win as o on
                d.race_id = o.race_id
                and d.horse_number = o.horse_number
            left join horse as h on d.horse_id = h.horse_id
            left join jockey as j on d.jockey_id = j.jockey_id
            left join trainer as t on d.trainer_id = t.trainer_id
        order by i.start_datetime, d.horse_number"""

        df = pd.read_sql(sql=sql, con=db_conn)

    logger.debug("#join_crawled_data: replace categorical")

    df.replace({"place_name": {
        "金沢.*": 1,
        "高知.*": 2,
        "門別.*": 3,
        "盛岡.*": 4,
        "水沢.*": 5,
        "園田.*": 6,
        "笠松.*": 7,
        "名古屋.*": 8,
        "佐賀.*": 9,
        "姫路.*": 10,
        "帯広.*": 11,
        "福山.*": 12,
        "荒尾.*": 13,
        "札幌.*": 14,
        "旭川.*": 15,
        "北見.*": 16,
        "岩見沢.*": 17,
    }}, regex=True, inplace=True)

    df.replace({
        "weather": {'晴れ': 1, '雨': 2, '小雨': 2, 'くもり': 3, '雪': 4, 'かみなり': 5},
        "course_type": {'ダ': 1, '芝': 2},
        "coat_color": {'栗毛': 1, '鹿毛': 2, '青鹿毛': 3, '黒鹿毛': 4, '芦毛': 5, '栃栗毛': 6, '青毛': 7, '青駁毛': 8, '鹿駁毛': 9, '白毛': 10, '鹿粕毛': 11, '栗駁毛': 12, '栗粕毛': 13, '駁栗毛': 14},
        "gender_horse": {'牡': 1, '牝': 2, 'セン': 3},
        "gender_jockey": {'男': 1, '女': 2},
        "gender_trainer": {'男': 1, '女': 2},
    }, inplace=True)

    logger.debug("#join_crawled_data: fillna")

    df.fillna({
        "result": 17,
        "arrival_time": 599.0,
        "horse_weight": 1999.0,
        "favorite": 17,
        "moisture": -1,
        "odds_win": 999.9,
        "birthday_horse": datetime(1900, 1, 1),
        "birthday_jockey": datetime(1900, 1, 1),
        "birthday_trainer": datetime(1900, 1, 1)
    }, inplace=True)
    df.fillna(0, inplace=True)

    logger.debug("#join_crawled_data: calc")

    df["speed"] = df["course_length"] / df["arrival_time"]
    df["birth_age_horse"] = (df["start_datetime"] - df["birthday_horse"]) / np.timedelta64(1, "D")
    df["birth_age_jockey"] = (df["start_datetime"] - df["birthday_jockey"]) / np.timedelta64(1, "D")
    df["birth_age_trainer"] = (df["start_datetime"] - df["birthday_trainer"]) / np.timedelta64(1, "D")
    df["licensing_age_jockey"] = df["start_datetime"].dt.year - df["first_licensing_year_jockey"]

    df.drop(["birthday_horse", "birthday_jockey", "birthday_trainer", "first_licensing_year_jockey"], axis=1, inplace=True)

    return df


def calc_horse_jockey_trainer_score(df):
    logger.info("#calc_horse_jockey_trainer_score: start")

    df_score = df[["race_id", "start_datetime", "horse_id", "jockey_id", "trainer_id", "result"]]
    df_score["score"] = 1 / df_score["result"]

    logger.debug("#calc_horse_jockey_trainer_score: calc horse score")

    df_score_horse = df_score[["race_id", "start_datetime", "horse_id", "score"]].sort_values(["horse_id", "start_datetime"])
    df_score_tmp = df_score_horse.groupby("horse_id").rolling(10000, min_periods=1)[["score"]].sum()
    df_score_horse["total_score"] = df_score_tmp["score"].values
    df_score_tmp = df_score_horse.shift(1)
    df_score_horse["horse_id_1"] = df_score_tmp["horse_id"].values
    df_score_horse["total_score"] = df_score_tmp["total_score"].values
    df_score_horse.loc[df_score_horse["horse_id"] != df_score_horse["horse_id_1"], "total_score"] = 0.0
    df_score_horse.drop(["horse_id_1"], axis=1, inplace=True)

    logger.debug("#calc_horse_jockey_trainer_score: calc jockey score")

    df_score_jockey = df_score[["race_id", "start_datetime", "jockey_id", "score"]].sort_values(["jockey_id", "start_datetime"])
    df_score_tmp = df_score_jockey.groupby("jockey_id").rolling(10000, min_periods=1)[["score"]].sum()
    df_score_jockey["total_score"] = df_score_tmp["score"].values
    df_score_tmp = df_score_jockey.shift(1)
    df_score_jockey["jockey_id_1"] = df_score_tmp["jockey_id"].values
    df_score_jockey["total_score"] = df_score_tmp["total_score"].values
    df_score_jockey.loc[df_score_jockey["jockey_id"] != df_score_jockey["jockey_id_1"], "total_score"] = 0.0
    df_score_jockey.drop(["jockey_id_1"], axis=1, inplace=True)

    logger.debug("#calc_horse_jockey_trainer_score: calc trainer score")

    df_score_trainer = df_score[["race_id", "start_datetime", "trainer_id", "score"]].sort_values(["trainer_id", "start_datetime"])
    df_score_tmp = df_score_trainer.groupby(["race_id", "trainer_id"])[["score"]].sum()
    df_score_trainer = pd.merge(df_score_trainer[["race_id", "start_datetime", "trainer_id"]], df_score_tmp, on=["race_id", "trainer_id"], how="left")
    df_score_tmp = df_score_trainer.groupby("trainer_id").rolling(10000, min_periods=1)[["score"]].sum()
    df_score_trainer["total_score"] = df_score_tmp["score"].values
    df_score_tmp = df_score_trainer.shift(1)
    df_score_trainer["trainer_id_1"] = df_score_tmp["trainer_id"].values
    df_score_trainer["total_score"] = df_score_tmp["total_score"].values
    df_score_trainer.loc[df_score_trainer["trainer_id"] != df_score_trainer["trainer_id_1"], "total_score"] = 0.0
    df_score_trainer.drop(["trainer_id_1"], axis=1, inplace=True)

    df_tmp = df.copy()
    df_tmp = pd.merge(df_tmp, df_score_horse[["race_id", "horse_id", "total_score"]], on=["race_id", "horse_id"], how="left")
    df_tmp.rename(columns={"total_score": "horse_score"}, inplace=True)
    df_tmp = pd.merge(df_tmp, df_score_jockey[["race_id", "jockey_id", "total_score"]], on=["race_id", "jockey_id"], how="left")
    df_tmp.rename(columns={"total_score": "jockey_score"}, inplace=True)
    df_tmp = pd.merge(df_tmp, df_score_trainer[["race_id", "trainer_id", "total_score"]], on=["race_id", "trainer_id"], how="left")
    df_tmp.rename(columns={"total_score": "trainer_score"}, inplace=True)

    return df_tmp


def merge_past_race(df):
    logger.info("#merge_past_race: start")

    df_all = df.copy()

    logger.debug("#merge_past_race: merge")

    for shift_i in range(1, 4):
        df_all = pd.merge(df_all, df.shift(shift_i), left_index=True, right_index=True, suffixes=("", f"_{shift_i}"))

    logger.debug("#merge_past_race: set none")

    for shift_i in range(1, 4):
        for col in df_all.columns:
            if col.endswith(f"_{shift_i}"):
                df_all.loc[df_all["horse_id"] != df_all[f"horse_id_{shift_i}"], col] = None

    logger.debug("#merge_past_race: drop")

    for shift_i in range(1, 4):
        df_all.drop([f"race_id_{shift_i}",
                     f"horse_id_{shift_i}",
                     f"jockey_id_{shift_i}",
                     f"trainer_id_{shift_i}",
                     f"start_datetime_{shift_i}"
                     ], axis=1, inplace=True)

    logger.debug("#merge_past_race: fillna")

    df_all.fillna({
        "result": 17,
        "horse_weight": 1999,
        "arrival_time": 599,
        "jockey_weight": 99,
        "favorite": 17,
        "moisture": -1,
        "course_length": 6000,
        "odds_win": 999,
        "speed": 10,
    }, inplace=True)

    for shift_i in range(1, 4):
        df_all.fillna({
            f"result_{shift_i}": 17,
            f"horse_weight_{shift_i}": 1999,
            f"arrival_time_{shift_i}": 599,
            f"jockey_weight_{shift_i}": 99,
            f"favorite_{shift_i}": 17,
            f"moisture_{shift_i}": -1,
            f"course_length_{shift_i}": 6000,
            f"odds_win_{shift_i}": 999,
            f"speed_{shift_i}": 10,
        }, inplace=True)

    df_all.fillna(0, inplace=True)

    return df_all


def split_data_query_label(df, race_id):
    logger.info(f"#split_data_query_label: start: race_id={race_id}")

    df_tmp = df.drop([
        "favorite",
        "odds_win",
        "arrival_time", "speed",
        "horse_id", "jockey_id", "trainer_id",
    ], axis=1)

    df_tmp = df_tmp.query(f"race_id=='{race_id}'")
    df_tmp.sort_values(["race_id", "horse_number"], inplace=True)

    df_tmp_label = df_tmp[["result"]]
    df_tmp_label["label"] = df_tmp_label["result"].apply(lambda r: 3 if r == 1 else (2 if r == 2 else (1 if r == 3 else 0)))
    df_tmp_query = pd.DataFrame(df_tmp.groupby("race_id").size())
    df_tmp_data = df_tmp.drop(["race_id", "start_datetime", "result"], axis=1)

    return df_tmp, df_tmp_data, df_tmp_query, df_tmp_label


def predict_result(df, df_data):
    logger.info("#predict_result: start")

    logger.debug("#predict_result: load model")

    model_data = load_json_from_url(os.getenv("RESULT_PREDICT_MODEL_URL"))
    lgb_model = pickle.loads(base64.b64decode(model_data["model"].encode()))
    logger.debug(f"#predict_result: algorithm={model_data['algorithm']}")

    logger.debug("#predict_result: predict")

    df_tmp = df[["race_id", "horse_number", "start_datetime", "race_round"]]
    df_tmp["pred"] = lgb_model.predict(df_data, num_iteration=lgb_model.best_iteration)

    for rank, index in enumerate(df_tmp.sort_values("pred", ascending=False).index):
        df_tmp.at[index, "pred_result"] = rank + 1

    logger.debug(f"#predict_result: result=1 record is {df_tmp.query('pred_result==1')}")

    return df_tmp, model_data["algorithm"]


def calc_vote_cost(asset, vote_cost_limit, race_id, horse_number):
    logger.info(f"#calc_vote_cost: start: asset={asset}, vote_cost_limit={vote_cost_limit}, race_id={race_id}, horse_number={horse_number}")

    logger.debug("#calc_vote_cost: load parameters")

    vote_parameters = load_json_from_url(os.getenv("VOTE_PREDICT_MODEL_URL"))
    logger.debug(f"#calc_vote_cost: vote_parameters={vote_parameters}")

    hit_rate = vote_parameters["parameters"]["hit_rate"]
    kelly_coefficient = vote_parameters["parameters"]["kelly_coefficient"]

    logger.debug("#calc_vote_cost: load odds")

    with flask.get_crawler_db() as db_conn:
        sql = f"select odds_win from odds_win where race_id = '{race_id}' and horse_number = '{horse_number}'"

        df = pd.read_sql(sql=sql, con=db_conn)
        odds_win = df["odds_win"].values[0]
        logger.debug(f"#calc_vote_cost: odds_win={odds_win}")

    logger.debug("#calc_vote_cost: calc")

    if odds_win > 1.0:
        kelly = (hit_rate * odds_win - 1.0) / (odds_win - 1.0)
    else:
        kelly = 0.0

    if kelly > 0.0:
        vote_cost = math.floor(asset * kelly * kelly_coefficient / 100.0) * 100
        if vote_cost > vote_cost_limit:
            vote_cost = vote_cost_limit
    else:
        vote_cost = 0

    vote_parameters["parameters"]["odds_win"] = odds_win
    vote_parameters["parameters"]["kelly"] = kelly

    logger.debug(f"#calc_vote_cost: vote_cost={vote_cost}, vote_parameters={vote_parameters}")

    return vote_cost, vote_parameters


def load_json_from_url(url):
    with urllib.request.urlopen(url) as response:
        data = json.load(response)

    return data
