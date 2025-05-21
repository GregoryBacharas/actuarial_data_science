import numpy as np
import pandas as pd
from urllib3 import request
import json
from geopy import distance
from datetime import datetime, timedelta
from io import StringIO


def convert_LV95_to_WGS84(easting, northing, altitute = None):
    if altitute:
        resp = request("GET",
                       "http://geodesy.geo.admin.ch/reframe/lv95towgs84?easting=" + str(easting) + "&northing=" + str(northing) + "&altitude=" + str(altitute) + "&format=json")
    else:
        resp = request("GET",
                       "http://geodesy.geo.admin.ch/reframe/lv95towgs84?easting=" + str(easting) + "&northing=" + str(northing) + "&format=json")
    location_point = json.loads(resp.data.decode('utf-8')[:-1])
    location_point['northing'] = round(float(location_point['northing']), 6)
    location_point['easting'] = round(float(location_point['easting']), 6)
    if altitute:
        location_point['altitude'] = round(float(location_point['altitude']), 2)
    else:
        location_point['altitude'] = ''
    return location_point
    # return str(location_point['northing']) + ', ' + str(location_point['easting']) + ', ' + str(location_point['altitude'])


def save_weather_data(coordinates, start_date, end_date, folder_name, filename):
    resp = request("GET",
                   "http://archive-api.open-meteo.com/v1/archive?latitude=" +
                   str(coordinates['northing']) +
                   "&longitude=" +
                   str(coordinates['easting']) +
                   "&start_date=" +
                   start_date +
                   "&end_date=" +
                   end_date +
                   "&daily=wind_direction_10m_dominant,precipitation_sum,rain_sum,snowfall_sum&hourly=temperature_2m,snowfall,rain,snow_depth,precipitation,wind_speed_10m,wind_speed_100m,wind_direction_10m,wind_direction_100m,wind_gusts_10m,sunshine_duration,cloud_cover&models=cerra,best_match&timezone=auto&elevation=" +
                   str(coordinates['altitude']) +
                   "&format=csv")
    weather_data = open(folder_name + "/" + filename + ".csv", "w")
    weather_data.write(resp.data.decode('utf-8'))
    weather_data.close()


def date_and_time_of_observation(raw_daytime_value):
    if '.' in raw_daytime_value:
        raw_date, raw_time = raw_daytime_value.split(' ')
        raw_day, raw_month, raw_year = raw_date.split('.')
        raw_hour, raw_minute = raw_time.split(':')
        return datetime(int(raw_year), int(raw_month), int(raw_day), int(raw_hour), int(raw_minute), 0)
    return datetime.fromisoformat(raw_daytime_value.replace(' ', 'T'))

# save_weather_data(convert_LV95_to_WGS84(2790000, 1190070, 2450), '2010-01-01', '2010-01-05', 'weather_data_instability', 'test')

def download_weather_data():
    for stability_measurement in snow_instability.itertuples():
        measurement_coordinates = convert_LV95_to_WGS84(int(2000000 + stability_measurement.X_Coordinate),
                                                        int(1000000 + stability_measurement.Y_Coordinate),
                                                        int(stability_measurement.Elevation))
        save_weather_data(measurement_coordinates,
                          str((date_and_time_of_observation(stability_measurement.Date_time) - timedelta(
                              days=14)).date()),
                          str(date_and_time_of_observation(stability_measurement.Date_time).date()),
                          'weather_data_instability',
                          'No' + str(int(stability_measurement.No)))

    for avalanche_accident in avalanche_accidents.itertuples():
        accident_coordinates = convert_LV95_to_WGS84(int(2000000 + avalanche_accident.start_zone_coordinates_x),
                                                     int(1000000 + avalanche_accident.start_zone_coordinates_y),
                                                     int(avalanche_accident.start_zone_elevation))
        save_weather_data(accident_coordinates,
                          str((datetime.fromisoformat(avalanche_accident.date) - timedelta(days=14)).date()),
                          avalanche_accident.date,
                          'weather_data_avalanches',
                          'ID' + str(int(avalanche_accident.avalanche_id)))


if __name__ == '__main__':
    snow_instability = pd.read_csv("snow_instability_field_data.csv", sep = ";")
    snow_instability = snow_instability[:-10]
    avalanche_accidents = pd.read_csv("avalanche_accidents_switzerland_since_1995.csv", sep = ",", encoding="ISO-8859-1")


    cor_matrix = snow_instability[['Profile_class', 'five_class_Stability', 'RB_score', 'RB_release_type', 'Fracture_plane_quality', 'S2008_1_RB', 'S2008_2_RT', 'S2008_3_Lemons', 'three_class_Stability', 'four_class_Stability_Techel', 'RB_height_cm', 'Snow_depth_cm', 'Slab_thickness_cm', 'FL_Thickness_cm', 'AL_Thickness_cm', 'FL_Grain_size_avg_mm', 'AL_Grain_size_avg_mm', 'FL_Grain_size_max_mm', 'AL_Grain_size_max_mm', 'FL_Grain_type1', 'FL_Grain_type2', 'FL_Hardness', 'FL_Top_Height_cm', 'FL_Bottom_Height_cm', 'AL_Top_Height_cm', 'AL_Bottom_Height_cm', 'AL_Hardness', 'Hard_Diff', 'Abs_Hard_Diff', 'Grain_Size_Diff_mm', 'FL_location', 'Lemon1_E', 'Lemon2_R', 'Lemon3_F', 'Lemon4_dE', 'Lemon5_dR', 'Lemon6_FLD', 'Lemons_FL', 'Whumpfs', 'Cracks', 'Avalanche_activity', 'LN_Local_danger_level_nowcast', 'LN_rounded', 'RF_Regional _danger_level_forecast', 'Deviation_LN_RF', 'SNPK_Index', 'SNPK_Index_Class']].corr()
    print(cor_matrix)
    cor_matrix.to_csv('correlation_matrix.csv', sep = ',')

    snow_instability.describe()

