import numpy as np
import pandas as pd
from urllib3 import request
import json
from geopy import distance
from datetime import datetime, timedelta
from io import StringIO
import cmath
import time
import statsmodels.formula.api as smf
import statsmodels.api as sm
from stat_model_diagnostics import LinearRegDiagnostic


def convert_LV95_to_WGS84(easting, northing, altitude = None):
    """
    Converts coordinates values from the swiss standard (LV95) to the global standard (WGS84).
    :param easting: Easting in Swiss coordinate system (always starts with 2).
    :param northing: Northing in Swiss coordinate system (always starts with 1).
    :param altitude: Elevation (m a.s.l.)
    :return: Dictionary with keys {'northing', 'easting', 'altitude'} for the values (as floats) in the global standard.
    """
    if altitude:
        resp = request("GET",
                       "http://geodesy.geo.admin.ch/reframe/lv95towgs84?easting=" + str(easting) + "&northing=" + str(northing) + "&altitude=" + str(altitude) + "&format=json")
    else:
        resp = request("GET",
                       "http://geodesy.geo.admin.ch/reframe/lv95towgs84?easting=" + str(easting) + "&northing=" + str(northing) + "&format=json")
    location_point = json.loads(resp.data.decode('utf-8')[:-1])
    location_point['northing'] = round(float(location_point['northing']), 6)
    location_point['easting'] = round(float(location_point['easting']), 6)
    if altitude:
        location_point['altitude'] = round(float(location_point['altitude']), 2)
    else:
        location_point['altitude'] = ''
    return location_point
    # return str(location_point['northing']) + ', ' + str(location_point['easting']) + ', ' + str(location_point['altitude'])


def save_weather_data(coordinates, start_date, end_date, folder_name, filename):
    """
    Downloads the weather data in a csv format.
    :param coordinates: Dictionary with keys {'northing', 'easting', 'altitude'} for the values (as floats) in the global standard (WGS84).
    :param start_date: String for the first day to be downloaded. Format: '%Y-%m-%d'.
    :param end_date: String for the last day to be downloaded. Format: '%Y-%m-%d'.
    :param folder_name: String with the folder path where the csv file will be saved.
    :param filename: String with the filename of the csv file (without extension).
    :return: None
    """
    resp = request("GET",
                   "http://archive-api.open-meteo.com/v1/archive?latitude=" +
                   str(coordinates['northing']) +
                   "&longitude=" +
                   str(coordinates['easting']) +
                   "&start_date=" +
                   start_date +
                   "&end_date=" +
                   end_date +
                   "&daily=wind_direction_10m_dominant,precipitation_sum,rain_sum,snowfall_sum&hourly=temperature_2m,snowfall,rain,snow_depth,precipitation,wind_speed_10m,wind_speed_100m,wind_direction_10m,wind_direction_100m,wind_gusts_10m,sunshine_duration,cloud_cover&models=cerra&timezone=auto&elevation=" +
                   str(coordinates['altitude']) +
                   "&format=csv")
    weather_data = open(folder_name + "/" + filename + ".csv", "w")
    weather_data.write(resp.data.decode('utf-8'))
    weather_data.close()


def date_and_time_of_observation(raw_daytime_value):
    """
    Reformats datetime string in the snow_instability dataset, in order to handle different formats used.
    :param raw_daytime_value: Datetime value from the snow_instability dataset.
    :return: Datetime object.
    """
    if '.' in raw_daytime_value:
        raw_date, raw_time = raw_daytime_value.split(' ')
        raw_day, raw_month, raw_year = raw_date.split('.')
        raw_hour, raw_minute = raw_time.split(':')
        return datetime(int(raw_year), int(raw_month), int(raw_day), int(raw_hour), int(raw_minute), 0)
    return datetime.fromisoformat(raw_daytime_value.replace(' ', 'T'))

# save_weather_data(convert_LV95_to_WGS84(2790000, 1190070, 2450), '2010-01-01', '2010-01-05', 'weather_data_instability', 'test')

def download_weather_data(instability, accidents):
    """
    Downloads all the weather data for the last 14 days before every observation.
    :param instability: Pandas dataframe, with the snow_instability dataset
    :param accidents: Pandas dataframe, with the avalanche_accidents dataset
    :return: Saves the weather data in a predefined folder structure.
    """
    for stability_measurement in instability.itertuples():
        measurement_coordinates = convert_LV95_to_WGS84(int(2000000 + stability_measurement.X_Coordinate),
                                                        int(1000000 + stability_measurement.Y_Coordinate),
                                                        int(stability_measurement.Elevation))
        if int(stability_measurement.No) in [100, 200, 300, 400, 500]:
            time.sleep(60) # Comply with API restrictions
        save_weather_data(measurement_coordinates,
                          str((date_and_time_of_observation(stability_measurement.Date_time) - timedelta(
                              days=14)).date()),
                          str(date_and_time_of_observation(stability_measurement.Date_time).date()),
                          'weather_data_instability',
                          'No' + str(int(stability_measurement.No)))

    for avalanche_accident in accidents.itertuples():
        accident_coordinates = convert_LV95_to_WGS84(int(2000000 + avalanche_accident.start_zone_coordinates_x),
                                                     int(1000000 + avalanche_accident.start_zone_coordinates_y),
                                                     int(avalanche_accident.start_zone_elevation))
        if avalanche_accident[0] in [100, 200, 300, 400, 500]:
            time.sleep(60) # Comply with API restrictions
        save_weather_data(accident_coordinates,
                          str((datetime.fromisoformat(avalanche_accident.date) - timedelta(days=14)).date()),
                          avalanche_accident.date,
                          'weather_data_avalanches',
                          'ID' + str(int(avalanche_accident.avalanche_id)))


def polar2complex(r, theta):
    return r * np.exp(1j * theta)

def weather_slice(weather_data_csv_path, measurement_window):
    """
    Returns a slice of the weather observations, to be used for predictor variables calculation.
    :param weather_data_csv_path: csv file with the weather data for the location we are interested in.
    :param measurement_window: Integer 1, 2, 3, or 4, that determined the measurement window: 0-24h, 72h-24h, 168h-72h, 336h-168h respectively (hours before observation).
    :return: Pandas dataframe with hourly observations within the specified measurement window.
    """
    with open(weather_data_csv_path) as weather_data_file:
        weather_hourly = pd.read_csv(StringIO(weather_data_file.read().split('\n\n')[1]), sep=',')
    measurement_date = datetime.strptime(weather_hourly.iloc[-1].time, '%Y-%m-%dT%H:%M')
    weather_hourly['time'] = pd.to_datetime(weather_hourly['time'])
    weather_hourly = weather_hourly.set_index('time')
    if measurement_window == 1:
        weather_in_window = weather_hourly.loc[(measurement_date - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'):]
    elif measurement_window == 2:
        weather_in_window = weather_hourly.loc[(measurement_date - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S'):(measurement_date - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S')]
    elif measurement_window == 3:
        weather_in_window = weather_hourly.loc[(measurement_date - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S'):(measurement_date - timedelta(days=4)).strftime('%Y-%m-%d %H:%M:%S')]
    elif measurement_window == 4:
        weather_in_window = weather_hourly.loc[:(measurement_date - timedelta(days=8)).strftime('%Y-%m-%d %H:%M:%S')]
    return weather_in_window


def snowfall_aspect_bias(weather_data_csv_path, measurement_window):
    """
    Sums up all the wind vectors (only during snowfall).
    :param weather_data_csv_path: csv file with the weather data for the location we are interested in.
    :param measurement_window: Integer 1, 2, 3, or 4, that determined the measurement window: 0-24h, 72h-24h, 168h-72h, 336h-168h respectively (hours before observation).
    :return: Complex number. To be interpreted as a vector. Determines the snowfall aspect bias.
    """
    wind_vector = 0j
    for _, row in weather_slice(weather_data_csv_path, measurement_window).iterrows():
        if row['precipitation (mm)'] != 0:
            wind_vector = wind_vector + cmath.rect(row['wind_speed_10m (km/h)'], -np.pi*row['wind_direction_10m (°)']/180 + np.pi/2) #mistake_ok
    return cmath.polar(wind_vector)


def accumulated_snow_calculation(weather_data_csv_path, measurement_window):
    """
    Sums up the snowfall that occurred during the specified weather window
    :param weather_data_csv_path: csv file with the weather data for the location we are interested in.
    :param measurement_window: Integer 1, 2, 3, or 4, that determined the measurement window: 0-24h, 72h-24h, 168h-72h, 336h-168h respectively (hours before observation).
    :return: Total snowfall in cm.
    """
    total_snowfall = 0
    for _, row in weather_slice(weather_data_csv_path, measurement_window).iterrows():
        if row['snowfall (cm)'] != 0:
            total_snowfall = total_snowfall + row['precipitation (mm)']
    return total_snowfall


def mean_temperature(weather_data_csv_path, measurement_window):
    """
    Calculates the mean temperature during the specified weather window.
    :param weather_data_csv_path: csv file with the weather data for the location we are interested in.
    :param measurement_window: Integer 1, 2, 3, or 4, that determined the measurement window: 0-24h, 72h-24h, 168h-72h, 336h-168h respectively (hours before observation).
    :return: Mean temperature in degrees Celsius.
    """
    return weather_slice(weather_data_csv_path, measurement_window)['temperature_2m (°C)'].mean()


def std_temperature(weather_data_csv_path, measurement_window):
    """
    Calculates the standard deviation in temperature during the specified weather window.
    :param weather_data_csv_path: csv file with the weather data for the location we are interested in.
    :param measurement_window: Integer 1, 2, 3, or 4, that determined the measurement window: 0-24h, 72h-24h, 168h-72h, 336h-168h respectively (hours before observation).
    :return: Standard deviation in temperature.
    """
    return weather_slice(weather_data_csv_path, measurement_window)['temperature_2m (°C)'].std()


def sunshine_percentage(weather_data_csv_path, measurement_window):
    """
    Calculates the sunshine duration during the specified weather window.
    :param weather_data_csv_path: csv file with the weather data for the location we are interested in.
    :param measurement_window: Integer 1, 2, 3, or 4, that determined the measurement window: 0-24h, 72h-24h, 168h-72h, 336h-168h respectively (hours before observation).
    :return: Sunshine duration as a percentage of the total time in the measurement window.
    """
    temperature_in_window = weather_slice(weather_data_csv_path, measurement_window)['sunshine_duration (s)']
    return temperature_in_window.sum()/(len(temperature_in_window) * 3600)


def create_df_for_instability_model(instability_df):
    """
    Builds a cleaned dataframe, as a copy of the original, with only the variables of interest for the models.
    :param instability_df: Pandas dataframe with the snow_instability dataset.
    :return: Pandas dataframe with all the variables necessary for the model.
    """
    # cleaned_data = instability_df[['No', 'Profile_ID', 'Date_time', 'Aspect', 'X_Coordinate', 'Y_Coordinate', 'Elevation', 'Slope_angle_degrees', 'RB_score', 'RB_release_type', 'RB_height_cm', 'FL_Grain_size_avg_mm', 'AL_Grain_size_avg_mm', 'SNPK_Index', 'HN24_cm', 'HN3d_cm']].copy()
    cleaned_data = instability_df.copy()
    aspect_radians = {
        'N': np.pi/2,
        'NNE': 3*np.pi/8,
        'NE': np.pi/4,
        'ENE': np.pi/8,
        'E': 0,
        'ESE': -np.pi/8,
        'SE': -np.pi/4,
        'SSE': -3*np.pi/8,
        'S': -np.pi/2,
        'SSW': -5*np.pi/8,
        'SW': -3*np.pi/4,
        'WSW': -7*np.pi/8,
        'W': np.pi,
        'WNW': 7*np.pi/8,
        'NW': 3*np.pi/4,
        'NNW': 5*np.pi/8
    }
    cleaned_data['Aspect'] = cleaned_data['Aspect'].map(aspect_radians)

    cleaned_data['Accumulated_Snow_1d'] = cleaned_data['No'].map(lambda x: accumulated_snow_calculation('weather_data_instability/No' + str(int(x)) + '.csv', 1))
    cleaned_data['Accumulated_Snow_3d'] = cleaned_data['No'].map(lambda x: accumulated_snow_calculation('weather_data_instability/No' + str(int(x)) + '.csv', 2))
    cleaned_data['Accumulated_Snow_7d'] = cleaned_data['No'].map(lambda x: accumulated_snow_calculation('weather_data_instability/No' + str(int(x)) + '.csv', 3))
    cleaned_data['Accumulated_Snow_14d'] = cleaned_data['No'].map(lambda x: accumulated_snow_calculation('weather_data_instability/No' + str(int(x)) + '.csv', 4))

    cleaned_data['Wind_Induced_Accumulation_Magnitude_1d'] = cleaned_data['No'].map(lambda x: snowfall_aspect_bias('weather_data_instability/No' + str(int(x)) + '.csv', 1)[0])
    cleaned_data['Wind_Induced_Accumulation_Magnitude_3d'] = cleaned_data['No'].map(lambda x: snowfall_aspect_bias('weather_data_instability/No' + str(int(x)) + '.csv', 2)[0])
    cleaned_data['Wind_Induced_Accumulation_Magnitude_7d'] = cleaned_data['No'].map(lambda x: snowfall_aspect_bias('weather_data_instability/No' + str(int(x)) + '.csv', 3)[0])
    cleaned_data['Wind_Induced_Accumulation_Magnitude_14d'] = cleaned_data['No'].map(lambda x: snowfall_aspect_bias('weather_data_instability/No' + str(int(x)) + '.csv', 4)[0])

    cleaned_data['Wind_Induced_Accumulation_Aspect_1d'] = cleaned_data['No'].map(lambda x: snowfall_aspect_bias('weather_data_instability/No' + str(int(x)) + '.csv', 1)[1])
    cleaned_data['Wind_Induced_Accumulation_Aspect_3d'] = cleaned_data['No'].map(lambda x: snowfall_aspect_bias('weather_data_instability/No' + str(int(x)) + '.csv', 2)[1])
    cleaned_data['Wind_Induced_Accumulation_Aspect_7d'] = cleaned_data['No'].map(lambda x: snowfall_aspect_bias('weather_data_instability/No' + str(int(x)) + '.csv', 3)[1])
    cleaned_data['Wind_Induced_Accumulation_Aspect_14d'] = cleaned_data['No'].map(lambda x: snowfall_aspect_bias('weather_data_instability/No' + str(int(x)) + '.csv', 4)[1])

    cleaned_data['Average_Temperature_1d'] = cleaned_data['No'].map(lambda x: mean_temperature('weather_data_instability/No' + str(int(x)) + '.csv', 1))
    cleaned_data['Average_Temperature_3d'] = cleaned_data['No'].map(lambda x: mean_temperature('weather_data_instability/No' + str(int(x)) + '.csv', 2))
    cleaned_data['Average_Temperature_7d'] = cleaned_data['No'].map(lambda x: mean_temperature('weather_data_instability/No' + str(int(x)) + '.csv', 3))
    cleaned_data['Average_Temperature_14d'] = cleaned_data['No'].map(lambda x: mean_temperature('weather_data_instability/No' + str(int(x)) + '.csv', 4))

    cleaned_data['SD_Temperature_1d'] = cleaned_data['No'].map(lambda x: std_temperature('weather_data_instability/No' + str(int(x)) + '.csv', 1))
    cleaned_data['SD_Temperature_3d'] = cleaned_data['No'].map(lambda x: std_temperature('weather_data_instability/No' + str(int(x)) + '.csv', 2))
    cleaned_data['SD_Temperature_7d'] = cleaned_data['No'].map(lambda x: std_temperature('weather_data_instability/No' + str(int(x)) + '.csv', 3))
    cleaned_data['SD_Temperature_14d'] = cleaned_data['No'].map(lambda x: std_temperature('weather_data_instability/No' + str(int(x)) + '.csv', 4))

    cleaned_data['Sunshine_Percentage_1d'] = cleaned_data['No'].map(lambda x: sunshine_percentage('weather_data_instability/No' + str(int(x)) + '.csv', 1))
    cleaned_data['Sunshine_Percentage_3d'] = cleaned_data['No'].map(lambda x: sunshine_percentage('weather_data_instability/No' + str(int(x)) + '.csv', 2))
    cleaned_data['Sunshine_Percentage_7d'] = cleaned_data['No'].map(lambda x: sunshine_percentage('weather_data_instability/No' + str(int(x)) + '.csv', 3))
    cleaned_data['Sunshine_Percentage_14d'] = cleaned_data['No'].map(lambda x: sunshine_percentage('weather_data_instability/No' + str(int(x)) + '.csv', 4))

    cleaned_data['Aspect_Delta_1d'] = np.minimum(
        abs(cleaned_data['Aspect'] - cleaned_data['Wind_Induced_Accumulation_Aspect_1d']),
        2*np.pi - abs(cleaned_data['Aspect'] - cleaned_data['Wind_Induced_Accumulation_Aspect_1d']))
    cleaned_data['Aspect_Delta_3d'] = np.minimum(
        abs(cleaned_data['Aspect'] - cleaned_data['Wind_Induced_Accumulation_Aspect_3d']),
        2 * np.pi - abs(cleaned_data['Aspect'] - cleaned_data['Wind_Induced_Accumulation_Aspect_3d']))
    cleaned_data['Aspect_Delta_7d'] = np.minimum(
        abs(cleaned_data['Aspect'] - cleaned_data['Wind_Induced_Accumulation_Aspect_7d']),
        2 * np.pi - abs(cleaned_data['Aspect'] - cleaned_data['Wind_Induced_Accumulation_Aspect_7d']))
    cleaned_data['Aspect_Delta_14d'] = np.minimum(
        abs(cleaned_data['Aspect'] - cleaned_data['Wind_Induced_Accumulation_Aspect_14d']),
        2 * np.pi - abs(cleaned_data['Aspect'] - cleaned_data['Wind_Induced_Accumulation_Aspect_14d']))

    return cleaned_data


def data_setup():
    """
    Saves the cleand data in a csv. Only run once.
    :return: None
    """
    snow_instability = pd.read_csv("snow_instability_field_data.csv", sep=";")
    snow_instability = snow_instability[:-10]
    cleand_data = create_df_for_instability_model(snow_instability)
    cleand_data.to_csv('cleand_data.csv', sep=',')


if __name__ == '__main__':
    snow_instability = pd.read_csv("snow_instability_field_data.csv", sep = ";")
    snow_instability = snow_instability[:-10]
    # avalanche_accidents = pd.read_csv("avalanche_accidents_switzerland_since_1995.csv", sep = ",", encoding="ISO-8859-1")

    # download_weather_data(snow_instability, avalanche_accidents)


    # cor_matrix = snow_instability[['Profile_class', 'five_class_Stability', 'RB_score', 'RB_release_type', 'Fracture_plane_quality', 'S2008_1_RB', 'S2008_2_RT', 'S2008_3_Lemons', 'three_class_Stability', 'four_class_Stability_Techel', 'RB_height_cm', 'Snow_depth_cm', 'Slab_thickness_cm', 'FL_Thickness_cm', 'AL_Thickness_cm', 'FL_Grain_size_avg_mm', 'AL_Grain_size_avg_mm', 'FL_Grain_size_max_mm', 'AL_Grain_size_max_mm', 'FL_Grain_type1', 'FL_Grain_type2', 'FL_Hardness', 'FL_Top_Height_cm', 'FL_Bottom_Height_cm', 'AL_Top_Height_cm', 'AL_Bottom_Height_cm', 'AL_Hardness', 'Hard_Diff', 'Abs_Hard_Diff', 'Grain_Size_Diff_mm', 'FL_location', 'Lemon1_E', 'Lemon2_R', 'Lemon3_F', 'Lemon4_dE', 'Lemon5_dR', 'Lemon6_FLD', 'Lemons_FL', 'Whumpfs', 'Cracks', 'Avalanche_activity', 'LN_Local_danger_level_nowcast', 'LN_rounded', 'RF_Regional _danger_level_forecast', 'Deviation_LN_RF', 'SNPK_Index', 'SNPK_Index_Class']].corr()
    # print(cor_matrix)
    # cor_matrix.to_csv('correlation_matrix.csv', sep = ',')
    #
    # snow_instability.describe()

    cleand = create_df_for_instability_model(snow_instability)

    # data_setup()

    # cleand = pd.read_csv('cleand_data.csv', sep=',')

    model1 = smf.ols(
        formula='RB_score ~ Slope_angle_degrees + Accumulated_Snow_1d + Accumulated_Snow_3d + Accumulated_Snow_7d + Accumulated_Snow_14d + Average_Temperature_1d + Average_Temperature_3d + Average_Temperature_7d + Average_Temperature_14d + SD_Temperature_1d + SD_Temperature_3d + SD_Temperature_7d + SD_Temperature_14d + Sunshine_Percentage_1d + Sunshine_Percentage_3d + Sunshine_Percentage_7d + Sunshine_Percentage_14d + Aspect_Delta_1d : Wind_Induced_Accumulation_Magnitude_1d + Aspect_Delta_3d : Wind_Induced_Accumulation_Magnitude_3d + Aspect_Delta_7d : Wind_Induced_Accumulation_Magnitude_7d + Aspect_Delta_14d : Wind_Induced_Accumulation_Magnitude_14d',
        data=cleand).fit()
    model2 = smf.ols(
        formula='RB_score ~ Slope_angle_degrees + Accumulated_Snow_1d + Accumulated_Snow_3d + Accumulated_Snow_7d + Accumulated_Snow_14d + Average_Temperature_1d + Average_Temperature_3d + Average_Temperature_7d + Average_Temperature_14d + SD_Temperature_1d + SD_Temperature_3d + SD_Temperature_7d + SD_Temperature_14d + Sunshine_Percentage_1d + Sunshine_Percentage_3d + Sunshine_Percentage_7d + Sunshine_Percentage_14d',
        data=cleand).fit()
    model3 = smf.ols(
        formula='RB_release_type ~ Slope_angle_degrees + Accumulated_Snow_1d + Accumulated_Snow_3d + Accumulated_Snow_7d + Accumulated_Snow_14d + Average_Temperature_1d + Average_Temperature_3d + Average_Temperature_7d + Average_Temperature_14d + SD_Temperature_1d + SD_Temperature_3d + SD_Temperature_7d + SD_Temperature_14d + Sunshine_Percentage_1d + Sunshine_Percentage_3d + Sunshine_Percentage_7d + Sunshine_Percentage_14d + Aspect_Delta_1d : Wind_Induced_Accumulation_Magnitude_1d + Aspect_Delta_3d : Wind_Induced_Accumulation_Magnitude_3d + Aspect_Delta_7d : Wind_Induced_Accumulation_Magnitude_7d + Aspect_Delta_14d : Wind_Induced_Accumulation_Magnitude_14d',
        data=cleand).fit()
    model4 = smf.ols(
        formula='RB_height_cm ~ Slope_angle_degrees + Accumulated_Snow_1d + Accumulated_Snow_3d + Accumulated_Snow_7d + Accumulated_Snow_14d + Average_Temperature_1d + Average_Temperature_3d + Average_Temperature_7d + Average_Temperature_14d + SD_Temperature_1d + SD_Temperature_3d + SD_Temperature_7d + SD_Temperature_14d + Sunshine_Percentage_1d + Sunshine_Percentage_3d + Sunshine_Percentage_7d + Sunshine_Percentage_14d + Aspect_Delta_1d : Wind_Induced_Accumulation_Magnitude_1d + Aspect_Delta_3d : Wind_Induced_Accumulation_Magnitude_3d + Aspect_Delta_7d : Wind_Induced_Accumulation_Magnitude_7d + Aspect_Delta_14d : Wind_Induced_Accumulation_Magnitude_14d',
        data=cleand).fit()
    model5 = smf.ols(
        formula='FL_Grain_size_avg_mm ~ Slope_angle_degrees + Accumulated_Snow_1d + Accumulated_Snow_3d + Accumulated_Snow_7d + Accumulated_Snow_14d + Average_Temperature_1d + Average_Temperature_3d + Average_Temperature_7d + Average_Temperature_14d + SD_Temperature_1d + SD_Temperature_3d + SD_Temperature_7d + SD_Temperature_14d + Sunshine_Percentage_1d + Sunshine_Percentage_3d + Sunshine_Percentage_7d + Sunshine_Percentage_14d + Aspect_Delta_1d : Wind_Induced_Accumulation_Magnitude_1d + Aspect_Delta_3d : Wind_Induced_Accumulation_Magnitude_3d + Aspect_Delta_7d : Wind_Induced_Accumulation_Magnitude_7d + Aspect_Delta_14d : Wind_Induced_Accumulation_Magnitude_14d',
        data=cleand).fit()
    model6 = smf.ols(
        formula='AL_Grain_size_avg_mm ~ Slope_angle_degrees + Accumulated_Snow_1d + Accumulated_Snow_3d + Accumulated_Snow_7d + Accumulated_Snow_14d + Average_Temperature_1d + Average_Temperature_3d + Average_Temperature_7d + Average_Temperature_14d + SD_Temperature_1d + SD_Temperature_3d + SD_Temperature_7d + SD_Temperature_14d + Sunshine_Percentage_1d + Sunshine_Percentage_3d + Sunshine_Percentage_7d + Sunshine_Percentage_14d + Aspect_Delta_1d : Wind_Induced_Accumulation_Magnitude_1d + Aspect_Delta_3d : Wind_Induced_Accumulation_Magnitude_3d + Aspect_Delta_7d : Wind_Induced_Accumulation_Magnitude_7d + Aspect_Delta_14d : Wind_Induced_Accumulation_Magnitude_14d',
        data=cleand).fit()
    model7 = smf.ols(
        formula='SNPK_Index ~ Slope_angle_degrees + Accumulated_Snow_1d + Accumulated_Snow_3d + Accumulated_Snow_7d + Accumulated_Snow_14d + Average_Temperature_1d + Average_Temperature_3d + Average_Temperature_7d + Average_Temperature_14d + SD_Temperature_1d + SD_Temperature_3d + SD_Temperature_7d + SD_Temperature_14d + Sunshine_Percentage_1d + Sunshine_Percentage_3d + Sunshine_Percentage_7d + Sunshine_Percentage_14d + Aspect_Delta_1d : Wind_Induced_Accumulation_Magnitude_1d + Aspect_Delta_3d : Wind_Induced_Accumulation_Magnitude_3d + Aspect_Delta_7d : Wind_Induced_Accumulation_Magnitude_7d + Aspect_Delta_14d : Wind_Induced_Accumulation_Magnitude_14d',
        data=cleand).fit()


    print(model1.summary())
    print(model2.summary())
    print(model3.summary())
    print(model4.summary())
    print(model5.summary())
    print(model6.summary())
    print(model7.summary())

if __name__ == '__main__':
    # cleand = pd.read_csv('cleand_data.csv', sep=',')

    # snow_instability = pd.read_csv("snow_instability_field_data.csv", sep=";")
    # snow_instability = snow_instability[:-10]
    # cleand = create_df_for_instability_model(snow_instability)

    model8 = smf.ols(
        formula='RB_score ~ HN3d_cm',
        data=cleand).fit()
    print(model8.summary())

    model9 = smf.ols(
        formula='RF_Regional_danger_level_forecast ~ Accumulated_Snow_1d + Accumulated_Snow_3d + Accumulated_Snow_7d + Accumulated_Snow_14d + Average_Temperature_1d + Average_Temperature_3d + Average_Temperature_7d + Average_Temperature_14d + SD_Temperature_1d + SD_Temperature_3d + SD_Temperature_7d + SD_Temperature_14d + Sunshine_Percentage_1d + Sunshine_Percentage_3d + Sunshine_Percentage_7d + Sunshine_Percentage_14d',
        data=cleand).fit()
    print(model9.summary())

    model10 = smf.ols(
        formula='LN_Local_danger_level_nowcast ~ Accumulated_Snow_1d + Accumulated_Snow_3d + Accumulated_Snow_7d + Accumulated_Snow_14d + Average_Temperature_1d + Average_Temperature_3d + Average_Temperature_7d + Average_Temperature_14d + SD_Temperature_1d + SD_Temperature_3d + SD_Temperature_7d + SD_Temperature_14d + Sunshine_Percentage_1d + Sunshine_Percentage_3d + Sunshine_Percentage_7d + Sunshine_Percentage_14d',
        data=cleand).fit()
    print(model10.summary())

    model11 = smf.ols(
        formula='RB_score ~ C(RF_Regional_danger_level_forecast) + LN_Local_danger_level_nowcast',
        data=cleand).fit()
    print(model11.summary())

    cleand.plot.scatter(x='Average_Temperature_1d', y='FL_Grain_size_avg_mm')

    diagnostics_model = LinearRegDiagnostic(model11)
    vif, fig, ax = diagnostics_model()
    print(vif)
