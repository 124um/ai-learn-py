import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 16, 8

### Загрузка данных: здания
# * primary_use - назначение
# * square_feet - площадь, кв.футы
# * year_built - год постройки
# * floor_count - число этажей
buildings = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz")
# print (buildings.head())

# ### Загрузка данных: погода
# * air_temperature - температура воздуха, С
# * dew_temperature - точка росы (влажность), С
# * cloud_coverage - облачность, %
# * precip_depth_1_hr - количество осадков, мм/час
# * sea_level_pressure - давление, мбар
# * wind_direction - направление ветра, градусы
# * wind_speed - скорость ветра, м/с
weather = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz")
# print (weather.head())

### Загрузка данных: потребление энергии здания 0
# * meter_reading - значение показателя (TOE, эквивалент тонн нефти)
energy_0 = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/train.0.0.csv.gz")
# print (energy_0.head())
# energy_0.set_index("timestamp")["meter_reading"].plot()
# plt.show()

### Объединение потребления энергии и информацию о здании
# Проводим объединение по building_id
energy_0 = pd.merge(left=energy_0, right=buildings, how="left", left_on="building_id", right_on="building_id")
# print (energy_0.head())

### Объединение потребления энергии и погоды
# Выставим индексы для объединения - timestamp, site_id
energy_0.set_index(["timestamp" , "site_id"], inplace=True)
weather.set_index(["timestamp" , "site_id"], inplace=True)

# Проведем объединение и сбросим индексы
energy_0 = pd.merge(left=energy_0, right=weather, how="left", left_index=True, right_index=True)
energy_0.reset_index(inplace=True)
# print (energy_0.head())


### Нахождение пропущенных данных
# Посчитаем количество пропусков данных по столбцам
for column in energy_0.columns:
    energy_nulls = energy_0[column].isnull().sum()
    if energy_nulls > 0:
        print (column + ": " + str(energy_nulls))
# print (energy_0[energy_0["precip_depth_1_hr"].isnull()])
# print (energy_0.info())


# ### Заполнение пропущенных данных
# * air_temperature: NaN -> 0
# * cloud_coverage: NaN -> 0
# * dew_temperature: NaN -> 0
# * precip_depth_1_hr: NaN -> 0, -1 -> 0
# * sea_level_pressure: NaN -> среднее
# * wind_direction: NaN -> среднее (роза ветров)

energy_0["air_temperature"].fillna(0, inplace=True)
energy_0["cloud_coverage"].fillna(0, inplace=True)
energy_0["dew_temperature"].fillna(0, inplace=True)
energy_0["precip_depth_1_hr"] = energy_0["precip_depth_1_hr"].apply(lambda x:x if x>0 else 0)
energy_0_sea_level_pressure_mean = energy_0["sea_level_pressure"].mean()
energy_0["sea_level_pressure"] = energy_0["sea_level_pressure"].apply(lambda x:energy_0_sea_level_pressure_mean if x!=x else x)
energy_0_wind_direction_mean = energy_0["wind_direction"].mean()
energy_0["wind_direction"] = energy_0["wind_direction"].apply(lambda x:energy_0_wind_direction_mean if x!=x else x)
energy_0.info()