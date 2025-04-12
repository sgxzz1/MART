import pickle
import random
import sys
import math
import numpy as np
from tqdm import tqdm
from pyproj import Geod
from config import Config
geod = Geod(ellps='WGS84')
# lat_min = 35.5
# lat_max = 57.5
# lon_min = -35.98
# lon_max = -15.98
# lat_min = 32
# lat_max = 52
# lon_min = 127
# lon_max = 140

length = 120
import math
def calculate_new_position2(lat, lon, speed, bearing, interval=None):
    """
    计算给定时间后的新位置
    :param lat: 初始纬度，单位为度
    :param lon: 初始经度，单位为度
    :param speed: 船速，单位为米/秒
    :param bearing: 航向，单位为度
    :param time_minutes: 时间，单位为分钟
    :return: 新的纬度和经度，单位为度
    """
    # 将时间转换为秒
    time_seconds = 10 * 60
    # length = len(lat)
    # 将速度转换为每秒的距离（假设速度单位是米/秒）
    if interval is not None:
        distance = speed * interval * 0.51444
    else:
        distance = speed * time_seconds * 0.51444
    # # 将纬度、经度和航向从度转换为弧度
    # lat = math.radians(lat)
    # lon = math.radians(lon)
    # bearing = math.radians(bearing)

    # 地球的半径，单位为米
    R = 6371e3

    # 计算新的纬度
    new_lat = np.arcsin(np.sin(lat) * np.cos(distance / R) +
                        np.cos(lat) * np.sin(distance / R) * np.cos(bearing))

    # 计算新的经度
    new_lon = lon + np.arctan2(np.sin(bearing) * np.sin(distance / R) * np.cos(lat),
                               np.cos(distance / R) - np.sin(lat) * np.sin(new_lat))

    # new_lat = new_lat[:-1]
    # new_lat = np.concatenate((lat[:1], new_lat), axis=-1)
    # new_lon = new_lon[:-1]
    # new_lon = np.concatenate((lon[:1], new_lon), axis=-1)
    # # 将新的纬度和经度从弧度转换为度
    new_lat = new_lat * 180.0 / np.pi
    new_lon = new_lon * 180.0 / np.pi
    cog = bearing * 180.0 / np.pi
    tmp = np.zeros((1, 4))
    tmp[:, 0] = new_lat
    tmp[:, 1] = new_lon
    tmp[:, 2] = speed
    tmp[:, 3] = cog

    return tmp
def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    计算从点1 (lat1, lon1) 到点2 (lat2, lon2) 的航向角
    参数:
    lat1, lon1: 起点的纬度和经度 (单位: 度)
    lat2, lon2: 终点的纬度和经度 (单位: 度)

    返回:
    航向角 (0到360度之间)
    """
    # 将度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    initial_bearing = math.atan2(x, y)

    # 将结果转换为0到360度之间
    bearing = math.degrees(initial_bearing)
    bearing = (bearing + 360) % 360

    return bearing


def calculate_speed(lat1, lon1, lat2, lon2):
    """
    计算船舶从点1到点2的速度
    参数:
    lat1, lon1: 起点的纬度和经度 (单位: 度)
    lat2, lon2: 终点的纬度和经度 (单位: 度)
    time: 两点之间的时间 (单位: 小时)

    返回:
    速度 (单位: 千米/小时)
    """
    distance = distance_c_km(lat1, lon1, lat2, lon2)
    speed = distance*6/1.852

    return speed


def cal_sog_cog(lon_end, lat_end, lon_start, lat_start):
    az, _, dist = geod.inv(lon_end,
                           lat_end,
                           lon_start,
                           lat_start)
    sog = dist / 1852*6
    cog = (az + 360) % 360
    return cog, sog


def distance_c_km(lat1, lon1, lat2, lon2):
    """求两地理位置之间的距离"""
    radius = 6371.  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c  # 单位：km
    return d


# 计算10min后的船舶位置
def calculate_new_position(lat, lon, speed, bearing):
    """
    计算给定时间后的新位置
    :param lat: 初始纬度，单位为度
    :param lon: 初始经度，单位为度
    :param speed: 船速，单位为米/秒
    :param bearing: 航向，单位为度
    :param time_minutes: 时间，单位为分钟
    :return: 新的纬度和经度，单位为度
    """
    # 将时间转换为秒
    time_seconds = 10 * 60

    # 将速度转换为每秒的距离（假设速度单位是米/秒）
    distance = speed * time_seconds * 0.51444

    # 将纬度、经度和航向从度转换为弧度
    lat = math.radians(lat)
    lon = math.radians(lon)
    bearing = math.radians(bearing)

    # 地球的半径，单位为米
    R = 6371e3

    # 计算新的纬度
    new_lat = math.asin(math.sin(lat) * math.cos(distance / R) +
                        math.cos(lat) * math.sin(distance / R) * math.cos(bearing))

    # 计算新的经度
    new_lon = lon + math.atan2(math.sin(bearing) * math.sin(distance / R) * math.cos(lat),
                               math.cos(distance / R) - math.sin(lat) * math.sin(new_lat))

    # 将新的纬度和经度从弧度转换为度
    new_lat = math.degrees(new_lat)
    new_lon = math.degrees(new_lon)

    return new_lat, new_lon


def generate_trajectory_true(number, length, cf):
    lat_min = cf.lat_min
    lat_max = cf.lat_max
    lon_min = cf.lon_min
    lon_max = cf.lon_max
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat, lon = generate_start_point_true(cf)

    traj = []
    for i in range(length):
        lon_next = 3000
        lat_next = 3000
        while lon_next >= lon_max or lat_next >= lat_max or lon_next < lon_min or lat_next < lat_min:
            lat_next, lon_next, sog_now, cog_now = generate_next_point_true(lon, lat)
        tmp = [(lat - lat_min) / lat_range, (lon - lon_min) / lon_range, sog_now / 30, cog_now / 360, number, 600 * i]
        lon = lon_next
        lat = lat_next
        traj.append(tmp)
    return np.array(traj)


def generate_trajectory_origin(number):
    x, y = generate_start_point_origin()

    traj = []
    for i in range(120):
        x_next = 10
        y_next = 10
        while x_next > 9 or y_next > 9 or x_next < 0 or y_next < 0:
            x_next, y_next, sog_now, cog_now = generate_next_point_origin(x, y)
        tmp = [y / 10, x / 10, sog_now / 2, cog_now / 8, number, 600 * i]
        x = x_next
        y = y_next
        traj.append(tmp)
    return np.array(traj)


# 随机生成速度和方向，并生成下一个点的位置
def generate_next_point_true(x, y):
    sog = random.randint(0, 29)
    cog = random.randint(0, 359)
    lat, lon = calculate_new_position(y, x, sog, cog)
    return lat, lon, sog, cog


def generate_next_point_origin(x, y):
    sog_ = random.randint(0, 1)
    sog = sog_ + 1
    cog = random.randint(0, 7)
    # 正上
    if cog == 0:
        y += sog
    # 左上
    elif cog == 1:
        y += sog
        x -= sog
    # 左
    elif cog == 2:
        x -= sog
    # 左下
    elif cog == 3:
        y -= sog
        x -= sog
    # 下
    elif cog == 4:
        y -= sog
    # 右下
    elif cog == 5:
        y -= sog
        x += sog
    # 右
    elif cog == 6:
        x += sog
    # 右上
    elif cog == 7:
        y += sog
        x += sog
    else:
        print('generate_next_point error')
        sys.exit()
    return x, y, sog_, cog


# 生成随机初始点位
def generate_start_point_origin():
    x = random.randint(0, 9)
    y = random.randint(0, 9)
    return x, y


def generate_start_point_true(cf):
    lon = random.uniform(cf.lon_min, cf.lon_max)
    lat = random.uniform(cf.lat_min, cf.lat_max)
    return lat, lon
name = 'ct_dma'

cf = Config(name)

if __name__ == '__main__':
    print('generate trajectory for', name)
    traj_list = []
    for number in tqdm(range(1000), total=1000):
        traj = generate_trajectory_true(number, length, cf)
        traj_list.append(traj)
    traj_np = np.array(traj_list)
    res = []
    for traj in traj_np:
        tmp = dict()
        tmp['mmsi'] = traj[0, 4]
        tmp['traj'] = traj
        res.append(tmp)
    with open('ceshi_valid.pkl', 'wb') as f:
        pickle.dump(res, f)
    # coor_range = np.array([cf.lat_max - cf.lat_min, cf.lon_max - cf.lon_min, 30., 360.], dtype=np.float64)
    # coor_start = np.array([cf.lat_min, cf.lon_min, 0., 0.], dtype=np.float64)
    # trajs = []
    # with open('data/ct_dma/ct_dma_train.pkl', 'rb') as f:
    #     obj = pickle.load(f)
    #     traj1 = obj[0]['traj']
    #     length = len(traj1)
    #     traj1[:, :4] = traj1[:, :4] * coor_range + coor_start
    #     lat_start = traj1[0, 0]
    #     lon_start = traj1[0, 1]
    #     for i in range(999):
    #         traj = []
    #         for j in range(length-1):
    #             lon_des = 0
    #             lat_des = 0
    #             while cf.lon_min > lon_des or lon_des > cf.lon_max or cf.lat_min > lat_des or lat_des > cf.lat_max:
    #                 a = random.randint(-2, 2)
    #                 b = random.randint(-2, 2)
    #                 lat_des = traj1[1 + j, 0] + a * 0.01
    #                 lon_des = traj1[1 + j, 1] + b * 0.01
    #
    #                 cog = calculate_bearing(lat_start, lon_start, lat_des, lon_des)
    #                 sog = calculate_speed(lat_start, lon_start, lat_des, lon_des)
    #                 tmp = np.array([lat_start, lon_start, sog, cog, traj1[j + 1, 4], traj1[j + 1, 5]])
    #             lat_start = lat_des
    #             lon_start = lon_des
    #             traj.append(tmp)
    #         traj.append(np.array([lat_des, lon_des, traj1[-1, 2], traj1[-1, 3], traj1[-1, 4], traj1[-1, 5]]))
    #         trajs.append(traj)
    #     traj2 = obj[7]['traj']
    #     traj2[:, :4] = traj2[:, :4]*coor_range+coor_start
    #     # print(traj2)
    #     trajs.append(traj2)
    # res = []
    # for i in trajs:
    #     tmp = dict()
    #     tmp['traj'] = np.array(i)
    #     tmp['traj'][:, 0] = (tmp['traj'][:, 0] - cf.lat_min) / (cf.lat_max - cf.lat_min)
    #     tmp['traj'][:, 1] = (tmp['traj'][:, 1] - cf.lon_min) / (cf.lon_max - cf.lon_min)
    #     tmp['traj'][:, 2] /= 30
    #     tmp['traj'][:, 3] /= 360
    #     tmp['mmsi'] = tmp['traj'][0, 5]
    #     res.append(tmp)
    # # print(res[999]['traj'])
    # with open('learn2.pkl', 'wb') as f:
    #     pickle.dump(res, f)
