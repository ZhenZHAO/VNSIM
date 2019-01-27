"""
@functions: Utility library
@author: Zhen ZHAO
@date: June 24, 2018
"""
import numpy as np
import load_conf as lc


# # # # # # # # # # # # # # #
# 1. time transformation  # #
# # # # # # # # # # # # # # #
def time_2_jde(year, month, day, hour, minute, sec):
    """
    儒略日的计算
    :return: jde time
    """
    if month == 1 or month == 2:
        f = year - 1
        g = month + 12
    else:  # month >= 3
        f = year
        g = month
    mid1 = np.floor(365.25 * f)
    mid2 = np.floor(30.6001 * (g+1))
    para_a = 2-np.floor(f/100)+np.floor(f/400)
    para_j = mid1 + mid2 + day + para_a + 1720994.5
    jde_time = para_j + hour / 24 + minute / 1440 + sec / 86400
    return jde_time


def time_2_mjd(year, month, day, hour, minute, sec, d_sec):
    """
    得到修正儒略日
    :return: mjd
    """
    YP = year
    MP = month
    if month <= 2:
        month += 12
        year = year - 1
    if (YP < 1582) or (YP == 1582 and MP < 10) or (YP == 1582 and MP == 10 and day <= 4):
        B = -2 + int((year + 4716) / 4) - 1179
    elif (YP > 1582) or (YP == 1582 and MP > 10) or (YP == 1582 and MP == 10 and day > 10):
        B = int(year / 400) - int(year / 100) + int(year / 4)

    mjd = 365.0 * np.double(year) - 679004.0 + np.double(B) + np.floor(30.6001 * np.double(month + 1)) + np.double(day)
    mjd += (np.double(3600 * hour + 60 * minute + sec) + d_sec) / 86400.00
    return mjd


def mjd_2_time(mjd_time):
    month_array = ((31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))
    year = int(0.0027379093 * mjd_time + 1858.877)
    day = int(mjd_time - time_2_mjd(year, 1, 0, 0, 0, 0, 0.0))
    if (year % 4 == 0 and year % 400 == 0) or (year % 4 == 0 and year % 100 != 0):
        m_flag = 0
    else:
        m_flag = 1
    month = 1
    for i in range(0, 12):
        day = day - month_array[m_flag][i]
        if day <= 0:
            day = day + month_array[m_flag][i]
            month = i+1
            break
        else:
            continue
    week = (int(mjd_time)-5) % 7
    mjd_time = mjd_time - time_2_mjd(year, month, day, 0, 0, 0, 0.0)
    mjd_time = mjd_time * 86400.0
    F = int(mjd_time)
    if np.fabs(mjd_time-np.floor(mjd_time)) >= 0.5:
        F = F+1
    hour = F//3600
    minute = np.mod(F, 3600)//60
    sec = np.mod(F, 3600)
    sec = np.mod(sec, 60)
    return week, year, month, day, hour, minute, sec  # W返回0代表星期一，返回6代表星期天


def time_2_day(day, hour, minute, sec):
    """
    将一段时间单位，转化为天数
    :param day:
    :param hour:
    :param minute:
    :param sec:
    :return:
    """
    day_num = np.double(3600 * hour + 60 * minute + sec) / 86400.00
    day_num += day
    return day_num


def time_2_rad(hour, minute, sec):
    """
    将时角转换为弧度
    :param hour:
    :param minute:
    :param sec:
    :return:
    """
    if hour < 0:
        flag = -1
    else:
        flag = 1
    hour = np.abs(hour)
    angle_rad = (hour + (60.0 * minute + sec) / 3600.0) / 12 * np.pi
    return angle_rad*flag


def time_str_2_rad(time_st):
    """
    将时间字符串转换为弧度
    :param time_st:"21h33m26s"
    :return: radian
    """
    time_str = time_st
    time_str = time_str.replace('h', ':')
    time_str = time_str.replace('m', ':')
    time_str = time_str.replace('s', '')
    time_str = time_str.split(':')
    time_h = int(time_str[0])
    time_m = int(time_str[1])
    time_s = float(time_str[2])
    time_rad = time_2_rad(time_h, time_m, time_s)
    return time_rad


def time_str_2_mjd(time_st):
    time_str = time_st
    time_year = int(time_str[0:4])
    time_month = int(time_str[4:6])
    time_day = int(time_str[6:8])
    time_hour = int(time_str[8:10])
    time_minute = int(time_str[10:12])
    time_second = int(time_str[12:14])
    mjd_time = time_2_mjd(time_year, time_month, time_day, time_hour, time_minute, time_second, 0)
    return mjd_time


def mjd_2_julian(mjd_time):
    """
    #J2000 2000年1月1日12时
    :param mjd_time:
    :return:
    """
    julian_time = (mjd_time-51544.5)/36525     # part4-1 p31
    return julian_time     # 以J2000作为参考，计算MJD和Julian时间


def mjd_2_gmst(mjd_time):
    """
    格林尼治平均恒星时
    :param mjd_time:
    :return:
    """
    jutime = mjd_2_julian(mjd_time)
    gmst = 67310.548 + 8640184.812866 * jutime + (mjd_time + 0.5 - int(mjd_time + 0.5)) * 86400
    gmst = gmst * np.pi / 43200
    return gmst


def mjd_2_gast(mjd_time):
    """
    格林尼治视恒星时
    :param mjd_time:
    :return:
    """
    gmst = mjd_2_gmst(mjd_time)
    eq_e = equinox_equation(mjd_time)  # part4-1 p31
    gast = gmst + eq_e
    return gast


def mjd_2_gst(time_mjd, delta_t, utc_ut1):
    dpi = 3.141592653589793238462643
    gst_offset = 0.7790572732640
    gst_factor = 1.00273781191135448
    t = time_mjd - 51544.5
    t = t * gst_factor
    t = t - np.double(np.int(t))  # 取其小数部分
    t = t + (delta_t + utc_ut1) / 86400.0 * gst_factor
    theta = 2.0 * dpi * (gst_offset + t)
    return theta


def ecliptic_obliquity(mjd_time):
    """
    计算黄赤交角
    :param mjd_time:
    :return:
    """
    ju_time = mjd_2_julian(mjd_time)

    # 4709636#将秒的单位转化为弧度1s/3600/180*np.pi=1/206264.80624709636
    epsilon = (84381.448 - 46.815 * ju_time - 0.00059 * ju_time ** 2 + 0.001813 * ju_time ** 3) / 206264.8062
    return epsilon


def nutation_omega(ju_time):
    """
    简化章动模型的基本参数Ω的计算
    :param ju_time:
    :return:
    """
    omega = (450160.28 - 6962890.539 * ju_time)  # 平均的月球轨道升交点经度
    omega = omega / 206264.8062
    return omega


def longitude_nutation(mjd_time):
    """
    黄经章动
    :param mjd_time:
    :return:
    """
    tim = (mjd_time - 51544.5) / 36525
    omega = nutation_omega(tim)  # Omega的单位为弧度
    delta_psi = -(17.1996 + 0.01742 * tim) * np.sin(omega) / 206264.8062
    return delta_psi


def equinox_equation(mjd_time):
    """
    春分方程
    :param mjd_time:
    :return:
    """
    epsilon = ecliptic_obliquity(mjd_time)
    delta_psi = longitude_nutation(mjd_time)
    e_e = delta_psi * np.cos(epsilon)
    return e_e


# # # # # # # # # # # # # # # # # #
# 2. coordinate transformation  # #
# # # # # # # # # # # # # # # # # #
def trans_matrix_uv_itrf(mjd_time, ra, dec):
    """
    生成从ITRF到UV坐标系的转换矩阵
    :param mjd_time:
    :param ra:
    :param dec:
    :return:
    """
    gast = mjd_2_gast(mjd_time)
    hour_angle = gast - ra
    hour_angle = np.mod(hour_angle, np.pi * 2)
    matrix = np.array([[np.sin(hour_angle), np.cos(hour_angle), 0],
                       [-np.sin(dec) * np.cos(hour_angle), np.sin(dec) * np.sin(hour_angle), np.cos(dec)],
                       [np.cos(dec) * np.cos(hour_angle), -np.cos(dec) * np.sin(hour_angle), np.sin(dec)]
                       ])
    return matrix


def geographic_2_itrf(longitude, latitude, height):
    """
    地理坐标系专为ITRF坐标
    :param longitude: 地理坐标的经度
    :param latitude: 纬度
    :param height:  高度
    :return: ITRF坐标位置(x,y,z)
    """
    e_square = lc.eccentricity_square
    temp = lc.earth_radius / np.sqrt(1 - e_square * (np.sin(latitude) ** 2))
    # 计算笛卡尔坐标(x,y,z)
    x = (temp + height) * np.cos(latitude) * np.cos(longitude)
    y = (temp + height) * np.cos(latitude) * np.sin(longitude)
    z = ((1 - e_square) * temp + height) * np.sin(latitude)
    return x, y, z


def itrf_2_geographic(cor_x, cor_y, cor_z):
    p = np.sqrt(cor_x ** 2 + cor_y ** 2)
    f = lc.earth_flattening
    e_square = lc.eccentricity_square

    # calculate longitude
    if (cor_x == 0) and (cor_y == 0):
        longitude = 0
        if cor_z == 0:
            latitude = 0
            height = -1 * lc.earth_radius
    else:
        longitude = np.arctan2(cor_y, cor_x)
    # calculate latitude
    if p == 0:
        if cor_z > 0:
            latitude = np.pi / 2
        elif cor_z < 0:
            latitude = -np.pi / 2
    else:
        r = np.sqrt(p ** 2 + cor_z ** 2)

        temp = cor_z / p * ((1 - f) + e_square * lc.earth_radius / r)
        u = np.arctan(temp)

        temp = (cor_z * (1 - f) + e_square * lc.earth_radius * ((np.sin(u)) ** 3)) / (
                (1 - f) * (p - e_square * lc.earth_radius * ((np.cos(u)) ** 3)))
        latitude = np.arctan(temp)  # -90度到+90度
    # calculate height
    if cor_z != 0:
        height = p * np.cos(latitude) + cor_z * np.sin(latitude) - lc.earth_radius * np.sqrt(
            1 - e_square * (np.sin(latitude) ** 2))
    return longitude, latitude, height


def rect_2_polar(x):
    """
    直角坐标系转换为极坐标（Long,Lat)
    :param x: 3维直角坐标
    :return:
    """
    r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    x[0] = x[0] / r
    x[1] = x[1] / r
    x[2] = x[2] / r
    if x[0] == 0 and x[1] == 0:
        Long = 0
    else:
        Long = np.arctan2(x[0], x[1])  # -180~180
        if Long < 0:
            Long = Long + np.pi * 2
    Lat = np.arcsin(x[2])
    return Long, Lat  # part4-2 p16


def polar_2_rect(long, lat):
    """
    极坐标到直角坐标的3维单位向量x
    :param long:
    :param lat:
    :return:
    """
    x1 = np.cos(lat) * np.cos(long)
    x2 = np.cos(lat) * np.sin(long)
    x3 = np.sin(lat)
    return x1, x2, x3


def equatorial_2_horizontal(time_mjd, ra_src, dec_src, long_station, lat_station):
    x, y, z = polar_2_rect(ra_src, dec_src)
    gast = mjd_2_gast(time_mjd)
    rz_pi = np.array([[np.cos(np.pi), np.sin(np.pi), 0],
                      [-np.sin(np.pi), np.cos(np.pi), 0],
                      [0, 0, 1]
                      ])
    ry_latitude = np.array([[np.cos(np.pi / 2 - lat_station), 0, -np.sin(np.pi / 2 - lat_station)],
                            [0, 1, 0],
                            [np.sin(np.pi / 2 - lat_station), 0, np.cos(np.pi / 2 - lat_station)]
                            ])
    matrix1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rz_longitude = np.array([[np.cos(gast + long_station), np.sin(gast + long_station), 0],
                             [-np.sin(gast + long_station), np.cos(gast + long_station), 0],
                             [0, 0, 1]
                             ])
    vec_xyz = np.array([[x], [y], [z]])
    temp_mat = np.dot(rz_pi, ry_latitude)
    temp_mat = np.dot(temp_mat, matrix1)
    temp_mat = np.dot(temp_mat, rz_longitude)
    horizon_xyz = np.dot(temp_mat, vec_xyz)
    horizon_xlst = [horizon_xyz[0][0], horizon_xyz[1][0], horizon_xyz[2][0]]
    azimuth, elevation = rect_2_polar(horizon_xlst)
    return azimuth, elevation


def drotate(x, e, axis):
    """
    :param x: 待旋转的向量
    :param e: 旋转的角度
    :param axis: 旋转轴
    :return: 旋转后的向量
    """
    u = x[0]
    v = x[1]
    w = x[2]
    cos = np.cos
    sin = np.sin
    if axis == 'x' or axis == 'X':
        x[1] = v * cos(e) - w * sin(e)
        x[2] = v * sin(e) + w * cos(e)
        return x
    elif axis == 'y' or axis == 'Y':
        x[2] = w * cos(e) - u * sin(e)
        x[0] = w * sin(e) + u * cos(e)
        return x
    elif axis == 'z' or axis == 'Z':
        x[0] = u * cos(e) - v * sin(e)
        x[1] = u * sin(e) + v * cos(e)
        return x
    else:
        print("drotate:bad flag to rotate %s\n" % axis)


def itrf_2_horizontal(satellite_lst, long_sta, lat_sta, height_sta):
    """
    从地面坐标到水平系统的转换
    :param satellite_lst: 依次存放的是数据是卫星位置和速度：x y z vx vy vz
    :param long_sta:  遥测站的经度
    :param lat_sta:   遥测站的维度
    :param height_sta:  遥测站的高度
    :return:
    """

    x0, y0, z0 = geographic_2_itrf(long_sta, lat_sta, height_sta)
    x = satellite_lst[0] - x0
    y = satellite_lst[1] - y0
    z = satellite_lst[2] - z0
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # 从地面坐标到水平系统的位置坐标转换
    matrix1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    Rz_pi = np.array([[np.cos(np.pi), np.sin(np.pi), 0],
                      [-np.sin(np.pi), np.cos(np.pi), 0],
                      [0, 0, 1]
                      ])
    Ry_Latitude = np.array([[np.cos(np.pi / 2 - lat_sta), 0, -np.sin(np.pi / 2 - lat_sta)],
                            [0, 1, 0],
                            [np.sin(np.pi / 2 - lat_sta), 0, np.cos(np.pi / 2 - lat_sta)]
                            ])
    Rz_Longitude = np.array([[np.cos(long_sta), np.sin(long_sta), 0],
                             [-np.sin(long_sta), np.cos(long_sta), 0],
                             [0, 0, 1]
                             ])
    xyzITRF = np.array([[x, y, z]])
    xyzITRF = xyzITRF.T
    TempMatrix = np.dot(matrix1, Rz_pi)
    TempMatrix = np.dot(TempMatrix, Ry_Latitude)
    TempMatrix = np.dot(TempMatrix, Rz_Longitude)
    xyzHorizon = np.dot(TempMatrix, xyzITRF)
    # 从地面坐标到水平坐标的速度坐标转换
    velocitymatrix = np.array([[satellite_lst[3], satellite_lst[4], satellite_lst[5]]])
    velocitymatrix = velocitymatrix.T
    VxVyVzHorizon = np.dot(TempMatrix, velocitymatrix)
    XVector = [xyzHorizon[0][0], xyzHorizon[1][0], xyzHorizon[2][0]]
    Azimuth, Elevation = rect_2_polar(XVector)  # 仰角和方位角
    # Radial,azimuthal and vertical nelocity componengts
    Ry_Elevation = np.array([[np.cos(-1 * Elevation), 0, -np.sin(-1 * Elevation)],
                             [0, 1, 0],
                             [np.sin(-1 * Elevation), 0, np.cos(-1 * Elevation)]
                             ])
    Rz_Azimuth = np.array([[np.cos(Azimuth), np.sin(Azimuth), 0],
                           [-np.sin(Azimuth), np.cos(Azimuth), 0],
                           [0, 0, 1]
                           ])
    TempMatrix = np.dot(Ry_Elevation, Rz_Azimuth)
    VelocityVector = np.dot(TempMatrix, VxVyVzHorizon)
    AzimuthVelocity = VelocityVector[1][0]
    ElevationVelocity = VelocityVector[2][0]
    AzimuthVelocity = AzimuthVelocity / r
    ElevationVelocity = ElevationVelocity / r  # part4-2 p11
    return [Azimuth, Elevation, AzimuthVelocity, ElevationVelocity]  # 返回元祖


def equatorial_2_ecliptic(equ, epsilon):
    """
    将赤道源坐标系转换为黄道坐标系
    :param equ: 赤道系统单位矢量[[x],[y],[z]]
    :param epsilon: 黄道倾角
    :return: 黄道系统单位矢量
    """
    rx_epsilon=np.array([[1, 0, 0],
                         [0, np.cos(epsilon), np.sin(epsilon)],
                         [0, -np.sin(epsilon), np.cos(epsilon)]
                        ])
    ecu = np.dot(rx_epsilon, equ)
    return ecu


def itrf_2_icrf(time_mjd, itrf_sat_x, itrf_sat_y, itrf_sat_z, itrf_sat_vx,
                itrf_sat_vy, itrf_sat_vz):
    gast = mjd_2_gast(time_mjd)
    rz_gast = np.array([[np.cos(-gast), np.sin(-gast), 0],
                        [-np.sin(-gast), np.cos(-gast), 0],
                        [0, 0, 1]
                        ])
    itrf_pos_vec = np.array([[itrf_sat_x], [itrf_sat_y], [itrf_sat_z]])
    icrf_pos_vec = np.dot(rz_gast, itrf_pos_vec)
    # 恒星时/平均时间比
    k = 1.002737909350795  # d(gast)/dt
    k = k * np.pi / 43200
    itrf_veliocity_vec = np.array([[itrf_sat_vx], [itrf_sat_vy], [itrf_sat_vz]])
    temp_mat_1 = np.dot(rz_gast, itrf_veliocity_vec)
    temp_mat_2 = np.array([[k * (-np.sin(gast)), k * (-np.cos(gast)), 0],
                            [k * np.cos(gast), k * (-np.sin(gast)), 0],
                            [0, 0, 0]
                            ])
    temp_mat_3 = np.dot(temp_mat_2, itrf_pos_vec)
    icrf_velocity_vec = temp_mat_1 + temp_mat_3
    return icrf_pos_vec, icrf_velocity_vec


def icrf_2_itrf(time_mjd, icrf_sat_x, icrf_sat_y, icrf_sat_z, icrf_sat_vx,
                icrf_sat_vy, icrf_sat_vz):
    gast = mjd_2_gast(time_mjd)
    rz_gast = np.array([[np.cos(gast), np.sin(gast), 0],
                        [-np.sin(gast), np.cos(gast), 0],
                        [0, 0, 1]
                        ])
    icrf_pos_vec = np.array([[icrf_sat_x], [icrf_sat_y], [icrf_sat_z]])
    itrf_pos_vec = np.dot(rz_gast, icrf_pos_vec)
    # 恒星时/平均时间比
    k = 1.002737909350795  # d(gast)/dt
    k = k * np.pi / 43200
    icrf_veliocity_mat = np.array([[icrf_sat_vx], [icrf_sat_vy], [icrf_sat_vz]])
    temp_mat_1 = np.dot(rz_gast, icrf_veliocity_mat)
    temp_mat_2 = np.array([[k * (-np.sin(gast)), k * np.cos(gast), 0],
                           [-k * np.cos(gast), k * (-np.sin(gast)), 0],
                           [0, 0, 0]
                           ])
    temp_mat_3 = np.dot(temp_mat_2, icrf_pos_vec)
    itrf_velocity_mat = temp_mat_1 + temp_mat_3
    return [itrf_pos_vec[0][0], itrf_pos_vec[1][0], itrf_pos_vec[2][0],
            itrf_velocity_mat[0][0], itrf_velocity_mat[1][0], itrf_velocity_mat[2][0]]


# # # # # # # # # # # # # # #
# 3. unit transformation  # #
# # # # # # # # # # # # # # #
def freq_2_wavelength(obs_freq):
    """
    freq to wavelength
    :param obs_freq:
    :return: wavelength
    """
    wavelength = lc.light_speed / obs_freq
    return wavelength


def angle_str_2_rad(angle_str):
    """
    sometimes the source info is given in the 'dms' format
    we transform it into the radian to facilitate the calculation
    :param angle_str: "23d43m54s"
    :return: 0.414195720319121
    """
    angle_str = angle_str.replace('d', ':')
    angle_str = angle_str.replace('m', ':')
    angle_str = angle_str.replace('s', '')
    angle_str = angle_str.split(':')
    angle_d = int(angle_str[0])
    angle_m = int(angle_str[1])
    angle_s = float(angle_str[2])
    angle_rad = angle_2_rad(angle_d, angle_m, angle_s)
    return angle_rad


def angle_2_rad(dd, mm, ss):
    """
    transform angle to radian
    :param dd:
    :param mm:
    :param ss:
    :return: radian
    """
    if dd < 0:
        flag = -1
        dd = np.abs(dd)
    else:
        flag = 1
    angle_rad = (dd + ((mm * 60.0 + ss) / 3600.0)) / 180 * np.pi

    return angle_rad * flag


def rad_2_angle(rad):
    """
    transform radian to angle
    :param rad:
    :return: angle [0,180]
    """
    return rad * 180 / np.pi


def sgn(x):
    """
    sign function
    :param x: an integer
    :return: the sign
    """
    return np.sign(x)


def angle_btw_vec(vec_x, vec_y):
    """
    calculate the included angle
    :param vec_x: 3x1 vector, in rad unit
    :param vec_y: 3x1 vector, in rad unit
    :return: included angle between two vectors, belongs to [0,pi]
    """
    arc = vec_x[0][0] * vec_y[0][0] + vec_x[1][0] * vec_y[1][0] + vec_x[2][0] * vec_y[2][0]
    if arc > 1 or arc < -1:
        arc = sgn(arc)
    arc = np.arccos(arc)
    return arc  # 两个单位向量的夹角，范围0-pi
