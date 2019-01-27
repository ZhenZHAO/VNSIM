"""
@functions: model the observation ability of stations and satellites
@author: Zhen ZHAO
@date: April 25, 2018
"""
import utility as ut
import model_effect as me
import model_satellite as ms
import numpy as np


# 问题
# 获得的是在某个时刻卫星的可观测性

def obs_all_active_sta(time_mjd, ra_src, dec_src, pos_mat_sat, pos_mat_telemetry,
                       pos_mat_vlbi, baseline_type, cutoff_dict, precession_mode):
    """
    输入卫星天球坐标系
    :param time_mjd: 观测时间
    :param ra_src:   源的位置
    :param dec_src:
    :param pos_mat_sat:  选择的所有卫星的位置信息
    :param pos_mat_telemetry:  选择的所有遥测站的位置信息
    :param pos_mat_vlbi:    选择的所有VLBI站的位置信息
    :param baseline_type:   基线类型：001(1)->select GtoG 010(2)->SELECT GtoS, 100(4)->StoS
    :param cutoff_dict:     观测站和遥测站的截止角
    :param precession_mode: 地球的进动模型
    :return: 返回这个时间点所有可观测的站信息
    """
    # 地面站的可观测性
    vlbi_visibility_result = []  # VLBI站可观测性
    for i in np.arange(len(pos_mat_vlbi)):
        vlbi_visibility_result.extend([pos_mat_vlbi[i][0], True])
    if (baseline_type & 3) != 0:  # GroundToGround, GroundToSpace
        for j in np.arange(len(pos_mat_vlbi)):
            long_vlbi, lat_vlbi, height_vlbi = ut.itrf_2_geographic(pos_mat_vlbi[j][1],
                                                                    pos_mat_vlbi[j][2], pos_mat_vlbi[j][3])
            # 1. 首先生成观测站在0-360度方向上的cutoff角度设置
            vlbi_loc_horiz = []
            if cutoff_dict['flag'] == 0:
                for _ in np.arange(0, 360):
                    vlbi_loc_horiz.append(pos_mat_vlbi[j][4])
            elif cutoff_dict['flag'] == 1:
                for _ in np.arange(0, 360):
                    vlbi_loc_horiz.append(cutoff_dict['CutAngle'])
            elif cutoff_dict['flag'] == 2:
                angle = np.max([pos_mat_vlbi[j][4], cutoff_dict['CutAngle']])
                for _ in np.arange(0, 360):
                    vlbi_loc_horiz.append(angle)
            elif cutoff_dict['flag'] == 3:
                angle = np.min([pos_mat_vlbi[j][4], cutoff_dict['CutAngle']])
                for _ in np.arange(0, 360):
                    vlbi_loc_horiz.append(angle)
            # 2. 获得该站的可观测性结果并记录
            visibility = obs_judge_active_vlbi_station(ra_src, dec_src, time_mjd, long_vlbi, lat_vlbi, vlbi_loc_horiz)
            vlbi_visibility_result[2 * j + 1] = visibility

    # 卫星的可观测性
    satellite_visibility_result = []
    for i in np.arange(len(pos_mat_sat)):
        satellite_visibility_result.extend([pos_mat_sat[i][0], True])
    sat_lst = []  # 存放卫星名称和ITRF坐标
    sat_itrf = []
    if (baseline_type & 6) != 0:  # if 'GroundToSpace' or 'SpaceToSpace'
        # 计算得卫星的ICRF坐标
        for j in np.arange(len(pos_mat_sat)):  # 卫星的输入参数 :卫星轨道椭圆半长轴a,偏心率e,
            # i为轨道平面倾角,(AOP)omega为近地点角,Omega(LOAN)为升交点赤经,M为平近点角
            satellite_a = pos_mat_sat[j][1]
            satellite_e = pos_mat_sat[j][2]
            satellite_i = pos_mat_sat[j][3] * np.pi / 180
            satellite_aop = pos_mat_sat[j][4] * np.pi / 180
            satellite_loan = pos_mat_sat[j][5] * np.pi / 180
            satellite_m = pos_mat_sat[j][6] * np.pi / 180
            satellite_epoch = pos_mat_sat[j][7]
            temp_lst = ms.get_satellite_position(satellite_a, satellite_e, satellite_i,
                                                 satellite_aop, satellite_loan, satellite_m,
                                                 satellite_epoch, time_mjd, precession_mode)
            sat_lst.append(temp_lst)  # 卫星在MJD时刻的位置，和速度
        # ICRF->ITRF
        for j in np.arange(len(sat_lst)):
            temp_lst = ut.icrf_2_itrf(time_mjd, sat_lst[j][6], sat_lst[j][7], sat_lst[j][8], sat_lst[j][9],
                                      sat_lst[j][10], sat_lst[j][11])  # 输入一个卫星的位置和速度信息
            sat_lst1 = [pos_mat_sat[j][0]]
            sat_lst1.extend(temp_lst)
            sat_itrf.append(sat_lst1)
            # 获取卫星的可见度
            visible = obs_judge_active_satellite(time_mjd, sat_lst1, pos_mat_telemetry, ra_src, dec_src, cutoff_dict)
            satellite_visibility_result[2 * j + 1] = visible  # visible
    # 返回结果
    if (baseline_type & 6) != 0:  # 如果跟卫星相关，返回卫星的itrf坐标
        return vlbi_visibility_result, satellite_visibility_result, sat_itrf
    else:  # 如果跟卫星无关，其实应该直接返回 vlbi_visibility_result 即可
        return vlbi_visibility_result, satellite_visibility_result, pos_mat_sat


def obs_all_active_vlbi(time_mjd, ra_src, dec_src, pos_mat_vlbi, cutoff_dict):
    vlbi_visibility_result = []  # VLBI站可观测性
    for i in np.arange(len(pos_mat_vlbi)):
        vlbi_visibility_result.extend([pos_mat_vlbi[i][0], True])
    # if (baseline_type & 3) != 0:  # GroundToGround, GroundToSpace
    for j in np.arange(len(pos_mat_vlbi)):
        long_vlbi, lat_vlbi, height_vlbi = ut.itrf_2_geographic(pos_mat_vlbi[j][1],
                                                                pos_mat_vlbi[j][2], pos_mat_vlbi[j][3])
        # 1. 首先生成观测站在0-360度方向上的cutoff角度设置
        vlbi_loc_horiz = []
        if cutoff_dict['flag'] == 0:
            for _ in np.arange(0, 360):
                vlbi_loc_horiz.append(pos_mat_vlbi[j][4])
        elif cutoff_dict['flag'] == 1:
            for _ in np.arange(0, 360):
                vlbi_loc_horiz.append(cutoff_dict['CutAngle'])
        elif cutoff_dict['flag'] == 2:
            angle = np.max([pos_mat_vlbi[j][4], cutoff_dict['CutAngle']])
            for _ in np.arange(0, 360):
                vlbi_loc_horiz.append(angle)
        elif cutoff_dict['flag'] == 3:
            angle = np.min([pos_mat_vlbi[j][4], cutoff_dict['CutAngle']])
            for _ in np.arange(0, 360):
                vlbi_loc_horiz.append(angle)
        # 2. 获得该站的可观测性结果并记录
        visibility = obs_judge_active_vlbi_station(ra_src, dec_src, time_mjd, long_vlbi, lat_vlbi, vlbi_loc_horiz)
        vlbi_visibility_result[2 * j + 1] = visibility
    return vlbi_visibility_result


def obs_judge_active_vlbi_station(ra_src, dec_src, time_mjd, long_vlbi_sta, lat_vlbi_sta, horizon_vlbi_sta):
    """
    VLBI站能否观测到源
    :param ra_src:  源的位置
    :param dec_src:
    :param time_mjd:  观测时间
    :param long_vlbi_sta:  台站的坐标
    :param lat_vlbi_sta:
    :param horizon_vlbi_sta:
    :return: yes / no
    """
    source_azimuth, source_elevation = ut.equatorial_2_horizontal(time_mjd, ra_src, dec_src,
                                                                  long_vlbi_sta, lat_vlbi_sta)
    azimuth_deg = ut.rad_2_angle(source_azimuth)
    elevation_deg = ut.rad_2_angle(source_elevation)
    # 将AzimuthDeg限定在0到360之间
    azimuth_deg = azimuth_deg - azimuth_deg // 360 * 360
    # 计算这个方向上能看到源的最小高度值
    azimuth_deg1 = int(azimuth_deg)
    if azimuth_deg1 == 359:
        azimuth_deg2 = 0
    else:
        azimuth_deg2 = azimuth_deg1 + 1
    elevation1 = horizon_vlbi_sta[azimuth_deg1]
    elevation2 = horizon_vlbi_sta[azimuth_deg2]
    min_elevation = elevation1 + (azimuth_deg - azimuth_deg1) * (
            elevation2 - elevation1)  # 斜率k=(elevation2-elevation1)/(AzimuthDeg2-azimuth_deg1)
    # print(" elevation_deg: %f, min_elevation, %f" % (elevation_deg, min_elevation))
    if elevation_deg > min_elevation:  # =(min_elevation-elevation1)/(azimuth_deg-azimuth_deg1)
        visibility = True
    else:
        visibility = False
    return visibility


def obs_all_active_sat(time_mjd, ra_src, dec_src, pos_mat_sat_kepler, pos_mat_telemetry,
                       baseline_type, cutoff_dict, precession_mode):
    # 卫星的可观测性
    satellite_visibility_result = []
    for i in np.arange(len(pos_mat_sat_kepler)):
        satellite_visibility_result.extend([pos_mat_sat_kepler[i][0], True])
    sat_lst = []  # 存放卫星名称和ITRF坐标
    sat_itrf = []
    if (baseline_type & 6) != 0:  # if 'GroundToSpace' or 'SpaceToSpace'
        # 计算得卫星的ICRF坐标
        for j in np.arange(len(pos_mat_sat_kepler)):  # 卫星的输入参数 :卫星轨道椭圆半长轴a,偏心率e,
            # i为轨道平面倾角,(AOP)omega为近地点角,Omega(LOAN)为升交点赤经,M为平近点角
            satellite_a = pos_mat_sat_kepler[j][1]
            satellite_e = pos_mat_sat_kepler[j][2]
            satellite_i = pos_mat_sat_kepler[j][3] * np.pi / 180
            satellite_aop = pos_mat_sat_kepler[j][4] * np.pi / 180
            satellite_loan = pos_mat_sat_kepler[j][5] * np.pi / 180
            satellite_m = pos_mat_sat_kepler[j][6] * np.pi / 180
            satellite_epoch = pos_mat_sat_kepler[j][7]
            temp_lst = ms.get_satellite_position(satellite_a, satellite_e, satellite_i,
                                                 satellite_aop, satellite_loan, satellite_m,
                                                 satellite_epoch, time_mjd, precession_mode)
            sat_lst.append(temp_lst)  # 卫星在MJD时刻的位置，和速度
        # ICRF->ITRF
        for j in np.arange(len(sat_lst)):
            temp_lst = ut.icrf_2_itrf(time_mjd, sat_lst[j][6], sat_lst[j][7], sat_lst[j][8], sat_lst[j][9],
                                      sat_lst[j][10], sat_lst[j][11])  # 输入一个卫星的位置和速度信息
            sat_lst1 = [pos_mat_sat_kepler[j][0]]
            sat_lst1.extend(temp_lst)
            sat_itrf.append(sat_lst1)
            # 获取卫星的可见度
            visible = obs_judge_active_satellite(time_mjd, sat_lst1, pos_mat_telemetry, ra_src, dec_src, cutoff_dict)
            satellite_visibility_result[2 * j + 1] = visible  # visible

    if (baseline_type & 6) != 0:  # 如果跟卫星相关，返回卫星的itrf坐标
        return satellite_visibility_result, sat_itrf
    else:  # 如果跟卫星无关，其实应该直接返回 vlbi_visibility_result 即可
        return satellite_visibility_result, pos_mat_sat_kepler


def obs_judge_active_satellite_with_kepler(time_mjd, ra_src, dec_src,
                                           pos_vec_sat_kepler, pos_mat_telemetry,
                                           baseline_type, cutoff_dict, precession_mode):
    if baseline_type & 6 == 0:
        return False

    sat_itrf = []  # 存放卫星名称和ITRF坐标
    sat_itrf[0] = pos_vec_sat_kepler[0]

    # 计算得卫星的ICRF坐标
    sat_lst = ms.get_satellite_position(pos_vec_sat_kepler[1], pos_vec_sat_kepler[2], pos_vec_sat_kepler[3],
                                        pos_vec_sat_kepler[4], pos_vec_sat_kepler[5], pos_vec_sat_kepler[6],
                                        pos_vec_sat_kepler[7], time_mjd, precession_mode)
    # ICRF->ITRF
    temp_lst = ut.icrf_2_itrf(time_mjd, sat_lst[6], sat_lst[7], sat_lst[8], sat_lst[9],
                              sat_lst[10], sat_lst[11])  # 输入一个卫星的位置和速度信息
    sat_itrf.extend(temp_lst)

    # check the visibility
    visible = obs_judge_active_satellite(time_mjd, sat_itrf, pos_mat_telemetry, ra_src, dec_src, cutoff_dict)
    return visible


def obs_judge_active_satellite(time_mjd, pos_vec_sat, pos_mat_telemetry, ra_src, dec_src, cutoff_dict):
    """
    当前用的遥测站和卫星能否观测到源
    :param time_mjd: 观测时间
    :param pos_vec_sat:  卫星的位置
    :param pos_mat_telemetry:  遥测站的位置
    :param ra_src:  源的信息
    :param dec_src:
    :param cutoff_dict:  {'flag':0,'CutAngle':10}
    :return:
    """
    # ["tele1", False, "tele2", False] -- 其实用字典更好，后面更正
    telemetry_visibility_result = []
    for i in np.arange(len(pos_mat_telemetry)):
        telemetry_visibility_result.extend([pos_mat_telemetry[i][0], False])
    # 遍历各个遥测站，看当前遥测站能否受到卫星的信号
    for i in np.arange(len(pos_mat_telemetry)):
        # 1. 首先生成遥测站在0-360度方向上的cutoff角度设置
        tele_loc_horiz = []
        if cutoff_dict['flag'] == 0:  # mod 1, 根据数据库中遥测站的信息设置cutoff，第5列
            for _ in np.arange(0, 360):
                tele_loc_horiz.append(pos_mat_telemetry[i][4])
        elif cutoff_dict['flag'] == 1:  # mod 2, 根据界面上直接设置遥测站的cutoff
            for _ in np.arange(0, 360):
                tele_loc_horiz.append(cutoff_dict['CutAngle'])
        elif cutoff_dict['flag'] == 2:  # mod 3 ＝ mod 1 and mod 2
            angle = np.max([pos_mat_telemetry[i][4], cutoff_dict['CutAngle']])
            for _ in np.arange(0, 360):
                tele_loc_horiz.append(angle)
        elif cutoff_dict['flag'] == 3:  # mod 3 ＝ mod 1 or mod 2
            angle = np.min([pos_mat_telemetry[i][4], cutoff_dict['CutAngle']])
            for _ in np.arange(0, 360):
                tele_loc_horiz.append(angle)
        # 2. 调用自函数去获得遥测站的可见性
        long_telemetry, lat_telemetry, height_telemetry = ut.itrf_2_geographic(pos_mat_telemetry[0][1],
                                                                               pos_mat_telemetry[0][2],
                                                                               pos_mat_telemetry[0][3])
        visibility = obs_telemetry_to_satellite(pos_vec_sat[1], pos_vec_sat[2], pos_vec_sat[3],
                                                long_telemetry, lat_telemetry, height_telemetry, tele_loc_horiz)
        # 3. 记录观测结果
        telemetry_visibility_result[2 * i + 1] = visibility
    # 若有遥测站可以看到卫星，则认为该卫星可用
    if True in telemetry_visibility_result:
        # 判断卫星能否看到源，若卫星能看到源，则返回True
        return obs_satellite_to_source(ra_src, dec_src, time_mjd, pos_vec_sat[1], pos_vec_sat[2], pos_vec_sat[3])
    else:
        return False


def obs_satellite_to_source(ra_src, dec_src, time_mjd, pos_sat_x, pos_sat_y, pos_sat_z):
    """
    卫星位置能否观察到源
    :param ra_src:   源赤经
    :param dec_src:  源赤纬
    :param time_mjd:  mjd时间
    :param pos_sat_x:  卫星ICRF位置坐标
    :param pos_sat_y:
    :param pos_sat_z:
    :return:
    """
    # 卫星地心距离
    r = np.sqrt(pos_sat_x ** 2 + pos_sat_y ** 2 + pos_sat_z ** 2)
    # 源赤道单位向量
    src_equ_x = np.cos(dec_src) * np.cos(ra_src)
    src_equ_y = np.cos(dec_src) * np.sin(ra_src)
    src_equ_z = np.sin(dec_src)
    # 计算黄赤交角
    epsilon = ut.ecliptic_obliquity(time_mjd)
    # 将源赤道单位向量转化为3x1矩阵
    src_equ = np.array([[src_equ_x], [src_equ_y], [src_equ_z]])
    # 计算源黄道单位矢量，返回类型为3X1矩阵
    src_ecliptic = ut.equatorial_2_ecliptic(src_equ, epsilon)
    # 计算太阳黄经  ?指的是太阳地心黄经还是太阳视黄经
    sun_ecliptic_long = me.sun_ecliptic_pos(time_mjd + 2400000.5)
    sun_ecliptic_x = np.cos(sun_ecliptic_long)
    sun_ecliptic_y = np.sin(sun_ecliptic_long)
    sun_ecliptic_z = 0
    sun_ecliptic = np.array([[sun_ecliptic_x], [sun_ecliptic_y], [sun_ecliptic_z]])
    # 检查约束？
    angle = ut.angle_btw_vec(src_ecliptic, sun_ecliptic)  # Angle的范围是多少 0~pi
    # print("angle Between SourceEcliptic and Sun Ecliptic",angle,"np.pi*7/18=",np.pi*7/18)
    if angle < np.pi * 7 / 18:
        visibility = False
    else:
        # 计算月球黄道坐标及月球地心距离 月球地心距离指的是月球距离地球的距离还是月球本身的半径？
        moon_ecliptic_long, moon_ecliptic_lat, moon_r = me.moon_ecliptic_pos(time_mjd + 2400000.5)
        # 计算月球地心黄道向量
        moon_ecliptic_x = moon_r * np.cos(moon_ecliptic_long) * np.cos(moon_ecliptic_lat)
        moon_ecliptic_y = moon_r * np.sin(moon_ecliptic_long) * np.cos(moon_ecliptic_lat)
        moon_ecliptic_z = moon_r * np.sin(moon_ecliptic_lat)
        # 计算卫星中心的黄道单位矢量
        sta_pos = np.array([[pos_sat_x], [pos_sat_y], [pos_sat_z]])
        earth_ecliptic = me.earth_ecliptic_pos(sta_pos, epsilon)
        # 计算月球卫星中心的黄道向量
        moon_ecliptic_x = moon_ecliptic_x + earth_ecliptic[0][0] * r
        moon_ecliptic_y = moon_ecliptic_y + earth_ecliptic[1][0] * r
        moon_ecliptic_z = moon_ecliptic_z + earth_ecliptic[2][0] * r
        # 计算月球卫星距离
        moon_sat_r = np.sqrt(moon_ecliptic_x ** 2 + moon_ecliptic_y ** 2 + moon_ecliptic_z ** 2)
        # 计算月球黄道单位向量
        moon_ecliptic_x = moon_ecliptic_x / moon_sat_r
        moon_ecliptic_y = moon_ecliptic_y / moon_sat_r
        moon_ecliptic_z = moon_ecliptic_z / moon_sat_r
        moon_ecliptic = np.array([[moon_ecliptic_x], [moon_ecliptic_y], [moon_ecliptic_z]])
        angle = ut.angle_btw_vec(src_ecliptic, moon_ecliptic)
        # 检查月球约束
        if angle < np.pi / 9:
            visibility = False
        else:
            visibility = True
    return visibility


def obs_telemetry_to_satellite(pos_x_sat, pos_y_sat, pos_z_sat, long_telemetry,
                               lat_telemetry, height_telemetry, loc_horiz_telemetry):
    """
    判断遥测站的可观测性, i.e., 当前卫星能不能被看到
    :param pos_x_sat:  卫星的位置信息
    :param pos_y_sat:
    :param pos_z_sat:
    :param long_telemetry:  遥测站的位置信息
    :param lat_telemetry:
    :param height_telemetry:
    :param loc_horiz_telemetry:  0-359角度所对应的弧度
    :return: yes / no
    """
    sat_lst = [pos_x_sat, pos_y_sat, pos_z_sat, 0, 0, 0]
    sat_az_el = ut.itrf_2_horizontal(sat_lst, long_telemetry, lat_telemetry, height_telemetry)
    azimuth = sat_az_el[0]
    elevation = sat_az_el[1]
    # 将方位角的单位从弧度转变成角度
    azimuth_deg = azimuth * 180 / np.pi
    elevation_deg = elevation * 180 / np.pi
    # 将方位角的范围限定在0-360之间
    azimuth_deg = azimuth_deg - int(azimuth_deg / 360) * 360  # 遥测站的方位角
    # 计算这个方向上能看到源的最小高度值
    azimuth_deg1 = int(azimuth_deg)
    if azimuth_deg1 == 359:
        azimuth_deg2 = 0
    else:
        azimuth_deg2 = azimuth_deg1 + 1
    elevation1 = loc_horiz_telemetry[azimuth_deg1]
    elevation2 = loc_horiz_telemetry[azimuth_deg2]  # 线性插值
    min_elevation = elevation1 + (azimuth_deg - azimuth_deg1) * (elevation2 - elevation1)
    if elevation_deg > min_elevation:
        visibility = True
    else:
        visibility = False
    return visibility
