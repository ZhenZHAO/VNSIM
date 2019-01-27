"""
@functions: model the effect from sun, moon, and earth on the satellite
@author: Zhen ZHAO
@date: April 24, 2018
"""
import numpy as np
import load_conf as lc
import utility as ut


def sun_ecliptic_pos(time_jd):
    """
    太阳视黄经的计算方法来自<天文算法>136页
    :param time_jd: JD是儒略日数
    :return: 太阳的位置
    """
    # JD是儒略日数
    T = (time_jd - 2451545.0) / 36525  # T是自J2000起算的儒略日世纪数
    L0 = 280.46645 + T * (36000.76983 + 0.0003032 * T)  # 太阳几何平黄经,UNIT:DEG
    M = 357.52910 + T * (35999.05030 - T * (0.0001559 + 0.00000048 * T))  # 太阳平近点角,DEG
    #  e=0.016708617-T*(0.000042037+0.0000001236*T)   #地球轨道离心率
    C = (1.914600 - T * (0.004817 + 0.000014 * T)) * np.sin(M / 180 * np.pi)  # 太阳中间方程 deg
    theta = L0 + C  # 太阳的真黄经,deg
    theta = np.mod(theta, 360)
    # v=M+C  #真近点角 deg
    #    R=(1.000001018*(1-e**2))/(1+e*np.cos(v*np.pi/180))#日地距离，单位为1个天文单位，一个天文单位为149597870700m
    omega = 125.04 - 1934.136 * T
    lamda = theta - 0.00569 - 0.00478 * np.sin(omega * np.pi / 180)  # 太阳视黄经deg，太阳黄纬由于不超过1.2秒,设置为0
    lamda = np.mod(lamda, 360)
    lamda = lamda / 180 * np.pi
    #    Epsilon=23.6  #黄赤交角 deg
    #    alpha=np.arctan2(np.cos(Epsilon/180*np.pi)*np.sin(theta/180*np.pi),np.cos(theta/180*np.pi))  #太阳地心赤经，rad
    #    delta=np.arcsin(np.sin(Epsilon/180*np.pi)*np.sin(theta/180*np.pi))  #太阳地心黄经
    return lamda  # rad


def moon_ecliptic_pos(time_jd):
    """
    月球位置的计算参照<天文算法>240页
    :param time_jd:
    :return:
    """
    T = (time_jd - 2451545.0) / 36525  # T是自J2000起算的儒略日世纪数
    # 计算月球的平黄经
    L1 = 218.3164591 + T * (481267.88134236 - T * (0.0013268 - T / 538841.0 + T * T / 65194000.0))
    # 计算月日距角
    D = 297.8502042 + T * (445267.1115168 - T * (0.0016300 - T / 545868.0 + T * T / 113065000.0))
    # 计算太阳平近点角
    M = 357.5291092 + T * (35999.0502909 - T * (0.0001536 + T / 24490000))
    # 计算月亮平近点角
    M1 = 134.9634114 + T * (477198.8675313 + T * (0.0089970 + T / 69699 - T * T / 14712000))
    # 计算月球经度参数（到升交点的平角距离)
    F = 93.2720993 + T * (483202.0175273 - T * (0.0034029 + T / 3526000 - T * T / 863310000))
    # 几个必要的摄动参数
    A1 = 119.75 + 131.849 * T  # 金星的摄动
    A2 = 53.09 + 479264.290 * T  # 木星的摄动
    A3 = 313.45 + 481266.484 * T
    E = 1 - 0.002516 * T - 0.0000074 * T * T  # 计算反映地球轨道偏心率变化
    DE = np.pi / 180
    # 月球黄经周期项角度D、M、M1、F的组合系数 每个列表有60个系数
    La = [0, 2, 2, 0, 0, 0, 2, 2, 2, 2,
          0, 1, 0, 2, 0, 0, 4, 0, 4, 2,
          2, 1, 1, 2, 2, 4, 2, 0, 2, 2,
          1, 2, 0, 0, 2, 2, 2, 4, 0, 3,
          2, 4, 0, 2, 2, 2, 4, 0, 4, 1,
          2, 0, 1, 3, 4, 2, 0, 1, 2, 2]
    Lb = [0, 0, 0, 0, 1, 0, 0, -1, 0, -1,
          1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
          1, 0, 1, -1, 0, 0, 0, 1, 0, -1,
          0, -2, 1, 2, -2, 0, 0, -1, 0, 0,
          1, -1, 2, 2, 1, -1, 0, 0, -1, 0,
          1, 0, 1, 0, 0, -1, 2, 1, 0, 0]
    Lc = [1, -1, 0, 2, 0, 0, -2, -1, 1, 0,
          -1, 0, 1, 0, 1, 1, -1, 3, -2, -1,
          0, -1, 0, 1, 2, 0, -3, -2, -1, -2,
          1, 0, 2, 0, -1, 1, 0, -1, 2, -1,
          1, -2, -1, -1, -2, 0, 1, 4, 0, -2,
          0, 2, 1, -2, -3, 2, 1, -1, 3, -1]
    Ld = [0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
          0, 0, 0, -2, 2, -2, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
          0, 0, 0, 0, 0, -2, 2, 0, 2, 0,
          0, 0, 0, 0, 0, -2, 0, 0, 0, 0,
          -2, -2, 0, 0, 0, 0, 0, 0, 0, -2]
    # 月球黄经正弦函数的各项振幅值
    Sl = [6288774, 1274027, 658314, 213618, -185116, -114332, 58793, 57066, 53322, 45758,
          -40923, -34720, -30383, 15327, -12528, 10980, 10675, 10034, 8548, -7888,
          -6766, -5163, 4987, 4036, 3994, 3861, 3665, -2689, -2602, 2390,
          -2348, 2236, -2120, -2069, 2048, -1773, -1595, 1215, -1110, -892,
          -810, 759, -713, -700, 691, 596, 549, 537, 520, -487,
          -399, -381, 351, -340, 330, 327, -323, 299, 294, 0]
    # 月球黄经余弦函数的各项振幅值
    Sr = [-20905355, -3699111, -2955968, -569925, 48888, -3149, 246158, -152138, -170733, -204586,
          -129620, 108743, 104755, 10321, 0, 79661, -34782, -23210, -21636, 24208,
          30824, -8379, -16675, -12831, -10445, -11650, 14403, -7003, 0, 10056,
          6322, -9884, 5751, 0, -4950, 4130, 0, -3958, 0, 3258,
          2616, 0, -2117, 2354, 0, 0, 0, 0, 0, 0,
          0, -4421, 0, 0, 0, 0, 1165, 0, 0, 8752]
    # 计算月球黄经周期项
    sumi = 0
    for i in range(0, 60):
        SIN1 = La[i] * D + Lb[i] * M + Lc[i] * M1 + Ld[i] * F + sumi
        sumi = sumi + Sl[i] * np.sin(SIN1 * DE) * 0.000001 * np.power(E, np.fabs(Lb[i]))
    # 计算月球黄经，单位为度
    lamda = L1 + sumi + (3958 * np.sin(A1 * DE) + 1962 * np.sin((L1 - F) * DE) + 318 * np.sin(A2 * DE)) / 1000000
    lamda = np.mod(lamda, 360)
    lamda = lamda / 180 * np.pi  # 将角度转换为弧度
    # 月球黄纬周期项角度D、M、M1、F的组合系数 每个列表有60个系数
    Ba = [0, 0, 0, 2, 2, 2, 2, 0, 2, 0,
          2, 2, 2, 2, 2, 2, 2, 0, 4, 0,
          0, 0, 1, 0, 0, 0, 1, 0, 4, 4,
          0, 4, 2, 2, 2, 2, 0, 2, 2, 2,
          2, 4, 2, 2, 0, 2, 1, 1, 0, 2,
          1, 2, 0, 4, 4, 1, 4, 1, 4, 2]
    Bb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          -1, 0, 0, 1, -1, -1, -1, 1, 0, 1,
          0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
          0, 0, 0, 0, -1, 0, 0, 0, 0, 1,
          1, 0, -1, -2, 0, 1, 1, 1, 1, 1,
          0, -1, 1, 0, -1, 0, 0, 0, -1, -2]
    Bc = [0, 1, 1, 0, -1, -1, 0, 2, 1, 2,
          0, -2, 1, 0, -1, 0, -1, -1, -1, 0,
          0, -1, 0, 1, 1, 0, 0, 3, 0, -1,
          1, -2, 0, 2, 1, -2, 3, 2, -3, -1,
          0, 0, 1, 0, 1, 1, 0, 0, -2, -1,
          1, -2, 2, -2, -1, 1, 1, -1, 0, 0]
    Bd = [1, 1, -1, -1, 1, -1, 1, 1, -1, -1,
          -1, -1, 1, -1, 1, 1, -1, -1, -1, 1,
          3, 1, 1, 1, -1, -1, -1, 1, -1, 1,
          -3, 1, -3, -1, -1, 1, -1, 1, -1, 1,
          1, 1, 1, -1, 3, -1, -1, 1, -1, -1,
          1, -1, 1, -1, -1, -1, -1, -1, -1, 1]
    # 月球黄纬周期项各项正弦振幅系数
    Sb = [5128122, 280602, 277693, 173237, 55413, 46271, 32573, 17198, 9266, 8822,
          8216, 4324, 4200, -3359, 2463, 2211, 2065, -1870, 1828, -1794,
          -1749, -1565, -1491, -1475, -1410, -1344, -1335, 1107, 1021, 833,
          777, 671, 607, 596, 491, -451, 439, 422, 421, -366,
          -351, 331, 315, 302, -283, -229, 223, 223, -220, -220,
          -185, 181, -177, 176, 166, -164, 132, -119, 115, 107]
    # 计算月球黄纬周期项
    sumb = 0
    for i in range(0, 60):
        SIN2 = Ba[i] * D + Bb[i] * M + Bc[i] * M1 + Bd[i] * F
        sumb = sumb + Sb[i] * 0.000001 * np.sin(SIN2 * DE) * np.power(E, np.fabs(Lb[i]))
    # 计算月球黄纬,单位为度
    beta = sumb + (-2235 * np.sin(L1 * DE) + 382 * np.sin(A3 * DE) + 175 * np.sin((A1 - F) * DE) + 175 * np.sin(
        (A1 + F) * DE) + 127 * np.sin((L1 - M1) * DE) - 115 * np.sin((L1 + M1) * DE)) / 1000000
    beta = beta / 180 * np.pi  # 将角度转换为弧度
    # 计算月球地心距离,单位km
    sumr = 0
    for i in range(0, 60):
        COS1 = La[i] * D + Lb[i] * M + Lc[i] * M1 + Ld[i] * F
        sumr = sumr + Sr[i] * 0.001 * np.cos(COS1 * DE) * np.power(E, np.fabs(Lb[i]))
    dist = 385000.56 + sumr  # 单位km

    return lamda, beta, dist  # rad rad km


def moon_ra_dec_cal(start_time_mjd, stop_time_mjd, time_step):
    moon_ra = []     # 2014.8.18. 天文算法书
    moon_dec = []
    jd_time = start_time_mjd + 2400000.5
    while True:
        lamda,beta,dist = moon_ecliptic_pos(jd_time)
        # lamda rad 黄经 beta rad 黄纬 dist km 什么距离？
        T = (jd_time-2451545.0)/36525   # T是自J2000起算的儒略日世纪数
        # 计算月球的平黄经
        L1 = 218.3164591+T*(481267.88134236-T*(0.0013268-T/538841.0+T*T/65194000.0))
        Theta = L1  #太阳的真黄经,deg
        epsilon = 23.439291
        # 计算月亮赤经 （黄道坐标转化为赤道坐标）天文算法书76页12.3
        # MRa = np.arctan2(np.cos(epsilon*np.pi/180)*np.sin(lamda*np.pi/180)-np.tan(beta*np.pi/180)*np.sin(epsilon*np.pi/180),np.cos(lamda*np.pi/180))
        MRa = np.arctan2(np.cos(epsilon*np.pi/180)*np.sin(Theta*np.pi/180),np.cos(Theta*np.pi/180))
        MRa = MRa*180/np.pi # rad 转换为 deg
        if MRa < 0:
            MRa = MRa + 360
        MRa = MRa/15  # rad to time
#        print 'MRa=',MRa
        # 计算月亮赤纬  （黄道坐标转化为赤道坐标）天文算法书76页12.4
        # MDec = np.arcsin(np.sin(beta*np.pi/180)*np.cos(epsilon*np.pi/180)+np.sin(epsilon*np.pi/180)*np.cos(beta*np.pi/180)*np.sin(lamda*np.pi/180))  # rad
        MDec = np.arcsin(np.sin(epsilon*np.pi/180)*np.sin(Theta*np.pi/180))  # rad
        MDec = MDec*180/np.pi  # rad 转换为 deg
        # print('MDec=', MDec)
        jd_time = jd_time + time_step
        moon_ra.append(MRa)
        moon_dec.append(MDec)
        if stop_time_mjd < (jd_time - 2400000.5):  # 浮点数的会出现问题 a=1.2,b=2.5 a>=a+b-b不一定为真
            break
    return moon_ra, moon_dec


def sun_ra_dec_cal(start_time_mjd, stop_time_mjd, time_step):
    sun_ra = []  # 2014.5.20. 依据天文算法book计算视赤经视赤纬
    sun_dec = []
    jd_time = start_time_mjd + 2400000.5
#    print "start_time_mjd=",start_time_mjd
#    print "stop_time_mjd=",stop_time_mjd
#    if(stop_time_mjd>=(jd_time - 2400000.5-1)):
#        a=stop_time_mjd
#        b=jd_time - 2400000.5
#        print "a=%.10f"%(a),"b=%.10f"%(b),"a>=b?"
#        if(a>=b):
#            print "True"
#        else:
#            print "false"
    while True:
        T = (jd_time-2451545.0)/36525   # T是自J2000起算的儒略日世纪数
        L0 = 280.46645+T*(36000.76983+0.0003032*T)  # 太阳(几何)平黄经,DEG
        Lmoon = 218.3165 + 481267.8813*T  # 月球平黄经,deg
        M = 357.52910+T*(35999.05030-T*(0.0001559+0.00000048*T))  # 太阳平近点角,DEG
        #  e=0.016708617-T*(0.000042037+0.0000001236*T)   # 地球轨道离心率
        C = (1.914600-T*(0.004817+0.000014*T))*np.sin(M/180*np.pi)  # 太阳中间方程 deg
        Theta = L0+C   # 太阳的真黄经,deg
#        Theta = np.mod(Theta,360) # 将Theta限制在0-360范围之内
        Omega = 125.04452 - T*(1934.136261 + T*(0.0020708 + T/450000))
        epsilon0 = (23+26/60+21.448/3600)-((46.8150/3600)-((0.00059/3600)+(0.001813/3600))*T)*T
        Thetaepsilon = (9.20/3600)*np.cos(Omega*np.pi/180)+(0.57/3600)*np.cos(2*L0*np.pi/180) - (0.23/3600)*np.sin(2*Lmoon*np.pi/180)-(0.09/3600)*np.cos(2*Omega*np.pi/180)
        epsilon = epsilon0 + Thetaepsilon
        SRa = np.arctan2(np.cos(epsilon*np.pi/180)*np.sin(Theta*np.pi/180),np.cos(Theta*np.pi/180))
        SRa = SRa*180/np.pi  # 转换为 deg
#        SRa = np.abs(SRa)
        if SRa < 0:
            SRa = SRa + 360
        SRa = SRa/15
        SDec = np.arcsin(np.sin(epsilon*np.pi/180)*np.sin(Theta*np.pi/180))  # rad
        SDec = SDec*180/np.pi  # 转换为 deg
        jd_time = jd_time + time_step
        sun_ra.append(SRa)
        sun_dec.append(SDec)
        if stop_time_mjd < (jd_time - 2400000.5):
            break
    return sun_ra, sun_dec


def earth_ecliptic_pos(pos_vec_sat, epsilon):
    """
    :param pos_vec_sat: 输入为卫星位置坐标三维矢量(ICRF),[[x],[y],[z]],
    :param epsilon: Epsilon 黄赤交角
    :return: 地球位置
    """
    rx_epsilon = np.array([[1, 0, 0],
                           [0, np.cos(epsilon), np.sin(epsilon)],
                           [0, -np.sin(epsilon), np.cos(epsilon)]
                           ])
    earth_ecliptic = np.dot(rx_epsilon, pos_vec_sat)
    # 计算卫星地心半径
    r = np.sqrt(earth_ecliptic[0][0]**2+earth_ecliptic[1][0]**2+earth_ecliptic[2][0]**2)
    earth_ecliptic[0][0] = earth_ecliptic[0][0]/r
    earth_ecliptic[1][0] = earth_ecliptic[1][0]/r
    earth_ecliptic[2][0] = earth_ecliptic[2][0]/r
    return earth_ecliptic                         # part4-1 P16


def sun_effect_src(time_mjd, ra_src, dec_src):
    """
    太阳是否遮挡源的观测
    :param time_mjd: 时间
    :param ra_src:  源的位置
    :param dec_src:
    :return: 是否受到影响
    """
    epsilon = ut.ecliptic_obliquity(time_mjd)  # 计算黄赤交角
    sin = np.sin
    cos = np.cos
    src_equ = np.zeros((3, 1))
    src_equ[0] = cos(dec_src) * cos(ra_src)  # 根据源赤经赤纬计算源赤道向量?
    src_equ[1] = cos(dec_src) * sin(ra_src)
    src_equ[2] = sin(dec_src)

    src_ecu = ut.equatorial_2_ecliptic(src_equ, epsilon)  # 根据源赤道单位向量计算源黄道单位向量
    jd_time = time_mjd + 2400000.5
    sun_lamb = sun_ecliptic_pos(jd_time)  # 根据儒略日计算太阳黄经
    sun_ecu = np.zeros((3, 1))
    sun_ecu[0] = cos(sun_lamb)
    sun_ecu[1] = sin(sun_lamb)  # 计算太阳黄道单位向量
    sun_angle = ut.angle_btw_vec(src_ecu, sun_ecu)  # 计算源和太阳之间的夹角
    if sun_angle > 50/180*np.pi:
        return True     # 源可见，太阳对源观测没有影响
    else:
        return False


def earth_shadow_sun(time_mjd, pos_vec_sat, amos_flag):
    """
    地球是否遮挡太阳
    :param time_mjd: 时间
    :param pos_vec_sat: 卫星的位置，向量
    :param amos_flag:
    :return: 是否受影响
    """
    cos = np.cos
    sin = np.sin
    epsilon = ut.ecliptic_obliquity(time_mjd)  # 计算黄赤交角
    # 计算卫星中心系统中地球黄道单位向量
    earth_ec = earth_ecliptic_pos(pos_vec_sat, epsilon)
    jd_time = time_mjd + 2400000.5
    sun_lamb = sun_ecliptic_pos(jd_time)  # 根据儒略日计算太阳黄经
    sun_ecu = np.zeros((3, 1))
    sun_ecu[0] = cos(sun_lamb)
    sun_ecu[1] = sin(sun_lamb)  # 计算太阳黄道单位向量
    r = np.sqrt(pos_vec_sat[0][0] ** 2 + pos_vec_sat[1][0] ** 2 + pos_vec_sat[2][0] ** 2)
    # 卫星视角看地球视半径 Earth Apparent Rad
    ear = np.arcsin(lc.earth_radius / r)
    # 卫星视角看太阳视半径 sun apparent rad
    sar = 0.00465421
    angle = ut.angle_btw_vec(earth_ec, sun_ecu)  # 计算源和太阳之间的夹角
    # 不考虑大气的影响
    if amos_flag == 0:
        if angle > (ear - sar):
            return True
        else:
            return False
    else:
        amp_angle = 5 / 180 * np.pi  # 受大气影响遭受的夹角
        if angle > (ear + amp_angle - sar):
            return True
        else:
            return False


def earth_shadow_src(time_mjd, pos_vec_sat, ra_src, dec_src, amos_flag):
    """
    地球是否遮挡源的观测
    :param time_mjd:  观测时间
    :param pos_vec_sat:  卫星的位置
    :param ra_src:  源的位置
    :param dec_src:
    :param amos_flag:  大气的影响
    :return:
    """
    cos = np.cos
    sin = np.sin
    epsilon = ut.ecliptic_obliquity(time_mjd)  # 计算黄赤交角
    # 计算卫星中心系统中地球黄道单位向量
    earth_ecu = earth_ecliptic_pos(pos_vec_sat, epsilon)
    src_equ = np.zeros((3, 1))
    src_equ[0] = cos(dec_src) * cos(ra_src)  # 根据源赤经赤纬计算源赤道向量?
    src_equ[1] = cos(dec_src) * sin(ra_src)
    src_equ[2] = sin(dec_src)
    src_ecu = ut.equatorial_2_ecliptic(src_equ, epsilon)  # 根据源赤道单位向量计算源黄道单位向量
    # 卫星视角看地球视半径 Earth Apparent Rad
    r = np.sqrt(pos_vec_sat[0][0] ** 2 + pos_vec_sat[1][0] ** 2 + pos_vec_sat[2][0] ** 2)
    ear = np.arcsin(lc.earth_radius / r)  # 地球视半径
    # 卫星视角看太阳视半径 sun apparent rad
    sar = 0.00465421
    angle = ut.angle_btw_vec(earth_ecu, src_ecu)  # 计算源和太阳之间的夹角
    # 不考虑大气的影响
    if amos_flag == 0:
        if angle > (ear - sar):
            return True
        else:
            return False
    else:
        amp_angle = 5 / 180 * np.pi  # 受大气影响遭受的夹角
        if angle > (ear + amp_angle - sar):
            return True
        else:
            return False
