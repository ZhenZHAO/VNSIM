"""
@functions: formulate the satellite model
@author: Zhen ZHAO
@date: April 24, 2018
"""
import numpy as np
import load_conf as lc


def semi_axis_cal(apogee, perigee):
    """
    计算卫星椭圆轨道半长轴
    :param apogee: 远地点 [km]
    :param perigee: 近地点 [km]
    :return: 半长轴 离心率
    """
    a = (apogee + perigee) / 2 + lc.earth_radius
    e = 1.0 - (lc.earth_radius + perigee) / a
    return a, e


def apo_per_cal(semi_axis, eccentricity):
    """
    计算远地点和近地点
    :param semi_axis: 半长轴
    :param eccentricity: 离心率
    :return:
    """
    apogee = semi_axis * (1 + eccentricity) - lc.earth_radius
    perigee = semi_axis * (1 - eccentricity) - lc.earth_radius
    return apogee, perigee


def satellite_orbit_period(a):
    """
    开普勒第三定律：卫星运行周期的平方与轨道椭圆长半径的立方之比为一常量，等于GM的倒数。
    :param a: 开普勒椭圆的长半径 [km]
    :return: 周期（单位：天）
    """
    cons = 4 * (np.pi ** 2) / lc.GM
    period_sec = np.sqrt((a ** 3) * cons)
    period_day = period_sec / 86400.0
    return period_day


def kepler_2_cartesian(a, e, i, aop, loan, m):
    """
    输入为卫星的六个轨道参数
    :param a: 椭圆轨道的半长轴 [km]
    :param e: 轨道偏心率
    :param i: 轨道平面倾角(deg)
    :param aop: omega为近地点角(deg)
    :param loan: Omega为升交点赤经(deg)
    :param m: 平近点角
    :return: 卡迪尔坐标
    """

    a = a * 1000 # 将半长轴a的单位化为m
    irad = np.mod(i, np.pi)  # 轨道倾角 0-180
    AOPrad = np.mod(aop, 2 * np.pi)  # 近地点角 0-360
    LOANrad = np.mod(loan, 2 * np.pi)  # 升交点赤经 0-360
    Mrad = np.mod(m, 2 * np.pi)
    # 求解开普勒方程 M=E-esinE
    Accuracy = 1e-7  # 定义要求的精度
    E0 = Mrad + e * np.sin(Mrad) / (1 - np.sin(Mrad + e) + np.sin(Mrad))  # 偏近点角的初始值

    # E0=np.mod(E0,np.pi*2)
    while True:
        E1 = E0 - (E0 - e * np.sin(E0) - Mrad) / (1 - e * np.cos(E0))  # 开普勒方程求解的牛顿微分法迭代
        # E1=np.mod(E1,np.pi*2)
        if (np.abs(E1 - E0) < Accuracy):
            break
        else:
            E0 = E1
    E = E1  # E为偏近点角
    r = a * (1 - e * np.cos(E))  # 地心半径
    SINF = np.sqrt(1 - e ** 2) * np.sin(E) / (1 - e * np.cos(E))
    COSF = (np.cos(E) - e) / (1 - e * np.cos(E))
    # 计算真近点角
    if SINF == 0:
        f = np.pi
    else:
        if (COSF == 0):
            f = np.pi / 2
        else:
            f = np.arctan(np.abs(SINF / COSF))  # 真近点角
            if ((SINF > 0) and (COSF < 0)):
                f = np.pi - f
            elif ((SINF < 0) and (COSF < 0)):
                f = np.pi + f
            elif ((SINF < 0) and (COSF > 0)):
                f = 2 * np.pi - f

    # print "f=",f
    u = AOPrad + f  # 纬度=近地点角+真近点角
    # print "u=",u
    # u的范围不确定?
    # 笛卡尔坐标
    x = r * (np.cos(LOANrad) * np.cos(u) - np.sin(LOANrad) * np.sin(u) * np.cos(irad))
    y = r * (np.sin(LOANrad) * np.cos(u) + np.cos(LOANrad) * np.sin(u) * np.cos(irad))
    z = r * np.sin(u) * np.sin(irad)
    # 计算速度
    p = a * (1 - e ** 2)
    # 径向和切向速度方向
    # GM=9.80665 #标准重力加速度
    GM = 3.986004418 * 1e14
    Vr = np.sqrt(GM / p) * e * SINF
    Vu = np.sqrt(GM / p) * (1 + e * COSF)
    # 计算在惯性参考系（ICRF)的速度分量
    Vx = Vr * (np.cos(LOANrad) * np.cos(u) - np.sin(LOANrad) * np.sin(u) * np.cos(irad)) - Vu * (
            np.cos(LOANrad) * np.sin(u) + np.sin(LOANrad) * np.cos(u) * np.cos(irad))
    Vy = Vr * (np.sin(LOANrad) * np.cos(u) + np.cos(LOANrad) * np.sin(u) * np.cos(irad)) - Vu * (
            np.sin(LOANrad) * np.sin(u) - np.cos(LOANrad) * np.cos(u) * np.cos(irad))
    Vz = Vr * np.sin(u) * np.sin(irad) + Vu * np.cos(u) * np.sin(irad)

    x = x / 1000
    y = y / 1000
    z = z / 1000
    Vx = Vx / 1000
    Vy = Vy / 1000
    Vz = Vz / 1000
    # print("satpositionx=",x,"satpositiony=",y,"satpositionz=",z)
    return x, y, z, Vx, Vy, Vz


def get_satellite_position(a, e, i, AOP, LOAN, M0, MJDEpoch, MJDTime, precession_mode):
    delta_T = (MJDTime - MJDEpoch) * 86400  # 时间差,单位时间转化为s 一天=86400s
    n = np.sqrt(lc.GM / (a ** 3))  # [rad/s]
    r0 = 6378.1363  # 地球赤道半径，单位为千米EarthRad=6378.1363
    J2 = 0.001082629832258  # 动力形状
    if precession_mode['flag'] == 0:
        dAOP_dt = 0
        dLOAN_dt = 0
        dM0_dt = 0
    elif precession_mode['flag'] == 1:
        dAOP_dt = 0.75 * J2 * n * ((r0 / a) ** 2) * ((5 * ((np.cos(i)) ** 2) - 1) / ((1 - e ** 2) ** 2))
        dLOAN_dt = -1.5 * J2 * n * ((r0 / a) ** 2) * (np.cos(i) / (1 - e ** 2) ** 2)
        dM0_dt = 0.75 * J2 * n * ((r0 / a) ** 2) * (3 * np.cos(i) ** 2 - 1) / np.sqrt((1 - e ** 2) ** 3)
    else:
        print("You have not choosed a precession model,Two-Body Instead")
        dAOP_dt = 0
        dLOAN_dt = 0
        dM0_dt = 0
    #    AOP=np.mod(AOP+dAOP_dt*delta_T,np.pi*2)  #将近地点角的范围限制在0-2*pi之间
    AOP = AOP + dAOP_dt * delta_T
    #  LOAN=np.mod(LOAN+dLOAN_dt*delta_T,np.pi*2)#将升交点赤经限制在0-2*pi之间
    LOAN = LOAN + dLOAN_dt * delta_T
    #    M0=np.mod(M0+dM0_dt*delta_T,np.pi*2)#将平近点角限定在0-2*pi之间
    M0 = M0 + dM0_dt * delta_T
    M = np.mod(M0 + n * delta_T, 2 * np.pi)  # 平近点角
    x, y, z, Vx, Vy, Vz = kepler_2_cartesian(a, e, i, AOP, LOAN, M)
    # print x,y,z
    return [a, e, i, AOP, LOAN, M, x, y, z, Vx, Vy, Vz]  # 这是ICRF坐标
