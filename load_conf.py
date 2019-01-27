# -*- coding:utf-8 -*-
"""
@functions: list all the parameters for testing
@author: Zhen ZHAO
@date: April 23, 2018
"""
import pickle

# velocity of light
light_speed = 299792458.8
# Earth radius [km]
earth_radius = 6378.1363
# Earth flattening
earth_flattening = 1 / 298.257
# square of ellipsoid eccentricity
eccentricity_square = earth_flattening * (2 - earth_flattening)
# GM constant     # GM=3.986004418*1e14    #地球（包括大气）引力常数  单位为m^3/s^-2 折合3.986004418*1e5
GM = 3.986004418 * 1e5  # [km^3/s^-2]

###########################################
# 1. 观测时间的设置
###########################################

# 起始时间全局变量
StartTimeGlobalYear = 2019
StartTimeGlobalMonth = 1
StartTimeGlobalDay = 20
StartTimeGlobalHour = 0
StartTimeGlobalMinute = 0
StartTimeGlobalSecond = 0

# 结束时间全局变量
StopTimeGlobalYear = 2019
StopTimeGlobalMonth = 1
StopTimeGlobalDay = 20
StopTimeGlobalHour = 10
StopTimeGlobalMinute = 0
StopTimeGlobalSecond = 0

# 时间步长
TimeStepGlobalDay = 0
TimeStepGlobalHour = 0
TimeStepGlobalMinute = 5
TimeStepGlobalSecond = 0

###########################################
# 2. 观测参数的设置
###########################################

# 三种基线类型的选择标志
baseline_flag_gg = 1
baseline_flag_gs = 0
baseline_flag_ss = 0
# baseline_type = baseline_flag_gg | baseline_flag_gs | baseline_flag_ss
baseline_type = baseline_flag_gg + baseline_flag_gs * 2 + baseline_flag_ss * 4
# 001(1)->select GtoG 010(2)->SELECT GtoS, 100(4)->StoS

# 观测频率和带宽
obs_freq = 22e9
bandwidth = 3.2e7

# 单位选择标志 km or lambda
unit_flag = 'km'

# cutoff_mode=1 #截止模式选择
cutoff_mode = {'flag': 1, 'CutAngle': 10}  # 截止模式选择，flag:0->取数据库中设置的水平角，1->取界面上设置的水平角 2->取大者，3->取小者
precession_mode = 0  # 进动模型选择，0->Two-Body,1->J2

###########################################
# 3. 源，观测站，卫星的信息
###########################################
# 源信息
pos_mat_src = [['0316+413', '3h19m48.160s', '41d30m42.10s'],
               ['0202+319', '2h5m4.925s', '32d12m30.095s']]
#
# pos_mat_src = [['0316+413', '3h19m48.160s', '41d30m42.10s'],
#                ['0202+319', '2h5m4.925s', '32d12m30.095s'],
#                ['0529+483', '5h33m15.866s', '48d22m52.808s'],
#                ['1030+415', '10h33m03.708s', '41d16m6.233s'],
#                ['1128+385', '11h30m53.283s', '38d15m18.547s'],
#                ['1418+546', '14h19m46.597s', '54d23m14.787s'],
#                ['1823+568', '18h24m7.068s', '56d51m01.491s'],
#                ['1828+487', '18h29m31.781s', '48d44m46.161s'],
#                ['1928+738', '19h27m48.495s', '73d58m1.57s'],
#                ['1954+513', '19h55m42.738s', '51d31m48.546s'],
#                ['2201+315', '22h3m14.976s', '31d45m38.27s']
#                ]


# VLBI站信息
pos_mat_vlbi = [['ShangHai', -2831.6870117, 4675.7338867, 3275.3276367, 10.0, 2],
                ['Tianma', -2826.7084961, 4679.2368164, 3274.6674805, 10.0, 2],
                ['Urumqi', 228.31073, 4631.9228516, 4367.0639648, 10.0, 2],
                ['GIFU11', -3787.123361, 3564.181694, 3680.274907, 10.0, 2],
                ['HITACHI', -3961.788796, 3243.597525, 3790.597709, 10.0, 2],
                ]

# pos_mat_vlbi = [['ShangHai', -2831.6870117, 4675.7338867, 3275.3276367, 10.0, 2],
#                 ['Tianma', -2826.7084961, 4679.2368164, 3274.6674805, 10.0, 2],
#                 ['Urumqi', 228.31073, 4631.9228516, 4367.0639648, 10.0, 2],
#                 ['GIFU11', -3787.123361, 3564.181694, 3680.274907, 10.0, 2],
#                 ['HITACHI', -3961.788796, 3243.597525, 3790.597709, 10.0, 2],
#                 ['KASHIM34', -3997.649236, 3276.6908071, 3724.2788924, 10.0, 2],
#                 ['TAKAHAGI32', -3961.881464, 3243.37261, 3790.687517, 10.0, 2],
#                 ['VERAIR', -3521.7195683, 4132.1747532, 3336.9943259, 10.0, 2],
#                 ['VERAIS', -3263.9946477, 4808.056357, 2619.9493953, 10.0, 2],
#                 ['VERAMZ', -3857.2418552, 3108.7848509, 4003.9005858, 10.0, 2],
#                 ['VERAOG', -4491.0688943, 3481.5448287, 2887.3996225, 10.0, 2],
#                 ['NOBEYA45', -3871.02349, 3428.1068, 3724.0395, 10.0, 2],
#                 ['SEJONG', -3110.0800133, 4082.0666386, 3775.0767703, 10.0, 2],
#                 ['KVNTN', -3171.7320009, 4292.679158, 3481.0392758, 10.0, 2],
#                 ['KVNUS', -3287.2690497, 4023.4507014, 3687.3804935, 10.0, 2],
#                 ['KVNYS', -3042.2788964, 4045.9020207, 3867.3761335, 10.0, 2],
#                 ]

# 遥测站信息
pos_mat_telemetry = []
# pos_mat_telemetry = [['Goldstone', -2353.62000, -4641.34000, 3677.05000]]
# 卫星列表信息,每一元祖数据对应的信息为Name,a,e,i,w,Ω,M0,Epoch
# # M0未知，这里设置为0，假设2020年3月1日通过近地点
pos_mat_sat = []
# pos_mat_sat = [['VSOP', 17367.457, 0.60150, 31.460, 106.755, 16.044, 66.210, 50868.00000],
#                ['RadioAstron', 46812.900, 0.8200, 51.000, 285.000, 255.000, 280.000, 50449.000000]]

###########################################
# 4. Imaging
###########################################
n_pix = 512
source_model = 'Point-source.model'  # Point-source.model  Five-Gauss.model
clean_gain = 0.9
clean_threshold = 0
clean_niter = 20
color_map_name = 'viridis'  # 'viridis', 'hot', 'jet', 'rainbow', 'Greys', 'hot', 'cool', 'nipy_spectral'
# https://matplotlib.org/examples/color/colormaps_reference.html

###########################################
# 5. Rad plot
###########################################
rad_plot_file = '0106+013_1.fits'

###########################################
# 6. play u,v
###########################################
fr = open('./DATABASE/playuv.pkl','rb')
just4fun_u = pickle.load(fr)
just4fun_v = pickle.load(fr)
just4fun_max = pickle.load(fr)
fr.close()


###################################################
# 7. update the values based on config_run_gui.ini
###################################################
def update_config_para():
    pass


###########################################
# 8. show the values
###########################################
def print_setting():
    info = []
    label1 = "Start Time: %d/%d/%d %d:%d:%d UT" % (StartTimeGlobalYear, StartTimeGlobalMonth, StartTimeGlobalDay,
                                                   StartTimeGlobalHour, StartTimeGlobalMinute, StartTimeGlobalSecond)
    info.append(label1)

    label2 = "Stop Time: %d/%d/%d %d:%d:%d UT" % (StopTimeGlobalYear, StopTimeGlobalMonth, StopTimeGlobalDay,
                                                  StopTimeGlobalHour, StopTimeGlobalMinute, StopTimeGlobalSecond)
    info.append(label2)

    label3 = "Time step: %dd %dh %dm %ds" % (TimeStepGlobalDay, TimeStepGlobalHour,
                                             TimeStepGlobalMinute, TimeStepGlobalSecond)
    info.append(label3)

    label4 = "Wavelength: %f" % (light_speed * 100 / obs_freq)
    info.append(label4 + '\n')

    label5 = "Source:\n\t"
    for item in pos_mat_src:
        label5 = label5 + item[0]
        label5 = label5 + '\n\tRA:     '
        label5 = label5 + str(item[1])
        label5 = label5 + '\n\tDEC:    '
        label5 = label5 + str(item[2])
        label5 = label5 + '\n'
    info.append(label5)

    label6 = "Satellite:\n\t"
    for item in pos_mat_sat:
        label6 = label6 + item[0]
        label6 = label6 + '\n\t'
    info.append(label6)

    label7 = 'VLBI Stations:\n\t'
    for item in pos_mat_vlbi:
        label7 = label7 + item[0]
        label7 = label7 + '\n\t'
    info.append(label7)
    print("=" * 30, '\n')
    print("\n".join(info))
    print("=" * 30)


if __name__ == '__main__':
    print_setting()
