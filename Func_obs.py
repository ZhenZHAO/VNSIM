"""
@functions: telescope visibility
@author: Zhen ZHAO
@date: Dec 26, 2018
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import argparse
import configparser
import os, sys
import pickle
import time

import utility as ut
import load_conf as lc
import model_effect as me
import model_satellite as ms
import model_obs_ability as mo


class FuncObs(object):
    def __init__(self, start_mjd, stop_mjd, t_step, pos_main_src, pos_vlbi, pos_sat, pos_telem,
                 baseline_type, cutoff_dict, procession):
        # 1. input parameter
        self.start_time_mjd = start_mjd
        self.stop_time_mjd = stop_mjd
        self.time_step = t_step
        self.pos_src = pos_main_src
        self.pos_mat_vlbi = pos_vlbi
        self.pos_mat_sat = pos_sat
        self.pos_mat_telem = pos_telem
        self.bl_type = baseline_type
        self.cutoff_angle = cutoff_dict['CutAngle']
        self.cutoff_dict = cutoff_dict
        self.procession_mode = procession

        # 2. Az-El result
        self.result_azimuth = []
        self.result_elevation = []
        self.result_hour = []
        self.result_gs = []

        # 3. sky survey
        self.result_pos_sun = []
        self.result_pos_moon = []
        self.num_array = None

        # 4. best observation
        self.result_obs_interval = []  # [[],]
        self.result_obs_best_inters = []  # [(),]
        self.result_obs_best_durations = []  # [0,]
        self.result_obs_optimal_inter = tuple()  # (a, b)

        # 5. best time string
        self.result_obs_optimal_time_string = ""

    # 0. get cutoff_angle
    def get_cutoff_angle(self):
        return self.cutoff_angle

    # 1. az_el
    def _func_tv_az_el(self):

        if type(self.pos_src[1]) == str:
            self.pos_src[1] = ut.time_str_2_rad(self.pos_src[1])
        if type(self.pos_src[2]) == str:
            self.pos_src[2] = ut.angle_str_2_rad(self.pos_src[2])
        ra_src = self.pos_src[1]
        dec_src = self.pos_src[2]

        lst_az = []
        lst_el = []
        lst_hour = []
        lst_gs = []

        for i in range(len(self.pos_mat_vlbi)):
            lst_gs.append(self.pos_mat_vlbi[i][0])
            lst_az_1 = []
            lst_el_1 = []
            lst_hour_1 = []
            long_vlbi, lat_vlbi, height_vlbi = ut.itrf_2_geographic(self.pos_mat_vlbi[i][1], self.pos_mat_vlbi[i][2],
                                                                    self.pos_mat_vlbi[i][3])
            for itr_mjd in np.arange(self.start_time_mjd, self.stop_time_mjd, self.time_step):
                source_azimuth, source_elevation = ut.equatorial_2_horizontal(itr_mjd, ra_src, dec_src, long_vlbi,
                                                                              lat_vlbi)
                azimuth_deg = ut.rad_2_angle(source_azimuth)
                elevation_deg = ut.rad_2_angle(source_elevation)
                if elevation_deg < 0:
                    elevation_deg = 0
                h1 = (itr_mjd - self.start_time_mjd) * 24  # mjd单位是天，这是把天转换成小时
                lst_az_1.append(azimuth_deg)
                lst_el_1.append(elevation_deg)
                lst_hour_1.append(h1)
            lst_az.append(lst_az_1)
            lst_el.append(lst_el_1)
            lst_hour.append(lst_hour_1)

        self.result_azimuth = lst_az
        self.result_elevation = lst_el
        self.result_hour = lst_hour
        self.result_gs = lst_gs

    def get_result_az_el_with_update(self):
        self._func_tv_az_el()
        return self.result_azimuth, self.result_elevation, self.result_hour

    def get_result_name_gs(self):
        return self.result_gs

    # for multiprocessing purpose (separate updating and getter)
    def update_result_az_el(self):
        self._func_tv_az_el()

    def get_result_az_el(self):
        return self.result_azimuth, self.result_elevation, self.result_hour

    # 2. sky survey
    def _func_sky_survey(self):
        # 统计可见望远镜/基线个数的数组
        num_array = []

        # cutoff angle setting
        vb = np.ones((1, 360), dtype=float)
        vb *= self.cutoff_angle
        vb = vb[0].tolist()

        # calculate the position of sun and moon
        sun_ra, sun_dec = me.sun_ra_dec_cal(self.start_time_mjd, self.start_time_mjd, 1)
        moon_ra, moon_dec = me.moon_ra_dec_cal(self.start_time_mjd, self.start_time_mjd, 1)

        # survey the whole sky
        ra_list = np.arange(0.125, 24, 0.25)
        dec_list = np.arange(-88.75, 90, 2.5)
        for src_dec in dec_list:
            src_dec = ut.angle_2_rad(src_dec, 0, 0)
            for src_ra in ra_list:
                src_ra = ut.time_2_rad(src_ra, 0, 0)
                num1 = 0  # sta和vlbi能观测到source的计数
                # test vlbi station
                for i in self.pos_mat_vlbi:  # get vlbi station
                    longitude, latitude, height = ut.itrf_2_geographic(i[1], i[2], i[3])
                    visibility = mo.obs_judge_active_vlbi_station(src_ra, src_dec, self.start_time_mjd, longitude, latitude, vb)
                    if visibility:
                        num1 = num1 + 1
                # # test satellite
                # for j in self.pos_mat_sat:
                #     visibility = mo.obs_judge_active_satellite_with_kepler(self.start_time_mjd,
                #                                                            src_ra, src_dec,
                #                                                            self.pos_mat_sat[j], self.pos_mat_telem,
                #                                                            self.bl_type, self.cutoff_dict,
                #                                                            self.procession_mode)
                #
                #     if visibility:
                #         num1 = num1 + 1
                # add num
                num_array.append(num1)

        num_array = np.array(num_array)
        num_array.shape = len(dec_list), len(ra_list)

        # 图像是，[0, 96], [0,72], 需要转化坐标
        img_sun_ra = sun_ra[0] * (96 / 24)
        img_sun_dec = sun_dec[0] * 0.4 + 36

        img_moon_ra = moon_ra[0] * (96 / 24)
        img_moon_dec = moon_dec[0] * 0.4 + 36

        # self.result_pos_sun = [sun_ra, sun_dec]
        # self.result_pos_moon = [moon_ra, moon_dec]
        self.result_pos_sun = [img_sun_ra, img_sun_dec]
        self.result_pos_moon = [img_moon_ra, img_moon_dec]
        self.num_array = num_array

    def get_result_sky_survey_with_update(self):
        self._func_sky_survey()
        return self.result_pos_sun, self.result_pos_moon, self.num_array

    # for multiprocessing purpose (separate updating and getter)
    def update_result_sky_survey(self):
        self._func_sky_survey()

    def get_result_sky_survey(self):
        return self.result_pos_sun, self.result_pos_moon, self.num_array

    # 3. best obs time duration
    def get_result_best_obs_time_el(self):
        self._func_best_obs_time_el()
        return self.result_obs_optimal_inter, self.result_obs_best_inters, self.result_obs_best_durations, self.result_obs_interval

    def _func_best_obs_time_el(self):
        # clear results
        self.result_obs_interval = []  # [[],]
        self.result_obs_best_inters = []  # [(),]
        self.result_obs_best_durations = []  # [0,]
        self.result_obs_optimal_inter = tuple()  # (a, b)

        # prepare data
        x = self.result_hour[0]
        # print("len of x axis", len(x))
        cut_line = np.ones_like(x) * self.cutoff_angle
        # calculate best interval for each station
        for el_line in self.result_elevation:
            # print(el_line)
            # find points
            tmp = np.argwhere(np.diff(np.sign(cut_line - el_line)) != 0).reshape(-1)
            tmp = [0] + list(tmp) + [len(x)-1]
            idx = list(set(tmp))
            idx.sort()
            # print(idx)
            # idx = [x[id_i] for id_i in idx_o]
            # find positive interval
            result = list()
            best_inter = tuple()
            max_interval_size = 0
            for i in range(1, len(idx)):
                tmp = (idx[i - 1], idx[i])
                test_point = int((tmp[0] + tmp[1]) / 2)
                if el_line[test_point] > cut_line[test_point]:
                    # index -> real x
                    tmp = (x[tmp[0]], x[tmp[1]])
                    result.append(tmp)
                    interval_size = tmp[1] - tmp[0]
                    if max_interval_size < interval_size:
                        max_interval_size = interval_size
                        best_inter = tmp

            # record results
            self.result_obs_interval.append(result)  # result
            self.result_obs_best_inters.append(best_inter)  # best_inter
            self.result_obs_best_durations.append(max_interval_size)  # max_interval_size

        # calculate the optimal obs interval for all
        # p_left = [t[0] for t in self.result_obs_best_inters]
        # p_right = [t[1] for t in self.result_obs_best_inters]
        op_left, op_right= x[0], x[-1]
        for t in self.result_obs_best_inters:
            if op_left < t[0]:
                op_left = t[0]
            if op_right > t[1]:
                op_right = t[1]
        if op_right > op_left:
            self.result_obs_optimal_inter = (op_left, op_right)

    # 4. get optimal best time string
    def get_result_best_time_string_with_update(self):
        self._func_best_obs_time_el()
        self._func_best_time_string()
        return self.result_obs_optimal_time_string

    def get_result_best_time_string_after_func_best_obs(self):
        self._func_best_time_string()
        return self.result_obs_optimal_time_string

    def _func_best_time_string(self):
        if len(self.result_obs_optimal_inter) != 0:
            timestamp1 = self.start_time_mjd + self.result_obs_optimal_inter[0] / 24  # unit of mjd is day
            week, year, month, day, hour, minute, sec = ut.mjd_2_time(timestamp1)
            str1 = "{}/{}/{} {}:{}:{}".format(year, month, day, hour, minute, sec)

            timestamp2 = self.start_time_mjd + self.result_obs_optimal_inter[1] / 24
            week, year, month, day, hour, minute, sec = ut.mjd_2_time(timestamp2)
            str2 = "{}/{}/{} {}:{}:{}".format(year, month, day, hour, minute, sec)

            self.result_obs_optimal_time_string = "Best Obs: from {} to {}".format(str1, str2)

    # 5. reset source position
    def reset_src_for_az_el(self, p_src):
        # 1. clean existing result
        # 1.1 az-el
        self.result_azimuth = []
        self.result_elevation = []
        self.result_hour = []
        self.result_gs = []

        # 1.2. best observation
        self.result_obs_interval = []  # [[],]
        self.result_obs_best_inters = []  # [(),]
        self.result_obs_best_durations = []  # [0,]
        self.result_obs_optimal_inter = tuple()  # (a, b)

        # 1.3. best time string
        self.result_obs_optimal_time_string = ""

        # 2. pass in new source
        self.pos_src = p_src


class ObsConfigParser(object):
    def __init__(self, _filename="config_obs.ini", _dbname='database.pkl'):
        # set the path of configFile and DBfile
        self.filename = os.path.join(os.path.join(os.getcwd(), 'CONFIG_FILE'), _filename)
        self.db_path = os.path.join(os.path.join(os.getcwd(), 'DATABASE'), _dbname)

        # time
        self.time_start = []
        self.time_end = []
        self.time_step = []
        # show info
        self.bs_flag_gg = 0
        self.bs_flag_gs = 0
        self.bs_flag_ss = 0
        self.baseline_type = 0
        self.cutoff_angle = 0
        self.precession_mode = 0

        self.str_source = ""
        self.str_vlbi = ""
        self.str_telemetry = ""
        self.str_sat = ""

        self.pos_mat_src = []
        self.pos_mat_vlbi = []
        self.pos_mat_telemetry = []
        self.pos_mat_sat = []

        self.parse_data()

    def parse_data(self):
        if not os.path.exists(self.filename):
            self.rewrite_config()
            return

        def parse_string_list(config, _string):
            tmp = config.get("station", _string)
            tmp_lst = [x.strip() for x in tmp.split(',')]
            return tmp_lst

        # create configparse
        config = configparser.ConfigParser()
        config.read(self.filename, encoding="utf-8")

        # obs_time
        tmp = config.get("obs_time", "start")
        self.time_start = [int(x) for x in tmp.split('/')]
        tmp = config.get("obs_time", "end")
        self.time_end = [int(x) for x in tmp.split('/')]
        tmp = config.get("obs_time", "step")
        self.time_step = [int(x) for x in tmp.split('/')]

        # bs_type
        self.bs_flag_gg = config.getint("bs_type", "bs_flag_gg")
        self.bs_flag_gs = config.getint("bs_type", "bs_flag_gs")
        self.bs_flag_ss = config.getint("bs_type", "bs_flag_ss")
        self.baseline_type = self.bs_flag_gg + self.bs_flag_gs * 2 + self.bs_flag_ss * 4

        # obs_mode
        self.cutoff_angle = config.getfloat("obs_mode", "cutoff_angle")
        self.precession_mode = config.getint("obs_mode", "precession_mode")

        # station
        self.str_source = parse_string_list(config, "pos_source")
        self.str_vlbi = parse_string_list(config, "pos_vlbi")
        self.str_telemetry = parse_string_list(config, "pos_telemetry")
        self.str_sat = parse_string_list(config, "pos_satellite")

        self.get_data_from_db()

    def show_info(self):
        print('*' * 15, " TIME ", '*' * 15)
        print("start=", self.time_start)
        print("end=", self.time_end)
        print("step=", self.time_step)
        print()

        print('*' * 15, " OBS ", '*' * 15)
        print("bs_type=", self.baseline_type)
        print("cutoff_angle=", self.cutoff_angle)
        print("precession_mode=", self.precession_mode)
        print()

        print('*' * 15, " Station ", '*' * 15)
        print("str_source=", self.str_source)
        print("str_vlbi=", self.str_vlbi)
        print("str_telemetry=", self.str_telemetry)
        print("str_sat=", self.str_sat)

        print('*' * 15, " Station with data", '*' * 15)
        print("\t source:", self.pos_mat_src)
        print("\t vlbi stations:", self.pos_mat_vlbi)
        print("\t telemetry stations:", self.pos_mat_telemetry)
        print("\t satellite:", self.pos_mat_sat)

    def rewrite_config(self):
        # create file
        if os.path.exists(self.filename):
            os.remove(self.filename)
            f = open(self.filename, 'w')
            f.close()
        else:
            f = open(self.filename, 'w')
            f.close()

        # create configparse
        config = configparser.ConfigParser()
        config.read(self.filename, encoding="utf-8")

        # add sections: obs_time
        config.add_section("obs_time")
        config.set("obs_time", "start", "2020/01/01/00/00/00")
        config.set("obs_time", "end", "2020/01/02/00/00/00")
        config.set("obs_time", "step", "00/00/05/00")
        self.time_start = [2020, 1, 1, 0, 0, 0]
        self.time_end = [2020, 1, 2, 0, 0, 0]
        self.time_step = [0, 0, 5, 0]

        # add sections: bs_type
        config.add_section("bs_type")
        config.set("bs_type", "bs_flag_gg", "1")
        config.set("bs_type", "bs_flag_gs", "0")
        config.set("bs_type", "bs_flag_ss", "0")
        self.bs_flag_gg, self.bs_flag_gs, self.bs_flag_ss = 1, 0, 0
        self.baseline_type = self.bs_flag_gg + self.bs_flag_gs * 2 + self.bs_flag_ss * 4

        # add sections: obs_mode
        config.add_section("obs_mode")
        config.set("obs_mode", "cutoff_angle", "10.0")
        config.set("obs_mode", "precession_mode", "0")
        self.cutoff_angle = 10.0
        self.precession_mode = 0

        # add sections: station
        config.add_section("station")
        config.set("station", "pos_source", "0316+413")
        config.set("station", "pos_vlbi", "ShangHai, Tianma, Urumqi, GIFU11, HITACHI,KASHIM34")
        config.set("station", "pos_telemetry", "")
        config.set("station", "pos_satellite", "")
        self.str_source = ['0316+413']
        self.str_vlbi = ['ShangHai', 'Tianma', 'Urumqi', 'GIFU11', 'HITACHI', 'KASHIM34']
        self.str_telemetry = ['']
        self.str_sat = ['']
        self.get_data_from_db()

        # write file
        config.write(open(self.filename, "w"))

    def get_data_from_db(self):
        with open(self.db_path, 'rb') as fr:
            db_src_dict = pickle.load(fr)
            db_sat_dict = pickle.load(fr)
            db_telem_dict = pickle.load(fr)
            db_vlbi_vlba_dict = pickle.load(fr)
            db_vlbi_evn_dict = pickle.load(fr)
            db_vlbi_eavn_dict = pickle.load(fr)
            db_vlbi_lba_dict = pickle.load(fr)
            db_vlbi_other_dict = pickle.load(fr)
            db_vlbi_all = pickle.load(fr)

        # source
        self.pos_mat_src = []
        if len(self.str_source) != 0:
            for each in self.str_source:
                if each in db_src_dict.keys():
                    self.pos_mat_src.append(list(db_src_dict[each]))

        # sat
        self.pos_mat_sat = []
        if len(self.str_sat) != 0:
            for each in self.str_sat:
                if each in db_sat_dict.keys():
                    self.pos_mat_sat.append(list(db_sat_dict[each]))

        # telem
        self.pos_mat_telemetry = []
        if len(self.str_telemetry) != 0:
            for each in self.str_telemetry:
                if each in db_telem_dict.keys():
                    self.pos_mat_telemetry.append(list(db_telem_dict[each]))

        # vlbi
        self.pos_mat_vlbi = []
        if len(self.str_vlbi) != 0:
            for each in self.str_vlbi:
                if each in db_vlbi_all.keys():
                    self.pos_mat_vlbi.append(list(db_vlbi_all[each]))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the observation survey, get the AZ-EL changes and sky Survey, "
                                                 "as well as best obs time duration")
    parser.add_argument('-c',
                        '--config',
                        default='config_obs.ini',
                        help='Specify the configuration file')
    parser.add_argument('-g',
                        '--show_gui',
                        action="store_true",
                        help='Choose to show GUI or not')
    parser.add_argument('-s',
                        '--save_az_el',
                        action="store_true",
                        help='Store the az_el data (/OUTPUT/uv_basic/az_el.txt)')
    parser.add_argument('-i',
                        '--obs_info',
                        action="store_true",
                        help='Choose to show best observation time duration', )
    parser.add_argument('-f',
                        '--img_fmt',
                        choices=['eps', 'png', 'pdf', 'svg', 'ps'],
                        help='Specify the img format (default:pdf)',
                        default='pdf')

    return parser.parse_args()


def run_obs():
    # 1. initialize parse and config objects
    args = parse_args()
    # args.obs_info = True
    args.show_gui = True
    if args.config != '':
        my_config_parser = ObsConfigParser(args.config)
    else:
        my_config_parser = ObsConfigParser()

    # output file path
    img_type = 'pdf'
    if args.img_fmt in ['eps', 'png', 'pdf', 'svg', 'ps']:
        img_type = args.img_fmt
    save_az_el_name = "az-el:" + time.asctime() + '.' + img_type
    path_az_el = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'obs_ability')
    path_az_el = os.path.join(path_az_el, save_az_el_name)

    save_sky_name = "sky-survey:" + time.asctime() + '.' + img_type
    path_sky_survey = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'obs_ability')
    path_sky_survey = os.path.join(path_sky_survey, save_sky_name)

    # mjd time
    start_time = ut.time_2_mjd(*my_config_parser.time_start, 0)
    stop_time = ut.time_2_mjd(*my_config_parser.time_end, 0)
    time_step = ut.time_2_day(*my_config_parser.time_step)
    cutoff_dict = {"flag": lc.cutoff_mode["flag"], "CutAngle": my_config_parser.cutoff_angle}
    myFuncObs = FuncObs(start_time, stop_time, time_step,
                        my_config_parser.pos_mat_src[0],
                        my_config_parser.pos_mat_vlbi,
                        my_config_parser.pos_mat_sat,
                        my_config_parser.pos_mat_telemetry,
                        my_config_parser.baseline_type,
                        cutoff_dict,
                        my_config_parser.precession_mode)
    # 2. calculate az - el
    azimuth, elevation, hour_lst = myFuncObs.get_result_az_el_with_update()
    gs_lst = myFuncObs.get_result_name_gs()
    x_hour_lmt = max(hour_lst[0])
    if x_hour_lmt > 24:
        x_hour_lmt = 24

    # save az-el data
    if args.save_az_el:
        name = "az-el-data:" + time.asctime() + '.txt'
        data_path = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'obs_ability')
        data_path = os.path.join(data_path, name)
        np.savetxt(data_path, [gs_lst, azimuth, elevation], fmt='%0.4f')

    # 3. calculate optimal interval
    optimal_inter, sta_best_inters, sta_best_durations, sta_all_inter = myFuncObs.get_result_best_obs_time_el()
    optimal_time_str = myFuncObs.get_result_best_time_string_after_func_best_obs()

    # 4. show optimal observation time
    if args.obs_info:
        print("optimal observation interval is :", optimal_inter)
        print(optimal_time_str)

    # 5. draw az-el
    fig1 = plt.figure(num=1, figsize=(8, 8))

    # draw az
    ax1_1 = fig1.add_subplot(211)
    for i in np.arange(0, len(azimuth)):
        az1 = azimuth[i]
        h1 = hour_lst[i]
        ax1_1.plot(h1, az1, '.-', markersize=1, label=gs_lst[i])
    ax1_1.set_xlim(0, x_hour_lmt)
    ax1_1.set_xlabel("Time(h)")
    ax1_1.set_ylabel("Azimuth($^\circ$)")
    ax1_1.set_title("The azimuth of source in VLBI stations")

    # draw el
    ax1_2 = fig1.add_subplot(212)
    for i in np.arange(0, len(elevation)):
        el1 = elevation[i]
        h1 = hour_lst[i]
        ax1_2.plot(h1, el1, '.-', markersize=1, label=gs_lst[i])
    ax1_2.set_xlim(0, x_hour_lmt)
    ax1_2.set_xlabel("Time(h)")
    ax1_2.set_ylabel("Elevation($^\circ$)")
    ax1_2.set_title("The elevation of source in VLBI stations")
    plt.legend(loc="best")
    tmp_cut = myFuncObs.get_cutoff_angle()
    ax1_2.plot([hour_lst[0][0], hour_lst[0][-1]], [tmp_cut, tmp_cut], '--k')

    # draw optimal time interval
    rect = plt.Rectangle((optimal_inter[0], 0), optimal_inter[1] - optimal_inter[0], 90, color='r', alpha=0.5)
    ax1_2.add_patch(rect)  # plt.gca().add_patch(rect)
    fig1.tight_layout()

    # save img
    plt.savefig(path_az_el)

    # 6. calculate sky survey
    pos_sun, pos_moon, num_array = myFuncObs.get_result_sky_survey_with_update()

    # 7. draw survey
    fig2 = plt.figure(num=2, figsize=(10, 8))
    fig2.add_subplot(111)
    array_max = np.max(num_array)
    bounds = np.arange(0, array_max + 1, 1)
    ax = pl.pcolor(num_array, edgecolors=(0.5, 0.5, 0.5), linewidths=1)
    fig2.colorbar(ax, ticks=bounds, shrink=1)
    plt.yticks([0, 24, 36, 48, 72], [-90, -30, 0, 30, 90])
    plt.xticks([0, 16, 32, 48, 64, 80, 96], [0, 4, 8, 12, 16, 20, 24])

    plt.plot([48, 48], [0, 72], color='black', linewidth=0.8, linestyle='-.', alpha=0.4)
    plt.plot([0, 96], [36, 36], color='black', linewidth=0.8, linestyle='-.', alpha=0.4)
    plt.xlabel("RA(H)")
    plt.ylabel(r'Dec ($^\circ$)')
    plt.title("SKY SURVEY")
    # draw soon, moon
    plt.plot(pos_sun[0], pos_sun[1], color='red', marker='o', markerfacecolor=(1, 0, 0), alpha=1, markersize=20)
    plt.plot(pos_moon[0], pos_moon[1], color='blue', marker='o', markerfacecolor='w', alpha=1, markersize=10)
    # save img
    plt.savefig(path_sky_survey)

    if args.show_gui:
        plt.show()


if __name__ == "__main__":
    run_obs()
