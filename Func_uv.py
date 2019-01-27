"""
@functions: basic uv coverage and sky coverage
@author: Zhen ZHAO
@date: Nov 2, 2018
"""

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import argparse
import configparser
import os
import pickle
import time

import load_conf as lc
import utility as ut
import model_effect as me
import model_satellite as ms
import model_obs_ability as mo


class FuncUv(object):
    def __init__(self, start_t, stop_t, step_t, p_main_src, p_multi_src, p_sat, p_vlbi, p_tele,
                 freq, bl_type, f_unit, cutoff_dict, precession_type):
        # 1. input parameter
        self.start_mjd = start_t
        self.stop_mjd = stop_t
        self.time_step = step_t

        self.pos_src = p_main_src
        self.pos_multi_src = p_multi_src

        self.pos_mat_sat = p_sat
        self.pos_mat_vlbi = p_vlbi
        self.pos_mat_telemetry = p_tele
        self.obs_freq = freq
        self.baseline_type = bl_type

        # self.flag_unit = f_unit
        self.flag_unit = 0 if f_unit == "lambda" else 1  # 0->lambda, 1->km
        self.cutoff_mode = cutoff_dict
        self.precession_mode = precession_type

        # obtain/sparse all the srcs
        self.src_num = len(self.pos_multi_src)
        for i in range(self.src_num):
            tmp_ra = self.pos_multi_src[i][1]
            tmp_dec = self.pos_multi_src[i][2]
            # print(self.pos_multi_src[i][1])
            if type(tmp_ra) == str:
                self.pos_multi_src[i][1] = ut.time_str_2_rad(tmp_ra)
            if type(tmp_dec) == str:
                self.pos_multi_src[i][2] = ut.angle_str_2_rad(tmp_dec)

        # 2. store old/the very first result
        self.result_u = []
        self.result_v = []
        self.max_range_single_uv = 1

        # 3. all year uv result
        self.result_time_u = []
        self.result_time_v = []
        self.max_range_time_uv = 1

        # 4. all sky uv result
        self.result_sky_u = []
        self.result_sky_v = []
        self.max_range_sky_uv = 1

        # 5. multiple src results
        self.result_multi_src_name = []
        self.result_multi_src_u = []
        self.result_multi_src_v = []
        self.max_range_multi_src = 1

        self._ini_para()

    def _ini_para(self):
        # 2. functional variables
        # 2.1 station info (self.lst_ground, self.lst_space)
        self.lst_ground = self.pos_mat_vlbi  # 将地面站看作是VLBI站
        self.lst_space = []
        for i in np.arange(len(self.pos_mat_sat)):
            if type(self.pos_mat_sat[i][7]) == str:
                # 将远地点和近地点数值转换成半长轴和离心率
                self.pos_mat_sat[i][1], self.pos_mat_sat[i][2] = ms.semi_axis_cal(self.pos_mat_sat[i][1],
                                                                                  self.pos_mat_sat[i][2])
                self.pos_mat_sat[i][7] = ut.time_str_2_mjd(self.pos_mat_sat[i][7])
            # 卫星名称，半长轴，偏心率
            self.lst_space.append([self.pos_mat_sat[i][0], self.pos_mat_sat[i][1], self.pos_mat_sat[i][2]])

        # 2.2 source info (self.src_ra, self.src_dec)
        if type(self.pos_src[1]) == str:
            self.src_ra = ut.time_str_2_rad(self.pos_src[1])
            # print(self.pos_src[1], self.src_ra)
        else:
            self.src_ra = self.pos_src[1]

        if type(self.pos_src[2]) == str:
            self.src_dec = ut.angle_str_2_rad(self.pos_src[2])
            # print(self.pos_src[2], self.src_dec)
        else:
            self.src_dec = self.pos_src[2]

        # 2.3 observation info (obs_wlen, max_baseline, max_range)
        self.obs_wlen = ut.freq_2_wavelength(self.obs_freq)
        self.max_baseline = self._get_max_baseline()

        # 3. temp single uv result
        self.dict_u = {"gg": None, "gs": None, "ss": None}
        self.dict_v = {"gg": None, "gs": None, "ss": None}
        self.dict_bl_sta1 = {"gg": None, "gs": None, "ss": None}
        self.dict_bl_sta2 = {"gg": None, "gs": None, "ss": None}
        self.dict_bl_duration = {"gg": None, "gs": None, "ss": None}

        self.result_tmp_u = []
        self.result_tmp_v = []
        self.max_range_tmp = 1

    # 1. multiple srcs
    def get_result_multi_src_with_update(self):
        self._func_multi_source_uv()
        return self.result_multi_src_name, self.result_multi_src_u, self.result_multi_src_v, self.max_range_multi_src

    def _func_multi_source_uv(self):
        for i in range(self.src_num):
            tmp_name = self.pos_multi_src[i][0]
            tmp_ra = self.pos_multi_src[i][1]
            tmp_dec = self.pos_multi_src[i][2]

            tmp_src = self.pos_src
            temp_u, temp_v, temp_max = self._get_reset_source_info([tmp_name, tmp_ra, tmp_dec])
            self.pos_src = tmp_src

            self.result_multi_src_name.append(tmp_name)
            self.result_multi_src_u.append(temp_u)
            self.result_multi_src_v.append(temp_v)

            if self.max_range_multi_src < temp_max:
                self.max_range_multi_src = temp_max

    # for multiprocessing purpose (separate updating and getter)
    def update_result_multi_src(self):
        self._func_multi_source_uv()

    def get_result_multi_src(self):
        return self.result_multi_src_name, self.result_multi_src_u, self.result_multi_src_v, self.max_range_multi_src

    # 2. all sky uv
    def get_result_sky_uv_with_update(self):
        self._func_all_sky_uv()
        return self.result_sky_u, self.result_sky_v, self.max_range_sky_uv * 1.3

    def _func_all_sky_uv(self):
        for i in (2, 6, 10, 14, 18, 22):  # dra
            for j in (-60, -30, 0, 30, 60):  # dra
                ra = ut.time_2_rad(i, 0, 0)
                dec = ut.angle_2_rad(j, 0, 0)
                # print(ra, dec)
                pos_src = ['source-%d-%d' % (i, j), ra, dec]
                record_source = self.pos_src
                temp_u, temp_v, temp_max = self._get_reset_source_info(pos_src)
                self.pos_src = record_source
                self.result_sky_u.append(temp_u)
                self.result_sky_v.append(temp_v)
                # calculate max {u,v}
                if self.max_range_sky_uv < temp_max:
                    self.max_range_sky_uv = temp_max

    # for multiprocessing purpose (separate updating and getter)
    def update_result_sky_uv(self):
        self._func_all_sky_uv()

    def get_result_sky_uv(self):
        return self.result_sky_u, self.result_sky_v, self.max_range_sky_uv * 1.3

    # 3. all year round uv
    def get_result_year_uv_with_update(self):
        self._func_all_year_uv()
        # print(self.start_mjd, self.pos_src[0], self.max_range_time_uv, self.result_time_u[0])
        return self.result_time_u, self.result_time_v, self.max_range_time_uv * 1.3

    def _func_all_year_uv(self):
        # generated 12 all year time, and calculate u,v
        date = ut.mjd_2_time(self.start_mjd)
        year = date[1]
        month = date[2]

        for _ in range(0, 12):
            # generate time
            if month > 13:
                year += 1
                month -= 12
                temp_start = ut.time_2_mjd(year, month, 1, 0, 0, 0, 0)
                temp_end = ut.time_2_mjd(year, month, 2, 0, 0, 0, 0)
            else:
                temp_start = ut.time_2_mjd(year, month, 1, 0, 0, 0, 0)
                temp_end = ut.time_2_mjd(year, month, 2, 0, 0, 0, 0)
            month += 1

            temp_u, temp_v, temp_max = self._get_reset_time_info(temp_start, temp_end, self.time_step)

            self.result_time_u.append(temp_u)
            self.result_time_v.append(temp_v)
            # calculate max {u,v}
            if self.max_range_time_uv < temp_max:
                self.max_range_time_uv = temp_max

    # for multiprocessing purpose (separate updating and getter)
    def update_result_year_uv(self):
        self._func_all_year_uv()

    def get_result_year_uv(self):
        return self.result_time_u, self.result_time_v, self.max_range_time_uv * 1.3

    # 4. single uv function
    def get_result_single_uv_with_update(self):
        self._func_uv()
        self._parse_result_dict()
        self.result_u, self.result_v, self.max_range_single_uv = self._get_tmp_single_uv()
        return self.result_u, self.result_v, self.max_range_single_uv

    # for multiprocessing purpose (separate updating and getter)
    def update_result_single_uv(self):
        self._func_uv()
        self._parse_result_dict()
        self.result_u, self.result_v, self.max_range_single_uv = self._get_tmp_single_uv()

    def get_result_single_uv(self):
        return self.result_u, self.result_v, self.max_range_single_uv

    # other implementations
    def _get_reset_source_info(self, p_src):
        self.pos_src = p_src
        self._ini_para()

        return self._get_tmp_single_uv()

    def _get_reset_time_info(self, temp_start, temp_end, time_step):
        self.start_mjd = temp_start
        self.stop_mjd = temp_end
        self.time_step = time_step
        self._ini_para()

        return self._get_tmp_single_uv()

    def _get_tmp_single_uv(self):
        self._func_uv()
        self._parse_result_dict()
        return self.result_tmp_u, self.result_tmp_v, self.max_range_tmp

    def _parse_result_dict(self):
        # 1. u,v
        for each in self.dict_u.values():
            if each is not None:
                self.result_tmp_u.extend(each)

        for each in self.dict_v.values():
            if each is not None:
                self.result_tmp_v.extend(each)

        # 2. calculate max {u,v}
        if len(self.result_tmp_u) > 0 and len(self.result_tmp_v) > 0:
            temp1 = np.max(np.abs(self.result_tmp_u))
            temp2 = np.max(np.abs(self.result_tmp_v))
            temp = max(temp1, temp2)
            if self.max_range_tmp < temp:
                self.max_range_tmp = temp

    def _func_uv(self):
        # according to the baseline type, calculate the corresponding uv coverage
        if (self.baseline_type & 1) != 0:  # ground to ground
            lst_u, lst_v, bl_sta1_name, bl_sta2_name, bl_duration \
                = self._func_uv_gg()
            self.dict_u["gg"] = lst_u
            self.dict_v["gg"] = lst_v
            self.dict_bl_sta1["gg"] = bl_sta1_name
            self.dict_bl_sta2["gg"] = bl_sta2_name
            self.dict_bl_duration["gg"] = bl_duration

        if (self.baseline_type & 2) != 0:  # ground to ground
            lst_u, lst_v, bl_sta1_name, bl_sta2_name, bl_duration \
                = self._func_uv_gg()
            self.dict_u["gs"] = lst_u
            self.dict_v["gs"] = lst_v
            self.dict_bl_sta1["gs"] = bl_sta1_name
            self.dict_bl_sta2["gs"] = bl_sta2_name
            self.dict_bl_duration["gs"] = bl_duration

        if (self.baseline_type & 4) != 0:  # ground to ground
            lst_u, lst_v, bl_sta1_name, bl_sta2_name, bl_duration \
                = self._func_uv_gg()
            self.dict_u["ss"] = lst_u
            self.dict_v["ss"] = lst_v
            self.dict_bl_sta1["ss"] = bl_sta1_name
            self.dict_bl_sta2["ss"] = bl_sta2_name
            self.dict_bl_duration["ss"] = bl_duration

    def _get_uv_coordination(self, mat_uv, pos_sta1, pos_sta2):
        d = np.array(pos_sta1) - np.array(pos_sta2)
        dtran = np.array([d])
        uvc = np.dot(mat_uv, dtran.T)
        if self.flag_unit == 0:
            return uvc[0][0] * 1000 / self.obs_wlen, uvc[1][0] * 1000 / self.obs_wlen, uvc[2][0] * 1000 / self.obs_wlen
        else:
            return uvc[0][0] * 1000, uvc[1][0] * 1000, uvc[2][0] * 1000

    def _get_max_baseline(self):
        max_baseline = 0
        lst_ground = self.lst_ground
        lst_space = self.lst_space
        if (self.baseline_type & 1) != 0:
            for i in np.arange(len(lst_ground)):
                for j in np.arange(i + 1, len(lst_ground)):
                    delta_x = lst_ground[i][1] - lst_ground[j][1]
                    delta_y = lst_ground[i][2] - lst_ground[j][2]
                    delta_z = lst_ground[i][3] - lst_ground[j][3]
                    distance = delta_x ** 2 + delta_y ** 2 + delta_z ** 2
                    baseline = np.sqrt(distance)
                    if max_baseline < baseline:
                        max_baseline = baseline

        if (self.baseline_type & 2) != 0:
            for m in range(len(lst_space)):
                baseline = lst_space[m][1] * (1 + lst_space[m][2])
                if baseline > max_baseline:
                    max_baseline = baseline
            max_baseline = max_baseline + lc.earth_radius

        elif (self.baseline_type & 4) != 0:
            max_apogee = lc.earth_radius  # 卫星的最大远地点距离
            second_max_apogee = 0
            for m in range(len(lst_space)):
                temp = lst_space[m][1] * (1 + lst_space[m][2])  # 半长轴 偏心率
                if temp > max_apogee:
                    second_max_apogee = max_apogee
                    max_apogee = temp
                elif temp > second_max_apogee:
                    second_max_apogee = temp
            max_baseline = max_apogee + second_max_apogee

        return max_baseline

    def _func_uv_gg(self):
        # define output
        lst_u = []
        lst_v = []
        lst_w = []
        baseline_sta1_name = []  # 一条地地基线对应的两个站名
        baseline_sta2_name = []
        baseline_duration = []  # 基线存在的时间

        # traverse all the time period
        for timestamp in np.arange(self.start_mjd, self.stop_mjd, self.time_step):
            active_station = mo.obs_all_active_vlbi(timestamp, self.src_ra, self.src_dec, self.pos_mat_vlbi,
                                                    self.cutoff_mode)
            uv_matrix = ut.trans_matrix_uv_itrf(timestamp, self.src_ra, self.src_dec)
            # traverse all the combinations of ground stations
            for i in np.arange(len(self.pos_mat_vlbi)):
                for j in np.arange(i + 1, len(self.pos_mat_vlbi)):
                    if active_station[2 * i + 1] is True and active_station[2 * j + 1] is True:
                        sta1_pos = self.pos_mat_vlbi[i][1:4]
                        sta2_pos = self.pos_mat_vlbi[j][1:4]
                        u, v, w = self._get_uv_coordination(uv_matrix, sta1_pos, sta2_pos)  # 单位为m
                        u /= 1000
                        v /= 1000
                        lst_u.extend([u, -u])
                        # lst_v.extend([-v, v])
                        lst_v.extend([v, -v])
                        lst_w.extend([w, -w])
                        baseline_sta1_name.extend([self.pos_mat_vlbi[i][0]])
                        baseline_sta2_name.extend([self.pos_mat_vlbi[j][0]])
                        baseline_duration.extend([timestamp])

        # return the value
        return lst_u, lst_v, baseline_sta1_name, baseline_sta2_name, baseline_duration

    def _func_uv_gs(self, start_mjd, stop_mjd, time_step, src_ra, src_dec, pos_mat_sat, pos_mat_telemetry,
                    pos_mat_vlbi, obs_freq, flag_unit, cutoff_mode, precession_mode):
        if len(self.lst_space) < 1:
            return None, None, None, None, None

        lst_u, lst_v, baseline_sta1_name, baseline_sta2_name, baseline_duration = \
            None, None, None, None, None
        # return the value
        return lst_u, lst_v, baseline_sta1_name, baseline_sta2_name, baseline_duration

    def _func_uv_ss(self, start_mjd, stop_mjd, time_step, src_ra, src_dec, pos_mat_sat, pos_mat_telemetry,
                    obs_freq, flag_unit, cutoff_mode, precession_mode):
        if len(self.lst_space) < 2:
            return None, None, None, None, None

        lst_u, lst_v, baseline_sta1_name, baseline_sta2_name, baseline_duration = \
            None, None, None, None, None
        # return the value
        return lst_u, lst_v, baseline_sta1_name, baseline_sta2_name, baseline_duration

    # 6. calculate beam size
    def _calculate_beam_size(self):
        u = np.array(self.result_u) * 1000 # km -> m
        v = np.array(self.result_v) * 1000
        max_baseline = self.max_baseline * 1000
        if self.flag_unit == 1:
            u = u / self.obs_wlen
            v = v / self.obs_wlen
            max_baseline = max_baseline / self.obs_wlen
        uv_bl = [np.sqrt(uu ** 2 + vv ** 2) for uu, vv in zip(u, v)]

        # calculating beam
        # maxuv = np.max(list(u) + list(v))  # 取uv最大值
        max_bl = np.max(uv_bl)
        min_beam = 1.1 / max_bl / np.pi * 180 * 3600 * 1000
        min_beam2 = 1.1 / max_baseline / np.pi * 180 * 3600 * 1000
        # print('maxuv=', maxuv, 'max_bl=', max_bl, 'min_beam=', min_beam, min_beam2)

        # The technique used was developed by Tim Pearson (estimate beam size) --> job moved to imaging file
        muu, mvv, muv = 0.0, 0.0, 0.0
        wsum, runwt = 0.0, 0.0
        # radial weighting
        for i in range(0, len(u)):
            # weight = 1.0
            # weight *= uv_bl[i]
            # wsum += weight
            # runwt = weight / wsum
            runwt = 1
            muu += runwt * (u[i] ** 2 - muu)
            mvv += runwt * (v[i] ** 2 - mvv)
            muv += runwt * (u[i] * v[i] - muv)
        # print('muu, mvv, muv =', muu, mvv, muv)

        e_bpa = -0.5 * np.arctan2(2.0 * muv, muu - mvv)
        e_bpa = e_bpa * 180 / np.pi

        # fudge = 0.7  # Empirical fudge factor of TJP's algorithm
        # ftmp = np.sqrt((muu - mvv) ** 2 + 4 * muv * muv)
        # # print('ftmp=', ftmp)
        # # print('sum=', muu + mvv)
        #
        #
        #
        # e_bmaj = fudge / (np.sqrt(2.0 * (muu + mvv - ftmp)))
        # e_bmaj = e_bmaj / np.pi * 180 * 3600 * 1000
        #
        # e_bmin = fudge / (np.sqrt(2.0 * (muu + mvv) + 2.0 * ftmp))
        # e_bmin = e_bmin / np.pi * 180 * 3600 * 1000

        # self.e_bpa, self.e_bmaj, self.e_bmin = e_bpa, e_bmaj, e_bmin
        self.e_bpa, self.e_bmin = e_bpa, min_beam
        return self.e_bpa, self.e_bmin, max_bl

    # other
    def get_max_uv(self):
        max_uv = self.max_range_single_uv
        if self.flag_unit == 1:
            max_uv = max_uv * 1000 / self.obs_wlen
        else:
            max_uv *= 1000
        return max_uv


class UVConfigParser(object):
    def __init__(self, _filename="config_uv.ini", _dbname='database.pkl'):
        #         path = os.path.abspath(path)
        #         path = os.getcwd()
        # path = "./CONFIG_FILE"
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
        self.unit_flag = "km"
        self.cutoff_angle = 0
        self.precession_mode = 0
        # obs
        self.obs_freq = 0

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
        self.obs_freq = config.getfloat("obs_mode", "obs_freq")
        self.cutoff_angle = config.getfloat("obs_mode", "cutoff_angle")
        self.precession_mode = config.getint("obs_mode", "precession_mode")
        self.unit_flag = config.get("obs_mode", "unit_flag")

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
        print("obs_freq=", self.obs_freq)
        print("cutoff_angle=", self.cutoff_angle)
        print("precession_mode=", self.precession_mode)
        print("unit_flag=", self.unit_flag)
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
        config.set("obs_mode", "obs_freq", "1.63e9")
        config.set("obs_mode", "bandwidth", "3.2e7")
        config.set("obs_mode", "cutoff_angle", "10.0")
        config.set("obs_mode", "precession_mode", "0")
        config.set("obs_mode", "unit_flag", "km")
        self.obs_freq = 1.63e9
        self.cutoff_angle = 10.0
        self.precession_mode = 0
        self.unit_flag = 'km'

        # add sections: station
        config.add_section("station")
        config.set("station", "pos_source", "0316+413, 0202+319")
        config.set("station", "pos_vlbi", "ShangHai, Tianma, Urumqi, GIFU11, HITACHI,KASHIM34")
        config.set("station", "pos_telemetry", "")
        config.set("station", "pos_satellite", "")
        self.str_source = ['0316+413', '0202+319']
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
    parser = argparse.ArgumentParser(description="Run the basic UV plots, calculate the beam size and position angle")
    parser.add_argument('-c',
                        '--config',
                        default='config_uv.ini',
                        help='Specify the configuration file')
    parser.add_argument('-g',
                        '--show_gui',
                        action="store_true",
                        help='Choose to show GUI or not')
    parser.add_argument('-s',
                        '--save_uv',
                        action="store_true",
                        help='Store the uv data (/OUTPUT/uv_basic/uvdata.txt)')
    parser.add_argument('-i',
                        '--img_info',
                        action="store_true",
                        help='Choose to show beam size and angle', )
    parser.add_argument('-n',
                        '--img_name',
                        help='Set the name (w/o suffix) of save imgs (under/OUTPUT/uv_basic/), '
                             'named with time by default',
                        default='bytime')
    parser.add_argument('-f',
                        '--img_fmt',
                        choices=['eps', 'png', 'pdf', 'svg', 'ps'],
                        help='Specify the img format (default:pdf)',
                        default='pdf')

    return parser.parse_args()


def run_uv_basic():
    # initialize parse and config objects
    args = parse_args()
    # for test in ide
    # args.img_info = True
    # args.show_gui = True
    if args.config != '':
        my_config_parser = UVConfigParser(args.config)
    else:
        my_config_parser = UVConfigParser()
        # print(my_config_parser.show_info())

    start_time = ut.time_2_mjd(*my_config_parser.time_start, 0)
    stop_time = ut.time_2_mjd(*my_config_parser.time_end, 0)
    time_step = ut.time_2_day(*my_config_parser.time_step)
    # invoke the calculation functions
    cutoff_dict = {"flag": lc.cutoff_mode["flag"], "CutAngle": my_config_parser.cutoff_angle}
    myFuncUV = FuncUv(start_time, stop_time, time_step,
                      my_config_parser.pos_mat_src[0],
                      my_config_parser.pos_mat_src,
                      my_config_parser.pos_mat_sat,
                      my_config_parser.pos_mat_vlbi,
                      my_config_parser.pos_mat_telemetry,
                      my_config_parser.obs_freq,
                      my_config_parser.baseline_type,
                      my_config_parser.unit_flag,
                      cutoff_dict,
                      my_config_parser.precession_mode
                      )

    # calculate u and v
    x, y, max_xy = myFuncUV.get_result_single_uv_with_update()

    # create the imgs
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, aspect='equal')
    if x is not None and y is not None:
        x = np.array(x)
        y = np.array(y)
        max_range = max_xy *1.1
        ax.scatter(x, y, s=1, marker='.', color='brown')
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_title("UV Plot: %s"% my_config_parser.str_source[0])
        if my_config_parser.unit_flag == 'km':
            ax.set_xlabel("u$(km)$")
            ax.set_ylabel("v$(km)$")
        else:
            ax.set_xlabel("u$(\lambda)$")
            ax.set_ylabel("v$(\lambda)$")
        ax.grid()
        # set science
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))

    # save uv data
    if args.save_uv:
        name = "uvdata" + time.asctime() + '.txt'
        uv_path = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'uv_basic')
        uv_path = os.path.join(uv_path, name)
        np.savetxt(uv_path, [x, y], fmt='%0.4f')
        # read_in = np.loadtxt(uv_path,dtype=np.float32)

    # show calculating info
    if args.img_info:
        e_bpa, e_bmin, max_bl = myFuncUV._calculate_beam_size()
        print('e_bpa={} degree, e_bmin = {} mas, max_baseline = {}'.format(e_bpa, e_bmin, max_bl))

    # save uv img
    img_type = 'pdf'
    if args.img_fmt in ['eps', 'png', 'pdf', 'svg', 'ps']:
        img_type = args.img_fmt
    if args.img_name == "bytime":
        name = "uv-" + time.asctime() + '.' + img_type
        uv_path = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'uv_basic')
        uv_path = os.path.join(uv_path, name)
        plt.savefig(uv_path)
    else:
        name = args.img_name + '.' + img_type
        uv_path = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'uv_basic')
        uv_path = os.path.join(uv_path, name)
        plt.savefig(uv_path)

    # show uv img
    if args.show_gui:
        plt.show()


if __name__ == "__main__":
    run_uv_basic()
