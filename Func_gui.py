import sys
import webbrowser
import os
import time
import logging
import pickle
from Func_cal import *
from Func_db import DbEditor

import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib.figure import Figure
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

import queue

from tkinter import ttk
import numpy as np
from tkinter import scrolledtext

import multiprocessing
from multiprocessing import Process
from threading import Thread

import load_conf as lc
import model_effect as me
import utility as ut
from Func_uv import FuncUv
from Func_img import FuncImg
from Func_obs import FuncObs
from Func_radplot import FuncRadPlot
# from Func_sched import FuncSched

__version__ = '1.0'
__author__ = 'Zhen Zhao'
_github_page_ = "https://github.com/ZhenZHAO/VNSIM"
__about_text__ = """ 
   VNSIM,  AN INTEGRATED VLBI SIMULATOR FOR EAST ASIA VLBI NETWORK

                     ZHEN ZHAO
  (Shanghai Astronomical Observatory)

VNSIM, an integrated VLBI simulator for East Asia VLBI Network(EAVN),  aims to assist radio astronomers to design, evaluate and simulate the VLBI experiments in a more friendly, flexible, and convenient fashion. It not only integrates the functionality of plotting (u, v) coverage, scheduling the observation, and displaying the dirty and CLEAN images, but also further extends to add new features such as the sensitivity calculation of a certain network. To facilitate the use for new astronomers, VNSIM provides flexible interactions on both command line and graphical user interface and offers friendly support for log report and database management. 

"""
__feedback_text__ = """ 
This software is still under developing by SKA group at Shanghai Astronomical Observatory. You are very welcome to leave your comments to us. 

Would you like to visit our group page?
"""
__help_text__ = """ 
VNSIM is an integrated VLBI simulator for East Asia VLBI Network. You can config your specific simulation on the left configure panel, where we support space VLBI and various constraints. 

You can choose different kinds of functions you like to explore on the left pane. There are three functional cluster, and you can switch them by clicking the corresponding label. In addition, there are two additional functions, parameter evaluation and database configuration, which can be accessible on the menu bar (Tool).  

You can add your self-defined VLBI stations or satellites by inserting new data into the database on the database page, and new data will be loaded into the GUI automatically.

You can save your configurations and your newly edited database, as well as various types of images. For All images, you can easily zoom in/out, move, adjust gamma, etc. 


Enjoy it!
"""
COPY_RIGHT_INFO = 'VNSIM \N{COPYRIGHT SIGN} 2018, %s' % __author__
TITLE = 'VNSIM - Shanghai Astronomical Observatory - version  %s' % __version__

# colors normalization
norm = mpl.colors.Normalize(vmin=0, vmax=0.6)

# addtional functions
POP_OUT_PARA_CAL = 'para'
POP_OUY_DB_EDIT = 'database'

# multiple choice type
MULTI_CHOICE_TYPE_SRC = 'source'
MULTI_CHOICE_TYPE_VLBA = 'vlba'
MULTI_CHOICE_TYPE_EAVN = 'eavn'
MULTI_CHOICE_TYPE_EVN = 'evn'
MULTI_CHOICE_TYPE_LBA = 'lba'
MULTI_CHOICE_TYPE_OTHER = 'other'
MULTI_CHOICE_TYPE_SAT = 'satellite'
MULTI_CHOICE_TYPE_TELE = 'telemetry'

# trigger to update (set multiple by addition)
UPDATE_TYPE_UV = 1
UPDATE_TYPE_OBS = 2
UPDATE_TYPE_IMG = 4
UPDATE_TYPE_RAD = 8
UPDATE_TYPE_UO = 1 + 2
UPDATE_TYPE_UI = 1 + 4
UPDATE_TYPE_UOI = 1 + 2 + 4

# multiprocess type
PROCESS_TYPE_UV_SINGLE = 0
PROCESS_TYPE_UV_SKY = 1
PROCESS_TYPE_UV_TIME = 2
PROCESS_TYPE_UV_SRC = 3
PROCESS_TYPE_OBS_AZ_EL = 4
PROCESS_TYPE_OBS_SURVEY = 5
PROCESS_TYPE_IMG = 6
PROCESS_TYPE_RAD = 7

# pop out uv window
UV_SHOW_POP_TYPE_YEAR = 'All Year Coverage'
UV_SHOW_POP_TYPE_SRC = 'Multi Source Coverage'

MY_IMG_FONT = {'family': 'Arial',
               'color': 'darkred',
               'weight': 'normal',
               'size': 10,
               }


def process_finish_call_back(args):
    # logger = logging.getLogger()
    # logger.info(time.asctime() + ": {0} updating - done !".format(str(args)))
    logging.info(time.asctime() + ": {0} updating - done !".format(str(args)))
    print(time.asctime() + ": {0} updating - done !".format(str(args)))


class AppData(object):
    def __init__(self, parent=None):
        # super().__init__()
        self.parent = parent

        # 1. initial parameters of calculating class
        self.init_true_cal_para_globally()
        self.update_time()

        # 2. calculating class
        self.myFuncUv = None
        self.myFuncObs = None
        self.myFuncImg = None
        self.myFuncRad = None

        # 3. all simulation result
        self.result_uv_single_u = []
        self.result_uv_single_v = []
        self.result_uv_single_max = 0.0

        self.result_uv_sky_u = []
        self.result_uv_sky_v = []
        self.result_uv_sky_max = 0.0

        self.result_uv_time_u = []
        self.result_uv_time_v = []
        self.result_uv_time_max = 0.0

        self.result_uv_src_name = []
        self.result_uv_src_u = []
        self.result_uv_src_v = []
        self.result_uv_src_max = 0.0

        self.result_obs_az = []
        self.result_obs_el = []
        self.result_obs_hour = []

        self.result_obs_survey_sun = []
        self.result_obs_survey_moon = []
        self.result_obs_survey_array = []

        self.result_img_model = []
        self.result_img_beam = []
        self.result_img_dirty = []
        self.result_img_clean = []
        self.result_img_x_max = 0.0

        self.result_rad_u = []
        self.result_rad_v = []
        self.result_rad_bl = []
        self.result_rad_vis = []

    def clear_result(self):
        self.result_uv_single_u = []
        self.result_uv_single_v = []
        self.result_uv_single_max = 0.0

        self.result_uv_sky_u = []
        self.result_uv_sky_v = []
        self.result_uv_sky_max = 0.0

        self.result_uv_time_u = []
        self.result_uv_time_v = []
        self.result_uv_time_max = 0.0

        self.result_uv_src_name = []
        self.result_uv_src_u = []
        self.result_uv_src_v = []
        self.result_uv_src_max = 0.0

        self.result_obs_az = []
        self.result_obs_el = []
        self.result_obs_hour = []

        self.result_obs_survey_sun = []
        self.result_obs_survey_moon = []
        self.result_obs_survey_array = []

        self.result_img_model = []
        self.result_img_beam = []
        self.result_img_dirty = []
        self.result_img_clean = []
        self.result_img_x_max = 0.0

        self.result_rad_u = []
        self.result_rad_v = []
        self.result_rad_bl = []
        self.result_rad_vis = []

    def update_time(self):
        self.var_start_time = ut.time_2_mjd(self.var_start_year, self.var_start_month,
                                            self.var_start_day, self.var_start_hour,
                                            self.var_start_minute, self.var_start_second, 0)
        self.var_stop_time = ut.time_2_mjd(self.var_stop_year, self.var_stop_month,
                                           self.var_stop_day, self.var_stop_hour,
                                           self.var_stop_minute, self.var_stop_second, 0)
        self.var_time_step = ut.time_2_day(0, self.var_step_hour, self.var_step_minute,
                                           self.var_step_second)

    def update_calculation_class(self):

        self.myFuncUv = FuncUv(self.var_start_time, self.var_stop_time, self.var_time_step,
                               self.var_main_src, self.var_src_lst,
                               self.var_sat_lst, self.var_vlbi_lst, self.var_telem_lst,
                               self.var_freq, self.var_baseline_type, self.var_unit,
                               self.var_cutoff_mode_dict, self.var_procession)

        self.myFuncObs = FuncObs(self.var_start_time, self.var_stop_time, self.var_time_step,
                                 self.var_main_src, self.var_vlbi_lst,
                                 self.var_sat_lst, self.var_telem_lst,
                                 self.var_baseline_type, self.var_cutoff_mode_dict, self.var_procession)

        self.myFuncImg = FuncImg(self.var_model_name, self.var_pix_size,
                                 self.result_uv_single_u, self.result_uv_single_v, self.result_uv_single_max,
                                 self.var_freq, True, self.var_clean_gain,
                                 self.var_clean_thresh, self.var_clean_niter, self.var_unit)

        self.myFuncRad = FuncRadPlot(self.var_rad_plot_file)

    def init_true_cal_para_globally(self):
        # time
        self.var_start_year = lc.StartTimeGlobalYear
        self.var_start_month = lc.StartTimeGlobalMonth
        self.var_start_day = lc.StartTimeGlobalDay
        self.var_start_hour = lc.StartTimeGlobalHour
        self.var_start_minute = lc.StartTimeGlobalMinute
        self.var_start_second = lc.StartTimeGlobalSecond

        self.var_stop_year = lc.StopTimeGlobalYear
        self.var_stop_month = lc.StopTimeGlobalMonth
        self.var_stop_day = lc.StopTimeGlobalDay
        self.var_stop_hour = lc.StopTimeGlobalHour
        self.var_stop_minute = lc.StopTimeGlobalMinute
        self.var_stop_second = lc.StopTimeGlobalSecond

        self.var_step_hour = lc.TimeStepGlobalHour
        self.var_step_minute = lc.TimeStepGlobalMinute
        self.var_step_second = lc.TimeStepGlobalSecond

        # source
        self.var_main_src = lc.pos_mat_src[0]
        self.var_src_lst = lc.pos_mat_src

        # vlbi
        self.var_vlbi_lst = lc.pos_mat_vlbi

        # satellite
        self.var_sat_lst = lc.pos_mat_sat
        self.var_telem_lst = lc.pos_mat_telemetry

        # parameters
        self.var_freq = lc.obs_freq
        self.var_unit = lc.unit_flag
        self.var_baseline_type = lc.baseline_type
        self.var_procession = lc.precession_mode
        self.var_cutoff_mode_dict = lc.cutoff_mode

        # imaging
        self.var_model_name = lc.source_model
        self.var_pix_size = lc.n_pix
        self.var_clean_gain = lc.clean_gain
        self.var_clean_thresh = lc.clean_threshold
        self.var_clean_niter = lc.clean_niter
        self.var_color_map_name = lc.color_map_name

        # rad plot
        self.var_rad_plot_file = lc.rad_plot_file

    def __call__(self, p_type, d_key, d_data):
        # print("call run!")
        if p_type == PROCESS_TYPE_UV_SINGLE:
            print('begin to update single uv')
            d_data[d_key] = self.update_data_uv_single()
            self.parent.set_after_call('Main UV Coverage - data')
            # process_finish_call_back('Main UV Coverage - data')

        elif p_type == PROCESS_TYPE_UV_SKY:
            print('begin to update all sky uv')
            d_data[d_key] = self.update_data_uv_sky()
            process_finish_call_back('SKY Coverage - data')

        elif p_type == PROCESS_TYPE_UV_TIME:
            print('begin to update all year uv')
            d_data[d_key] = self.update_data_uv_time()
            process_finish_call_back('AllYear Coverage - data')

        elif p_type == PROCESS_TYPE_UV_SRC:
            print('begin to update multi source uv')
            d_data[d_key] = self.update_data_uv_src()
            process_finish_call_back('MultiSrc Coverage  - data')

        elif p_type == PROCESS_TYPE_OBS_AZ_EL:
            print('begin to update Az-EL')
            d_data[d_key] = self.update_data_obs_az_el()
            process_finish_call_back('AZ-EL  - data')

        elif p_type == PROCESS_TYPE_OBS_SURVEY:
            print('begin to update Az-EL')
            d_data[d_key] = self.update_data_obs_survey()
            process_finish_call_back('SKY SURVEY  - data')

        elif p_type == PROCESS_TYPE_IMG:
            print('begin to update img')
            d_data[d_key] = self.update_data_img()
            process_finish_call_back('ALL Imaging - data')

        elif p_type == PROCESS_TYPE_RAD:
            print('begin to update radplot')
            d_data[d_key] = self.update_data_rad()
            process_finish_call_back('Rad Plot - data')

        else:
            pass

    def _update_uv_with_multiprocess(self):
        # 1. update para
        self.update_data_uv_para()
        # 2. set up the passing arguments
        mng = multiprocessing.Manager()
        arg1 = [PROCESS_TYPE_UV_SINGLE, PROCESS_TYPE_UV_SKY, PROCESS_TYPE_UV_TIME, PROCESS_TYPE_UV_SRC]
        arg2 = ['single_uv', 'sky_uv', 'time_uv', 'src_uv']
        data_dict = mng.dict()
        # 3. run multiprocess
        jobs = [Process(target=self, args=(arg1[i], arg2[i], data_dict)) for i in range(len(arg1))]

        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        # 4. parse the result
        for each in arg2:
            if each == 'single_uv':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_uv_single_u = value[0]
                    self.result_uv_single_v = value[1]
                    self.result_uv_single_max = value[2]
            elif each == 'sky_uv':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_uv_sky_u = value[0]
                    self.result_uv_sky_v = value[1]
                    self.result_uv_sky_max = value[2]
            elif each == 'time_uv':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_uv_time_u = value[0]
                    self.result_uv_time_v = value[1]
                    self.result_uv_time_max = value[2]
            elif each == 'src_uv':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_uv_src_name = value[0]
                    self.result_uv_src_u = value[1]
                    self.result_uv_src_v = value[2]
                    self.result_uv_src_max = value[3]

    def _update_obs_with_multiprocess(self):
        self.update_data_obs_para()
        mng = multiprocessing.Manager()
        arg1 = [PROCESS_TYPE_OBS_AZ_EL, PROCESS_TYPE_OBS_SURVEY]
        arg2 = ['az_el', 'survey']
        data_dict = mng.dict()

        jobs = [Process(target=self, args=(arg1[i], arg2[i], data_dict)) for i in range(len(arg1))]

        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        for each in arg2:
            if each == 'az_el':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_obs_az = value[0]
                    self.result_obs_el = value[1]
                    self.result_obs_hour = value[2]
            elif each == 'survey':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_obs_survey_sun = value[0]
                    self.result_obs_survey_moon = value[1]
                    self.result_obs_survey_array = value[2]

    def _update_uv_obs_with_multiprocess(self):
        # 1. update para
        self.update_data_uv_para()
        self.update_data_obs_para()
        # 2. set up the passing arguments
        mng = multiprocessing.Manager()
        arg1 = [PROCESS_TYPE_UV_SINGLE, PROCESS_TYPE_UV_SKY, PROCESS_TYPE_UV_TIME,
                PROCESS_TYPE_UV_SRC, PROCESS_TYPE_OBS_AZ_EL, PROCESS_TYPE_OBS_SURVEY]
        arg2 = ['single_uv', 'sky_uv', 'time_uv', 'src_uv', 'az_el', 'survey']
        data_dict = mng.dict()
        # 3. run multiprocess
        jobs = [Process(target=self, args=(arg1[i], arg2[i], data_dict)) for i in range(len(arg1))]

        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        # 4. parse the result
        for each in arg2:
            if each == 'single_uv':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_uv_single_u = value[0]
                    self.result_uv_single_v = value[1]
                    self.result_uv_single_max = value[2]
            elif each == 'sky_uv':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_uv_sky_u = value[0]
                    self.result_uv_sky_v = value[1]
                    self.result_uv_sky_max = value[2]
            elif each == 'time_uv':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_uv_time_u = value[0]
                    self.result_uv_time_v = value[1]
                    self.result_uv_time_max = value[2]
            elif each == 'src_uv':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_uv_src_name = value[0]
                    self.result_uv_src_u = value[1]
                    self.result_uv_src_v = value[2]
                    self.result_uv_src_max = value[3]
            elif each == 'az_el':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_obs_az = value[0]
                    self.result_obs_el = value[1]
                    self.result_obs_hour = value[2]
            elif each == 'survey':
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_obs_survey_sun = value[0]
                    self.result_obs_survey_moon = value[1]
                    self.result_obs_survey_array = value[2]

    def _update_all_first_update_with_accelerate(self):
        # update all uv
        self._update_uv_with_multiprocess()
        # update obs, img, radplot
        self.update_data_obs_para()
        self.update_data_img_para()
        self.update_data_rad_para()

        mng = multiprocessing.Manager()
        arg1 = [PROCESS_TYPE_OBS_AZ_EL, PROCESS_TYPE_OBS_SURVEY, PROCESS_TYPE_IMG, PROCESS_TYPE_RAD]
        arg2 = ['az_el', 'survey', 'imaging', 'radplot']
        data_dict = mng.dict()

        jobs = [Process(target=self, args=(arg1[i], arg2[i], data_dict)) for i in range(len(arg1))]

        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        for each in arg2:
            if each == arg2[0]:
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_obs_az = value[0]
                    self.result_obs_el = value[1]
                    self.result_obs_hour = value[2]
            elif each == arg2[1]:
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_obs_survey_sun = value[0]
                    self.result_obs_survey_moon = value[1]
                    self.result_obs_survey_array = value[2]
            elif each == arg2[2]:
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_img_model = value[0]
                    self.result_img_beam = value[1]
                    self.result_img_dirty = value[2]
                    self.result_img_clean = value[3]
                    self.result_img_x_max = value[4]

            elif each == arg2[3]:
                value = data_dict.get(each, None)
                if value is not None:
                    self.result_rad_u = value[0]
                    self.result_rad_v = value[1]
                    self.result_rad_bl = value[2]
                    self.result_rad_vis = value[3]

    def update_all_with_flag(self, update_flag_first_run, update_flag_uv,
                             update_flag_obs, update_flag_img,
                             update_flag_rad):

        # Process(target=self.parent.gress_bar.start(), args=(arg1[i], arg2[i], data_dict))
        # self.parent.gress_bar.quit()

        if update_flag_first_run:
            self._update_all_first_update_with_accelerate()
            return

        # update time
        if update_flag_uv or update_flag_obs:
            self.update_time()

        # update uv, obs with multiprocessing
        if update_flag_uv and not update_flag_obs:
            self._update_uv_with_multiprocess()
        elif update_flag_obs and not update_flag_uv:
            self._update_obs_with_multiprocess()
        elif update_flag_obs and update_flag_uv:
            self._update_uv_obs_with_multiprocess()

        # update img
        if update_flag_img:
            print('begin to update img')
            self.update_data_img_para()
            tmp = self.update_data_img()
            if tmp is not None:
                self.result_img_model = tmp[0]
                self.result_img_beam = tmp[1]
                self.result_img_dirty = tmp[2]
                self.result_img_clean = tmp[3]
                self.result_img_x_max = tmp[4]
            process_finish_call_back('ALL Imaging - data')

        # update radplot
        if update_flag_rad:
            print('begin to update radplot')
            self.update_data_rad_para()
            tmp = self.update_data_rad()
            if tmp is not None:
                self.result_rad_u = tmp[0]
                self.result_rad_v = tmp[1]
                self.result_rad_bl = tmp[2]
                self.result_rad_vis = tmp[3]
            process_finish_call_back('Rad Plot - data')

        # self.parent.notify_queue.put((1,))

    def update_all_with_flag_wo_speed(self, update_flag_first_run, update_flag_uv,
                                      update_flag_obs, update_flag_img,
                                      update_flag_rad):
        if update_flag_first_run or update_flag_uv:
            print('begin to update uv')
            self.update_data_uv_para()
            tmp1 = self.update_data_uv_single()
            tmp2 = self.update_data_uv_sky()
            tmp3 = self.update_data_uv_time()
            tmp4 = self.update_data_uv_src()
            if tmp1 is not None:
                self.result_uv_single_u = tmp1[0]
                self.result_uv_single_v = tmp1[1]
                self.result_uv_single_max = tmp1[2]
            if tmp2 is not None:
                self.result_uv_sky_u = tmp2[0]
                self.result_uv_sky_v = tmp2[1]
                self.result_uv_sky_max = tmp2[2]
            if tmp3 is not None:
                self.result_uv_time_u = tmp3[0]
                self.result_uv_time_v = tmp3[1]
                self.result_uv_time_max = tmp3[2]
            if tmp4 is not None:
                self.result_uv_src_name = tmp4[0]
                self.result_uv_src_u = tmp4[1]
                self.result_uv_src_v = tmp4[2]
                self.result_uv_src_max = tmp4[3]
            process_finish_call_back('UV Coverage - data')

        if update_flag_first_run or update_flag_obs:
            print('begin to update obs')
            self.update_data_obs_para()
            tmp1 = self.update_data_obs_az_el()
            if tmp1 is not None:
                self.result_obs_az = tmp1[0]
                self.result_obs_el = tmp1[1]
                self.result_obs_hour = tmp1[2]
            tmp2 = self.update_data_obs_survey()
            if tmp2 is not None:
                self.result_obs_survey_sun = tmp2[0]
                self.result_obs_survey_moon = tmp2[1]
                self.result_obs_survey_array = tmp2[2]
            process_finish_call_back('Observation - data')

        if update_flag_first_run or update_flag_img:
            print('begin to update img')
            self.update_data_img_para()
            tmp = self.update_data_img()
            if tmp is not None:
                self.result_img_model = tmp[0]
                self.result_img_beam = tmp[1]
                self.result_img_dirty = tmp[2]
                self.result_img_clean = tmp[3]
                self.result_img_x_max = tmp[4]
            process_finish_call_back('ALL Imaging - data')

        if update_flag_first_run or update_flag_rad:
            print('begin to update radplot')
            self.update_data_rad_para()
            tmp = self.update_data_rad()
            if tmp is not None:
                self.result_rad_u = tmp[0]
                self.result_rad_v = tmp[1]
                self.result_rad_bl = tmp[2]
                self.result_rad_vis = tmp[3]
            process_finish_call_back('Rad Plot - data')

    # update data
    def update_data_uv_para(self):
        self.myFuncUv = FuncUv(self.var_start_time, self.var_stop_time, self.var_time_step,
                               self.var_main_src, self.var_src_lst,
                               self.var_sat_lst, self.var_vlbi_lst, self.var_telem_lst,
                               self.var_freq, self.var_baseline_type, self.var_unit,
                               self.var_cutoff_mode_dict, self.var_procession)

    def update_data_uv_single(self):
        return self.myFuncUv.get_result_single_uv_with_update()

    def update_data_uv_sky(self):
        return self.myFuncUv.get_result_sky_uv_with_update()

    def update_data_uv_time(self):
        return self.myFuncUv.get_result_year_uv_with_update()

    def update_data_uv_src(self):
        return self.myFuncUv.get_result_multi_src_with_update()

    def update_data_obs_para(self):
        self.myFuncObs = FuncObs(self.var_start_time, self.var_stop_time, self.var_time_step,
                                 self.var_main_src, self.var_vlbi_lst,
                                 self.var_sat_lst, self.var_telem_lst,
                                 self.var_baseline_type, self.var_cutoff_mode_dict, self.var_procession)

    def update_data_obs_az_el(self):
        return self.myFuncObs.get_result_az_el_with_update()

    def update_data_obs_survey(self):
        return self.myFuncObs.get_result_sky_survey_with_update()

    def update_data_img_para(self):
        self.myFuncImg = FuncImg(self.var_model_name, self.var_pix_size,
                                 self.result_uv_single_u, self.result_uv_single_v, self.result_uv_single_max,
                                 self.var_freq, True, self.var_clean_gain,
                                 self.var_clean_thresh, self.var_clean_niter, self.var_unit)

    def update_data_img(self):
        if len(self.result_uv_single_u) > 0 and len(self.result_uv_single_v) > 0 and self.var_model_name != '':
            # 2. update results
            tmp_model, tmp_x_max = self.myFuncImg.get_result_src_model_with_update()
            tmp_beam = self.myFuncImg.get_result_dirty_beam_with_update()
            tmp_dirty = self.myFuncImg.get_result_dirty_map_with_update()
            tmp_clean, tmp_res, show_src, show_cln_beam = self.myFuncImg.get_result_clean_map_with_update()

            return tmp_model, tmp_beam, tmp_dirty, tmp_clean, tmp_x_max
        else:
            return None

    def update_data_rad_para(self):
        self.myFuncRad = FuncRadPlot(self.var_rad_plot_file)

    def update_data_rad(self):
        if self.myFuncRad.get_data_state():
            tmp_u, tmp_v = self.myFuncRad.get_result_uv_data()
            tmp_bl, tmp_vis = self.myFuncRad.get_result_rad_data()

            return tmp_u, tmp_v, tmp_bl, tmp_vis
        else:
            return None

    # getters - uv
    def get_data_uv_single(self):
        return self.result_uv_single_u, self.result_uv_single_v, self.result_uv_single_max

    def get_data_uv_sky(self):
        return self.result_uv_sky_u, self.result_uv_sky_v, self.result_uv_sky_max

    def get_data_uv_time(self):
        return self.result_uv_time_u, self.result_uv_time_v, self.result_uv_time_max

    def get_data_uv_src(self):
        return self.result_uv_src_name, self.result_uv_src_u, self.result_uv_src_v, self.result_uv_src_max

    # getters - obs
    def get_data_obs_az_el(self):
        return self.result_obs_az, self.result_obs_el, self.result_obs_hour

    def get_data_obs_survey(self):
        return self.result_obs_survey_sun, self.result_obs_survey_moon, self.result_obs_survey_array

    # getters - img
    def get_data_img_model(self):
        return self.result_img_model, self.result_img_x_max

    def get_data_img_beam(self):
        return self.result_img_beam, self.result_img_x_max

    def get_data_img_dirty(self):
        return self.result_img_dirty, self.result_img_x_max

    def get_data_img_clean(self):
        return self.result_img_clean, self.result_img_x_max

    # getters - rad
    def get_data_rad_uv(self):
        return self.result_rad_u, self.result_rad_v

    def get_data_rad_vis(self):
        return self.result_rad_bl, self.result_rad_vis


class AppGUI(object):
    def __init__(self, master=None, db_file='', pkl_file=''):
        self.notify_queue = queue.Queue()
        self.window = master
        self.window.protocol("WM_DELETE_WINDOW", self.quit)
        self.db_name = db_file
        self.pkl_name = pkl_file
        self.pkl_path = os.path.join(os.path.join(os.getcwd(), 'DATABASE'), self.pkl_name)

        # 0. real-time input checking
        self.test_float = self.window.register(self.test_input_float)

        # 1. initialize pop out windows
        self.PopOutParaCal = None
        self.is_pop_out_para = False
        self.PopOutDbEditor = None
        self.is_pop_out_db = False

        # 2. init cal class/parameters
        self.myData = AppData(self)

        # 3. load database (db->ui_choice_para->ui_cal_para->true_cal_para)
        self._pickle_load_database()

        # 4. init ui parameters
        self._init_ui_cal_para()
        self._init_ui_choice_para()
        self._load_cal_para_to_ui_para()

        # 5. whether need to update the calculation
        self.if_update_first_time = True
        self.if_update_uv = False
        self.if_update_img = False
        self.if_update_obs = False
        self.if_update_rad = False

        # 6. show info on uv plane
        self.tab_uv_ui_var_info = tk.StringVar('')

        # 7. color bar on obs plane
        self.obs_panel_color_bar = None

        # 8. gui after call
        self.gress_bar = GressBar()
        self.is_start_process_msg = False
        # self.process_msg()
        # self.after_call_new = False
        # self.after_call_str = ''
        # self.gui_after_call()

        # 9. init gui
        self.gui_var_status = tk.StringVar()  # status
        self._gui_int()

    def start_process_msg(self):
        self.is_start_process_msg = True
        self.process_msg()

    def stop_process_msg(self):
        self.is_start_process_msg = False

    def process_msg(self):
        if not self.is_start_process_msg:
            return
        self.window.after(400, self.process_msg)
        # print("heloo")
        # if self.gress_bar.master is not None:
        #     print("heloo")
        # if self.gress_bar.master is not None:
        #     self.gress_bar.master.overrideredirect(True)
        # print("=-="*20)
        while not self.notify_queue.empty():
            try:
                msg = self.notify_queue.get()
                if msg[0] == 1:
                    time.sleep(0.5)
                    self.gress_bar.quit()
                    self.stop_process_msg()
            except queue.Empty:
                pass

    def reset_all(self):
        self.myData.init_true_cal_para_globally()
        self._load_cal_para_to_ui_para()
        self.logger.info(time.asctime() + ": reset all gui para - done!")
        self.gui_var_status.set("Ready")

        self.reset_panel_uv()
        logging.info(time.asctime() + ": reset panel-coverage - done!")

        self.reset_panel_obs()
        logging.info(time.asctime() + ": reset panel-observe - done!")

        self.reset_panel_img()
        logging.info(time.asctime() + ": reset panel-imaging - done!")

        self.reset_panel_radplot()
        logging.info(time.asctime() + ": reset panel-radplot - done!")

        self.if_update_first_time = True

    def set_after_call(self, info):
        self.after_call_new = True
        self.after_call_str = info

    def gui_after_call(self):
        self.window.after(200, self.gui_after_call)
        if self.after_call_new:
            print('hell0')
            process_finish_call_back(self.after_call_str)
            self.after_call_new = False

    def apply_all_with_multiprocess(self):
        self._load_ui_para_to_cal_para()
        self.logger.info(time.asctime() + ": reading gui para - done!")
        self.gui_var_status.set('Running')

        self.start_time = time.time()

        self.myData.update_all_with_flag(self.if_update_first_time,
                                         self.if_update_uv,
                                         self.if_update_obs,
                                         self.if_update_img,
                                         self.if_update_rad)
        self.update_all_panel()

    def apply_all(self):
        self._load_ui_para_to_cal_para()
        self.logger.info(time.asctime() + ": reading gui para - done!")
        self.gui_var_status.set('Running')

        self.start_time = time.time()

        self.start_process_msg()

        def cal_para_normal(_queue):
            self.update_all_data()
            self.update_all_panel()
            _queue.put((1,))

        th1 = Thread(target=cal_para_normal, args=(self.notify_queue,))
        th1.start()
        self.gress_bar.start()
        th1.join()

    def update_all_data(self):
        self.myData.update_all_with_flag_wo_speed(self.if_update_first_time,
                                                  self.if_update_uv,
                                                  self.if_update_obs,
                                                  self.if_update_img,
                                                  self.if_update_rad)

    def update_all_panel(self):
        # update uv coverage functions
        if self.if_update_first_time or self.if_update_uv:
            self.update_panel_uv()
            process_finish_call_back("UV - panel")
            self.if_update_uv = False

        # update uv observation functions
        if self.if_update_first_time or self.if_update_obs:
            self.update_panel_obs()
            process_finish_call_back("OBS - panel")
            self.if_update_obs = False

        # update uv imaging functions
        if self.if_update_first_time or self.if_update_img:
            self.update_panel_img()
            process_finish_call_back("IMG - panel")
            self.if_update_img = False

        # update uv radplot functions
        if self.if_update_first_time or self.if_update_rad:
            self.update_panel_radplot()
            process_finish_call_back("RAD - panel")
            self.if_update_rad = False

        self.if_update_first_time = False
        self.gui_var_status.set('Done')

        print("RUNING TIME IS %s" % (time.time() - self.start_time))

    def trigger_panel_update(self, t_type):
        self.if_update_uv = (t_type & 1 != 0)
        self.if_update_obs = (t_type & 2 != 0)
        self.if_update_img = (t_type & 4 != 0)
        self.if_update_rad = (t_type & 8 != 0)
        # print(t_type, self.if_update_uv, self.if_update_obs, self.if_update_img, self.if_update_rad)

    def trigger_update_event(self, event, t_type):
        self.if_update_uv = (t_type & 1 != 0)
        self.if_update_obs = (t_type & 2 != 0)
        self.if_update_img = (t_type & 4 != 0)
        self.if_update_rad = (t_type & 8 != 0)

    def refresh_ui_config_with_db(self):
        self._pickle_load_database()
        self._init_ui_choice_para()
        # 多选框 是不需要单独更新的，因为每次弹出都会读取新的 choice_lst
        # 需要更新与数据库相关的单选box - 只有一个，ui_main_src 处
        self.selec_single_src.config(values=self.ui_choice_src)

    def _pickle_load_database(self):
        with open(self.pkl_path, 'rb') as fr:
            self.db_src_dict = pickle.load(fr)
            self.db_sat_dict = pickle.load(fr)
            self.db_telem_dict = pickle.load(fr)
            self.db_vlbi_vlba_dict = pickle.load(fr)
            self.db_vlbi_evn_dict = pickle.load(fr)
            self.db_vlbi_eavn_dict = pickle.load(fr)
            self.db_vlbi_lba_dict = pickle.load(fr)
            self.db_vlbi_other_dict = pickle.load(fr)

    def _init_ui_cal_para(self):

        # time
        self.ui_start_year = tk.StringVar('')
        self.ui_start_month = tk.StringVar('')
        self.ui_start_day = tk.StringVar('')
        self.ui_start_hour = tk.StringVar('')
        self.ui_start_minute = tk.StringVar('')
        self.ui_start_second = tk.StringVar('')

        self.ui_stop_year = tk.StringVar('')
        self.ui_stop_month = tk.StringVar('')
        self.ui_stop_day = tk.StringVar('')
        self.ui_stop_hour = tk.StringVar('')
        self.ui_stop_minute = tk.StringVar('')
        self.ui_stop_second = tk.StringVar('')

        self.ui_step_hour = tk.StringVar('')
        self.ui_step_minute = tk.StringVar('')
        self.ui_step_second = tk.StringVar('')

        # source
        self.ui_main_src = tk.StringVar('')
        self.ui_src_lst = []

        # vlbi
        self.ui_vlbi_vlba_lst = []
        self.ui_vlbi_evn_lst = []
        self.ui_vlbi_eavn_lst = []
        self.ui_vlbi_lba_lst = []
        self.ui_vlbi_othter_lst = []

        # satellite
        self.ui_sat_lst = []
        self.ui_telem_lst = []

        # parameters
        self.ui_freq = tk.StringVar('')
        self.ui_unit = tk.StringVar("")
        self.ui_cutoff_angle = tk.StringVar('')
        self.ui_procession = tk.StringVar('')
        self.ui_baseline_gg = tk.IntVar()
        self.ui_baseline_gs = tk.IntVar()
        self.ui_baseline_ss = tk.IntVar()

        # imaging
        self.ui_model_name = tk.StringVar('')
        self.ui_pix_size = tk.StringVar('')
        self.ui_clean_gain = tk.StringVar('')
        self.ui_clean_thresh = tk.StringVar('')
        self.ui_clean_niter = tk.StringVar('')
        self.ui_color_map = tk.StringVar('')

        # radplot
        self.ui_rad_plot_file = tk.StringVar('')

    def _init_ui_choice_para(self):
        # database related
        self.ui_choice_src = list(self.db_src_dict.keys())

        self.ui_choice_vlbi_vlba = list(self.db_vlbi_vlba_dict.keys())
        self.ui_choice_vlbi_evn = list(self.db_vlbi_evn_dict.keys())
        self.ui_choice_vlbi_eavn = list(self.db_vlbi_eavn_dict.keys())
        self.ui_choice_vlbi_lba = list(self.db_vlbi_lba_dict.keys())
        self.ui_choice_vlbi_other = list(self.db_vlbi_other_dict.keys())

        self.ui_choice_sat = list(self.db_sat_dict.keys())
        self.ui_choice_telem = list(self.db_telem_dict.keys())

        # time
        self.ui_choice_year = list(range(1980, 2201))
        self.ui_choice_month = list(range(1, 13))
        self.ui_choice_day = list(range(1, 32))
        self.ui_choice_hour = list(range(0, 24))
        # self.ui_choice_step_hour = list(range(0, 50))
        self.ui_choice_minute = list(range(0, 60))
        self.ui_choice_second = list(range(0, 60))

        # para
        # self.ui_choice_unit = ['lambda', 'km']  # 'km',
        self.ui_choice_unit = ['km']  # 'km'
        self.ui_choice_precession = [0, 1, 2]

        # imaging
        self.ui_choice_model_name = ['Cloud.model',
                                     'Discs.model',
                                     'Double-source.model',
                                     'Faceon-Galaxy.model',
                                     'Five-Gauss.model',
                                     'Nebula.model',
                                     'One-Disc.model',
                                     'point.model',
                                     'RadioGalaxy.model']
        self.ui_choice_pix = [128, 256, 512, 1024]
        self.ui_choice_clean_niter = list(range(20, 160, 10))
        self.ui_choice_clean_thred = list(range(-100, 110, 10))
        self.ui_choice_clean_gain = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        # https://matplotlib.org/examples/color/colormaps_reference.html
        self.ui_choice_color_map = ['jet', 'rainbow', 'Greys', 'hot', 'cool', 'nipy_spectral', 'viridis']

    def _load_ui_para_to_cal_para(self):
        # time
        self.myData.var_start_year = int(self.ui_start_year.get())
        self.myData.var_start_month = int(self.ui_start_month.get())
        self.myData.var_start_day = int(self.ui_start_day.get())
        self.myData.var_start_hour = int(self.ui_start_hour.get())
        self.myData.var_start_minute = int(self.ui_start_minute.get())
        self.myData.var_start_second = int(self.ui_start_second.get())

        self.myData.var_stop_year = int(self.ui_stop_year.get())
        self.myData.var_stop_month = int(self.ui_stop_month.get())
        self.myData.var_stop_day = int(self.ui_stop_day.get())
        self.myData.var_stop_hour = int(self.ui_stop_hour.get())
        self.myData.var_stop_minute = int(self.ui_stop_minute.get())
        self.myData.var_stop_second = int(self.ui_stop_second.get())

        self.myData.var_step_hour = int(self.ui_step_hour.get())
        self.myData.var_step_minute = int(self.ui_step_minute.get())
        self.myData.var_step_second = int(self.ui_step_second.get())

        # source
        self.myData.var_main_src = list(self.db_src_dict[self.ui_main_src.get()])
        # print(self.var_main_src)
        self.myData.var_src_lst = []
        for each in self.ui_src_lst:
            self.myData.var_src_lst.append(list(self.db_src_dict[each]))

        # vlbi
        self.myData.var_vlbi_lst = []
        for each in self.ui_vlbi_vlba_lst:
            self.myData.var_vlbi_lst.append(list(self.db_vlbi_vlba_dict[each]))
        for each in self.ui_vlbi_evn_lst:
            self.myData.var_vlbi_lst.append(list(self.db_vlbi_evn_dict[each]))
        for each in self.ui_vlbi_eavn_lst:
            self.myData.var_vlbi_lst.append(list(self.db_vlbi_eavn_dict[each]))
        for each in self.ui_vlbi_lba_lst:
            self.myData.var_vlbi_lst.append(list(self.db_vlbi_lba_dict[each]))
        for each in self.ui_vlbi_othter_lst:
            self.myData.var_vlbi_lst.append(list(self.db_vlbi_other_dict[each]))

        # satellite
        self.myData.var_sat_lst = []
        for each in self.ui_sat_lst:
            self.myData.var_sat_lst.append(list(self.db_sat_dict[each]))

        self.myData.var_telem_lst = []
        for each in self.ui_telem_lst:
            self.myData.var_telem_lst.append(list(self.db_telem_dict[each]))

        # parameters
        self.myData.var_freq = float(self.ui_freq.get()) * 1e9
        self.myData.var_unit = self.ui_unit.get()
        # print(self.myData.var_unit)
        self.myData.var_baseline_type = int(self.ui_baseline_gg.get()) + int(self.ui_baseline_gs.get()) * 2 + int(
            self.ui_baseline_ss.get()) * 4
        self.myData.var_procession = int(self.ui_procession.get())
        # print(self.var_procession)
        self.myData.var_cutoff_mode_dict['CutAngle'] = float(self.ui_cutoff_angle.get())

        # imaging
        self.myData.var_model_name = self.ui_model_name.get()
        self.myData.var_pix_size = int(self.ui_pix_size.get())
        self.myData.var_clean_gain = float(self.ui_clean_gain.get())
        self.myData.var_clean_thresh = int(self.ui_clean_thresh.get())
        self.myData.var_clean_niter = int(self.ui_clean_niter.get())
        self.myData.var_color_map_name = self.ui_color_map.get()

        # rad plot
        self.myData.var_rad_plot_file = self.ui_rad_plot_file.get()

    def _load_cal_para_to_ui_para(self):
        # time
        self.ui_start_year.set(self.myData.var_start_year)
        self.ui_start_month.set(self.myData.var_start_month)
        self.ui_start_day.set(self.myData.var_start_day)
        self.ui_start_hour.set(self.myData.var_start_hour)
        self.ui_start_minute.set(self.myData.var_start_minute)
        self.ui_start_second.set(self.myData.var_start_second)

        self.ui_stop_year.set(self.myData.var_stop_year)
        self.ui_stop_month.set(self.myData.var_stop_month)
        self.ui_stop_day.set(self.myData.var_stop_day)
        self.ui_stop_hour.set(self.myData.var_stop_hour)
        self.ui_stop_minute.set(self.myData.var_stop_minute)
        self.ui_stop_second.set(self.myData.var_stop_second)

        self.ui_step_hour.set(self.myData.var_step_hour)
        self.ui_step_minute.set(self.myData.var_step_minute)
        self.ui_step_second.set(self.myData.var_step_second)

        # source
        self.ui_main_src.set(self.myData.var_main_src[0])
        src = np.array(self.myData.var_src_lst)
        self.ui_src_lst = list(src[:, 0]) if (len(src) > 0) else []

        # vlbi
        self.ui_vlbi_vlba_lst = []
        self.ui_vlbi_evn_lst = []
        self.ui_vlbi_eavn_lst = []
        self.ui_vlbi_lba_lst = []
        self.ui_vlbi_othter_lst = []
        for each in self.myData.var_vlbi_lst:
            if each[-1] == 0:
                self.ui_vlbi_vlba_lst.append(each[0])
            if each[-1] == 1:
                self.ui_vlbi_evn_lst.append(each[0])
            if each[-1] == 2:
                self.ui_vlbi_eavn_lst.append(each[0])
            if each[-1] == 3:
                self.ui_vlbi_lba_lst.append(each[0])
            if each[-1] == 4:
                self.ui_vlbi_othter_lst.append(each[0])

        # satellite
        sat = np.array(self.myData.var_sat_lst)
        self.ui_sat_lst = list(sat[:, 0]) if (len(sat) > 0) else []
        tele = np.array(self.myData.var_telem_lst)
        self.ui_telem_lst = list(tele[:, 0]) if (len(tele) > 0) else []

        # parameters
        self.ui_freq.set(self.myData.var_freq / 1e9)
        # self.ui_unit.set(self.ui_choice_unit[self.myData.var_unit])
        self.ui_unit.set(self.myData.var_unit)
        self.ui_cutoff_angle.set(self.myData.var_cutoff_mode_dict['CutAngle'])
        self.ui_procession.set(self.myData.var_procession)

        self.ui_baseline_gg.set((self.myData.var_baseline_type & 1) != 0)
        self.ui_baseline_gs.set((self.myData.var_baseline_type & 2) != 0)
        self.ui_baseline_ss.set((self.myData.var_baseline_type & 4) != 0)

        # imaging
        self.ui_model_name.set(self.myData.var_model_name)
        self.ui_pix_size.set(self.myData.var_pix_size)
        self.ui_clean_gain.set(self.myData.var_clean_gain)
        self.ui_clean_thresh.set(self.myData.var_clean_thresh)
        self.ui_clean_niter.set(self.myData.var_clean_niter)
        self.ui_color_map.set(self.myData.var_color_map_name)

        # rad plot
        self.ui_rad_plot_file.set(self.myData.var_rad_plot_file)

    def gui_data_file_chooser(self, type):
        if type == 'radplot':
            file_path = tk.filedialog.askopenfilename(filetypes=[("all files", "*.fits")])
            file_name = os.path.basename(file_path)
            if file_name != '':
                self.logger.info(time.asctime() + ': choose read data ' + file_name)
                self.ui_rad_plot_file.set(file_name)
                # trigger radplot updating
                self.trigger_panel_update(UPDATE_TYPE_RAD)

    def draw_panel_config(self):

        # 1. show/hide config sections
        def config_hide_show_frm(btn_index):  # index start from 1
            def get_string_of_list(tmp_lst):
                result = ''
                for each in tmp_lst:
                    result += str(each) + ' '
                return result.strip()

            def change_to_pos(string):
                return '+ ' + get_string_of_list(string.split()[1:])

            def change_to_neg(string):
                return '- ' + get_string_of_list(string.split()[1:])

            if btn_index == self.config_current_active_index:
                return

            tmp_str = self.config_btn_lst[btn_index - 1]['text']
            # print(tmp_str, change_to_pos(tmp_str))
            self.config_btn_lst[btn_index - 1]['text'] = change_to_neg(tmp_str)

            if btn_index < len(self.config_btn_lst):
                self.config_frm_lst[self.config_current_active_index - 1].pack_forget()
                # update name of above
                for i in range(0, btn_index - 1):
                    tmp_str = self.config_btn_lst[i]['text']
                    self.config_btn_lst[i]['text'] = change_to_pos(tmp_str)
                # forget belowing
                for i in range(btn_index, len(self.config_btn_lst)):
                    self.config_btn_lst[i].pack_forget()
                    self.config_frm_lst[btn_index - 1].pack(side=tk.TOP, fill=tk.X)
                # repacking following
                for i in range(btn_index, len(self.config_btn_lst)):
                    tmp_str = self.config_btn_lst[i]['text']
                    self.config_btn_lst[i]['text'] = change_to_pos(tmp_str)
                    self.config_btn_lst[i].pack(side=tk.TOP, fill=tk.X)
            elif btn_index == len(self.config_btn_lst):
                # update name of above
                for i in range(0, btn_index - 1):
                    tmp_str = self.config_btn_lst[i]['text']
                    self.config_btn_lst[i]['text'] = change_to_pos(tmp_str)
                self.config_frm_lst[self.config_current_active_index - 1].pack_forget()
                self.config_frm_lst[-1].pack(side=tk.TOP, fill=tk.X)

            self.config_current_active_index = btn_index

        # 1. upper frm - configuration
        # 1.1 buttons: save/load/reset
        tmp_frm = tk.Frame(self.config_upper_frm, bg='lightgrey', bd=3)
        tmp_frm.pack(side=tk.TOP, fill=tk.X)
        config_handle_frm = tk.Frame(tmp_frm, bd=3)
        config_handle_frm.pack(side=tk.TOP, fill=tk.X)
        self.ui_btn_load = tk.Button(config_handle_frm, text='Load', bd=10, overrelief=tk.SUNKEN)
        self.ui_btn_save = tk.Button(config_handle_frm, text='Save', bg='grey')
        self.ui_btn_reset = tk.Button(config_handle_frm, text='Reset', bg='grey', command=self.reset_all)
        # normal running with ui reminder
        self.ui_btn_apply = tk.Button(config_handle_frm, text='Apply', bg='grey', command=self.apply_all)
        # accelarate running w/o ui reminder
        # self.ui_btn_apply = tk.Button(config_handle_frm, text='Apply', bg='grey', command=self.apply_all_with_multiprocess)

        self.ui_btn_load.grid(row=0, column=0, sticky=tk.W)  # sticky=tk.NSEW
        self.ui_btn_save.grid(row=0, column=1, sticky=tk.W)
        ttk.Separator(config_handle_frm, orient=tk.VERTICAL).grid(row=0, column=2, sticky=tk.N + tk.S)
        self.ui_btn_reset.grid(row=0, column=3, sticky=tk.E)
        self.ui_btn_apply.grid(row=0, column=4, sticky=tk.E)

        ttk.Separator(self.config_upper_frm).pack(side=tk.TOP, fill=tk.X)
        # make the grid stretchable
        for i in range(5):
            if i != 2:
                config_handle_frm.grid_columnconfigure(i, weight=1)

        # 1.2 show/hide config
        btn1 = tk.Button(self.config_upper_frm, text='- Observe Settings', bg='grey', justify='left',
                         command=lambda: config_hide_show_frm(1))
        btn1.pack(side=tk.TOP, fill=tk.X)
        self.config_btn_lst.append(btn1)

        frm1 = tk.Frame(self.config_upper_frm, height=50, width=100, bg='yellow', bd=2)
        frm1.pack(side=tk.TOP, fill=tk.X)
        self.config_frm_lst.append(frm1)

        btn2 = tk.Button(self.config_upper_frm, text='+ Station Settings', bg='grey', justify='left',
                         command=lambda: config_hide_show_frm(2))
        btn2.pack(side=tk.TOP, anchor='nw', fill=tk.X)
        self.config_btn_lst.append(btn2)

        frm2 = tk.Frame(self.config_upper_frm, height=50, width=100, bg='blue', bd=2)
        self.config_frm_lst.append(frm2)

        btn3 = tk.Button(self.config_upper_frm, text='+ Para Settings', bg='grey', justify='left',
                         command=lambda: config_hide_show_frm(3))
        btn3.pack(side=tk.TOP, anchor='nw', fill=tk.X)
        self.config_btn_lst.append(btn3)

        frm3 = tk.Frame(self.config_upper_frm, height=50, width=100, bg='green', bd=2)
        self.config_frm_lst.append(frm3)

        btn4 = tk.Button(self.config_upper_frm, text='+ Imaging Settings', bg='grey', justify='left',
                         command=lambda: config_hide_show_frm(4))
        btn4.pack(side=tk.TOP, fill=tk.X)
        self.config_btn_lst.append(btn4)

        frm4 = tk.Frame(self.config_upper_frm, height=50, width=100, bg='MediumOrchid', bd=2)
        self.config_frm_lst.append(frm4)

        btn5 = tk.Button(self.config_upper_frm, text='+ File Settings', bg='grey', justify='left',
                         command=lambda: config_hide_show_frm(5))
        btn5.pack(side=tk.TOP, fill=tk.X)
        self.config_btn_lst.append(btn5)

        frm5 = tk.Frame(self.config_upper_frm, height=50, width=100, bg='#00FFFF', bd=2)
        self.config_frm_lst.append(frm5)

        # 1.3 Observe setting / frm1
        group_time = tk.LabelFrame(frm1, text="Time:", fg='red', bd=5, relief=tk.RIDGE)
        group_time.pack(padx=2, pady=2, fill=tk.X, side=tk.TOP)

        tk.Label(group_time, text='Start:').grid(row=0, column=0, sticky=tk.E)
        ttk.Combobox(group_time, textvariable=self.ui_start_year, values=self.ui_choice_year,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=0, column=1)
        ttk.Combobox(group_time, textvariable=self.ui_start_month, values=self.ui_choice_month,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=0, column=2)
        ttk.Combobox(group_time, textvariable=self.ui_start_day, values=self.ui_choice_day,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=0, column=3)
        ttk.Combobox(group_time, textvariable=self.ui_start_hour, values=self.ui_choice_hour,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=1, column=1)
        ttk.Combobox(group_time, textvariable=self.ui_start_minute, values=self.ui_choice_minute,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=1, column=2)
        ttk.Combobox(group_time, textvariable=self.ui_start_second, values=self.ui_choice_second,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=1, column=3)

        tk.Label(group_time, text='Stop:').grid(row=2, column=0, sticky=tk.E)
        ttk.Combobox(group_time, textvariable=self.ui_stop_year, values=self.ui_choice_year,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=2, column=1)
        ttk.Combobox(group_time, textvariable=self.ui_stop_month, values=self.ui_choice_month,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=2, column=2)
        ttk.Combobox(group_time, textvariable=self.ui_stop_day, values=self.ui_choice_day,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=2, column=3)
        ttk.Combobox(group_time, textvariable=self.ui_stop_hour, values=self.ui_choice_hour,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=3, column=1)
        ttk.Combobox(group_time, textvariable=self.ui_stop_minute, values=self.ui_choice_minute,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=3, column=2)
        ttk.Combobox(group_time, textvariable=self.ui_stop_second, values=self.ui_choice_second,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=3, column=3)

        tk.Label(group_time, text='Step:').grid(row=4, column=0, sticky=tk.E)
        ttk.Combobox(group_time, textvariable=self.ui_step_hour, values=self.ui_choice_hour,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=4, column=1)
        ttk.Combobox(group_time, textvariable=self.ui_step_minute, values=self.ui_choice_minute,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=4, column=2)
        ttk.Combobox(group_time, textvariable=self.ui_step_second, values=self.ui_choice_second,
                     width='5', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=4, column=3)
        for i in range(4):
            group_time.grid_columnconfigure(i, weight=1)

        group_src = tk.LabelFrame(frm1, text="Source:", fg='red', bd=5, relief=tk.RIDGE)
        group_src.pack(padx=2, pady=2, fill=tk.X, side=tk.TOP)

        tk.Label(group_src, text='Main Src:').grid(row=0, column=0, sticky=tk.E)
        self.selec_single_src = ttk.Combobox(group_src, textvariable=self.ui_main_src,
                                             values=self.ui_choice_src, width='10',
                                             postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI))
        self.selec_single_src.grid(row=0, column=1)

        tk.Label(group_src, text='MultiSrc:').grid(row=0, column=2, sticky=tk.E)
        tk.Button(group_src, text="Choose",
                  command=lambda: self.multi_choice_call_back(type=MULTI_CHOICE_TYPE_SRC)).grid(row=0,
                                                                                                column=3)
        for i in range(4):
            group_src.grid_columnconfigure(i, weight=1)

        # 1.4 Station setting / frm2
        group_vlbi = tk.LabelFrame(frm2, text="VLBI Stat:", fg='red', bd=5, relief=tk.RIDGE)  # bg='grey'
        group_vlbi.pack(padx=2, pady=2, fill=tk.X, side=tk.TOP)
        tk.Button(group_vlbi, text="VLBA",
                  command=lambda: self.multi_choice_call_back(type=MULTI_CHOICE_TYPE_VLBA)).grid(row=0, column=0,
                                                                                                 padx=5, sticky=tk.NSEW)
        tk.Button(group_vlbi, text="EVN",
                  command=lambda: self.multi_choice_call_back(type=MULTI_CHOICE_TYPE_EVN)).grid(row=0, column=1,
                                                                                                padx=5, sticky=tk.NSEW)
        tk.Button(group_vlbi, text="EAVN",
                  command=lambda: self.multi_choice_call_back(type=MULTI_CHOICE_TYPE_EAVN)).grid(row=0, column=2,
                                                                                                 padx=5, sticky=tk.NSEW)
        tk.Button(group_vlbi, text="LBA",
                  command=lambda: self.multi_choice_call_back(type=MULTI_CHOICE_TYPE_LBA)).grid(row=1, column=0,
                                                                                                padx=5, sticky=tk.NSEW)
        tk.Button(group_vlbi, text="OTHER",
                  command=lambda: self.multi_choice_call_back(type=MULTI_CHOICE_TYPE_OTHER)).grid(row=1, column=1,
                                                                                                  padx=5,
                                                                                                  sticky=tk.NSEW)
        for i in range(3):
            group_vlbi.grid_columnconfigure(i, weight=1)

        group_sat = tk.LabelFrame(frm2, text="Satellite:", fg='red', bd=5, relief=tk.RIDGE)  # bg='grey'
        group_sat.pack(padx=2, pady=2, fill=tk.X, side=tk.TOP)
        tk.Button(group_sat, text="Satellite",
                  command=lambda: self.multi_choice_call_back(type=MULTI_CHOICE_TYPE_SAT)).grid(row=0, column=0,
                                                                                                padx=5,
                                                                                                sticky=tk.NSEW)

        tk.Button(group_sat, text="Telemetry",
                  command=lambda: self.multi_choice_call_back(type=MULTI_CHOICE_TYPE_TELE)).grid(row=0, column=2,
                                                                                                 padx=5,
                                                                                                 sticky=tk.NSEW)
        for i in range(3):
            group_sat.grid_columnconfigure(i, weight=1)

        # 1.5 Para setting / frm3
        group_para = tk.LabelFrame(frm3, text="Parameter:", fg='red', bd=5, relief=tk.RIDGE)  # bg='grey'
        group_para.pack(padx=2, pady=2, fill=tk.X, side=tk.TOP)

        tk.Label(group_para, text='Freq/Ghz:').grid(row=0, column=0, sticky=tk.E)
        freq_input = tk.Entry(group_para, textvariable=self.ui_freq, bg="#282B2B", fg="white",
                              width='10', validate="key", validatecommand=(self.test_float, '%P'))
        freq_input.grid(row=0, column=1, sticky=tk.EW)

        # ttk.Combobox(group_para, textvariable=self.ui_freq, values=self.ui_choice_freq,
        #              width='10').grid(row=0, column=1, sticky=tk.EW)

        tk.Label(group_para, text='Unit:').grid(row=0, column=2, sticky=tk.E)
        ttk.Combobox(group_para, textvariable=self.ui_unit, values=self.ui_choice_unit,
                     width='10', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UV)).grid(row=0, column=3,
                                                                                                     sticky=tk.EW)

        tk.Label(group_para, text='CutAngle:').grid(row=1, column=0, sticky=tk.E)
        tk.Entry(group_para, textvariable=self.ui_cutoff_angle, bg="#282B2B", fg="white",
                 width='10', validate="key", validatecommand=(self.test_float, '%P')).grid(row=1, column=1,
                                                                                           sticky=tk.EW)

        tk.Label(group_para, text='P-model:').grid(row=1, column=2, sticky=tk.E)
        ttk.Combobox(group_para, textvariable=self.ui_procession, values=self.ui_choice_precession,
                     width='10', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(row=1, column=3,
                                                                                                      sticky=tk.EW)

        tk.Label(group_para, text='Baseline:').grid(row=2, column=0, sticky=tk.E)
        tk.Checkbutton(group_para, text='gg', onvalue=1, offvalue=0,
                       variable=self.ui_baseline_gg, command=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(
            row=2, column=1)
        tk.Checkbutton(group_para, text='gs', onvalue=1, offvalue=0,
                       variable=self.ui_baseline_gs, command=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(
            row=2, column=2)
        tk.Checkbutton(group_para, text='ss', onvalue=1, offvalue=0,
                       variable=self.ui_baseline_ss, command=lambda: self.trigger_panel_update(UPDATE_TYPE_UOI)).grid(
            row=2, column=3)

        for i in range(4):
            group_para.grid_columnconfigure(i, weight=1)

        # 1.6 Imaging setting / frm4
        group_img = tk.LabelFrame(frm4, text="Imaging:", fg='red', bd=5, relief=tk.RIDGE)  # bg='grey'
        group_img.pack(padx=2, pady=2, fill=tk.X, side=tk.TOP)

        tk.Label(group_img, text='Model:').grid(row=0, column=0, sticky=tk.E)
        ttk.Combobox(group_img, textvariable=self.ui_model_name, values=self.ui_choice_model_name,
                     width='10', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_IMG)).grid(row=0, column=1,
                                                                                                      sticky=tk.EW)

        tk.Label(group_img, text='Pixel:').grid(row=0, column=2, sticky=tk.E)
        ttk.Combobox(group_img, textvariable=self.ui_pix_size, values=self.ui_choice_pix,
                     width='10', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_IMG)).grid(row=0, column=3,
                                                                                                      sticky=tk.EW)

        tk.Label(group_img, text='Colormap:').grid(row=1, column=0, sticky=tk.E)
        ttk.Combobox(group_img, textvariable=self.ui_color_map, values=self.ui_choice_color_map,
                     width='10', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_IMG)).grid(row=1, column=1,
                                                                                                      sticky=tk.EW)

        tk.Label(group_img, text='C-niter:').grid(row=1, column=2, sticky=tk.E)
        ttk.Combobox(group_img, textvariable=self.ui_clean_niter, values=self.ui_choice_clean_niter,
                     width='10', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_IMG)).grid(row=1, column=3,
                                                                                                      sticky=tk.EW)

        tk.Label(group_img, text='C-gain:').grid(row=2, column=0, sticky=tk.E)
        ttk.Combobox(group_img, textvariable=self.ui_clean_gain, values=self.ui_choice_clean_gain,
                     width='10', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_IMG)).grid(row=2, column=1,
                                                                                                      sticky=tk.EW)

        tk.Label(group_img, text='C-thred:').grid(row=2, column=2, sticky=tk.E)
        ttk.Combobox(group_img, textvariable=self.ui_clean_thresh, values=self.ui_choice_clean_thred,
                     width='10', postcommand=lambda: self.trigger_panel_update(UPDATE_TYPE_IMG)).grid(row=2, column=3,
                                                                                                      sticky=tk.EW)

        for i in range(4):
            group_img.grid_columnconfigure(i, weight=1)

        # 1.7 File settings / frm5
        group_file = tk.LabelFrame(frm5, text="Select Data:", fg='red', bd=5, relief=tk.RIDGE)
        group_file.pack(padx=2, pady=2, fill=tk.X, side=tk.TOP)
        tk.Label(group_file, text="Data: ", ).grid(row=0, column=0, padx=5, sticky=tk.E)
        entry_rad = tk.Entry(group_file, textvariable=self.ui_rad_plot_file, bg="#282B2B", fg="white",
                             width='15', state=tk.DISABLED)
        entry_rad.grid(row=0, column=1, padx=5, sticky=tk.NSEW)
        tk.Button(group_file, text="Choose",
                  command=lambda: self.gui_data_file_chooser('radplot')).grid(row=0, column=2, padx=5, sticky=tk.NSEW)

        for i in range(3):
            group_file.grid_columnconfigure(i, weight=1)

        # 2. lower frm - configuration
        # self.ui_logs = tk.Text(self.config_lower_frm, width=20, wrap=tk.WORD,
        #                        bg='black', fg='white', state=tk.DISABLED)
        # self.ui_logs.pack(fill=tk.X, side=tk.LEFT, expand=1, anchor='nw')
        # sb = tk.Scrollbar(self.config_lower_frm,
        #                   orient=tk.VERTICAL, )
        # sb.pack(fill=tk.Y, side=tk.LEFT)
        # self.ui_logs.config(yscrollcommand=sb.set)
        # sb['command'] = self.ui_logs.yview

        st = scrolledtext.ScrolledText(self.config_lower_frm, bg='black', fg='white', width=0,
                                       height=0)  # , state='disabled'
        st.configure(font='TkFixedFont')
        st.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # Create textLogger
        text_handler = TextHandler(st)
        # Logging configuration
        logging.basicConfig(filename='soft_running.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        # Add the handler to logger
        self.logger = logging.getLogger()
        self.logger.addHandler(text_handler)

        # when using
        # logging.info(time.asctime() + 'what the fuck')
        # msg = time.asctime() + "some message"
        # logging.info(msg)

    def quit(self):
        self.window.destroy()
        sys.exit()

    def show_dialog_info(self, _str):
        if _str.lower() == 'about':
            print("info", tk.messagebox.showinfo("About", __about_text__))
        if _str.lower() == 'support':
            res = tk.messagebox.askyesno("Feedback", __feedback_text__)
            if res is True:
                webbrowser.open('http://202.127.29.4/CRATIV/en/home.html')

        if _str.lower() == 'help':
            # print("info", tk.messagebox.showinfo("Help", __help_text__))
            win = tk.Toplevel(self.window)
            win.title("Help")
            win.resizable(False, False)
            center_window(win, 400, 150)
            frm1 = tk.Frame(win)
            frm1.pack(side=tk.TOP, fill=tk.X, expand=1)
            helptext = scrolledtext.ScrolledText(frm1, height=8)
            helptext.insert('1.0', __help_text__)
            helptext.config(state=tk.DISABLED)
            helptext.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            tk.Button(win, text='OK', bg='grey', command=win.destroy).pack()

    def _gui_int(self):
        # # # # # # # # # # # # # # # # # #
        # GUI AREA 1: set up the Menu bar #
        # # # # # # # # # # # # # # # # # #
        menu = tk.Menu(self.window)
        self.window.config(menu=menu)

        # 1.1. VNSIM (2 dialog)
        menu_sim = tk.Menu(menu, tearoff=True)
        menu.add_cascade(label="VNSIM", menu=menu_sim)
        menu_sim.add_command(label="About", command=lambda: self.show_dialog_info('about'), accelerator="Cmd-i")
        menu_sim.add_command(label="Feedback", command=lambda: self.show_dialog_info('support'))
        menu_sim.add_separator()
        menu_sim.add_command(label="Quit", command=self.window.quit)

        # 1.2. Tool (3 radio and 2 check btn))
        self.menu_tool = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Tool", menu=self.menu_tool)
        self.gui_var_func = tk.IntVar()
        self.gui_var_func.set(0)
        self.menu_tool.add_radiobutton(label="UV Coverage", value=0, variable=self.gui_var_func,
                                       command=self._gui_switch_func)
        self.menu_tool.add_radiobutton(label="Obs Survey", value=1, variable=self.gui_var_func,
                                       command=self._gui_switch_func)
        self.menu_tool.add_radiobutton(label="Imaging", value=2, variable=self.gui_var_func,
                                       command=self._gui_switch_func)
        self.menu_tool.add_radiobutton(label="RadPlot", value=3, variable=self.gui_var_func,
                                       command=self._gui_switch_func)
        self.menu_tool.add_separator()
        self.gui_var_tool_para = tk.BooleanVar()
        self.gui_var_tool_db = tk.BooleanVar()
        self.menu_tool.add_checkbutton(label="Para Evaluate", onvalue='True', offvalue='False',
                                       variable=self.gui_var_tool_para,
                                       command=lambda: self._gui_show_add_func(POP_OUT_PARA_CAL))
        self.menu_tool.add_checkbutton(label="Edit Database", onvalue='True', offvalue='False',
                                       variable=self.gui_var_tool_db,
                                       command=lambda: self._gui_show_add_func(POP_OUY_DB_EDIT))

        # 1.3. View (radio btn)
        self.menu_view = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="View", menu=self.menu_view)
        self.gui_var_view = tk.IntVar()
        self.gui_var_view.set(0)
        self.menu_view.add_radiobutton(label='Image+Config', value=0, variable=self.gui_var_view,
                                       command=self._gui_switch_view)
        self.menu_view.add_radiobutton(label='Image Only', value=1, variable=self.gui_var_view,
                                       command=self._gui_switch_view)
        self.menu_view.entryconfig(0, state=tk.DISABLED)

        # 1.4. Help (link)
        menu_help = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Help", menu=menu_help)
        menu_help.add_command(label="User Tips", command=lambda: self.show_dialog_info('Help'))
        menu_help.add_command(label="Project Github", command=lambda: webbrowser.open(_github_page_))

        # # # # # # # # # # # # # # # # # # #
        # GUI AREA 2: set up the Main Area  #
        # # # # # # # # # # # # # # # # # # #
        main_frm = tk.Frame(self.window, bg='red')
        main_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.paned_w = tk.PanedWindow(main_frm, showhandle=True, sashrelief=tk.SUNKEN, handlepad=0)
        self.paned_w.pack(fill=tk.BOTH, expand=1)

        # # # # # # # # # # # # # # # # # # #
        # 2.1 Configure panel
        # # # # # # # # # # # # # # # # # # #
        self.config_frm = tk.Frame(self.paned_w, bg='lightgrey')
        self.paned_w.add(self.config_frm)

        config_paned_w = tk.PanedWindow(self.config_frm, orient=tk.VERTICAL, showhandle=True, sashrelief=tk.SUNKEN,
                                        handlepad=0)
        config_paned_w.pack(fill=tk.BOTH, expand=1)

        # show config
        self.config_upper_frm = tk.LabelFrame(self.paned_w, text="Configure",
                                              font=('Arial ', 15, "bold"))
        config_paned_w.add(self.config_upper_frm)

        # show logs
        self.config_lower_frm = tk.LabelFrame(self.config_frm, text="Logs",
                                              font=('Arial ', 15, "bold"))
        config_paned_w.add(self.config_lower_frm)

        # for show/hide config
        self.config_current_active_index = 1
        self.config_btn_lst = []
        self.config_frm_lst = []
        self.draw_panel_config()

        # # # # # # # # # # # # # # # # # # #
        # 2.2 Image panel
        # # # # # # # # # # # # # # # # # # #
        image_frm = tk.Frame(self.paned_w, bg='green')
        self.paned_w.add(image_frm)

        self.tab_control = ttk.Notebook(image_frm)
        self.tab_control.bind('<ButtonRelease-1>', self._gui_switch_func_tab_click)
        self.tab_control.pack(expand=1, fill="both")

        # 1st tab: uv coverage
        self.tab_uv = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_uv, text='UV Funcs')
        self.figs_uv = pl.Figure(figsize=(3, 2))
        self.figs_uv = Figure(figsize=(3, 2))
        # pl.xticks(fontsize=3)
        self.canvas_uv = FigureCanvasTkAgg(self.figs_uv, master=self.tab_uv)
        self.canvas_uv.show()
        self.canvas_uv.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.draw_panel_uv()

        # 2nd tab: observation
        self.tab_obs = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_obs, text='OBS Funcs')
        self.figs_obs = Figure(figsize=(3, 2))
        self.canvas_obs = FigureCanvasTkAgg(self.figs_obs, master=self.tab_obs)
        self.canvas_obs.show()
        self.canvas_obs.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.draw_panel_obs()

        # 3rd tab: imaging
        self.tab_img = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_img, text='Imaging')
        self.figs_img = Figure(figsize=(3, 2))
        self.canvas_img = FigureCanvasTkAgg(self.figs_img, master=self.tab_img)
        self.canvas_img.show()
        self.canvas_img.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.draw_panel_img()

        # 4th tab: radplot
        self.tab_radplot = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_radplot, text='Rad Plot')
        self.figs_radplot = Figure(figsize=(3, 2))
        self.canvas_radplot = FigureCanvasTkAgg(self.figs_radplot, master=self.tab_radplot)
        self.canvas_radplot.show()
        self.canvas_radplot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.draw_panel_radplot()

        # initial setting
        self.tab_control.select(0)
        # print(self.tab_control.index("current"))

        # # # # # # # # # # # # # # # # # # #
        # GUI AREA 3: set up the status bar #
        # # # # # # # # # # # # # # # # # # #
        # status : Ready / running / done / input error
        status_frm1 = tk.Frame(self.window, bg='grey')
        status_frm1.pack(side=tk.TOP, fill=tk.X)
        self.gui_var_status.set('Ready')
        tk.Label(status_frm1, text='Status:',
                 relief='flat', anchor=tk.E, bg='grey',
                 font=('Arial ', 11, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Label(status_frm1, textvariable=self.gui_var_status,
                 relief='groove', anchor=tk.E, bg='grey', fg='blue',
                 font=('Arial ', 10, "bold italic")).pack(side=tk.LEFT)
        tk.Label(status_frm1, text=COPY_RIGHT_INFO,
                 relief='flat', anchor=tk.E, bg='grey',
                 font=('Arial', 10, "bold italic")).pack(side=tk.RIGHT)

    def _gui_switch_func_tab_click(self, event):
        # tab change through tab clicking
        id_click = self.tab_control.index('current')
        # tab change through menu
        id_menu = self.gui_var_func.get()
        if id_click != id_menu:
            self.gui_var_func.set(id_click)
            self._gui_switch_func()

    def _gui_switch_func(self):

        if self.gui_var_func.get() == 0:
            # print("select UV")
            # logging.info(time.asctime() + ': switch to UV function Tab')
            self.tab_control.select(0)

        elif self.gui_var_func.get() == 1:
            # print("select Obs")
            # logging.info(time.asctime() + ': switch to Obs function Tab')
            self.tab_control.select(1)

        elif self.gui_var_func.get() == 2:
            # print("select Imaging")
            # logging.info(time.asctime() + ': switch to Imaging Tab')
            self.tab_control.select(2)

        elif self.gui_var_func.get() == 3:
            # print("select Radplot")
            # logging.info(time.asctime() + ': switch to Rad Plot Tab')
            self.tab_control.select(3)

        else:
            print("error happen")

    def _gui_switch_view(self):
        if self.gui_var_view.get() == 0:
            # print("image with configuration")
            logging.info(time.asctime() + ': set to show both image and config')
            self.paned_w.sash_place(0, 340, 0)
            self.menu_view.entryconfig(0, state=tk.DISABLED)
            self.menu_view.entryconfig(1, state=tk.ACTIVE)

        if self.gui_var_view.get() == 1:
            # print("image only")
            logging.info(time.asctime() + ': set to show image only')
            self.paned_w.sash_place(0, 10, 0)
            self.menu_view.entryconfig(0, state=tk.ACTIVE)
            self.menu_view.entryconfig(1, state=tk.DISABLED)
            # self.paned_w.sash_place(1, 100, 0)

    def _gui_show_add_func(self, show_type):
        if show_type == POP_OUT_PARA_CAL and self.window is not None:
            # self.gui_var_tool_para.set(True)
            self.menu_tool.entryconfig(5, state=tk.DISABLED)
            self.PopOutParaCal = TopLevelParaCal(self)
        if show_type == POP_OUY_DB_EDIT and self.window is not None:
            # self.gui_var_tool_db.set(True)
            self.menu_tool.entryconfig(6, state=tk.DISABLED)
            self.PopOutDbEditor = TopLevelDbEditor(self, self.db_name, self.pkl_name)

    def _clear_pop_out_win_state(self, win_name):
        if win_name == POP_OUT_PARA_CAL:
            self.gui_var_tool_para.set(False)
            self.menu_tool.entryconfig(5, state=tk.ACTIVE)

        if win_name == POP_OUY_DB_EDIT:
            self.gui_var_tool_db.set(False)
            self.menu_tool.entryconfig(6, state=tk.ACTIVE)

    def multi_choice_call_back(self, type):
        # enable to trigger updating
        if type != MULTI_CHOICE_TYPE_SRC:
            self.trigger_panel_update(UPDATE_TYPE_UOI)

        # judge the choice type
        if type == MULTI_CHOICE_TYPE_SRC:
            result = self.ask_choice("Source", self.ui_choice_src)
            if result is not None:
                self.ui_src_lst = result

        elif type == MULTI_CHOICE_TYPE_VLBA:
            result = self.ask_choice("VLBI-vlba", self.ui_choice_vlbi_vlba)
            if result is not None:
                self.ui_vlbi_vlba_lst = result

        elif type == MULTI_CHOICE_TYPE_EVN:
            result = self.ask_choice("VLBI-evn", self.ui_choice_vlbi_evn)
            if result is not None:
                self.ui_vlbi_evn_lst = result

        elif type == MULTI_CHOICE_TYPE_EAVN:
            result = self.ask_choice("VLBI-eavn", self.ui_choice_vlbi_eavn)
            if result is not None:
                self.ui_vlbi_eavn_lst = result

        elif type == MULTI_CHOICE_TYPE_LBA:
            result = self.ask_choice("VLBI-lba", self.ui_choice_vlbi_lba)
            if result is not None:
                self.ui_vlbi_lba_lst = result

        elif type == MULTI_CHOICE_TYPE_OTHER:
            result = self.ask_choice("VLBI-other", self.ui_choice_vlbi_other)
            if result is not None:
                self.ui_vlbi_othter_lst = result

        elif type == MULTI_CHOICE_TYPE_SAT:
            result = self.ask_choice("Satellite", self.ui_choice_sat)
            if result is not None:
                self.ui_sat_lst = result

        elif type == MULTI_CHOICE_TYPE_TELE:
            result = self.ask_choice("Telemetry", self.ui_choice_telem)
            if result is not None:
                self.ui_telem_lst = result

        else:
            return

    def ask_choice(self, op_name, op_list):
        myDialog = MultiChoiceDialog(op_name, op_list)
        self.window.wait_window(myDialog)
        return myDialog.get_choice_result()

    # uv panel text
    def _panel_uv_show_info(self):
        src = np.array(self.myData.var_src_lst)
        src_lst = list(src[:, 0]) if (len(src) > 0) else []
        main_src = str(self.myData.var_main_src[0])
        vlbi = np.array(self.myData.var_vlbi_lst)
        vlbi_lst = list(vlbi[:, 0]) if (len(vlbi) > 0) else []
        # vlbi_str = ' '.join(vlbi_lst)
        sat = np.array(self.myData.var_sat_lst)
        sat_lst = list(sat[:, 0]) if (len(sat) > 0) else []
        tele = np.array(self.myData.var_telem_lst)
        tele_lst = list(tele[:, 0]) if (len(tele) > 0) else []

        my_str = "Main Src:\n{4}\nMore Src:\n{0}\nVLBI:\n{1}\nSat:\n{2}\nTele:\n{3}".format(','.join(src_lst),
                                                                                            ','.join(vlbi_lst),
                                                                                            ','.join(sat_lst),
                                                                                            ','.join(tele_lst),
                                                                                            main_src)
        return my_str

    def _panel_uv_info_set_to_scroll_text(self):
        self.panel_uv_text_info.delete(0.0, tk.END)
        self.panel_uv_text_info.insert(0.0, self._panel_uv_show_info())

    # uv
    def draw_panel_uv(self):
        new_frm = tk.LabelFrame(self.tab_uv, text="More Funcs", fg='darkred', width=100, height=100)
        new_frm.place(relx=0.8, rely=0.26, anchor=tk.CENTER, )

        tk.Label(new_frm, text='Information:', bg='lightgrey', fg='black',
                 font={'Arial', 12, 'bold'}).grid(row=0, column=0, columnspan=2)  # row=0, column=0, columnspan=2
        self.panel_uv_text_info = scrolledtext.ScrolledText(new_frm, bg='black', fg='white',
                                                            width=0, height=8)
        self.panel_uv_text_info.grid(row=1, column=0, columnspan=4, rowspan=3, sticky=tk.NSEW)
        self._panel_uv_info_set_to_scroll_text()

        tk.Label(new_frm, text='More Funcs:', bg='lightgrey', fg='black',
                 font={'Arial', 12}).grid(row=4, column=0, columnspan=2)
        tk.Button(new_frm, text="All Year UV", bg='yellow', command=self._panel_uv_show_time_uv).grid(row=5, column=1)
        tk.Button(new_frm, text="MultiSource UV", bg='yellow', command=self._panel_uv_show_multi_src).grid(row=5,
                                                                                                           column=2)
        for i in range(6):
            if i < 4:
                new_frm.grid_columnconfigure(i, weight=1)
            new_frm.grid_rowconfigure(i, weight=1)

        gs = gridspec.GridSpec(2, 2)
        self.figs_uv_fig_uv_single = self.figs_uv.add_subplot(gs[0, 0], aspect='equal', facecolor=(0.4, 0.4, 0.4))
        self.figs_uv_fig_all_sky = self.figs_uv.add_subplot(gs[1, :])
        self.figs_uv.tight_layout(pad=0.15)

        # data settings - single uv
        self.figs_uv_fig_uv_single.scatter(lc.just4fun_u, lc.just4fun_v, s=5, marker='.')
        self.figs_uv_fig_uv_single.set_xlim(-lc.just4fun_max, +lc.just4fun_max)
        self.figs_uv_fig_uv_single.set_ylim(-lc.just4fun_max, +lc.just4fun_max)

        # data settings -  multiSrc uv
        self.figs_uv_fig_all_sky.text(6, 0, r"$Please\ apply\ your\ config.$", fontdict={"size": 12, "color": 'r'})

        # label settings
        self.figs_uv_fig_uv_single.set_title("UV Plot", fontdict=MY_IMG_FONT)
        if self.myData.var_unit == 0:
            self.figs_uv_fig_uv_single.set_xlabel("u$(\lambda)$")
            self.figs_uv_fig_uv_single.set_ylabel("v$(\lambda)$")
        else:
            self.figs_uv_fig_uv_single.set_xlabel("u$(km)$")
            self.figs_uv_fig_uv_single.set_ylabel("v$(km)$")
        self.figs_uv_fig_uv_single.grid()

        self.figs_uv_fig_all_sky.set_title("ALL SKY UV Plots", fontdict=MY_IMG_FONT)
        self.figs_uv_fig_all_sky.set_xlabel(r"Ra($H$)")
        self.figs_uv_fig_all_sky.set_ylabel(r'Dec ($^\circ$)')
        self.figs_uv_fig_all_sky.set_xticks([0, 2, 6, 10, 14, 18, 22, 24])
        self.figs_uv_fig_all_sky.set_yticks([-90, -60, -30, 0, 30, 60, 90], )
        self.figs_uv_fig_all_sky.set_xlim(0, 24)
        self.figs_uv_fig_all_sky.set_ylim(-90, +90)

        self.canvas_uv.show()

        # image save tool
        toolbar = NavigationToolbar2TkAgg(self.canvas_uv, self.tab_uv)
        toolbar.update()
        self.canvas_uv._tkcanvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

    def reset_panel_uv(self):
        # clear
        self.figs_uv_fig_uv_single.cla()
        self.figs_uv_fig_all_sky.cla()

        # initial label settings
        self.figs_uv_fig_uv_single.set_title("UV Plot", fontdict=MY_IMG_FONT)
        if self.myData.var_unit == 0:
            self.figs_uv_fig_uv_single.set_xlabel("u$(\lambda)$")
            self.figs_uv_fig_uv_single.set_ylabel("v$(\lambda)$")
        else:
            self.figs_uv_fig_uv_single.set_xlabel("u$(m)$")
            self.figs_uv_fig_uv_single.set_ylabel("v$(m)$")
        self.figs_uv_fig_uv_single.grid()

        self.figs_uv_fig_all_sky.set_title("ALL Sky UV Plots", fontdict=MY_IMG_FONT)
        self.figs_uv_fig_all_sky.set_xlabel(r"Ra($H$)")
        self.figs_uv_fig_all_sky.set_ylabel(r'Dec ($^\circ$)')
        self.figs_uv_fig_all_sky.set_xticks([0, 2, 6, 10, 14, 18, 22, 24], )
        self.figs_uv_fig_all_sky.set_yticks([-90, -60, -30, 0, 30, 60, 90], )
        self.figs_uv_fig_all_sky.set_xlim(0, 24)
        self.figs_uv_fig_all_sky.set_ylim(-90, +90)
        self.figs_uv_fig_all_sky.grid()
        # self.figs_uv_fig_all_sky.text(6, 0, r"$Please\ apply\ your\ config.$", fontdict={"size": 12, "color": 'r'})

        self.canvas_uv.draw()
        # reset show info
        self._panel_uv_info_set_to_scroll_text()

    def update_panel_uv(self):
        # update the show info
        self.reset_panel_uv()

        # draw single uv
        single_u, single_v, single_max = self.myData.get_data_uv_single()

        self.figs_uv_fig_uv_single.scatter(single_u, single_v, s=3, marker='.', color='brown')
        self.figs_uv_fig_uv_single.set_xlim(-single_max, +single_max)
        self.figs_uv_fig_uv_single.set_ylim(-single_max, +single_max)
        self.figs_uv_fig_uv_single.grid()

        # draw all sky uv
        result_mat_u, result_mat_v, max_range = self.myData.get_data_uv_sky()
        if len(result_mat_u) != 0 and len(result_mat_v) != 0:
            k = 0
            for i in (2, 6, 10, 14, 18, 22):
                for j in (-60, -30, 0, 30, 60):
                    if len(result_mat_u[k]) > 0 and len(result_mat_v[k]) > 0:
                        temp_u = np.array(result_mat_u[k]) / max_range
                        temp_v = np.array(result_mat_v[k]) / max_range * 10
                        temp_u += i
                        temp_v += j
                        self.figs_uv_fig_all_sky.scatter(temp_u, temp_v, s=1, marker='.', color='b')
                    k += 1

        print()
        # plot sun position
        sun_ra, sun_dec = me.sun_ra_dec_cal(self.myData.var_start_time, self.myData.var_stop_time,
                                            self.myData.var_time_step)
        self.figs_uv_fig_all_sky.plot(np.array(sun_ra), np.array(sun_dec), '.k', linewidth=2)
        self.figs_uv_fig_all_sky.plot(sun_ra[0], sun_dec[0], 'or', alpha=0.5, markersize=20)

        # update label of single uv
        if self.myData.var_unit == 0:
            self.figs_uv_fig_uv_single.set_xlabel("u$(\lambda)$")
            self.figs_uv_fig_uv_single.set_ylabel("v$(\lambda)$")
        else:
            self.figs_uv_fig_uv_single.set_xlabel("u$(m)$")
            self.figs_uv_fig_uv_single.set_ylabel("v$(m)$")
        self.figs_uv_fig_uv_single.grid()

        # show update on GUI
        self.canvas_uv.draw()

    # obs
    def draw_panel_obs(self):  # self.figs_obs
        gs = gridspec.GridSpec(4, 1)

        self.figs_obs_fig_az = self.figs_obs.add_subplot(gs[0, :])  # , facecolor=(0.4, 0.4, 0.4)
        self.figs_obs_fig_el = self.figs_obs.add_subplot(gs[1, :], sharex=self.figs_obs_fig_az)
        self.figs_obs_fig_survey = self.figs_obs.add_subplot(gs[2:, :])
        self.figs_obs.subplots_adjust(wspace=0, hspace=0.5)

        # initial label and data settings
        self.reset_panel_obs()

        # image save tool
        toolbar = NavigationToolbar2TkAgg(self.canvas_obs, self.tab_obs)
        toolbar.update()
        self.canvas_obs._tkcanvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

    def reset_panel_obs(self):
        self.figs_obs_fig_az.cla()
        self.figs_obs_fig_el.cla()
        self.figs_obs_fig_survey.cla()

        self.figs_obs_fig_az.set_title('AZ-EL of VLBI Stations', fontdict=MY_IMG_FONT)
        self.figs_obs_fig_az.set_ylabel("Azimuth($^\circ$)")
        self.figs_obs_fig_el.set_xlabel("Time(h)")
        self.figs_obs_fig_el.set_ylabel("Elevation($^\circ$)")

        self.figs_obs_fig_survey.set_title('SKY SURVEY', fontdict=MY_IMG_FONT)
        self.figs_obs_fig_survey.set_xlabel("RA(H)")
        self.figs_obs_fig_survey.set_ylabel(r'Dec ($^\circ$)')

        pl.setp(self.figs_obs_fig_el.get_xticklabels(), visible=False)

        if self.obs_panel_color_bar is not None:
            self.obs_panel_color_bar.remove()
            self.obs_panel_color_bar = None
        self.canvas_obs.show()

    def update_panel_obs(self):
        # reset ui
        self.reset_panel_obs()

        # draw new - az - el
        azimuth, elevation, hour_lst = self.myData.get_data_obs_az_el()
        for i in range(0, len(elevation)):
            az1 = azimuth[i]
            el1 = elevation[i]
            h1 = hour_lst[i]
            self.figs_obs_fig_az.plot(h1, az1, '.-', markersize=0.5)
            self.figs_obs_fig_el.plot(h1, el1, '.-', markersize=0.5)

        # draw new - sky survey
        pos_sun, pos_moon, num_array = self.myData.get_data_obs_survey()
        array_max = np.max(num_array)
        bounds = np.arange(0, array_max + 1, 1)
        ax = self.figs_obs_fig_survey.pcolor(num_array, edgecolors=(0.5, 0.5, 0.5), linewidths=1)
        self.obs_panel_color_bar = self.figs_obs.colorbar(ax, ticks=bounds, shrink=1)

        # set ticks
        self.figs_obs_fig_survey.set_xticks([0, 16, 32, 48, 64, 80, 96])
        self.figs_obs_fig_survey.set_yticks([0, 24, 36, 48, 72])
        self.figs_obs_fig_survey.set_xticklabels([0, 4, 8, 12, 16, 20, 24])
        self.figs_obs_fig_survey.set_yticklabels([-90, -30, 0, 30, 90])

        # draw axis
        self.figs_obs_fig_survey.plot([48, 48], [0, 72], color='black', linewidth=0.8, linestyle='-.', alpha=0.4)
        self.figs_obs_fig_survey.plot([0, 96], [36, 36], color='black', linewidth=0.8, linestyle='-.', alpha=0.4)

        # draw soon, moon
        self.figs_obs_fig_survey.plot(pos_sun[0], pos_sun[1], color='red', marker='o', markerfacecolor=(1, 0, 0), alpha=1, markersize=20)
        self.figs_obs_fig_survey.plot(pos_moon[0], pos_moon[1], color='blue', marker='o', markerfacecolor='w', alpha=1, markersize=10)

        # show new
        self.canvas_obs.show()

    # img
    def draw_panel_img(self):
        self.figs_img_fig_beam = self.figs_img.add_subplot(221)
        self.figs_img_fig_src = self.figs_img.add_subplot(222)
        self.figs_img_fig_dirty = self.figs_img.add_subplot(223)
        self.figs_img_fig_clean = self.figs_img.add_subplot(224)
        self.figs_img.tight_layout(pad=0.15)

        # initial label and settings
        self.reset_panel_img()

        # image save tool
        toolbar = NavigationToolbar2TkAgg(self.canvas_img, self.tab_img)
        toolbar.update()
        self.canvas_img._tkcanvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

    def reset_panel_img(self):
        self.figs_img_fig_beam.cla()
        self.figs_img_fig_src.cla()
        self.figs_img_fig_dirty.cla()
        self.figs_img_fig_clean.cla()

        self.figs_img_fig_beam.set_title('Beam', fontdict=MY_IMG_FONT)
        self.figs_img_fig_src.set_title('Source', fontdict=MY_IMG_FONT)
        self.figs_img_fig_dirty.set_title('Dirty Map', fontdict=MY_IMG_FONT)
        self.figs_img_fig_clean.set_title('Clean Img', fontdict=MY_IMG_FONT)

        self.canvas_img.show()

    def update_panel_img(self):
        gamma = 0.5
        colormap = cm.get_cmap(self.myData.var_color_map_name)

        # reset ui
        self.reset_panel_img()

        # draw new - src
        show_model, src_max = self.myData.get_data_img_model()
        x_range = self.myData.myFuncUv.get_max_uv() // 4
        model_plot = self.figs_img_fig_src.imshow(np.power(show_model, gamma), picker=True,
                                                  origin='lower', cmap=colormap, norm=norm)
        pl.setp(model_plot, extent=(-src_max / 2., src_max / 2., -src_max / 2., src_max / 2.))

        # draw new - beam
        show_beam, x_max = self.myData.get_data_img_beam()
        beam_plot = self.figs_img_fig_beam.imshow(show_beam, picker=True, origin='lower',
                                                  aspect='equal', cmap=colormap, norm=norm)
        pl.setp(beam_plot, extent=(-x_range / 2., x_range / 2., -x_range / 2., x_range / 2.))

        # draw new - dirty
        show_map, x_max = self.myData.get_data_img_dirty()
        dirty_plot = self.figs_img_fig_dirty.imshow(show_map, picker=True, origin='lower', aspect='equal',
                                                    cmap=colormap, norm=norm)
        pl.setp(dirty_plot, extent=(-x_range / 2., x_range / 2., -x_range / 2., x_range / 2.))

        # draw new - clean
        clean_img, res_img = self.myData.get_data_img_clean()
        clean_plot = self.figs_img_fig_clean.imshow(clean_img, picker=True, origin='lower', aspect='equal',
                                                    cmap=colormap, norm=norm)
        pl.setp(clean_plot, extent=(-x_range / 2., x_range / 2., -x_range / 2., x_range / 2.))

        # show new
        self.canvas_img.show()

    # radplot
    def draw_panel_radplot(self):
        gs = gridspec.GridSpec(3, 1)
        self.figs_radplot_fig_uv = self.figs_radplot.add_subplot(gs[:2, :])
        self.figs_radplot_fig_rad = self.figs_radplot.add_subplot(gs[2, :])
        self.figs_radplot.tight_layout(pad=0.15)

        # initial label and data settings
        self.reset_panel_radplot()

        # image save tool
        toolbar = NavigationToolbar2TkAgg(self.canvas_radplot, self.tab_radplot)
        toolbar.update()
        self.canvas_radplot._tkcanvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

    def reset_panel_radplot(self):
        self.figs_radplot_fig_uv.cla()
        self.figs_radplot_fig_rad.cla()

        self.figs_radplot_fig_uv.set_title('UV PLOT', fontdict=MY_IMG_FONT)
        self.figs_radplot_fig_uv.set_xlabel("U (m)", fontdict=MY_IMG_FONT)
        self.figs_radplot_fig_uv.set_ylabel("V (m)", fontdict=MY_IMG_FONT)

        self.figs_radplot_fig_rad.set_title('RAD PLOT', fontdict=MY_IMG_FONT)
        self.figs_radplot_fig_rad.set_xlabel("UV Distance", fontdict=MY_IMG_FONT)
        self.figs_radplot_fig_rad.set_ylabel("Visibility Amplitude", fontdict=MY_IMG_FONT)

        self.canvas_radplot.show()

    def update_panel_radplot(self):
        # reset ui
        self.reset_panel_radplot()

        # draw new
        tmp_baseline, tmp_vis = self.myData.get_data_rad_vis()
        tmp_u, tmp_v = self.myData.get_data_rad_uv()
        self.figs_radplot_fig_rad.plot(tmp_baseline, abs(tmp_vis), 'ko', markersize=2)
        self.figs_radplot_fig_uv.plot(tmp_u, tmp_v, 'ko', markersize=1)
        max_u = max(np.abs(tmp_u))
        max_v = max(np.abs(tmp_v))
        self.figs_radplot_fig_uv.set_xlim(-max_u, max_u)
        self.figs_radplot_fig_uv.set_ylim(-max_v, max_v)

        # show new
        self.canvas_radplot.show()

    # input checking
    def test_input_float(self, content):
        # trigger the update if any input
        self.trigger_panel_update(UPDATE_TYPE_UOI)
        return True
        # if content.isdigit() or is_float(content):
        # if content.isdigit() or '.' in content:
        #     return True
        # else:
        #     return False

    # draw time uv and sky uv
    def _panel_uv_show_time_uv(self):
        myWinYearUv = ImagePopWin(self, UV_SHOW_POP_TYPE_YEAR)
        self.window.wait_window(myWinYearUv)

    def _panel_uv_show_multi_src(self):
        myWinSrcUv = ImagePopWin(self, UV_SHOW_POP_TYPE_SRC)
        self.window.wait_window(myWinSrcUv)


class GressBar(object):
    def __init__(self):
        self.master = None

    def start(self):
        top = tk.Toplevel()
        self.master = top
        color = '#E3E3E3'
        top.config(bg=color)
        top.overrideredirect(True)
        top.title("Progress Bar")
        # tk.Label(top, text="任务正在运行中,请稍等……", fg="green").pack(pady=2)
        tk.Label(top, text="Task is running, please wait……", bg=color,
                 fg="darkgreen", font=('Arial', 12, 'bold italic')).pack(pady=2)
        prog = ttk.Progressbar(top, mode='indeterminate', length=200)
        prog.pack(pady=10, padx=35)
        prog.start()
        top.wm_attributes("-topmost", 1)
        top.grab_set()

        curWidth = top.winfo_width()
        curHeight = top.winfo_height()
        scnWidth, scnHeight = top.maxsize()
        tmpcnf = '+%d+%d' % ((scnWidth - curWidth) / 2 - 100, (scnHeight - curHeight) / 2 - 50)
        top.geometry(tmpcnf)

        top.resizable(False, False)
        top.update()
        top.mainloop()

    def quit(self):
        if self.master:
            self.master.destroy()
            self.master = None


class ImagePopWin(tk.Toplevel):
    def __init__(self, parent, show_type):
        super().__init__()
        self.parent = parent
        # self.overrideredirect(True)
        self.show_type = show_type
        self.title('Show ' + show_type)
        center_window(self, 600, 600)
        self.protocol("WM_DELETE_WINDOW", self.quit)
        self._gui_ini()

    def quit(self):
        self.destroy()

    def _gui_ini(self):
        fig = Figure(figsize=(5, 4))
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # my_plot = fig.add_subplot(111, aspect='equal')

        is_nothing_data_obtained = False

        if self.show_type == UV_SHOW_POP_TYPE_YEAR:
            result_time_u, result_time_v, max_range = self.parent.myData.get_data_uv_time()
            print(max_range)
            my_plot = fig.add_subplot(111, aspect='equal')
            if len(result_time_u) != 0 and len(result_time_v) != 0:
                # 横着有3个是6分, 纵轴有4个是8分, 所以取最小公倍数, 24
                k = 0
                for irow in (21, 15, 9, 3):  # 24份对应的画点的位置
                    for icol in (20, 12, 4):
                        if len(result_time_u[k]) > 0 and len(result_time_v[k]) > 0:
                            temp_u = result_time_u[k] / max_range * 4
                            temp_v = result_time_v[k] / max_range * 3
                            temp_u += icol
                            temp_v += irow
                            my_plot.scatter(temp_u, temp_v, s=3, marker='.', color='b')
                        k += 1
                my_plot.set_title("All Year Round UV Plot")
                my_plot.set_xlim(0, 24)
                my_plot.set_ylim(0, 24)
                my_plot.set_xticks([4, 12, 20])
                my_plot.set_xticklabels([1, 2, 3])
                my_plot.set_yticks([3, 9, 15, 21])
                my_plot.set_yticklabels([4, 3, 2, 1])
                my_plot.set_xlabel(r"$i_{th}\ month$")
                my_plot.set_ylabel(r"$Quarter$")
                my_plot.grid()

            else:
                is_nothing_data_obtained = True

        elif self.show_type == UV_SHOW_POP_TYPE_SRC:
            result_src_name, result_src_u, result_src_v, max_range = self.parent.myData.get_data_uv_src()
            num_src = len(result_src_name)
            # print(num_src, result_src_name)
            if num_src > 0:
                num_col = int(np.ceil(np.sqrt(num_src)))
                num_row = int(np.ceil(num_src / num_col))
                # print(num_col, num_row)
                for k in range(num_src):
                    tmp_plot = fig.add_subplot(num_row, num_col, k+1, aspect='equal')
                    if len(result_src_u[k]) > 0 and len(result_src_v[k]) > 0:
                        tmp_plot.scatter(result_src_u[k] / max_range, result_src_v[k] / max_range, s=3, marker='.',
                                         color='b')
                        tmp_plot.set_xlim([-1, 1])
                        tmp_plot.set_ylim([-1, 1])
                    tmp_plot.set_title(result_src_name[k])
                fig.tight_layout(pad=0.15)
            else:
                is_nothing_data_obtained = True

        if is_nothing_data_obtained:
            my_plot = fig.add_subplot(111, aspect='equal')
            my_plot.set_xlim(-6, 6)
            my_plot.set_ylim(-6, 6)
            my_plot.text(-2, 2, r"$Please\ apply\ your\ config.$", fontdict={"size": 10, "color": 'r'})

        # toolbox
        canvas.show()
        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # tk.Button(self, text="OK", command=self.quit).pack(side=tk.BOTTOM)


class MultiChoiceDialog(tk.Toplevel):
    def __init__(self, choice_name, option_lst):
        super().__init__()
        self.option_lst = option_lst
        self.title('Select ' + choice_name)
        center_window(self, 300, 250)
        self.protocol("WM_DELETE_WINDOW", self.quit)

        self.choice_result = None
        self._gui_ini()

    def quit(self):
        self.choice_result = None
        self.destroy()

    def _gui_ini(self):
        row1 = tk.Frame(self)
        row1.pack(fill="x")

        sb = tk.Scrollbar(row1)
        listbox_var = tk.StringVar()
        self.listbox = tk.Listbox(row1,
                                  listvariable=listbox_var,
                                  selectmode=tk.MULTIPLE,  # SINGLE, MULTIPLE
                                  exportselection=True,
                                  bg='grey',
                                  fg='blue',
                                  bd=10,
                                  relief='ridge',
                                  font=('Arial', 15, 'bold'),
                                  selectbackground='red',
                                  selectforeground='white',
                                  selectborderwidth=2,
                                  height=6,
                                  # setgrid=True
                                  )
        # self.listbox.grid(row=0, column=0, columnspan=4, rowspan=6, sticky=tk.NSEW)
        # sb.grid(row=0, column=3, rowspan=6, sticky=tk.E)
        listbox_var.set(self.option_lst)

        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.listbox.config(yscrollcommand=sb.set)
        sb['command'] = self.listbox.yview

        # row 2
        row2 = tk.Frame(self)
        row2.pack(pady=10)
        tk.Button(row2, text="Cancel", command=self.quit).pack(side=tk.RIGHT, padx=5)
        tk.Button(row2, text="Apply", command=self.call_back_ok).pack(side=tk.RIGHT, padx=5)

    def call_back_ok(self):
        self.choice_result = [self.option_lst[x] for x in self.listbox.curselection()]
        # print(self.choice_result)
        self.destroy()

    def get_choice_result(self):
        return self.choice_result


class TextHandler(logging.Handler):
    # This class allows you to log to a Tkinter Text or ScrolledText widget
    # Adapted from Moshe Kaplan: https://gist.github.com/moshekaplan/c425f861de7bbf28ef06

    def __init__(self, text):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Store a reference to the Text it will log to
        self.text = text

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text.configure(state='normal')
            self.text.insert(tk.END, msg + '\n')
            self.text.configure(state='disabled')
            # Autoscroll to the bottom
            self.text.yview(tk.END)

        # This is necessary because we can't modify the Text from other threads
        self.text.after(0, append)


class TopLevelParaCal(object):
    def __init__(self, parent):
        self.parent = parent
        self.master = tk.Toplevel(parent.window)
        self.master.protocol("WM_DELETE_WINDOW", self.quit)
        self.master.title("Parameter Evaluation")
        self.master.resizable(False, False)
        ParaCal(self.master)

    def quit(self):
        # clear the window object
        self.parent.PopOutParaCal = None
        self.parent._clear_pop_out_win_state(POP_OUT_PARA_CAL)
        self.master.destroy()


# only when closing the dbeditor, the main ui will be refresh, which should be changed later.
class TopLevelDbEditor(object):
    def __init__(self, parent, database='', db_pickle=''):
        self.parent = parent
        self.master = tk.Toplevel(parent.window)
        self.master.protocol("WM_DELETE_WINDOW", self.quit)
        self.master.title("Database Editor")
        self.master.resizable(False, False)
        self.myDbEditor = DbEditor(self.master, database, db_pickle)

    def quit(self):
        # refresh the db_pickle
        self.myDbEditor.update_db_pkl()

        # refresh the ui config
        self.parent.refresh_ui_config_with_db()

        # clear the window object
        self.parent.PopOutDbEditor = None
        self.parent._clear_pop_out_win_state(POP_OUY_DB_EDIT)
        self.master.destroy()


def center_window(master, width, height):
    screenwidth = master.winfo_screenwidth()
    screenheight = master.winfo_screenheight()
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    # print(size)
    master.geometry(size)


def limit_window_size(master, vec_max, vec_min):
    master.maxsize(vec_max[0], vec_max[1])
    master.minsize(vec_min[0], vec_min[1])


# load config_file to initialize the global var
def load_config_file(config_file):
    if len(config_file) == 0:
        # print("Please input a correct file name")
        return
    else:
        config_path = os.path.join(os.getcwd(), config_file)
        if not os.path.exists(config_path):
            print("Cannot find configuration file")
            return
        else:
            pass


def test_class_my_data():
    my_data = AppData()
    start_time = time.time()
    my_data.update_all_with_flag(True, True, True, True, True)
    print("updating needs {}".format(time.time() - start_time))
    print(len(my_data.get_data_img_clean()[0]))


if __name__ == "__main__":
    # # 1. load the parameter file
    # config_file = ''
    # load_config_file(config_file)

    # 2. Start the GUI
    root = tk.Tk()
    root.title(TITLE)
    # customize the geometry of UI
    center_window(root, 1000, 700)
    limit_window_size(root, (1400, 1000), (900, 600))
    # run GUI
    db_file = 'database.db'
    pkl_file = 'database.pkl'
    my_gui = AppGUI(root, db_file, pkl_file)
    # gui loop
    tk.mainloop()
