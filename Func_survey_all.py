"""
@functions: run the multiple source survey, show its obs ability, uvplots, beam, map, clean and corresponding parameters
@author: Zhen ZHAO
@date: Jan 16, 2019
"""
import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.image as plimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage.interpolation as spndint
import scipy.optimize as spfit

import argparse
import configparser
import pickle
import time
import multiprocessing

import load_conf as lc
import utility as ut
import model_effect as me
import model_satellite as ms
import model_obs_ability as mo

from Func_uv import FuncUv
from Func_img import FuncImg, overlap_indices
from Func_obs import FuncObs


# colors normalization
norm = mpl.colors.Normalize(vmin=0, vmax=0.6)
gamma = 0.3


class FuncSurvey(object):
    def __init__(self, start_t, stop_t, step_t, p_main_src, p_multi_src, p_sat, p_vlbi, p_tele,
                 freq, bl_type, f_unit, cutoff_angle, precession_type,
                 model_name, n_pix, set_clean_window, clean_gain, clean_threshold, clean_niter,
                 dir_output, is_show_para, is_save_uv, num_sub_proc, color_map, img_type):
        cutoff_dict = {"flag": lc.cutoff_mode["flag"], "CutAngle": cutoff_angle}
        # FuncUV paras
        self.myFuncUv = FuncUv(start_t, stop_t, step_t, p_main_src, p_multi_src, p_sat, p_vlbi, p_tele,
                               freq, bl_type, f_unit, cutoff_dict, precession_type)
        # FuncObs paras
        self.myFuncObs = FuncObs(start_t, stop_t, step_t, p_main_src, p_vlbi, p_sat, p_tele,
                            bl_type, cutoff_dict, precession_type)
        # FuncImg paras
        self.obs_freq = freq
        self.unit_flag = f_unit
        self.model_name = model_name
        self.n_pix = n_pix
        self.set_clean_window = set_clean_window
        self.clean_gain = clean_gain
        self.clean_threshold = clean_threshold
        self.clean_niter = clean_niter
        # operational paras
        self.dir_ouput = dir_output
        self.is_show_para = is_show_para
        self.is_save_uv = is_save_uv
        self.num_sub_proc = num_sub_proc
        self.color_map = color_map
        self.img_type = img_type

        # parameters for sub process
        self.sp_args_srcs = []

    def show_running_info(self, args):
        print(time.asctime() + ": {}-Done".format(args))

    def __call__(self, arg_lst):
        # 0. parameters
        para_info = ""
        src_name = arg_lst[0]
        ##############
        # 1. obs img
        # 1.1 reset and calculate az - el
        self.myFuncObs.reset_src_for_az_el(arg_lst)
        azimuth, elevation, hour_lst = self.myFuncObs.get_result_az_el_with_update()
        gs_lst = self.myFuncObs.get_result_name_gs()
        x_hour_lmt = max(hour_lst[0])
        if x_hour_lmt > 24:
            x_hour_lmt = 24

        # 1.2 calculate optimal interval
        optimal_inter, sta_best_inters, sta_best_durations, sta_all_inter = self.myFuncObs.get_result_best_obs_time_el()
        optimal_time_str = self.myFuncObs.get_result_best_time_string_after_func_best_obs()
        para_info += "optimal observation interval is : {}".format(optimal_inter) + "\n"
        para_info += optimal_time_str + '\n'
        # 1.3 draw az-el
        if len(azimuth) != 0 and len(elevation) != 0:
            fig1 = plt.figure(figsize=(8, 8))
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
            tmp_cut = self.myFuncObs.get_cutoff_angle()
            ax1_2.plot([hour_lst[0][0], hour_lst[0][-1]], [tmp_cut, tmp_cut], '--k')

            # draw optimal time interval
            rect = plt.Rectangle((optimal_inter[0], 0), optimal_inter[1] - optimal_inter[0], 90, color='r', alpha=0.5)
            ax1_2.add_patch(rect)
            fig1.tight_layout()

            # save fig
            tmp_az_path = os.path.join(self.dir_ouput, "{}-obs-az-el.{}".format(src_name, self.img_type))
            plt.savefig(tmp_az_path)

        ##############
        # 2. uvplot and imaging
        # 2.1 create img object
        data_u, data_v, max_uv = self.myFuncUv._get_reset_source_info(arg_lst)
        myFuncImg = FuncImg(self.model_name, self.n_pix,
                            data_u, data_v, max_uv,
                            self.obs_freq, self.set_clean_window,
                            self.clean_gain, self.clean_threshold,
                            self.clean_niter, self.unit_flag)
        # 2.2 src model
        data_img_src, data_img_range = myFuncImg.get_result_src_model_with_update()
        # 2.3 dirty beam
        data_img_bm = myFuncImg.get_result_dirty_beam_with_update()
        # 2.4 dirty map
        data_img_map = myFuncImg.get_result_dirty_map_with_update()
        # 2.5 clean map, resual map, clean beam
        data_img_cmap, data_img_res, data_pure_point, data_img_cbm = myFuncImg.get_result_clean_map_with_update()
        data_img_range = myFuncImg.get_result_img_range()
        show_range = data_img_range // 2

        # 2.6 record parameters
        para_info += myFuncImg.show_result_para_cal()

        ##########
        # 3. draw uv, model, beam, map, clean
        tmp_path_save_uv = os.path.join(self.dir_ouput, "{}-uv-plot.{}".format(src_name, self.img_type))
        tmp_path_save_src = os.path.join(self.dir_ouput, "{}-src-model.{}".format(src_name, self.img_type))
        tmp_path_save_bm = os.path.join(self.dir_ouput, "{}-dirty-beam.{}".format(src_name, self.img_type))
        tmp_path_save_map = os.path.join(self.dir_ouput, "{}-dirty-map.{}".format(src_name, self.img_type))
        tmp_path_save_cmap = os.path.join(self.dir_ouput, "{}-clean-map.{}".format(src_name, self.img_type))
        # 3.1 u,v
        fig6 = plt.figure(figsize=(4, 4))
        fig_uv = fig6.add_subplot(111, aspect='equal')
        x = np.array(data_u)
        y = np.array(data_v)
        max_range = max_uv * 1.1
        fig_uv.scatter(x, y, s=1, marker='.', color='brown')
        fig_uv.set_xlim([-max_range, max_range])
        fig_uv.set_ylim([-max_range, max_range])
        fig_uv.set_title("UV Plot: %s" % src_name)
        if self.unit_flag == 'km':
            fig_uv.set_xlabel("u$(km)$")
            fig_uv.set_ylabel("v$(km)$")
        else:
            fig_uv.set_xlabel("u$(\lambda)$")
            fig_uv.set_ylabel("v$(\lambda)$")
        fig_uv.grid()
        # set science
        fig_uv.yaxis.get_major_formatter().set_powerlimits((0, 1))
        fig_uv.xaxis.get_major_formatter().set_powerlimits((0, 1))
        # save uv
        plt.savefig(tmp_path_save_uv)

        # 3.2 dirty beam
        fig2 = plt.figure(figsize=(4, 4))
        fig_bm = fig2.add_subplot(111, aspect='equal')
        plot_beam = fig_bm.imshow(data_img_bm, origin='lower', aspect='equal', cmap=self.color_map, norm=norm)
        plt.setp(plot_beam, extent=(-show_range, show_range, -show_range, show_range))
        fig_bm.set_xlabel('Relative RA (mas)')
        fig_bm.set_ylabel('Relative DEC (mas)')
        fig_bm.set_title('DIRTY BEAM')
        fig2.colorbar(plot_beam, shrink=0.9)
        plt.savefig(tmp_path_save_bm)

        # 3.3 src model
        fig3 = plt.figure(figsize=(4, 4))
        fig_model = fig3.add_subplot(111, aspect='equal')
        plot_model = fig_model.imshow(np.power(data_img_src, gamma), origin='lower', aspect='equal',
                                      cmap=self.color_map, norm=norm)
        plt.setp(plot_model, extent=(-show_range, show_range, -show_range, show_range))
        fig_model.set_xlabel('Relative RA (mas)')
        fig_model.set_ylabel('Relative DEC (mas)')
        fig_model.set_title('MODEL IMAGE')
        fig3.colorbar(plot_model, shrink=0.9)
        plt.savefig(tmp_path_save_src)

        # 3.4 dirty map
        fig4 = plt.figure(figsize=(4, 4))
        fig_map = fig4.add_subplot(111, aspect='equal')
        plot_map = fig_map.imshow(data_img_map, origin='lower', aspect='equal', cmap=self.color_map, norm=norm)
        plt.setp(plot_map, extent=(-show_range, show_range, -show_range, show_range))
        fig_map.set_xlabel('Relative RA (mas)')
        fig_map.set_ylabel('Relative DEC (mas)')
        fig_map.set_title('DIRTY IMAGE')
        fig4.colorbar(plot_map, shrink=0.9)
        plt.savefig(tmp_path_save_map)

        # 3.5 clean map
        fig5 = plt.figure(figsize=(4, 4))
        fig_cmap = fig5.add_subplot(111, aspect='equal')
        plot_cmap = fig_cmap.imshow(data_img_cmap, origin='lower', aspect='equal', picker=True, interpolation='nearest',
                                    cmap=self.color_map, norm=norm)
        plt.setp(plot_cmap, extent=(-show_range, show_range, -show_range, show_range))
        fig_cmap.set_xlabel('Relative RA (mas)')
        fig_cmap.set_ylabel('Relative DEC (mas)')
        fig_cmap.set_title('CLEAN IMAGE')
        fig5.colorbar(plot_cmap, shrink=0.9)
        plt.savefig(tmp_path_save_cmap)

        ##########
        # 4. save uv data
        if self.is_save_uv:
            tmp_path_data_uv = os.path.join(self.dir_ouput, "{}-uv-data.txt".format(src_name))
            np.savetxt(tmp_path_data_uv, [data_u, data_v], fmt='%0.4f')

        ##########
        # 4. save parameters
        tmp_path_data_para = os.path.join(self.dir_ouput, "{}-src-info.txt".format(src_name))
        with open(tmp_path_data_para, 'w') as f:
            f.write(para_info)
        if self.is_show_para:
            print(para_info)

        return src_name

    def run_survey_all(self):
        # 1. prepare args (multi src) for subprocess
        self.sp_args_srcs = []
        run_time_start = time.time()
        for i in range(self.myFuncUv.src_num):
            tmp_name = self.myFuncUv.pos_multi_src[i][0]
            tmp_ra = self.myFuncUv.pos_multi_src[i][1]
            tmp_dec = self.myFuncUv.pos_multi_src[i][2]
            self.sp_args_srcs.append([tmp_name, tmp_ra, tmp_dec])
        # 2. create processing pool
        if self.num_sub_proc > 0:
            pool = multiprocessing.Pool(self.num_sub_proc)
        else:
            pool = multiprocessing.Pool()  # the number of the cores(defalut), obtained by multiprocessing.cpu_count()

        # 3. run sub process
        for each in self.sp_args_srcs:
            pool.apply_async(func=self, args=(each,), callback=self.show_running_info)
        pool.close()
        pool.join()
        print("== All Sub-process(es) done.===")
        print("The time cost is: ", time.time() - run_time_start)
        print("Please check the result at:", self.dir_ouput)


class SurveyConfigParser(object):
    def __init__(self, _filename="config_survey.ini", _dbname='database.pkl'):
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

        # position
        self.str_source = ""
        self.str_vlbi = ""
        self.str_telemetry = ""
        self.str_sat = ""

        self.pos_mat_src = []
        self.pos_mat_vlbi = []
        self.pos_mat_telemetry = []
        self.pos_mat_sat = []

        # imaging
        self.n_pix = 0
        self.source_model = ""
        self.clean_gain = 0
        self.clean_threshold = 0
        self.clean_niter = 0
        self.color_map_name = ""

        # parse data
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

        # imaging
        self.n_pix = config.getint("imaging", "n_pix")
        self.source_model = config.get("imaging", "source_model")
        self.clean_gain = config.getfloat("imaging", "clean_gain")
        self.clean_threshold = config.getfloat("imaging", "clean_threshold")
        self.clean_niter = config.getint("imaging", "clean_niter")
        self.color_map_name = config.get("imaging", "color_map_name")

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

        print('*' * 15, " Imaging", '*' * 15)
        print("\t n_pix:", self.n_pix)
        print("\t source model:", self.source_model)
        print("\t clean gain:", self.clean_gain)
        print("\t clean threshold:", self.clean_threshold)
        print("\t clean iterations:", self.clean_niter)
        print("\t colormap name:", self.color_map_name)

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
        config.set("station", "pos_source", "0316+413")
        config.set("station", "pos_vlbi", "ShangHai, Tianma, Urumqi, GIFU11, HITACHI,KASHIM34")
        config.set("station", "pos_telemetry", "")
        config.set("station", "pos_satellite", "")
        self.str_source = ['0316+413']
        self.str_vlbi = ['ShangHai', 'Tianma', 'Urumqi', 'GIFU11', 'HITACHI', 'KASHIM34']
        self.str_telemetry = ['']
        self.str_sat = ['']
        self.get_data_from_db()

        # add section: imaging
        config.add_section("imaging")
        config.set("imaging", "n_pix", "512")
        config.set("imaging", "source_model", "Point-source.model")
        config.set("imaging", "clean_gain", "0.9")
        config.set("imaging", "clean_threshold", "0.01")
        config.set("imaging", "clean_niter", "20")
        config.set("imaging", "color_map_name", "viridis")
        self.n_pix = 512
        self.source_model = "Point-source.model"
        self.clean_gain = 0.9
        self.clean_threshold = 0.01
        self.clean_niter = 20
        self.color_map_name = "viridis"

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
    parser = argparse.ArgumentParser(description="Run the multiple source survey, show its uvplots, obs ability, beam, map, clean and corresponding parameters")
    parser.add_argument('-c',
                        '--config',
                        default='config_survey.ini',
                        help='Specify the configuration file')
    parser.add_argument('-i',
                        '--show_info',
                        action="store_true",
                        help='Choose to show the best obs time, beam size, position angle, dynamic range and rms noise')
    parser.add_argument('-f',
                        '--img_fmt',
                        choices=['eps', 'png', 'pdf', 'svg', 'ps'],
                        help='Specify the img format (default:pdf)',
                        default='pdf')
    parser.add_argument('-s',
                        '--save_uv',
                        action="store_true",
                        help='Store the uv data (/OUTPUT/survey_all/<time_str>/uv_data.txt)')
    parser.add_argument('-n',
                        '--num_procs',
                        help='Specify the number of subprocess you wanna use',
                        default='bydefault')
    # parser.add_argument('-m',
    #                     '--color_map',
    #                     choices=['viridis', 'hot', 'jet', 'rainbow', 'Greys', 'cool', 'nipy_spectral'],
    #                     help='Specify the color map',
    #                     default='viridis')

    return parser.parse_args()


def run_muiltisrc_survey():
    # 1. initialize parse
    args = parse_args()
    # 1.1 config file
    if args.config != '':
        my_config_parser = SurveyConfigParser(args.config)
    else:
        my_config_parser = SurveyConfigParser()
    # 1.2 img type
    img_type = 'pdf'
    if args.img_fmt in ['eps', 'png', 'pdf', 'svg', 'ps']:
        img_type = args.img_fmt
    # 1.3 output path
    path_dir_out_pre = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'survey_all')
    path_dir_out = os.path.join(path_dir_out_pre, time.ctime())
    if not os.path.exists(path_dir_out):
        os.mkdir(path_dir_out)
    # 1.4 print parameters or not
    is_show_para = False
    if args.show_info:
        is_show_para = True
    # 1.5 save uv data or not
    is_save_uv = False
    if args.save_uv:
        is_save_uv = True
    # 1.6 specify process num
    num_procs = 0
    args_num_procs = str(args.num_procs)
    if args_num_procs.isdigit():
        num_procs = int(args_num_procs)
    # 1.7 colormap
    # colormap = 'viridis'
    # if args.color_map in ['viridis', 'hot', 'jet', 'rainbow', 'Greys', 'cool', 'nipy_spectral']:
    #     colormap = args.color_map
    colormap = my_config_parser.color_map_name
    # 2. config objects
    start_time = ut.time_2_mjd(*my_config_parser.time_start, 0)
    stop_time = ut.time_2_mjd(*my_config_parser.time_end, 0)
    time_step = ut.time_2_day(*my_config_parser.time_step)

    myFuncSurvey = FuncSurvey(start_time, stop_time, time_step,
                              my_config_parser.pos_mat_src[0],
                              my_config_parser.pos_mat_src,
                              my_config_parser.pos_mat_sat,
                              my_config_parser.pos_mat_vlbi,
                              my_config_parser.pos_mat_telemetry,
                              my_config_parser.obs_freq,
                              my_config_parser.baseline_type,
                              my_config_parser.unit_flag,
                              my_config_parser.cutoff_angle,
                              my_config_parser.precession_mode,
                              my_config_parser.source_model,
                              my_config_parser.n_pix, True,
                              my_config_parser.clean_gain,
                              my_config_parser.clean_threshold,
                              my_config_parser.clean_niter,
                              path_dir_out, is_show_para, is_save_uv, num_procs, colormap, img_type)
    # 3. run survey
    myFuncSurvey.run_survey_all()


if __name__ == "__main__":
    run_muiltisrc_survey()
