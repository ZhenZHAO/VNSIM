"""
@functions: basic uv coverage and sky coverage
@author: Zhen ZHAO
@date: Nov 3, 2018
"""
import load_conf as lc
import utility as ut
import model_effect as me
import model_satellite as ms
import model_obs_ability as mo
import numpy as np
import matplotlib.pyplot as plt
import argparse
import configparser
import os
import pickle
import time
import multiprocessing

from Func_uv import FuncUv, UVConfigParser

# run more uv function  if num&xx == xx
RUN_FUNC_UV_SRCS = 1  # 001
RUN_FUNC_UV_SKY = 2   # 010
RUN_FUNC_UV_YEAR = 4  # 100


# multiprocess type
SUB_PROCESS_TYPE_UV_SRCS = 0
SUB_PROCESS_TYPE_UV_SKY = 1
SUB_PROCESS_TYPE_UV_YEAR = 2


# function design ideas
_design_idea = """
In this scripts, I provide the following three advanced functions about uv plots:
- multiple-source uvplots: show the (u,v) coverage of all radio sources specified in the configuration file
- all-sky uvplots: by evenly divid- ing the whole sky into 5×6 blocks, this function roughly shows the survey ability of selected station combinations at a given observing time;
- all-year-round uvplots: show 12 (u, v) coverage plots generated through simulating the first-day observation of each month, which is designed to observe the coverage evolution of space VLBI due to the satellite orbit precession;
"""


class FuncUvMore(object):
    def __init__(self, start_t, stop_t, step_t, p_main_src, p_multi_src, p_sat, p_vlbi, p_tele,
                 freq, bl_type, f_unit, cutoff_dict, precession_type, run_type):
        self.myFuncUv = FuncUv(start_t, stop_t, step_t, p_main_src, p_multi_src, p_sat, p_vlbi, p_tele,
                               freq, bl_type, f_unit, cutoff_dict, precession_type)
        self.run_func_type = run_type
        self.reset_results()

    def reset_results(self):
        # 1. all year uv result
        self.result_year_u = [[]] * 12  # [[],[],[],...,[]]
        self.result_year_v = [[]] * 12
        self.result_mrange_year = [[]] * 12
        self.max_range_year_uv = -1

        # 2. all sky uv result
        self.result_sky_u = [[]] * 30
        self.result_sky_v = [[]] * 30
        self.result_mrange_sky = [[]] * 30
        self.max_range_sky_uv = -1

        # 3. multiple src results
        self.result_multi_src_name = []
        self.result_multi_src_u = []
        self.result_multi_src_v = []
        self.result_mrange_src = []
        self.max_range_multi_src = -1

        # 4. args for multiprocess
        self.sp_args_srcs = []
        self.sp_args_sky = []
        self.sp_args_year = []

    def get_run_result_by_type(self):
        pass

    # must invoke run_uv_more before you get results
    def get_run_result_srcs(self):
        return self.result_multi_src_name, self.result_multi_src_u, self.result_multi_src_v, self.result_mrange_src, self.max_range_multi_src

    def get_run_result_sky(self):
        return self.result_sky_u, self.result_sky_v, self.max_range_sky_uv * 1.3

    def get_run_result_year(self):
        return self.result_year_u, self.result_year_v, self.max_range_year_uv * 1.3

    # multiprocessing workers
    def __call__(self, p_type, arg_lst, res_queue):
        # parse format: {"type":xxx, "name":xxx, "u":xxx, "v":xxx, "maxuv":xxx}
        if p_type == SUB_PROCESS_TYPE_UV_SRCS:
            temp_u, temp_v, temp_max = self.myFuncUv._get_reset_source_info(arg_lst)
            put_in_data = {"type": SUB_PROCESS_TYPE_UV_SRCS, "name": arg_lst[0],
                           "u": temp_u, "v": temp_v, "maxuv": temp_max}
            res_queue.put(put_in_data)
        elif p_type == SUB_PROCESS_TYPE_UV_SKY:
            temp_u, temp_v, temp_max = self.myFuncUv._get_reset_source_info(arg_lst)
            put_in_data = {"type": SUB_PROCESS_TYPE_UV_SKY, "name": arg_lst[0],
                           "u": temp_u, "v": temp_v, "maxuv": temp_max}
            res_queue.put(put_in_data)
        elif p_type == SUB_PROCESS_TYPE_UV_YEAR:
            temp_start, temp_end, time_step, tmp_name = arg_lst
            temp_u, temp_v, temp_max = self.myFuncUv._get_reset_time_info(temp_start, temp_end, time_step)
            put_in_data = {"type": SUB_PROCESS_TYPE_UV_YEAR, "name": tmp_name,
                           "u": temp_u, "v": temp_v, "maxuv": temp_max}
            res_queue.put(put_in_data)
        else:
            pass

    def run_uv_more(self):
        self._prepare_args_for_multiprocess()
        # 1. create processing pool and queue
        run_time_start = time.time()
        pool = multiprocessing.Pool()
        res_queue = multiprocessing.Manager().Queue()

        # 2. run subprocess
        if self.run_func_type & RUN_FUNC_UV_SRCS == RUN_FUNC_UV_SRCS:
            for each in self.sp_args_srcs:
                pool.apply_async(func=self, args=(SUB_PROCESS_TYPE_UV_SRCS, each, res_queue))
        if self.run_func_type & RUN_FUNC_UV_SKY == RUN_FUNC_UV_SKY:
            for each in self.sp_args_sky:
                pool.apply_async(func=self, args=(SUB_PROCESS_TYPE_UV_SKY, each, res_queue))
        if self.run_func_type & RUN_FUNC_UV_YEAR == RUN_FUNC_UV_YEAR:
            for each in self.sp_args_year:
                pool.apply_async(func=self, args=(SUB_PROCESS_TYPE_UV_YEAR, each, res_queue))
        pool.close()
        pool.join()
        print("== Sub-process(es) done.===")
        print("The time cost by multiprocessing is: ",  time.time() - run_time_start)
        print("The core number of cpu", multiprocessing.cpu_count())

        # 3. parse result
        self.reset_results()
        # print("===start to parse data===")
        while not res_queue.empty():
            tmp_result = res_queue.get()
            tmp_type = tmp_result["type"]
            # print(tmp_type, tmp_result["name"], len(tmp_result["u"]))
            # parse format: {"type":xxx, "name":xxx, "u":xxx, "v":xxx, "maxuv":xxx}
            if tmp_type == SUB_PROCESS_TYPE_UV_SRCS:
                self.result_multi_src_name.append(tmp_result["name"])
                self.result_multi_src_u.append(tmp_result["u"])
                self.result_multi_src_v.append(tmp_result["v"])
                self.result_mrange_src.append(tmp_result["maxuv"])
                if self.max_range_multi_src < tmp_result["maxuv"]:
                    self.max_range_multi_src = tmp_result["maxuv"]

            if tmp_type == SUB_PROCESS_TYPE_UV_SKY:  # source-%d
                s_index = int(tmp_result["name"].split('-')[1])
                self.result_sky_u[s_index] = tmp_result["u"]
                self.result_sky_v[s_index] = tmp_result["v"]
                # self.result_mrange_sky[s_index] = tmp_result["maxuv"]  # useless for sky
                if self.max_range_sky_uv < tmp_result["maxuv"]:
                    self.max_range_sky_uv = tmp_result["maxuv"]

            if tmp_type == SUB_PROCESS_TYPE_UV_YEAR:  # year-{}
                t_index = int(tmp_result["name"].split('-')[1])
                self.result_year_u[t_index] = tmp_result["u"]
                self.result_year_v[t_index] = tmp_result["v"]
                # self.result_mrange_time[t_index] = tmp_result["maxuv"]  # useless for year
                if self.max_range_year_uv < tmp_result["maxuv"]:
                    self.max_range_year_uv = tmp_result["maxuv"]

        # print("Parse data done.")

    def _prepare_args_for_multiprocess(self):
        # 1. prepare args for subprocess
        # 1.1 for multi src
        self.sp_args_srcs = []
        for i in range(self.myFuncUv.src_num):
            tmp_name = self.myFuncUv.pos_multi_src[i][0]
            tmp_ra = self.myFuncUv.pos_multi_src[i][1]
            tmp_dec = self.myFuncUv.pos_multi_src[i][2]
            self.sp_args_srcs.append([tmp_name, tmp_ra, tmp_dec])
        # 1.2 for all sky
        self.sp_args_sky = []
        index = 0
        for i in (2, 6, 10, 14, 18, 22):  # dra
            for j in (-60, -30, 0, 30, 60):  # dra
                ra = ut.time_2_rad(i, 0, 0)
                dec = ut.angle_2_rad(j, 0, 0)
                pos_src = ['source-%d' % (index), ra, dec]
                index = index + 1
                self.sp_args_sky.append(pos_src)
        # 1.3 for all year
        self.sp_args_year = []
        date = ut.mjd_2_time(self.myFuncUv.start_mjd)
        year, month = date[1], date[2]
        for i in range(0, 12):
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
            time_id_str = "year-{}".format(i)
            self.sp_args_year.append([temp_start, temp_end, self.myFuncUv.time_step, time_id_str])


def parse_args():
    parser = argparse.ArgumentParser(description="Show more UV plots: multisource uvplot, all-sky uvplots, all-year uvplots")
    parser.add_argument('-c',
                        '--config',
                        default='config_uv.ini',
                        help='Specify the configuration file')
    parser.add_argument('-t',
                        '--run_type',
                        choices=['src', 'sky', 'time', 'src+sky', 'all'],
                        help='Specify the functions you wanna run',
                        default='src+sky')
    parser.add_argument('-i',
                        '--show_funcs',
                        action="store_true",
                        help='Choose to show more details about the function descriptions')
    parser.add_argument('-g',
                        '--show_gui',
                        action="store_true",
                        help='Choose to show GUI or not')
    parser.add_argument('-p',
                        '--separate_srcs',
                        action="store_true",
                        help='To save uvplots of multiple sources seperately')
    parser.add_argument('-f',
                        '--img_fmt',
                        choices=['eps', 'png', 'pdf', 'svg', 'ps'],
                        help='Specify the img format (default:pdf)',
                        default='pdf')

    return parser.parse_args()


def run_uv_advanced():
    # 1. initialize parse and config objects
    args = parse_args()
    # args.show_gui = True
    # args.separate_srcs = True
    if args.config != '':
        my_config_parser = UVConfigParser(args.config)
    else:
        my_config_parser = UVConfigParser()
        # print(my_config_parser.show_info())
    # 2. parse config file
    start_time = ut.time_2_mjd(*my_config_parser.time_start, 0)
    stop_time = ut.time_2_mjd(*my_config_parser.time_end, 0)
    time_step = ut.time_2_day(*my_config_parser.time_step)
    run_type_str = "src+sky"
    if args.run_type in ['src', 'sky', 'time', 'src+sky', 'all']:
        run_type_str = args.run_type
    run_type = 0
    if run_type_str == 'src':
        run_type = RUN_FUNC_UV_SRCS
    elif run_type_str == 'sky':
        run_type = RUN_FUNC_UV_SKY
    elif run_type_str == 'time':
        run_type = RUN_FUNC_UV_YEAR
    elif run_type_str == 'src+sky':
        run_type = RUN_FUNC_UV_SRCS | RUN_FUNC_UV_SKY
    elif run_type_str == "all":
        run_type = RUN_FUNC_UV_SRCS | RUN_FUNC_UV_SKY | RUN_FUNC_UV_YEAR
    else:
        pass
    img_type = 'pdf'
    time_str = time.ctime()
    path_out = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'uv_advance')
    if args.img_fmt in ['eps', 'png', 'pdf', 'svg', 'ps']:
        img_type = args.img_fmt
    path_dir_sky = os.path.join(path_out, "sky-uv-{}.{}".format(time_str, img_type))
    path_dir_year = os.path.join(path_out, "year-uv-{}.{}".format(time_str, img_type))
    path_dir_srcs_all = os.path.join(path_out, "multi-uv-all-{}.{}".format(time_str, img_type))
    if args.show_funcs:
        print(_design_idea)

    # 3. invoke the calculation functions
    cutoff_dict = {"flag": lc.cutoff_mode["flag"], "CutAngle": my_config_parser.cutoff_angle}
    myFuncUvMore = FuncUvMore(start_time, stop_time, time_step,
                              my_config_parser.pos_mat_src[0],
                              my_config_parser.pos_mat_src,
                              my_config_parser.pos_mat_sat,
                              my_config_parser.pos_mat_vlbi,
                              my_config_parser.pos_mat_telemetry,
                              my_config_parser.obs_freq,
                              my_config_parser.baseline_type,
                              my_config_parser.unit_flag,
                              cutoff_dict,
                              my_config_parser.precession_mode,
                              run_type)
    myFuncUvMore.run_uv_more()

    # 4. draw sky uv
    if run_type & RUN_FUNC_UV_SKY == RUN_FUNC_UV_SKY:
        run_result_sky_u, run_result_sky_v, run_result_sky_max_range = myFuncUvMore.get_run_result_sky()
        if len(run_result_sky_u) != 0 and len(run_result_sky_v) != 0:
            plt.figure()
            k = 0
            for i in (2, 6, 10, 14, 18, 22):
                for j in (-60, -30, 0, 30, 60):
                    if len(run_result_sky_u[k]) > 0 and len(run_result_sky_v[k]) > 0:
                        temp_u = np.array(run_result_sky_u[k]) / run_result_sky_max_range
                        temp_v = np.array(run_result_sky_v[k]) / run_result_sky_max_range * 10
                        temp_u += i
                        temp_v += j
                        plt.scatter(temp_u, temp_v, s=3, marker='.', color='b')
                    k += 1
            # plot sun position
            sun_ra, sun_dec = me.sun_ra_dec_cal(start_time, stop_time, time_step)
            plt.plot(np.array(sun_ra), np.array(sun_dec), '.k', linewidth=2)
            plt.plot(sun_ra[0], sun_dec[0], 'or', alpha=0.5, markersize=20)
            # ticks
            plt.title("All Sky UV Plot")
            plt.xlabel(r"Ra($H$)")
            plt.ylabel(r'Dec ($^\circ$)')
            plt.xticks([0, 2, 6, 10, 14, 18, 22, 24])
            plt.yticks([-90, -60, -30, 0, 30, 60, 90])
            plt.xlim(0, 24)
            plt.ylim(-90, +90)
            plt.grid()
            plt.savefig(path_dir_sky)

    # 5. draw year uv
    if run_type & RUN_FUNC_UV_YEAR == RUN_FUNC_UV_YEAR:
        run_result_time_u, run_result_time_v, run_result_time_max_range = myFuncUvMore.get_run_result_year()
        if len(run_result_time_u) != 0 and len(run_result_time_v) != 0:
            plt.figure()
            k = 0
            for irow in (21, 15, 9, 3):
                for icol in (20, 12, 4):
                    if len(run_result_time_u[k]) > 0 and len(run_result_time_v[k]) > 0:
                        temp_u = np.array(run_result_time_u[k]) / run_result_time_max_range * 4
                        temp_v = np.array(run_result_time_v[k]) / run_result_time_max_range * 3
                        temp_u += icol
                        temp_v += irow
                        plt.scatter(temp_u, temp_v, s=3, marker='.', color='b')
                    k += 1
            plt.title("All Year Round UV Plot")
            plt.xlim(0, 24)
            plt.ylim(0, 24)
            plt.xticks([4, 12, 20], [1, 2, 3])
            plt.yticks([3, 9, 15, 21], [4, 3, 2, 1])
            plt.xlabel(r"$i_{th}$\ month")
            plt.ylabel(r"Quarter")
            plt.grid()
            plt.savefig(path_dir_year)

    # 6. draw multisrc
    num_src = 0
    if run_type & RUN_FUNC_UV_SRCS == RUN_FUNC_UV_SRCS:
        run_result_src_name, run_result_src_u, run_result_src_v, run_result_mrange_src, run_result_src_max_range = myFuncUvMore.get_run_result_srcs()
        num_src = len(run_result_src_name)
        if not args.separate_srcs:
            if num_src > 0:
                plt.figure()
                num_col = int(np.ceil(np.sqrt(num_src)))
                num_row = int(np.ceil(num_src / num_col))
                # num_col, num_row= 4, 7
                for k in range(num_src):
                    plt.subplot(num_row, num_col, k + 1, aspect='equal')
                    if len(run_result_src_u[k]) > 0 and len(run_result_src_v[k]) > 0:
                        plt.scatter(run_result_src_u[k], run_result_src_v[k], s=1, marker='.', color='brown')
                        plt.xlim([-run_result_src_max_range, run_result_src_max_range])
                        plt.ylim([-run_result_src_max_range, run_result_src_max_range])
                    plt.title(run_result_src_name[k])
                    # science
                    ax = plt.gca()
                    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
                    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))

                plt.xlabel('u')
                plt.ylabel('v')
                plt.tight_layout()
                plt.savefig(path_dir_srcs_all)
        else:
            if num_src > 0:
                for k in range(num_src):
                    if len(run_result_src_u[k]) > 0 and len(run_result_src_v[k]) > 0:
                        plt.figure()
                        plt.scatter(run_result_src_u[k], run_result_src_v[k], s=1, marker='.', color='brown')
                        plt.xlim([-run_result_src_max_range, run_result_src_max_range])
                        plt.ylim([-run_result_src_max_range, run_result_src_max_range])
                        plt.title(run_result_src_name[k])
                        plt.xlabel('u')
                        plt.ylabel('v')
                        temp_dir_src = os.path.join(path_out, "multi-uv-{}-{}.{}".format(run_result_src_name[k], time_str, img_type))
                        # science
                        ax = plt.gca()
                        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
                        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
                        plt.savefig(temp_dir_src)

    # 7.show uv img
    if args.show_gui:
        if args.separate_srcs:
            if run_type & RUN_FUNC_UV_SRCS == RUN_FUNC_UV_SRCS and num_src > 5:
                print("There are too many figures, please check the output file instead of show them directly")
            else:
                plt.show()
        else:
            plt.show()


# for comparision purpose: run with single process
def run_uv_advanced_single_process(is_show_gui=True):
    # 1. initialize parse and config objects
    args = parse_args()
    if args.config != '':
        my_config_parser = UVConfigParser(args.config)
    else:
        my_config_parser = UVConfigParser()

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
    # do calculations
    run_time_start = time.time()
    run_result_sky_u, run_result_sky_v, run_result_sky_max_range = myFuncUV.get_result_sky_uv_with_update()
    run_result_src_name, run_result_src_u, run_result_src_v, run_result_src_max_range = myFuncUV.get_result_multi_src_with_update()
    run_time_end = time.time()

    # draw sky uv
    plt.figure()
    if len(run_result_sky_u) != 0 and len(run_result_sky_v) != 0:
        k = 0
        # print(len(result_mat_u))
        for i in (2, 6, 10, 14, 18, 22):
            for j in (-60, -30, 0, 30, 60):
                if len(run_result_sky_u[k]) > 0 and len(run_result_sky_v[k]) > 0:
                    temp_u = np.array(run_result_sky_u[k]) / run_result_sky_max_range
                    temp_v = np.array(run_result_sky_v[k]) / run_result_sky_max_range * 10
                    temp_u += i
                    temp_v += j
                    plt.scatter(temp_u, temp_v, s=3, marker='.', color='b')
                k += 1

        # plot sun position
        sun_ra, sun_dec = me.sun_ra_dec_cal(start_time, stop_time, time_step)
        plt.plot(np.array(sun_ra), np.array(sun_dec), '.k', linewidth=2)
        plt.plot(sun_ra[0], sun_dec[0], 'or', alpha=0.5, markersize=20)
        # ticks
        plt.title("All Sky UV Plot")
        plt.xlabel(r"Ra($H$)")
        plt.ylabel(r'Dec ($^\circ$)')
        plt.xticks([0, 2, 6, 10, 14, 18, 22, 24])
        plt.yticks([-90, -60, -30, 0, 30, 60, 90])
        plt.xlim(0, 24)
        plt.ylim(-90, +90)
        plt.grid()

    # draw multi src
    plt.figure()
    num_src = len(run_result_src_name)
    if num_src > 0:
        num_col = int(np.ceil(np.sqrt(num_src)))
        num_row = int(np.ceil(num_src/num_col))
        # num_col, num_row= 4, 7
        for k in range(num_src):
            plt.subplot(num_row, num_col, k + 1, aspect='equal')
            if len(run_result_src_u[k]) > 0 and len(run_result_src_v[k]) > 0:
                plt.scatter(run_result_src_u[k], run_result_src_v[k], s=1, marker='.', color='brown')
                plt.xlim([-run_result_src_max_range, run_result_src_max_range])
                plt.ylim([-run_result_src_max_range, run_result_src_max_range])
            plt.title(run_result_src_name[k])
        plt.xlabel('u')
        plt.ylabel('v')
        plt.tight_layout()
    if is_show_gui:
        plt.show()

    return run_time_end - run_time_start


def test_accelerate_result():
    run_uv_advanced()
    run_time_single_process = run_uv_advanced_single_process(is_show_gui=False)
    print("The time cost by single process: ", run_time_single_process)


if __name__ == "__main__":
    run_uv_advanced()
    # test_accelerate_result()
