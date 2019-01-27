"""
@functions: visibility amplitude versus projected UV distance
@author: Zhen ZHAO
@date: May 19, 2018
"""
import tkinter as tk
from tkinter import messagebox

import load_conf as lc
import astropy.io.fits as pf
import numpy as np
import cmath
import os
import argparse
import time
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

# Data of the input file is instored in the following format:
# ['UU---SIN', 'VV---SIN', 'WW---SIN', 'DATE', '_DATE', 'BASELINE', 'INTTIM', 'GATEID', 'CORR-ID', 'DATA']


class FuncRadPlot(object):
    def __init__(self, file):
        self.file_name = file
        obs_fits_dir = os.path.join(os.getcwd(), 'OBSERVE_DATA')
        self.fits_file = os.path.join(obs_fits_dir, str(self.file_name))

        self.plot_u = None
        self.plot_v = None
        self.baseline = None
        self.vis = None

        self.data_state = self._extract_rad_plot_data()

    def get_data_state(self):
        if self.data_state == 0:
            return True
        return False

    def reset_file(self, file):
        self.__init__(file)

    def _extract_rad_plot_data(self):
        if len(self.file_name) == 0:
            # print("\n\nWrong input!!\n\n")
            # print("info", tk.messagebox.showinfo("About", "Wrong input!!"))
            # return False
            return 1
        else:
            if not os.path.exists(self.fits_file):
                # print("\n\nModel file %s does not exist!\n\n" % fits_file)
                # print("info", tk.messagebox.showinfo("About", "Model file %s does not exist!" % self.fits_file))
                # return False
                return 2
            else:
                hdu_lst = pf.open(self.fits_file)
                PSCAL2 = hdu_lst[0].header['PSCAL2']
                data_in = hdu_lst[0].data
                uu = data_in['UU---SIN'] / PSCAL2 / 1e6
                temp_u = list(uu)
                temp_u.extend(list(-uu))
                self.plot_u = np.array(temp_u)
                vv = data_in['VV---SIN'] / PSCAL2 / 1e6
                temp_v = list(vv)
                temp_v.extend(list(-vv))
                self.plot_v = np.array(temp_v)

                DATA = data_in['DATA']
                vis_re = DATA[:, 0, 0, 0, 0, 0, 0]
                vis_im = DATA[:, 0, 0, 0, 0, 0, 1]
                self.vis = vis_re + vis_im * cmath.sqrt(-1)
                self.baseline = np.sqrt(uu ** 2 + vv ** 2)

                return 0

    def get_result_uv_data(self):
        if self.data_state == 0:
            return self.plot_u, self.plot_v
        elif self.data_state == 1:
            print("info", tk.messagebox.showinfo("About", "Wrong input!!"))
        elif self.data_state == 2:
            print("info", tk.messagebox.showinfo("About", "Model file %s does not exist!" % self.fits_file))

    def get_result_rad_data(self):
        if self.data_state == 0:
            return self.baseline, self.vis
        elif self.data_state == 1:
            print("info", tk.messagebox.showinfo("About", "Wrong input!!"))
        elif self.data_state == 2:
            print("info", tk.messagebox.showinfo("About", "Model file %s does not exist!" % self.fits_file))

    def test_rad_plot(self):
        if self.data_state == 0:
            plt.figure(num=1)
            plt.plot(self.plot_u, self.plot_v, 'ko', markersize=1)
            max_u = max(np.abs(self.plot_u))
            max_v = max(np.abs(self.plot_v))
            plt.xlim(-max_u, max_u)
            plt.ylim(-max_v, max_v)
            plt.title('UV PLOT')
            plt.xlabel("U (m)")
            plt.xlabel("V (m)")

            plt.figure(num=2)
            plt.plot(self.baseline, abs(self.vis), 'ko', markersize=2)
            plt.xlabel("UV Distance")
            plt.ylabel("Visibility Amplitude")
            plt.title('RAD PLOT')

        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Read in the data record in fits format and show you uvplot and radplot")
    parser.add_argument('-f',
                        '--fits_file',
                        default='0106+013_1.fits',
                        help='Specify the data file you want to analyze (put it under ./OBSERVE_DATA/*)')
    parser.add_argument('-g',
                        '--show_gui',
                        action="store_true",
                        help='Choose to show the UV Plot and Rad Plot')

    parser.add_argument('-s',
                        '--save_data',
                        action="store_true",
                        help='Store the uv and vis data in txt(/OUTPUT/data_analyze/*.txt)')

    parser.add_argument('-t',
                        '--img_type',
                        choices=['eps', 'png', 'pdf', 'svg', 'ps'],
                        help='Specify the img format (default:pdf)',
                        default='pdf')

    return parser.parse_args()


def run_radplot():
    args = parse_args()
    # for test in ide
    # args.show_gui = True
    # fits data
    if args.fits_file != '':
        myFuncRad = FuncRadPlot(args.fits_file)
    else:
        myFuncRad = FuncRadPlot("0106+013_1.fits")

    if not myFuncRad.get_data_state():
        print("The file is not in legal format")
        return

    # output file path
    img_type = 'pdf'
    if args.img_type in ['eps', 'png', 'pdf', 'svg', 'ps']:
        img_type = args.img_type
    save_uv_plot = "uv-plot:" + time.asctime() + '.' + img_type
    path_uv_plot = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'data_analyze')
    path_uv_plot = os.path.join(path_uv_plot, save_uv_plot)

    save_rad_plot = "rad-plot:" + time.asctime() + '.' + img_type
    path_rad_plot = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'data_analyze')
    path_rad_plot = os.path.join(path_rad_plot, save_rad_plot)

    # do calculation
    data_u, data_v = myFuncRad.get_result_uv_data()
    data_bl, data_vis = myFuncRad.get_result_rad_data()

    # save data into csv
    if args.save_data and myFuncRad.get_data_state():
        name = "u-v:" + time.asctime() + '.txt'
        rad_path = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'data_analyze')
        rad_path = os.path.join(rad_path, name)
        # np.savetxt(rad_path, [data_u, data_v, data_bl, np.abs(data_vis)])
        np.savetxt(rad_path, [data_u, data_v])
        f_name = "bl-vis:" + time.asctime() + '.txt'
        f_rad_path = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'data_analyze')
        f_rad_path = os.path.join(f_rad_path, f_name)
        np.savetxt(f_rad_path, [data_bl, np.abs(data_vis)])

    # draw figs
    if myFuncRad.get_data_state():
        # draw uv
        plt.figure(num=1, figsize=(8, 8))
        ax = plt.subplot(111, aspect='equal')
        plt.plot(data_u, data_v, 'ko', markersize=1)
        max_uv = max(max(np.abs(data_u)), max(np.abs(data_v)))
        plt.xlim(-max_uv, max_uv)
        plt.ylim(-max_uv, max_uv)
        plt.title('UV PLOT')
        plt.xlabel("U (m)")
        plt.xlabel("V (m)")
        # set science
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        plt.savefig(path_uv_plot)

        # draw rad
        plt.figure(num=2)
        plt.plot(data_bl, abs(data_vis), 'ko', markersize=2)
        plt.xlabel("UV Distance")
        plt.ylabel("Visibility Amplitude")
        plt.title('RAD PLOT')
        plt.savefig(path_rad_plot)

    if args.show_gui:
        plt.show()


if __name__ == "__main__":
    run_radplot()
