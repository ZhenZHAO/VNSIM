"""
@functions: parameter calculation
@author: Zhen ZHAO
@date: Dec 20, 2018
"""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from collections import OrderedDict
import argparse

_copyright_ = 'Copyright \N{COPYRIGHT SIGN} 2018, Zhen Zhao'


class ParaCal(object):
    def __init__(self, window=None):
        # # # # # # # # # # # # # # # #
        # 0. UI parameters
        # 1)stations
        teleLst = ['Hn', 'Fd', 'La', 'Kp', 'Pt', 'Ov', 'Br', 'Nl', 'Tm65',
                   'My', 'Mk', 'Sv', 'Zc', 'Bd', 'Wz', 'Ka', 'W1', 'Ro34',
                   'Nt', 'Sh', 'Ny', 'Mc', 'Mh', 'Ys', 'Wb', 'Cm', 'Ro70',
                   'Pv', 'Hh', 'Mp', 'Km', 'Ku', 'Ky', 'Kt', 'Y1', 'Y27',
                   'Pa', 'Ho', 'Cd', 'Ap', 'Go', 'Gb', 'Ar', 'Jb1', 'Jb2',
                   'Ef', 'Ur', 'On', 'Tr', 'At', 'Sr', 'Pb', 'Sc', 'ALMA',
                   'SAT1', 'SAT2', 'FAST', 'SKA1-mid', 'SKA1-low', 'Ir',
                   'VERAIR', 'VERAIS', 'VERAMZ', 'VERAOG', 'NRO45']
        teleLst.sort()
        # self.teleDict = {x: 0 for x in teleLst}
        self.teleDict = OrderedDict().fromkeys(teleLst, 0)
        # print(self.teleDict)

        # 2)observed band
        self.obevBandDict = {'P - 92cm': '92cm',
                             'P - 49cm': '49cm',
                             'UHF - 30cm': '30cm',
                             'L - 21cm': '21cm',
                             'L - 18cm': '18cm',
                             'S - 13cm': '13cm',
                             'C - 6cm': '6cm',
                             'C - 5cm': '5cm',
                             'X - 3.6cm': '3.6cm',
                             'U - 2cm': '2cm',
                             'K - 1.3cm': '1.3cm',
                             'Ka - 9mm': '9mm',
                             'Q - 7mm': '7mm',
                             'W - 3mm': '3mm'}
        self.obevBandList = list(self.obevBandDict.keys())
        # 3)data rate
        self.datarateList = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]
        # 4)channel number
        self.channelNumberDist = {'8192 ch': '8192', '4096 ch': '4096', '2048 ch': '2048', '1024 ch': '1024',
                                  '512 ch': '512', '256 ch': '256', '128 ch': '128',
                                  '64 ch': '64', '32 ch': '32', '16 ch': '16'}
        self.channelNumberList = list(self.channelNumberDist.keys())
        # 5)integration time
        self.integrateTimeDist = {'16 s': '16', '8 s': '8', '4 s': '4', '2 s': '2', '1 s': '1', '1/2 s': '0.5',
                                  '1/4 s': '0.25', '1/8 s': '0.125', '60 ms': '0.060',
                                  '30 ms': '0.030', '15 ms': '0.015'}
        self.integrateTimeList = list(self.integrateTimeDist.keys())
        # 6)polarizations number
        self.polarNumberDist = {'4 pols': '4', '2 pols': '2', '1 pol': '1'}
        self.polarNumberList = list(self.polarNumberDist.keys())
        # 7)subband number
        self.subBandNumberDist = {'16 sb': '16', '8 sb': '8', '4 sb': '4', '2 sb': '2', '1 sb': '1'}
        self.subBandNumberList = list(self.subBandNumberDist.keys())
        # 8)subband bandwidth
        self.subBandBandwidthDist = {'128 MHz': '128', '64 MHz': '64', '32 MHz': '32', '16 MHz': '16', '8 MHz': '8',
                                     '4 MHz': '4', '2 MHz': '2', '1 MHz': '1', '0.5 MHz': '0.5'}
        self.subBandBandwidthList = list(self.subBandBandwidthDist.keys())
        # 9)baseline length
        self.baselineLenDist = {'12000 km (EVN+VLBA)': '12000.0', '10000 km (Full EVN)': '10000.0',
                                '9000 km (VLBA)': '9000.0',
                                '5000 km': '5000.0', '2500 km (Western EVN)': '2500.0', '1000 km': '1000.0'}
        self.baselineLenList = list(self.baselineLenDist.keys())
        # # # # # # # # # # # # # # # #
        # 1. main window
        # moving the window to specified position
        center_window(window, 750, 550)
        # # # # # # # # # # # # # # # #
        # 2. framework #flat, groove, raised, ridge, solid, or sunken
        tk.Label(window, text='   Configuration   ', relief='ridge', bg='lightblue', font=('Arial', 18)).pack(
            side='top', anchor='w', padx=5)
        frm_config = tk.Frame(window, border=5)
        frm_config.pack(side='top', anchor='w')
        tk.Label(window, text='   Calculation   ', relief='ridge', bg='lightblue', font=('Arial', 18)).pack(side='top',
                                                                                                            anchor='w',
                                                                                                            padx=5)
        frm_run = tk.Frame(window)
        frm_run.pack(side='top', anchor='w')

        frm_result = tk.Frame(window, border=1)
        frm_result.pack(side='top', anchor='w')

        # # # # # # # # # # # # # # # #
        # 3. config frame
        config_left_frm = tk.Frame(frm_config, border=1)
        config_left_frm.grid(row=0, column=0, padx=5)

        iRow, iColumn, showNum = 0, 0, 0
        for teleStation in self.teleDict:
            self.teleDict[teleStation] = tk.IntVar()
            tmp_str = str(teleStation).ljust(5)
            tk.Checkbutton(config_left_frm, text=tmp_str, variable=self.teleDict[teleStation],
                           onvalue=1, offvalue=0).grid(row=iRow, column=iColumn, ipadx=2, pady=1)
            showNum += 1
            iRow = showNum // 9
            iColumn = showNum % 9

        config_right_frm = tk.Frame(frm_config, border=3)
        config_right_frm.grid(row=1, column=0)

        # waveband and datarate
        self.waveband = tk.StringVar(window, 'L - 18cm')
        tk.Label(config_right_frm, text='Observe Band:').grid(row=0, column=0)
        ttk.Combobox(config_right_frm, textvariable=self.waveband, values=self.obevBandList, width='12').grid(row=0,
                                                                                                              column=1)

        self.datarate = tk.StringVar(window, self.datarateList[1])
        tk.Label(config_right_frm, text='Data Rate(Mbps):').grid(row=0, column=2)
        ttk.Combobox(config_right_frm, textvariable=self.datarate, values=self.datarateList, width='18').grid(row=0,
                                                                                                              column=3)

        # spectrum channel number and integration time
        self.specChNum = tk.StringVar(window, '16 ch')
        tk.Label(config_right_frm, text='Channel Num:').grid(row=1, column=0)
        ttk.Combobox(config_right_frm, textvariable=self.specChNum, values=self.channelNumberList, width='12').grid(
            row=1, column=1)

        self.integrateTime = tk.StringVar(window, '2 s')
        tk.Label(config_right_frm, text='Integrate Time:').grid(row=1, column=2)
        ttk.Combobox(config_right_frm, textvariable=self.integrateTime, values=self.integrateTimeList, width='18').grid(
            row=1, column=3)

        # Number of polarizations, bandwidth of a subband [MHz]
        self.polarNum = tk.StringVar(window, '2 pols')
        tk.Label(config_right_frm, text='Polarization Num:').grid(row=2, column=0)
        ttk.Combobox(config_right_frm, textvariable=self.polarNum, values=self.polarNumberList, width='12').grid(row=2,
                                                                                                                 column=1)

        self.subBandBandwidth = tk.StringVar(window, '16 MHz')
        tk.Label(config_right_frm, text='Subband BW:').grid(row=2, column=2)
        ttk.Combobox(config_right_frm, textvariable=self.subBandBandwidth, values=self.subBandBandwidthList,
                     width='18').grid(row=2, column=3)

        # subbands per polarizations and baseline
        self.subBandNum = tk.StringVar(window, '8 sb')
        tk.Label(config_right_frm, text='Subband Num:').grid(row=3, column=0)
        ttk.Combobox(config_right_frm, textvariable=self.subBandNum, values=self.subBandNumberList, width='12').grid(
            row=3,
            column=1)

        self.baselineLen = tk.StringVar(window, '10000 km (Full EVN)')
        tk.Label(config_right_frm, text='Baseline Len:').grid(row=3, column=2)
        ttk.Combobox(config_right_frm, textvariable=self.baselineLen, values=self.baselineLenList, width='18').grid(
            row=3,
            column=3)

        # On-source integration time [min]
        self.onSourceTime = tk.StringVar(window, '150')
        self.onSourceTime.trace('w', self.lmt_input_size)

        self.testCMD = window.register(test)
        tk.Label(config_right_frm, text='On-Source Time [min]:').grid(row=4, column=1)
        tk.Entry(config_right_frm, bg="#282B2B", fg="white", width=12, textvariable=self.onSourceTime,
                 validate="key", validatecommand=(self.testCMD, '%P')).grid(row=4, column=2)
        # # # # # # # # # # # # # # # #
        # 4. Run frame
        # 4.1 parameter definition and value-obtain
        self.selectedStationArray = []
        self.missTelescopeArray = []
        self.stationNum = 0
        self.SEFD = {}
        self.wavelength = 0.0
        self.dRate, self.tObs = 0.0, 0.0
        self.stationEffectNum, self.specNum, self.subNum, self.polNum = 0, 0, 0, 0
        self.baseLen, self.subBW, self.tInt = 0.0, 0.0, 0.0
        self.trueDatarate = 0.0
        self.parNum, self.crossNum = 0, 0

        btn_run = tk.Button(frm_run, text='RUN', width=10, height=2, command=self.run_calculation)
        btn_run.pack(side='left', padx=100)

        btn_reset = tk.Button(frm_run, text='RESET', width=10, height=2, command=self.reset_all)
        btn_reset.pack(side='left', padx=30)

        # # # # # # # # # # # # # # # #
        # 5. Result frame
        frm_result_top = tk.Frame(frm_result)
        frm_result_top.pack(side='top', anchor='w')

        self.thermalNoise = tk.StringVar(window, "")
        tk.Label(frm_result_top, text="Thermal Noise:").grid(row=0, column=0)
        self.noiseOutWin = tk.Label(frm_result_top, textvariable=self.thermalNoise, width=20, bg='lightblue',
                                    fg='white')
        self.noiseOutWin.grid(row=0, column=1, pady=5)

        self.fitsFile = tk.StringVar(window, "")
        tk.Label(frm_result_top, text="FITS File Size:").grid(row=1, column=0)
        self.fitsCapOutWin = tk.Label(frm_result_top, textvariable=self.fitsFile, width=20, bg='lightblue', fg='white')
        self.fitsCapOutWin.grid(row=1, column=1)

        self.bwFOV = tk.StringVar(window, "")
        tk.Label(frm_result_top, text="Bandwidth Smearing:").grid(row=0, column=2)
        self.bwFovOutWin = tk.Label(frm_result_top, textvariable=self.bwFOV, width=20, bg='lightblue', fg='white')
        self.bwFovOutWin.grid(row=0, column=3)

        self.tmFOV = tk.StringVar(window, "")
        tk.Label(frm_result_top, text="Time Smearing:").grid(row=1, column=2)
        self.tmFovOutWin = tk.Label(frm_result_top, textvariable=self.tmFOV, width=20, bg='lightblue', fg='white')
        self.tmFovOutWin.grid(row=1, column=3)

        # self.debugInfo = tk.StringVar()
        # self.debugView = tk.Label(window, textvariable=debugInfo, bg='red', font=('Arial',12),width=600,height=2)
        # self.debugView.pack(side='top', anchor='w')

        # 6. show window
        label = tk.Label(window, text=_copyright_, bd=1, relief='sunken',
                         anchor='e')
        label.pack(side='bottom', fill='x')

        window.mainloop()

    def lmt_input_size(self, *args):
        value = self.onSourceTime.get()
        if len(value) > 4:
            self.onSourceTime.set(value[:5])

    def obtain_input(self):
        # 1. SEFD and wavelength
        self.SEFD = {}
        self.wavelength = 0.0
        temp_wb = self.obevBandDict[self.waveband.get()]
        if temp_wb == '92cm':
            self.SEFD = _band_92cm
            self.wavelength = 92.0
        elif temp_wb == '49cm':
            self.SEFD = _band_49cm
            self.wavelength = 49.0
        elif temp_wb == '30cm':
            self.SEFD = _band_UFH
            self.wavelength = 30.0
        elif temp_wb == '21cm':
            self.SEFD = _band_21cm
            self.wavelength = 21.0
        elif temp_wb == '18cm':
            self.SEFD = _band_18cm
            self.wavelength = 18.0
        elif temp_wb == '13cm':
            self.SEFD = _band_13cm
            self.wavelength = 13.0
        elif temp_wb == '6cm':
            self.SEFD = _band_6cm
            self.wavelength = 6.0
        elif temp_wb == '5cm':
            self.SEFD = _band_5cm
            self.wavelength = 5.0
        elif temp_wb == '3.6cm':
            self.SEFD = _band_4cm
            self.wavelength = 3.6
        elif temp_wb == '2cm':
            self.SEFD = _band_2cm
            self.wavelength = 2.0
        elif temp_wb == '1.3cm':
            self.SEFD = _band_13mm
            self.wavelength = 1.3
        elif temp_wb == '9mm':
            self.SEFD = _band_9mm
            self.wavelength = 0.9
        elif temp_wb == '7mm':
            self.SEFD = _band_7mm
            self.wavelength = 0.7
        else:  # 3mm
            self.SEFD = _band_3mm
            self.wavelength = 0.3

        # 2. stations: selectedStationArray, stationNum, missTelescopeArray
        self.selectedStationArray = []
        self.missTelescopeArray = []
        for teleStation in self.teleDict:
            if self.teleDict[teleStation].get() != 0:
                self.selectedStationArray.append(teleStation)
                if self.SEFD[teleStation] == -1:
                    self.missTelescopeArray.append(teleStation)
        self.stationNum = len(self.selectedStationArray)

        # 3. drate, Tobs
        self.dRate = float(self.datarate.get())
        self.tObs = float(self.onSourceTime.get()) * 60

        # 4. stationEffectNum, specNum, subNum, polNum
        self.specNum = int(self.channelNumberDist[self.specChNum.get()])
        self.subNum = int(self.subBandNumberDist[self.subBandNum.get()])
        self.polNum = int(self.polarNumberDist[self.polarNum.get()])

        # 5. baseLen, subBW, tInt
        self.baseLen = float(self.baselineLenDist[self.baselineLen.get()])
        self.subBW = float(self.subBandBandwidthDist[self.subBandBandwidth.get()])
        self.tInt = float(self.integrateTimeDist[self.integrateTime.get()])

        if self.stationNum % 4 == 0:
            self.stationEffectNum = int(self.stationNum / 4) * 4.0
        else:
            self.stationEffectNum = int(1 + self.stationNum / 4) * 4.0

        # 6. parNum, crossNum
        if self.polNum == 1:
            self.parNum, self.crossNum = 1, 1
        else:
            self.parNum, self.crossNum = 2, 1
        if self.polNum == 4:
            self.crossNum = 2

        # 7. trueDatarate, $Nsb*$Nparpol*$BWsb*4;
        self.trueDatarate = self.subNum * self.parNum * self.subBW * 4

    def calculate(self):
        # selectedStationArray, missTelescopeArray, stationNum, SEFD, wavelength
        # dRate, tObs, stationEffectNum, specNum, subNum, polNum,baseLen, subBW, tInt
        m, sum1, sum2 = 2, 0, 0
        for tel1 in self.selectedStationArray:
            for tel2 in self.selectedStationArray:
                if tel1 != tel2:
                    t1 = self.SEFD[tel1]
                    t2 = self.SEFD[tel2]
                    sum1 += (t1 * t2) ** (1 - m)
                    sum2 += (t1 * t2) ** (-m / 2)

        sum1 *= 0.5
        sum2 *= 0.5
        mean_sefd = sum1 ** (1 / 2) / sum2
        return 1000 * 1.43 * mean_sefd / ((self.dRate * 1000000.0 / 2.0 * self.tObs) ** (1 / 2))

    def noise_calculation(self):
        unit = ' mJy'
        err_msg = ''
        if len(self.missTelescopeArray) > 0:
            err_msg = 'There are no receivers in this band (or SEFD is not ' \
                      'yet available) at following stations: ' + str(self.missTelescopeArray)
            return 'N/A', err_msg
        elif self.tObs <= 0:
            err_msg = 'Please specify a reasonable observation time'
            return 'N/A', err_msg
        else:
            if self.stationNum == 1:
                noise = self.SEFD[self.selectedStationArray[0]]
                unit = ' Jy'
            else:
                noise = self.calculate()
                if noise > 100:
                    unit = " Jy"
                    noise /= 1000.0
                elif noise < 0.1:
                    unit = " uJy"
                    noise *= 1000.0
                if self.stationNum == 2:
                    unit += '(1 sigma)'
                else:
                    unit += '/beam'
                    if self.dRate != self.trueDatarate:
                        err_msg = "Warning: the total data rate " + str(self.dRate) \
                                  + "Mbps does not math the subBand Bandwidth setting"
        return '{:.3f}'.format(noise) + unit, err_msg

    def fov_bw_calculation(self):
        fov_bw = 49500.0 * self.specNum / (self.baseLen * self.subBW)
        unit = ' arcsec'
        if fov_bw >= 60.0:
            fov_bw /= 60.0
            unit = ' arcmin'
        err_msg = 'We assuming {} km for the maximum baseline, '.format(self.baseLen)
        return '{:.3f}'.format(fov_bw) + unit, err_msg

    def fov_tm_calculation(self):
        fov_tm = 18560.0 * self.wavelength / (self.baseLen * self.tInt)
        unit = ' arcsec'
        if fov_tm >= 60.0:
            fov_tm /= 60.0
            unit = ' arcmin'
        err_msg = 'and the smearing values are calculated for 10% loss in the response of a point source.'
        return '{:.3f}'.format(fov_tm) + unit, err_msg

    def capacity_calculation(self):
        unit = ' GB'
        err_msg = ''

        if self.parNum * self.subNum > 16:
            err_msg = "Warning: The number of subbands*polarizations exceeds 16. This has to be correlated in multiple passes.Decrease the number of subbands or polarizations to see the results for a single pass."
            return "N/A", err_msg
        elif self.stationNum > 16:
            err_msg = "Warning: More than 16 stations. This has to be correlated in multiple passes.Decrease the number of subbands or polarizations to see the results for a single pass."
            return "N/A", err_msg
        else:
            corr_usage_1 = (
                                   self.stationNum * self.stationNum * self.parNum * self.crossNum * self.subNum * self.specNum) / (
                           131072.0)
            cor_cap = 1.75 * corr_usage_1 * (self.tObs / 3600.0) / self.tInt
            if cor_cap < 1.0:
                cor_cap *= 1000.0
                unit = ' MB'
            return '{:.3f}'.format(cor_cap) + unit, err_msg

    def run_calculation(self):
        # selectedStationArray, missTelescopeArray, stationNum, SEFD, wavelength
        # dRate, tObs, stationEffectNum, specNum, subNum, polNum,baseLen, subBW, tInt
        self.obtain_input()
        # print(self.dRate, self.tInt, self.subBW, self.baseLen, self.tObs)
        # print(self.stationNum, self.stationEffectNum, self.wavelength, self.specNum, self.polNum, self.subNum)

        # reset result
        self.tmFOV.set("")
        self.bwFOV.set("")
        self.fitsFile.set("")
        self.thermalNoise.set("")

        # format cleaning
        self.noiseOutWin.config(bg='lightblue')
        self.bwFovOutWin.config(bg='lightblue')
        self.tmFovOutWin.config(bg='lightblue')
        self.fitsCapOutWin.config(bg='lightblue')

        # start calculation
        if self.stationNum > 0:
            noise_set, err_noise = self.noise_calculation()
            self.thermalNoise.set(noise_set)
            if noise_set == 'N/A':
                self.noiseOutWin.config(bg='red')

            if self.stationNum < 3:
                self.bwFOV.set('N/A')
                self.bwFovOutWin.config(bg='red')
                self.tmFOV.set('N/A')
                self.tmFovOutWin.config(bg='red')
                self.fitsFile.set('N/A')
                self.fitsCapOutWin.config(bg='red')
                messagebox.showinfo(title="Warning",
                                    message=err_noise + "\n\nNote: Please select a station array (N>2) if you wanna see the 'smearing' information")
            else:
                fov_bw_set, err_fov_bw = self.fov_bw_calculation()
                self.bwFOV.set(fov_bw_set)
                fov_tm_set, err_fov_tm = self.fov_tm_calculation()
                self.tmFOV.set(fov_tm_set)
                fits_cap_set, err_fits_cap = self.capacity_calculation()
                self.fitsFile.set(fits_cap_set)
                if fits_cap_set == 'N/A':
                    self.fitsCapOutWin.config(bg='red')
                if err_noise != "":
                    if err_fits_cap != "":
                        messagebox.showinfo(title="Warning", message=err_noise + '\n' + err_fits_cap)
                    else:
                        messagebox.showinfo(title="Warning", message=err_noise)
        else:
            messagebox.showinfo(title="Warning", message="Warning:Please select the observation stations!")

    def reset_all(self):
        # reset all UI parameters
        for teleStation in self.teleDict:
            self.teleDict[teleStation].set(0)
        self.waveband.set('L - 18cm')
        self.datarate.set(self.datarateList[1])
        self.specChNum.set('16 ch')
        self.integrateTime.set('2 s')
        self.baselineLen.set('10000 km (Full EVN)')
        self.polarNum.set('2 pols')
        self.subBandNum.set('8 sb')
        self.subBandBandwidth.set('16 MHz')
        self.onSourceTime.set('150')
        # reset calculation parameters
        self.obtain_input()
        # reset result
        self.tmFOV.set("")
        self.bwFOV.set("")
        self.fitsFile.set("")
        self.thermalNoise.set("")
        # format cleaning
        self.noiseOutWin.config(bg='lightblue')
        self.bwFovOutWin.config(bg='lightblue')
        self.tmFovOutWin.config(bg='lightblue')
        self.fitsCapOutWin.config(bg='lightblue')


def center_window(root, width, height):
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    position_y = max((screenheight - height) / 2 - 50, 0)
    size = '%dx%d+%d+%d' % (width, height, (screenwidth + width // 2) / 2, position_y)
    # print(size)
    root.geometry(size)


def test(content):
    return content.isdigit() or content == ""


# pre-known information
_band_92cm = {'Jb1': 132, 'Jb2': -1, 'Cm': -1, 'Wb': 150, 'W1': 2100, 'Ef': 600, 'Mc': -1, 'Nt': -1, 'On': -1, 'Sh': -1,
              'Tm65': -1, 'Ur': 3020, 'Tr': -1, 'Mh': -1, 'Ys': -1, 'Sr': 76, 'Ar': 12, 'Wz': -1, 'Hh': -1, 'My': -1,
              'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 3900,
              'Y27': 167, 'Ro34': -1, 'Go': -1, 'Gb': 35, 'Sc': 2742, 'Hn': 2742, 'Nl': 2742, 'Fd': 2742, 'La': 2742,
              'Kp': 2742, 'Pt': 2742, 'Ov': 2742, 'Br': 2742, 'Mk': 2742, 'Pv': -1, 'Pb': -1, 'At': -1, 'Mp': -1,
              'Pa': -1, 'Ho': -1, 'Cd': -1, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1,
              'SAT1': 2100, 'SAT2': 2100, 'FAST': 12, 'SKA1-mid': 9.4, 'SKA1-low': 4.2, 'Ir':-1,
              'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}
_band_49cm = {'Jb1': 83, 'Jb2': -1, 'Cm': -1, 'Wb': -1, 'W1': 1260, 'Ef': 600, 'Mc': -1, 'Nt': -1, 'On': -1, 'Sh': -1,
              'Tm65': -1, 'Ur': -1, 'Tr': -1, 'Mh': -1, 'Ys': -1, 'Sr': -1, 'Ar': -1, 'Wz': -1, 'Hh': -1, 'My': -1,
              'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': -1, 'Y27': -1,
              'Ro34': -1, 'Go': -1, 'Gb': 24, 'Sc': 2744, 'Hn': 2744, 'Nl': 2744, 'Fd': 2744, 'La': 2744, 'Kp': 2744,
              'Pt': 2744, 'Ov': 2744, 'Br': 2744, 'Mk': 2744, 'Pv': -1, 'Pb': -1, 'At': -1, 'Mp': -1, 'Pa': -1,
              'Ho': -1, 'Cd': -1, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1,
              'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
              'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}
_band_UFH = {'Jb1': 100, 'Jb2': -1, 'Cm': -1, 'Wb': 120, 'W1': 1680, 'Ef': 65, 'Mc': -1, 'Nt': -1, 'On': 900, 'Sh': -1,
             'Tm65': -1, 'Ur': 2400, 'Tr': 2000, 'Mh': -1, 'Ys': -1, 'Sr': -1, 'Ar': 3, 'Wz': -1, 'Hh': -1, 'My': -1,
             'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': -1, 'Y27': -1,
             'Ro34': -1, 'Go': -1, 'Gb': 13, 'Sc': -1, 'Hn': -1, 'Nl': -1, 'Fd': -1, 'La': -1, 'Kp': -1, 'Pt': -1,
             'Ov': -1, 'Br': -1, 'Mk': -1, 'Pv': -1, 'Pb': -1, 'At': -1, 'Mp': -1, 'Pa': -1, 'Ho': -1, 'Cd': -1,
             'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1,
             'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
             'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}
_band_21cm = {'Jb1': 65, 'Jb2': 350, 'Cm': 220, 'Wb': 40, 'W1': 420, 'Ef': 20, 'Mc': 700, 'Nt': 820, 'On': 320,
              'Sh': -1, 'Tm65': 39, 'Ur': 350, 'Tr': 300, 'Mh': -1, 'Ys': -1, 'Sr': 67, 'Ar': 3.5, 'Wz': -1, 'Hh': -1,
              'My': -1, 'Km': -1, 'Sv': 360, 'Zc': 300, 'Bd': 330, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1,
              'Y1': 420, 'Y27': 17.9, 'Ro34': -1, 'Go': -1, 'Gb': 10, 'Sc': 289, 'Hn': 289, 'Nl': 289, 'Fd': 289,
              'La': 289, 'Kp': 289, 'Pt': 289, 'Ov': 289, 'Br': 289, 'Mk': 289, 'Pv': -1, 'Pb': -1, 'At': 68, 'Mp': 240,
              'Pa': 40, 'Ho': 470, 'Cd': 1000, 'Ap': 6000, 'Ku': -1, 'Ky': -1, 'Kt': -1,
              'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
              'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}
_band_18cm = {'Jb1': 65, 'Jb2': 320, 'Cm': 212, 'Wb': 40, 'W1': 420, 'Ef': 19, 'Mc': 700, 'Nt': 784, 'On': 320,
              'Sh': 670, 'Tm65': 39, 'Ur': 270, 'Tr': 300, 'Mh': -1, 'Ys': -1, 'Sr': 67, 'Ar': 3, 'Wz': -1, 'Hh': 450,
              'My': -1, 'Km': -1, 'Sv': 360, 'Zc': 300, 'Bd': 330, 'Ro70': 35, 'Ka': -1, 'Ny': -1, 'ALMA': -1,
              'Y1': 420, 'Y27': 17.9, 'Ro34': -1, 'Go': 49, 'Gb': 10, 'Sc': 314, 'Hn': 314, 'Nl': 314, 'Fd': 314,
              'La': 314, 'Kp': 314, 'Pt': 314, 'Ov': 314, 'Br': 314, 'Mk': 314, 'Pv': -1, 'Pb': -1, 'At': 68, 'Mp': 240,
              'Pa': 40, 'Ho': 470, 'Cd': 1000, 'Ap': 6000, 'Ku': -1, 'Ky': -1, 'Kt': -1,
              'SAT1': 420, 'SAT2': 420, 'FAST': 3, 'SKA1-mid': 1.53, 'SKA1-low': -1, 'Ir':3600,
              'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}
_band_13cm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': 60, 'W1': 840, 'Ef': 300, 'Mc': 400, 'Nt': 770, 'On': 1110,
              'Sh': 800, 'Tm65': 46, 'Ur': 680, 'Tr': -1, 'Mh': 4500, 'Ys': -1, 'Sr': -1, 'Ar': 3, 'Wz': 1250,
              'Hh': 380, 'My': -1, 'Km': 350, 'Sv': 330, 'Zc': 330, 'Bd': 330, 'Ro70': 20, 'Ka': 240, 'Ny': 850,
              'ALMA': -1, 'Y1': 370, 'Y27': 15.8, 'Ro34': 150, 'Go': -1, 'Gb': 12, 'Sc': 347, 'Hn': 347, 'Nl': 347,
              'Fd': 347, 'La': 347, 'Kp': 347, 'Pt': 347, 'Ov': 347, 'Br': 347, 'Mk': 347, 'Pv': -1, 'Pb': -1,
              'At': 106, 'Mp': 530, 'Pa': 30, 'Ho': 650, 'Cd': 400, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1,
              'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
              'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}
_band_6cm = {'Jb1': 80, 'Jb2': 320, 'Cm': 136, 'Wb': 120, 'W1': 840, 'Ef': 20, 'Mc': 170, 'Nt': 260, 'On': 600,
             'Sh': 720, 'Tm65': 26, 'Ur': 200, 'Tr': 220, 'Mh': -1, 'Ys': 160, 'Sr': -1, 'Ar': 5, 'Wz': -1, 'Hh': 795,
             'My': -1, 'Km': -1, 'Sv': 250, 'Zc': 400, 'Bd': 200, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 310,
             'Y27': 13.2, 'Ro34': -1, 'Go': -1, 'Gb': 13, 'Sc': 210, 'Hn': 210, 'Nl': 210, 'Fd': 210, 'La': 210,
             'Kp': 210, 'Pt': 210, 'Ov': 210, 'Br': 210, 'Mk': 210, 'Pv': -1, 'Pb': -1, 'At': 70, 'Mp': 350, 'Pa': 110,
             'Ho': 640, 'Cd': 450, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1,
             'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
             'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}
_band_5cm = {'Jb1': -1, 'Jb2': 300, 'Cm': 410, 'Wb': -1, 'W1': 1600, 'Ef': 25, 'Mc': 840, 'Nt': 1100, 'On': 1500,
             'Sh': 1500, 'Tm65': 26, 'Ur': -1, 'Tr': 400, 'Mh': -1, 'Ys': 160, 'Sr': 50, 'Ar': 5, 'Wz': -1, 'Hh': 680,
             'My': -1, 'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 310,
             'Y27': 13.2, 'Ro34': -1, 'Go': -1, 'Gb': 13, 'Sc': 278, 'Hn': 278, 'Nl': 278, 'Fd': 278, 'La': 278,
             'Kp': 278, 'Pt': 278, 'Ov': 278, 'Br': 278, 'Mk': 278, 'Pv': -1, 'Pb': -1, 'At': 70, 'Mp': 350, 'Pa': 110,
             'Ho': 640, 'Cd': 450, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1,
             'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
             'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}
_band_4cm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': 120, 'W1': 1680, 'Ef': 20, 'Mc': 320, 'Nt': 770, 'On': 1000,
             'Sh': 800, 'Tm65': 48, 'Ur': 480, 'Tr': -1, 'Mh': 3200, 'Ys': 210, 'Sr': -1, 'Ar': 6, 'Wz': 750, 'Hh': 940,
             'My': -1, 'Km': 480, 'Sv': 200, 'Zc': 200, 'Bd': 200, 'Ro70': 18, 'Ka': 300, 'Ny': 1255, 'ALMA': -1,
             'Y1': 250, 'Y27': 10.7, 'Ro34': 106, 'Go': -1, 'Gb': 15, 'Sc': 327, 'Hn': 327, 'Nl': 327, 'Fd': 327,
             'La': 327, 'Kp': 327, 'Pt': 327, 'Ov': 327, 'Br': 327, 'Mk': 327, 'Pv': -1, 'Pb': -1, 'At': 86, 'Mp': 430,
             'Pa': 43, 'Ho': 560, 'Cd': 600, 'Ap': 3500, 'Ku': 1000, 'Ky': 1000, 'Kt': 1000,
             'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
             'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}
_band_2cm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': -1, 'W1': -1, 'Ef': 45, 'Mc': -1, 'Nt': -1, 'On': -1, 'Sh': -1,
             'Tm65': -1, 'Ur': -1, 'Tr': -1, 'Mh': -1, 'Ys': -1, 'Sr': -1, 'Ar': -1, 'Wz': -1, 'Hh': -1, 'My': -1,
             'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 350, 'Y27': 15.0,
             'Ro34': -1, 'Go': -1, 'Gb': 20, 'Sc': 543, 'Hn': 543, 'Nl': 543, 'Fd': 543, 'La': 543, 'Kp': 543,
             'Pt': 543, 'Ov': 543, 'Br': 543, 'Mk': 543, 'Pv': -1, 'Pb': -1, 'At': -1, 'Mp': -1, 'Pa': -1, 'Ho': -1,
             'Cd': -1, 'Ap': -1, 'Ku': 1000, 'Ky': 1000, 'Kt': 1000,
             'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
             'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}
_band_13mm = {'Jb1': -1, 'Jb2': 910, 'Cm': 720, 'Wb': -1, 'W1': -1, 'Ef': 90, 'Mc': 700, 'Nt': 800, 'On': 1380,
              'Sh': -1, 'Tm65': 70, 'Ur': 2950, 'Tr': 500, 'Mh': 2608, 'Ys': 295, 'Sr': 138, 'Ar': -1, 'Wz': -1,
              'Hh': 3000, 'My': -1, 'Km': -1, 'Sv': 710, 'Zc': 710, 'Bd': 710, 'Ro70': 83, 'Ka': -1, 'Ny': -1,
              'ALMA': -1, 'Y1': 560, 'Y27': 23.9, 'Ro34': -1, 'Go': 65, 'Gb': 30, 'Sc': 640, 'Hn': 640, 'Nl': 640,
              'Fd': 640, 'La': 640, 'Kp': 640, 'Pt': 640, 'Ov': 640, 'Br': 640, 'Mk': 640, 'Pv': -1, 'Pb': -1,
              'At': 106, 'Mp': 675, 'Pa': 810, 'Ho': 1800, 'Cd': 2500, 'Ap': -1, 'Ku': 1288, 'Ky': 1288, 'Kt': 1288,
              'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
              'VERAIR': 2110, 'VERAIS': 2110, 'VERAMZ': 2110, 'VERAOG': 2110, 'NRO45': -1}
_band_9mm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': -1, 'W1': -1, 'Ef': -1, 'Mc': -1, 'Nt': -1, 'On': -1, 'Sh': -1,
             'Tm65': -1, 'Ur': -1, 'Tr': -1, 'Mh': -1, 'Ys': -1, 'Sr': -1, 'Ar': -1, 'Wz': -1, 'Hh': -1, 'My': -1,
             'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 710, 'Y27': 30.3,
             'Ro34': -1, 'Go': -1, 'Gb': -1, 'Sc': -1, 'Hn': -1, 'Nl': -1, 'Fd': -1, 'La': -1, 'Kp': -1, 'Pt': -1,
             'Ov': -1, 'Br': -1, 'Mk': -1, 'Pv': -1, 'Pb': -1, 'At': -1, 'Mp': -1, 'Pa': -1, 'Ho': -1, 'Cd': -1,
             'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1,
             'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
             'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}
_band_7mm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': -1, 'W1': -1, 'Ef': 200, 'Mc': -1, 'Nt': 900, 'On': 1310, 'Sh': -1,
             'Tm65': 110, 'Ur': -1, 'Tr': -1, 'Mh': 4500, 'Ys': -1, 'Sr': 98, 'Ar': -1, 'Wz': -1, 'Hh': -1, 'My': -1,
             'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 1260,
             'Y27': 53.8, 'Ro34': -1, 'Go': -1, 'Gb': 60, 'Sc': 1181, 'Hn': 1181, 'Nl': 1181, 'Fd': 1181, 'La': 1181,
             'Kp': 1181, 'Pt': 1181, 'Ov': 1181, 'Br': 1181, 'Mk': 1181, 'Pv': -1, 'Pb': -1, 'At': 180, 'Mp': 900,
             'Pa': -1, 'Ho': -1, 'Cd': -1, 'Ap': -1, 'Ku': 1919, 'Ky': 1919, 'Kt': 1919,
             'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
             'VERAIR': 4395, 'VERAIS': 4395, 'VERAMZ': 4395, 'VERAOG': 4395, 'NRO45': 655}
_band_3mm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': -1, 'W1': -1, 'Ef': 930, 'Mc': -1, 'Nt': -1, 'On': 5100, 'Sh': -1,
             'Tm65': -1, 'Ur': -1, 'Tr': -1, 'Mh': 17650, 'Ys': 2540, 'Sr': 367, 'Ar': -1, 'Wz': -1, 'Hh': -1, 'My': -1,
             'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': 70, 'Y1': -1, 'Y27': -1,
             'Ro34': -1, 'Go': -1, 'Gb': 140, 'Sc': -1, 'Hn': -1, 'Nl': 4236, 'Fd': 4236, 'La': 4236, 'Kp': 4236,
             'Pt': 4236, 'Ov': 4236, 'Br': 4236, 'Mk': 4236, 'Pv': 640, 'Pb': 450, 'At': 1440, 'Mp': 3750, 'Pa': -1,
             'Ho': -1, 'Cd': -1, 'Ap': -1, 'Ku': 3000, 'Ky': 3000, 'Kt': 3000,
             'SAT1': -1, 'SAT2': -1, 'FAST': -1, 'SKA1-mid': -1, 'SKA1-low': -1, 'Ir': -1,
             'VERAIR': -1, 'VERAIS': -1, 'VERAMZ': -1, 'VERAOG': -1, 'NRO45': -1}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Parameter Evaluation")
    group = parser.add_mutually_exclusive_group()

    group.add_argument('-g',
                       '--show_gui',
                       action="store_true",
                       help='Choose to use my developed parameter calculators')

    parser.add_argument('-i',
                        '--show_info',
                        action="store_true",
                        help='Show info about our design', )

    return parser.parse_args()


def run_calculator():
    my_window = tk.Tk()
    # my_window.resizable(False, False)
    my_window.title("Parameter Calculator")

    ParaCal(window=my_window)
    my_window.mainloop()


_design_info = """
Referring to the EVN Calculator, this function provides the following parameter calculation: 
- the image thermal noise
- bandwidth-smearing-limited field of view 
- time-smearing-limited field of view
- an estimate of the FITS file size C

The calculation details please refer to our paper on 
https://arxiv.org/abs/1808.06726v1. 
Please inform us if SEFD value of your station was updated
"""

if __name__ == "__main__":
    args = parse_args()
    # args.show_gui = True
    if args.show_info:
        print(_design_info)

    if args.show_gui:
        run_calculator()
