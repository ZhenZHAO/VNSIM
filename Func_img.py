"""
@functions: source model, dirty map, clean img, radplot
@author: Zhen ZHAO
@date: Dec 16, 2018
"""
import os
import matplotlib as mpl
import matplotlib.image as plimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage.interpolation as spndint
import scipy.optimize as spfit
import numpy as np
import load_conf as lc
import utility as ut
from Func_uv import FuncUv
import argparse
import configparser
import pickle
import time

# colors normalization
norm = mpl.colors.Normalize(vmin=0, vmax=0.6)


class FuncImg(object):
    def __init__(self, model_name, n_pix, coverage_u, coverage_v, max_uv, obs_freq,
                 set_clean_window, clean_gain, clean_threshold, clean_niter, uv_unit="km"):
        self.n_pix = n_pix
        self.n_phf = self.n_pix // 2
        # 1. source model
        # 1.1 get source model file directory
        self.source_model = model_name
        model_dir = os.path.join(os.getcwd(), 'SOURCE_MODELS')
        self.model_file = os.path.join(model_dir, self.source_model)
        self.unit_flag = 0 if uv_unit == "lambda" else 1
        self.obs_freq = obs_freq
        self.obs_wlen = 299792458.8 / self.obs_freq
        # 1.2 source model result
        self.img_size = 4.
        self.img_file = []
        self.models = []
        self.x_max = 0

        self.model_img = []
        self.model_fft = []

        # 2. dirty beam
        # 2.1 parameter settings
        self.u = []
        self.v = []
        self.max_u = 0

        u = np.array(coverage_u)
        v = np.array(coverage_v)
        max_u = max_uv
        if len(u) != 0 and len(v) != 0:
            if self.unit_flag != 0:  # unit is not lambda
                self.u = u * 1000 / self.obs_wlen
                self.v = v * 1000 / self.obs_wlen
                self.max_u = max_u * 1000 / self.obs_wlen
            else:
                self.u = u
                self.v = v
                self.max_u = max_u

        # 2.2 dirty beam result
        self.dirty_beam = []
        self.mask = []
        self.beam_scale = 0

        # 3. dirty map
        self.dirty_map = np.zeros((self.n_pix, self.n_pix), dtype=np.float32)

        # 4. cleaner
        # 4.1 settings
        self.clean_window = set_clean_window
        self.clean_gain = clean_gain
        self.clean_thresh = clean_threshold
        self.clean_niter = clean_niter
        # 4.2 clean results
        self.clean_img = []
        self.res_img = []

        # to avoid multiple runing
        self.is_model_obtained = False
        self.is_beam_obtained = False
        self.is_map_obtained = False

        # 5. parameter calculation
        self.result_e_bpa = 0
        self.result_e_bmaj = 0
        self.result_e_bmin = 0
        self.result_e_range = 0
        self.result_dynamic_range = 0
        self.result_rms_noise = 0

    # 1.source model
    def _read_model(self):
        """
        :return: models, img_size, Xaxmax, img_file
        """

        if len(self.model_file) == 0:
            self.models = [['G', 0., 0.4, 1.0, 0.1], ['D', 0., 0., 2., 0.5], ['P', -0.4, -0.5, 0.1]]
            self.x_max = self.img_size / 2.
            return True

        if len(self.model_file) > 0:
            if not os.path.exists(self.model_file):
                print("\n\nModel file %s does not exist!\n\n" % self.model_file)
                return False
            else:
                fix_size = False
                temp_model = []
                temp_img_files = []
                temp_img_size = self.img_size
                Xmax = 0.0
                fi = open(self.model_file)
                for li, l in enumerate(fi.readlines()):
                    comm = l.find('#')
                    if comm >= 0:
                        l = l[:comm]
                    it = l.split()
                    if len(it) > 0:
                        if it[0] == 'IMAGE':
                            temp_img_files.append([str(it[1]), float(it[2])])
                        elif it[0] in ['G', 'D', 'P']:
                            temp_model.append([it[0]] + list(map(float, it[1:])))
                            if temp_model[-1][0] != 'P':
                                temp_model[-1][4] = np.abs(temp_model[-1][4])
                                Xmax = np.max([np.abs(temp_model[-1][1]) + temp_model[-1][4],
                                               np.abs(temp_model[-1][2]) + temp_model[-1][4], Xmax])
                        elif it[0] == 'IMSIZE':
                            temp_img_size = 2. * float(it[1])
                            fix_size = True
                        else:
                            print("\n\nWRONG SYNTAX IN LINE %i:\n\n %s...\n\n" % (li + 1, l[:max(10, len(l))]))
                if len(temp_model) + len(temp_img_files) == 0:
                    print("\n\nThere should be at least 1 model component!\n\n")

                self.models = temp_model
                self.imsize = temp_img_size
                self.imfiles = temp_img_files
                if not fix_size:
                    self.imsize = Xmax * 1.1
                self.x_max = self.imsize / 2
                fi.close()

                return True

        return False

    def _prepare_model(self):
        """
        :return: modelim, modelfft
        """
        if self._read_model():
            # create temp variable
            models = self.models
            imsize = self.imsize
            imfiles = self.imfiles
            Npix = self.n_pix
            Nphf = self.n_phf

            pixsize = float(imsize) / Npix
            xx = np.linspace(-imsize / 2., imsize / 2., Npix)
            yy = np.ones(Npix, dtype=np.float32)
            distmat = np.zeros((Npix, Npix), dtype=np.float32)
            modelim = np.zeros((Npix, Npix), dtype=np.float32)

            # read model
            for model in models:
                xsh = -model[1]
                ysh = -model[2]
                xpix = np.rint(xsh / pixsize).astype(np.int32)
                ypix = np.rint(ysh / pixsize).astype(np.int32)
                centy = np.roll(xx, ypix)
                centx = np.roll(xx, xpix)
                distmat[:] = np.outer(centy ** 2., yy) + np.outer(yy, centx ** 2.)
                if model[0] == 'D':
                    mask = np.logical_or(distmat <= model[4] ** 2., distmat == np.min(distmat))
                    modelim[mask] += float(model[3]) / np.sum(mask)
                elif model[0] == 'G':
                    gauss = np.exp(-distmat / (2. * model[4] ** 2.))
                    modelim[:] += float(model[3]) * gauss / np.sum(gauss)
                elif model[0] == 'P':
                    if np.abs(xpix + Nphf) < Npix and np.abs(ypix + Nphf) < Npix:
                        yint = ypix + Nphf
                        xint = xpix + Nphf
                        modelim[yint, xint] += float(model[3])

            # read image file
            for imfile in imfiles:
                if not os.path.exists(imfile[0]):
                    imfile[0] = os.path.join(os.path.join(os.getcwd(), 'PICTURES'), imfile[0])
                    if not os.path.exists(imfile[0]):
                        print('File %s does NOT exist. Cannot read the model!' % imfile[0])
                        return False

                Np4 = Npix // 4
                img = plimg.imread(imfile[0]).astype(np.float32)
                dims = np.shape(img)
                d3 = min(2, dims[2])
                d1 = float(max(dims))
                avimg = np.average(img[:, :, :d3], axis=2)
                avimg -= np.min(avimg)
                avimg *= imfile[1] / np.max(avimg)
                if d1 == Nphf:
                    pass
                else:
                    zoomimg = spndint.zoom(avimg, float(Nphf) / d1)
                    zdims = np.shape(zoomimg)
                    zd0 = min(zdims[0], Nphf)
                    zd1 = min(zdims[1], Nphf)
                    sh0 = (Nphf - zdims[0]) // 2
                    sh1 = (Nphf - zdims[1]) // 2
                    # print(sh0, Np4, zd0, sh1, zd1)
                    modelim[sh0 + Np4:sh0 + Np4 + zd0, sh1 + Np4:sh1 + Np4 + zd1] += zoomimg[:zd0, :zd1]

            # obtain modelim, modelfft
            modelim[modelim < 0.0] = 0.0
            self.model_img = modelim
            self.model_fft = np.fft.fft2(np.fft.fftshift(modelim))
            return True
        else:
            print("wrong model settings")
            return False

    def get_result_src_model_with_update(self):
        """
        :return: model_img, max_range
        """
        if self._prepare_model():
            self.is_model_obtained = True
            Npix = self.n_pix
            Np4 = Npix // 4
            show_modelim = self.model_img[Np4:(Npix - Np4), Np4:(Npix - Np4)]
            return show_modelim, self.x_max
        else:
            return None, None

    def update_result_src_model(self):
        if self._prepare_model():
            self.is_model_obtained = True
        else:
            self.is_model_obtained = False

    def get_result_src_model(self):
        if self.is_model_obtained:
            Npix = self.n_pix
            Np4 = Npix // 4
            show_modelim = self.model_img[Np4:(Npix - Np4), Np4:(Npix - Np4)]
            return show_modelim, self.x_max
        else:
            return [], 0.0

    # 2.dirty beam
    def _prepare_beam(self):
        mask = np.zeros((self.n_pix, self.n_pix), dtype=np.float32)
        beam = []

        # 1. griding uv
        ctr = self.n_pix // 2
        scale_uv = self.n_pix / 2 / self.max_u * 0.95 * 0.5
        for index in np.arange(len(self.u)):
            mask[int(ctr + round(self.u[index] * scale_uv)), int(ctr + round(self.v[index] * scale_uv))] += 1
        # mask = np.transpose(mask)
        mask[mask > 1] = 1

        # 2. robust sampling
        # robust = 0.0
        # Nbas = len(u)
        # nH = 200 # time_duration // time_step
        # robfac = (5. * 10. ** (-robust)) ** 2. * (2. * Nbas * nH) / np.sum( mask** 2.)
        # robustsamp = np.zeros((Npix, Npix), dtype=np.float32)
        # robustsamp[:] = mask / (1. + robfac * mask)

        # 3. beam
        # beam = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(mask))))
        # beam = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(mask))))
        beam = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(mask))))

        beam_scale = np.max(beam)
        # beam_scale = np.max(beam[self.n_phf:self.n_phf + 1, self.n_phf:self.n_phf + 1])
        beam /= beam_scale

        # return
        self.dirty_beam = beam
        self.mask = mask
        self.beam_scale = beam_scale
        # print("="*20)
        # print(self.beam_scale)
        # print("=" * 20)

    def get_result_dirty_beam_with_update(self):
        self._prepare_beam()
        self.is_beam_obtained = True
        Npix = self.n_pix
        Np4 = Npix // 4
        show_beam = self.dirty_beam[Np4:(Npix - Np4), Np4:(Npix - Np4)]
        return show_beam

    # for multiprocessing purpose (separate updating and getter)
    def update_result_dirty_beam(self):
        self._prepare_beam()
        self.is_beam_obtained = True

    def get_result_dirty_beam(self):
        if self.is_beam_obtained:
            Npix = self.n_pix
            Np4 = Npix // 4
            show_beam = self.dirty_beam[Np4:Npix - Np4, Np4:Npix - Np4]
            return show_beam
        else:
            return []

    # 3.dirty map
    def _prepare_map(self):
        if not self.is_model_obtained:
            self._prepare_model()
        if not self.is_beam_obtained:
            self._prepare_beam()

        # Be1=np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.dirty_beam))))
        # Ga1=np.real(self.model_fft)
        # C_BG1=np.copy(self.dirty_map)
        #
        # for ii in np.arange(len(Be1)):
        #     for jj in np.arange(len(Ga1)):
        #         C_BG1[ii][jj] = Be1[ii][jj]*Ga1[ii][jj]
        #
        # self.dirty_map[:] = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(C_BG1))))

        # self.dirty_map[:] = np.fft.fftshift(
        # np.fft.ifft2(self.model_fft * np.fft.ifftshift(self.mask))).real / self.beam_scale
        self.dirty_map[:] = np.fft.fftshift(np.fft.ifft2(self.model_fft * np.fft.ifftshift(self.mask))).real / (
                    self.beam_scale * 1.5)

    def get_result_dirty_map_with_update(self):
        self._prepare_map()
        self.is_map_obtained = True
        Np4 = self.n_pix // 4
        show_dirty = self.dirty_map[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
        return show_dirty

    # for multiprocessing purpose (separate updating and getter)
    def update_result_dirty_map(self):
        self._prepare_map()
        self.is_map_obtained = True

    def get_result_dirty_map(self):
        if self.is_map_obtained:
            Np4 = self.n_pix // 4
            show_dirty = self.dirty_map[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
            return show_dirty
        else:
            return []

    # 4.cleaner
    def overlap_indices(self):
        pass

    def do_clean(self):
        # clean_img, res_img = do_clean(dirty_map, dirty_beam, True, 0.2, 0, 100)
        if not self.is_map_obtained:
            self._prepare_map()
        self.get_clean_beam()
        clean_beam = self.clean_beam
        Npix = self.n_pix
        image_shape = self.dirty_map.shape
        # clean_img = np.zeros(image_shape)
        # res_img = np.array(self.dirty_map)
        clean_img = np.zeros(np.shape(self.dirty_map))
        source_img = np.zeros(np.shape(self.dirty_map))
        res_img = np.copy(self.dirty_map)
        # clean window
        window = []
        if self.clean_window is True:
            window = np.ones(image_shape, np.bool)
        # clean iterations
        for i in range(self.clean_niter):
            mx, my = np.unravel_index(np.fabs(res_img[window]).argmax(), res_img.shape)
            mval = res_img[mx, my] * self.clean_gain
            source_img[mx, my] += mval
            clean_img += mval * np.roll(np.roll(clean_beam, mx - Npix // 2, axis=0),
                                        my - Npix // 2, axis=1)

            a1o, a2o = overlap_indices(self.dirty_map, self.dirty_beam,
                                       mx - image_shape[0] / 2,
                                       my - image_shape[1] / 2)
            # print(a1o, a2o)
            res_img[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= self.dirty_beam[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
            if np.fabs(res_img).max() < self.clean_thresh:
                break
        # result
        # print("="*20, self.clean_niter, "="*20)
        self.clean_img = clean_img
        self.res_img = res_img
        self.source_img = source_img

    def get_clean_beam(self):
        beam = self.dirty_beam
        main_lobe = np.where(beam > 0.6 * np.max(beam))
        clean_beam = np.zeros(np.shape(beam))
        Npix = self.n_pix
        # print(Npix)

        if len(main_lobe[0]) < 5:
            print('ERROR!', 'The main lobe of the PSF is too narrow!\n CLEAN model will not be restored')
            clean_beam[:] = 0.0
            clean_beam[Npix // 2, Npix // 2] = 1.0
        else:
            dX = main_lobe[0] - Npix // 2
            dY = main_lobe[1] - Npix // 2
            #  if True:
            try:
                fit = spfit.leastsq(
                    lambda x: np.exp(-(dX * dX * x[0] + dY * dY * x[1] + dX * dY * x[2])) - beam[main_lobe],
                    [1., 1., 0.])
                ddX = np.outer(np.ones(Npix),
                               np.arange(-Npix // 2, Npix // 2).astype(np.float64))
                ddY = np.outer(np.arange(-Npix // 2, Npix // 2).astype(np.float64),
                               np.ones(Npix))

                clean_beam[:] = np.exp(-(ddY * ddY * fit[0][0] + ddX * ddX * fit[0][1] + ddY * ddX * fit[0][2]))

                del ddX, ddY
            except:
                print('ERROR!', 'Problems fitting the PSF main lobe!\n CLEAN model will not be restored')
                clean_beam[:] = 0.0
                clean_beam[Npix // 2, Npix // 2] = 1.0

        self.clean_beam = clean_beam

    def get_result_clean_map_with_update(self):
        self.do_clean()
        Np4 = self.n_pix // 4
        show_clean = self.clean_img[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
        show_res = self.res_img[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
        show_src = self.source_img[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
        show_cln_beam = self.clean_beam[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
        return show_clean + show_res, show_res, show_src, show_cln_beam

    # for multiprocessing purpose (separate updating and getter)
    def update_result_clean_map(self):
        self.do_clean()

    def get_result_clean_map(self):
        if self.is_map_obtained:
            Np4 = self.n_pix // 4
            show_clean = self.clean_img[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
            show_res = self.res_img[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
            return show_clean, show_res
        else:
            return [], []

    # 5. calculation
    def get_result_img_range(self):
        self.update_result_para_cal()
        return self.result_e_range

    def show_result_para_cal(self):
        str1 = "e_bpa={} degree\ne_bmaj={} mas\ne_bmin={} mas\ne_range={}\nrms_noise={}\ndr={}".format(
            self.result_e_bpa, self.result_e_bmaj, self.result_e_bmin, self.result_e_range,
            self.result_rms_noise, self.result_dynamic_range)
        return str1

    def update_result_para_cal(self):
        # 1. calculate beam size and position angle
        # the unit of u,v in my code is km or lambda
        u = np.array(self.u)
        v = np.array(self.v)
        max_uv = self.max_u
        if len(self.u) != 0 and len(self.v) != 0:
            uv_bl = [np.sqrt(uu ** 2 + vv ** 2) for uu, vv in zip(u, v)]
            max_bl = np.max(uv_bl)
            muu, mvv, muv = 0.0, 0.0, 0.0
            wsum, runwt = 0.0, 1.0
            for i in range(0, len(u)):
                weight = 1.0
                # if True: # do radial weighting
                #     weight *= uv_bl[i]
                if True:  # do uniform weighting
                    weight /= max_bl
                wsum += weight
                runwt = weight / wsum

                muu += runwt * (u[i] ** 2 - muu)
                mvv += runwt * (v[i] ** 2 - mvv)
                muv += runwt * (u[i] * v[i] - muv)
            # http://www.astro.caltech.edu/~tjp/  Timothy J. Pearson
            # https://www.eso.org/sci/meetings/2015/eris2015/ERIS-T4.pdf
            fudge = 0.7  # Empirical fudge factor of TJP's algorithm
            ftmp = np.sqrt((muu - mvv) ** 2 + 4 * muv * muv)

            e_bpa = -0.5 * np.arctan2(2.0 * muv, muu - mvv)
            e_bpa = e_bpa * 180 / np.pi

            e_bmaj = fudge / (np.sqrt(2.0 * (muu + mvv - ftmp)))
            e_bmaj = e_bmaj / np.pi * 180 * 3600 * 1000

            e_bmin = fudge / (np.sqrt(2.0 * (muu + mvv) + 2.0 * ftmp))
            e_bmin = e_bmin / np.pi * 180 * 3600 * 1000

            self.result_e_bpa = e_bpa
            self.result_e_bmaj, self.result_e_bmin = e_bmaj, e_bmin

        # 2. calculate the image axis

        u_range = np.linspace(-max_uv, max_uv, self.n_pix)
        u_reso = np.abs(u_range[3] - u_range[2])  # delta u
        l_extent = 1 / u_reso
        l_angle = np.arcsin(l_extent) * 180.0 / np.pi  # rad to degree
        l_angle = l_angle * 3600  # degree to as
        l_angle = l_angle * 1000  # as to mas
        self.result_e_range = l_angle // 4

        # 3. calculate the rms noise and dr, self.clean_img,  self.res_img
        clean_img = np.abs(self.clean_img + self.res_img)
        mean_noise = np.mean(np.abs(self.res_img))
        self.result_rms_noise = mean_noise
        self.result_dynamic_range = np.max(clean_img) / mean_noise

        return self.result_e_bpa, self.result_e_bmaj, self.result_e_bmin, self.result_e_range, self.result_rms_noise, self.result_dynamic_range


class ImgConfigParser(object):
    def __init__(self, _filename="config_img.ini", _dbname='database.pkl'):
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


def overlap_indices(a1, a2, shiftx, shifty):
    if shiftx >= 0:
        a1xbeg = shiftx
        a2xbeg = 0
        a1xend = a1.shape[0]
        a2xend = a1.shape[0] - shiftx
    else:
        a1xbeg = 0
        a2xbeg = -shiftx
        a1xend = a1.shape[0] + shiftx
        a2xend = a1.shape[0]

    if shifty >= 0:
        a1ybeg = shifty
        a2ybeg = 0
        a1yend = a1.shape[1]
        a2yend = a1.shape[1] - shifty
    else:
        a1ybeg = 0
        a2ybeg = -shifty
        a1yend = a1.shape[1] + shifty
        a2yend = a1.shape[1]

    return (int(a1xbeg), int(a1xend), int(a1ybeg), int(a1yend)), (int(a2xbeg), int(a2xend), int(a2ybeg), int(a2yend))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the imaging func, show the source model, dirty beam, dirty map, clean map and corresponding parameter info")
    parser.add_argument('-c',
                        '--config',
                        default='config_img.ini',
                        help='Specify the configuration file')
    parser.add_argument('-u',
                        '--uv_file',
                        default="",
                        help="Load your own u,v data instead of configuring the obs parameters (under ./)")
    parser.add_argument('-p',
                        '--group_img',
                        action="store_true",
                        help="To save 4 imgs in a single one or separately"
                        )
    parser.add_argument('-g',
                        '--show_img',
                        action="store_true",
                        help='Choose to show GUI or not')
    parser.add_argument('-i',
                        '--show_info',
                        action="store_true",
                        help='Choose to show beam size, position angle, dynamic range and rms noise', )
    parser.add_argument('-f',
                        '--img_fmt',
                        choices=['eps', 'png', 'pdf', 'svg', 'ps'],
                        help='Specify the img format (default:pdf)',
                        default='pdf')
    # parser.add_argument('-m',
    #                     '--color_map',
    #                     choices=['viridis', 'hot', 'jet', 'rainbow', 'Greys', 'cool', 'nipy_spectral'],
    #                     help='Specify the color map',
    #                     default='viridis')

    return parser.parse_args()


def run_img():
    # 1.initialize parse and config objects
    args = parse_args()
    # for test in ide
    # args.show_img = True
    # args.group_img = True
    # args.show_info = True

    if args.config != '':
        my_config_parser = ImgConfigParser(args.config)
    else:
        my_config_parser = ImgConfigParser()

    # 2. show-image parameters
    # colormap = 'viridis'
    # if args.color_map in ['viridis', 'hot', 'jet', 'rainbow', 'Greys', 'cool', 'nipy_spectral']:
    #     colormap = args.color_map
    colormap = my_config_parser.color_map_name
    gamma = 0.3
    set_clean_window = True
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)

    # 3. results data
    data_u, data_v = [], []
    max_uv = 0
    data_img_src, data_img_bm, data_img_map, data_img_cbm, data_img_cmap = 0, 0, 0, 0, 0
    data_img_range = 0

    # 4. u,v
    use_uv_file = False
    uv_file_path = ''
    if args.uv_file != "":
        uv_file_name = args.uv_file
        uv_file_path = os.path.join(os.getcwd(), uv_file_name)
        if os.path.exists(uv_file_path):
            use_uv_file = True

    if use_uv_file:
        read_in = np.loadtxt(uv_file_path, dtype=np.float32)
        # my Func_uv.py will save u,v data in row fashion
        data_u, data_v = read_in[0], read_in[1]
        max_uv = max(np.max(np.abs(data_u)), np.max(np.abs(data_v)))
    else:
        start_time = ut.time_2_mjd(*my_config_parser.time_start, 0)
        stop_time = ut.time_2_mjd(*my_config_parser.time_end, 0)
        time_step = ut.time_2_day(*my_config_parser.time_step)
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
        data_u, data_v, max_uv = myFuncUV.get_result_single_uv_with_update()

    # 5. img calculation
    if len(data_u) == 0 or len(data_v) == 0:
        print("U,V data is not properly configured!")
        return
    # 5.1 initialize FuncImg object
    myFuncImg = FuncImg(my_config_parser.source_model,
                        my_config_parser.n_pix,
                        data_u, data_v, max_uv,
                        my_config_parser.obs_freq,
                        set_clean_window,
                        my_config_parser.clean_gain,
                        my_config_parser.clean_threshold,
                        my_config_parser.clean_niter,
                        my_config_parser.unit_flag)
    # 5.2 src model
    data_img_src, data_img_range = myFuncImg.get_result_src_model_with_update()
    # 5.3 dirty beam
    data_img_bm = myFuncImg.get_result_dirty_beam_with_update()
    # 5.4 dirty map
    data_img_map = myFuncImg.get_result_dirty_map_with_update()
    # 5.5 clean map, resual map, clean beam
    data_img_cmap, data_img_res, data_pure_point, data_img_cbm = myFuncImg.get_result_clean_map_with_update()
    data_img_range = myFuncImg.get_result_img_range()
    show_range = data_img_range // 2

    # 7. show parameter info
    if args.show_info:
        print(myFuncImg.show_result_para_cal())

    # 8. Imaging
    img_type = 'pdf'
    if args.img_fmt in ['eps', 'png', 'pdf', 'svg', 'ps']:
        img_type = args.img_fmt
    # 8.1 specify img type and output directory
    img_out_path = os.path.join(os.path.join(os.getcwd(), 'OUTPUT'), 'imaging')
    path_time_str = time.asctime()
    path_save_uv = os.path.join(img_out_path, "uv-{}.{}".format(path_time_str, img_type))
    path_save_bm = os.path.join(img_out_path, "dirty-beam-{}.{}".format(path_time_str, img_type))
    path_save_cbm = os.path.join(img_out_path, "clean-beam-{}.{}".format(path_time_str, img_type))
    path_save_src = os.path.join(img_out_path, "src-model-{}.{}".format(path_time_str, img_type))
    path_save_map = os.path.join(img_out_path, "dirty-map-{}.{}".format(path_time_str, img_type))
    path_save_cmap = os.path.join(img_out_path,"clean-map-{}.{}".format(path_time_str, img_type))
    path_save_integrate = os.path.join(img_out_path,"Integrated-all-{}.{}".format(path_time_str, img_type))

    # 8.2 draw imgs
    if args.group_img:
        figs = plt.figure(figsize=(8, 4))
        # 1) u,v
        fig_uv = figs.add_subplot(231, aspect='equal')
        x = np.array(data_u)
        y = np.array(data_v)
        max_range = max_uv * 1.1
        fig_uv.scatter(x, y, s=1, marker='.', color='brown')
        fig_uv.set_xlim([-max_range, max_range])
        fig_uv.set_ylim([-max_range, max_range])
        fig_uv.set_title("UV Plot: %s" % my_config_parser.str_source[0])
        if my_config_parser.unit_flag == 'km':
            fig_uv.set_xlabel("u$(km)$")
            fig_uv.set_ylabel("v$(km)$")
        else:
            fig_uv.set_xlabel("u$(\lambda)$")
            fig_uv.set_ylabel("v$(\lambda)$")
        fig_uv.grid()
        # set science
        fig_uv.yaxis.get_major_formatter().set_powerlimits((0, 1))
        fig_uv.xaxis.get_major_formatter().set_powerlimits((0, 1))

        # 2) dirty beam
        fig_bm = figs.add_subplot(232, aspect='equal')
        plot_beam = fig_bm.imshow(data_img_bm, origin='lower', aspect='equal', picker=True, interpolation='nearest', cmap=colormap, norm=norm)
        plt.setp(plot_beam, extent=(-show_range, show_range, -show_range, show_range))
        fig_bm.set_xlabel('Relative RA (mas)')
        fig_bm.set_ylabel('Relative DEC (mas)')
        fig_bm.set_title('DIRTY BEAM')

        # 3) clean beam
        fig_cbm = figs.add_subplot(233, aspect='equal')
        plot_cbeam = fig_cbm.imshow(data_img_cbm, origin='lower', aspect='equal', picker=True, interpolation='nearest', cmap=colormap, norm=norm)
        plt.setp(plot_cbeam, extent=(-show_range, show_range, -show_range, show_range))
        fig_cbm.set_xlabel('Relative RA (mas)')
        fig_cbm.set_ylabel('Relative DEC (mas)')
        fig_cbm.set_title('CLEAN BEAM')

        figs.colorbar(plot_cbeam, shrink=0.9)

        # 4) src model
        fig_model = figs.add_subplot(234, aspect='equal')
        plot_model = fig_model.imshow(np.power(data_img_src, gamma), origin='lower', aspect='equal', picker=True, cmap=colormap, norm=norm)
        plt.setp(plot_model, extent=(-show_range, show_range, -show_range, show_range))
        fig_model.set_xlabel('Relative RA (mas)')
        fig_model.set_ylabel('Relative DEC (mas)')
        fig_model.set_title('MODEL IMAGE')

        # 5) dirty map
        fig_map = figs.add_subplot(235, aspect='equal')
        plot_map = fig_map.imshow(data_img_map, origin='lower', aspect='equal', cmap=colormap, norm=norm)
        plt.setp(plot_map, extent=(-show_range, show_range, -show_range, show_range))
        fig_map.set_xlabel('Relative RA (mas)')
        fig_map.set_ylabel('Relative DEC (mas)')
        fig_map.set_title('DIRTY IMAGE')

        # 6) clean map
        fig_cmap = figs.add_subplot(236, aspect='equal')
        plot_cmap = fig_cmap.imshow(data_img_cmap, origin='lower', aspect='equal',picker=True, interpolation='nearest', cmap=colormap, norm=norm)
        plt.setp(plot_cmap, extent=(-show_range, show_range, -show_range, show_range))
        fig_cmap.set_xlabel('Relative RA (mas)')
        fig_cmap.set_ylabel('Relative DEC (mas)')
        fig_cmap.set_title('CLEAN IMAGE')
        figs.colorbar(plot_cmap, shrink=0.9)

        figs.tight_layout()
        plt.savefig(path_save_integrate)
    else:
        # 1) u,v
        fig1 = plt.figure(figsize=(4, 4))
        fig_uv = fig1.add_subplot(111, aspect='equal')
        x = np.array(data_u)
        y = np.array(data_v)
        max_range = max_uv * 1.1
        fig_uv.scatter(x, y, s=1, marker='.', color='brown')
        fig_uv.set_xlim([-max_range, max_range])
        fig_uv.set_ylim([-max_range, max_range])
        fig_uv.set_title("UV Plot: %s" % my_config_parser.str_source[0])
        if my_config_parser.unit_flag == 'km':
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
        plt.savefig(path_save_uv)

        # 2) dirty beam
        fig2 = plt.figure(figsize=(4, 4))
        fig_bm = fig2.add_subplot(111, aspect='equal')
        # plot_beam = fig_bm.imshow(data_img_bm, origin='lower', aspect='equal', vmin=-0, vmax=1.0, cmap=colormap)
        # plot_beam = fig_bm.imshow(data_img_bm, picker=True, cmap=colormap, norm=norm) # interpolation='nearest',
        plot_beam = fig_bm.imshow(data_img_bm, origin='lower', aspect='equal', cmap=colormap, norm=norm)
        plt.setp(plot_beam, extent=(-show_range, show_range, -show_range, show_range))
        fig_bm.set_xlabel('Relative RA (mas)')
        fig_bm.set_ylabel('Relative DEC (mas)')
        fig_bm.set_title('DIRTY BEAM')
        fig2.colorbar(plot_beam, shrink=0.9)
        plt.savefig(path_save_bm)

        # 3) clean beam
        fig3 = plt.figure(figsize=(4, 4))
        fig_cbm = fig3.add_subplot(111, aspect='equal')
        plot_cbeam = fig_cbm.imshow(data_img_cbm, origin='lower', aspect='equal',picker=True, interpolation='nearest', cmap=colormap, norm=norm)
        plt.setp(plot_cbeam, extent=(-show_range, show_range, -show_range, show_range))
        fig_cbm.set_xlabel('Relative RA (mas)')
        fig_cbm.set_ylabel('Relative DEC (mas)')
        fig_cbm.set_title('CLEAN BEAM')
        fig3.colorbar(plot_cbeam, shrink=0.9)
        plt.savefig(path_save_cbm)

        # 4) src model
        fig4 = plt.figure(figsize=(4, 4))
        fig_model = fig4.add_subplot(111, aspect='equal')
        plot_model = fig_model.imshow(np.power(data_img_src, gamma), origin='lower', aspect='equal',picker=True, cmap=colormap, norm=norm)
        plt.setp(plot_model, extent=(-show_range, show_range, -show_range, show_range))
        fig_model.set_xlabel('Relative RA (mas)')
        fig_model.set_ylabel('Relative DEC (mas)')
        fig_model.set_title('MODEL IMAGE')
        fig4.colorbar(plot_model, shrink=0.9)
        plt.savefig(path_save_src)

        # 5) dirty map
        fig5 = plt.figure(figsize=(4, 4))
        fig_map = fig5.add_subplot(111, aspect='equal')
        plot_map = fig_map.imshow(data_img_map, origin='lower', aspect='equal', cmap=colormap, norm=norm)
        plt.setp(plot_map, extent=(-show_range, show_range, -show_range, show_range))
        fig_map.set_xlabel('Relative RA (mas)')
        fig_map.set_ylabel('Relative DEC (mas)')
        fig_map.set_title('DIRTY IMAGE')
        fig5.colorbar(plot_map, shrink=0.9)
        plt.savefig(path_save_map)

        # 6) clean map
        fig6 = plt.figure(figsize=(4, 4))
        fig_cmap = fig6.add_subplot(111, aspect='equal')
        plot_cmap = fig_cmap.imshow(data_img_cmap, origin='lower', aspect='equal', picker=True, interpolation='nearest', cmap=colormap, norm=norm)
        plt.setp(plot_cmap, extent=(-show_range, show_range, -show_range, show_range))
        fig_cmap.set_xlabel('Relative RA (mas)')
        fig_cmap.set_ylabel('Relative DEC (mas)')
        fig_cmap.set_title('CLEAN IMAGE')
        fig6.colorbar(plot_cmap, shrink=0.9)
        plt.savefig(path_save_cmap)

    if args.show_img:
        plt.show()


if __name__ == "__main__":
    run_img()
