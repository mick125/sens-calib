import time
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1 import make_axes_locatable


class CalibReader:
    """
    Reader for DRNU calibration file. Additional features like plotting are available, too.
    """

    def __init__(self, file_path, output_path):
        self.chip_dim = (240, 320)
        self.n_delay_steps = 50
        self.maxphase = 3e4
        self.file_path = Path(file_path)
        self.output_path = Path(output_path)
        self.raw_data = np.zeros((1, 1, 1))
        self.calibrated_data = []
        self.sigma_map = np.zeros((1, 1, 1))
        self.discr_map = np.zeros((self.n_delay_steps, self.chip_dim[0], self.chip_dim[1]))
        # self.discr_map = np.ma.masked_array(np.zeros((self.n_delay_steps, self.chip_dim[0], self.chip_dim[1])))
        self.mod_frequency = int(self.file_path.parts[-1].split('_')[2]) * 1000
        self.chip = '_'.join(self.file_path.parts[-1].split('_')[:2])
        self.mean_vs_dll = np.zeros((1))
        self.stdev_vs_dll = np.zeros((1))
        self.fit_params_all = np.zeros((1))

    def load_raw_file(self):
        """
        Parses and loads binary DRNU calibration file into variable self.raw_data.
        Calculates mean and std. deviation for each DL step.
        """
        with open(self.file_path, 'rb') as file:
            self.raw_data = self.lsb_to_mm(np.fromfile(file, dtype=np.uint16), self.mod_frequency)
            # 50 delay line steps, 240 x 320 pixels
            self.raw_data = self.raw_data.reshape((self.n_delay_steps, self.chip_dim[0], self.chip_dim[1]))

        self.mean_vs_dll = np.array([self.raw_data[dll].mean() for dll in range(self.n_delay_steps)])
        self.stdev_vs_dll = np.array([self.raw_data[dll].std() for dll in range(self.n_delay_steps)])

        print('Calibration data loaded,', np.count_nonzero(self.raw_data), 'non-zero elements found.\n')

    def load_fit_params_ext(self):
        """
        Load fit parameters from external file.
        """
        with open(self.output_path / 'pickl' / (self.chip + '_all_pix_fit.pkl'), 'rb') as file:
            self.fit_params_all = pkl.load(file)

        print('Full set of fit parameters loaded from external file.')

    def load_calib_data_ext(self, pickl_id):
        """
        Loads calibrated data from external file and appends it to list self.calibrated_data.
        """
        with open(self.output_path / 'pickl' / f'{self.chip}_calibr_data_{pickl_id}.pkl', 'rb') as file:
            self.calibrated_data.append(pkl.load(file))

        print('Calibrated measurement data loaded from external file.')

    def compensate_rollover(self):
        """
        Shift calibration points up after rollover to extend the calibration curve seamlessly.
        """
        rollover_dl = 0
        rollover_shift = CalibReader.lsb_to_mm(self.maxphase, self.mod_frequency)

        # find the DL step in which the rollover takes place
        for i in range(self.n_delay_steps - 1):
            if self.mean_vs_dll[i] - self.mean_vs_dll[i + 1] > rollover_shift * 0.9:
                rollover_dl = i + 1
                print('Rollover DL is', rollover_dl, '. Rollover will be compensated.')
                break

        for i in range(rollover_dl, self.n_delay_steps):
            self.mean_vs_dll[i] += rollover_shift
            self.raw_data[i] += rollover_shift

    @staticmethod
    def lsb_to_mm(lst, frequency, maxphase=3e4):
        """
        Converts LST units to mm.
        :param lst: Input LST value to be converted
        :param frequency: Source modulation frequency
        :param maxphase: Max. LSB value
        :return: Distance in mm
        """
        return 3e8 / frequency / maxphase / 2 * lst * 1000

    @staticmethod
    def dll_to_mm(nsteps):
        """
        Converts delay line step to mm.
        :param nsteps: Number of DL steps to be converted.
        :return: Equivalent distance in mm.
        """
        return 345 + np.array(nsteps) * 315

    def plot_heatmap(self, delay_step, data, sub_folder='', name='heatmap'):
        """
        Plot one frame at given DL step.
        :param delay_step: Delay line step
        :param data: calibration data to be plotted, variable containing all DL steps
        :param sub_folder: optional subfolder to which the heat map shall be saved
        :param name: optional string which will be placed in the file name
        """
        fig, ax = plt.subplots()
        # im = ax.imshow(data[delay_step], interpolation=None, origin='lower', cmap='brg')
        # im = ax.pcolor(data[delay_step], cmap='brg')
        im = ax.pcolorfast(data[delay_step])

        # set labels
        ax.set_xlabel('Pixel [-]')
        ax.set_ylabel('Pixel [-]')
        ax.set_title(f'{self.chip}, DLL = {delay_step + 1}')

        # Create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('Distance [mm]')
        # issue - colorbar title outside of plottable area

        out_path = Path(self.output_path).joinpath(sub_folder,
                                                   self.chip + '_' + name + '_DLL-' + f'{delay_step + 1:02d}' + '.png')
        plt.savefig(out_path, dpi=200)
        plt.close()

    def plot_distance(self, delay_step):
        """
        Wrapper for plotting raw calibration data
        :param delay_step: DL step
        """
        self.plot_heatmap(delay_step, self.raw_data, sub_folder='distance', name='distance')

    def plot_discr(self, delay_step):
        """
        Wrapper for plotting discriminated heat map.
        Method create_discr_map must be run prior to this method to create the discriminated plots.
        :param delay_step: DL step
        """
        self.plot_heatmap(delay_step, self.discr_map, sub_folder='discr_maps', name='discr-map')

    def plot_hist_meas_dist(self, delay_step):
        """
        Plot histogram of one frame for one DL step.
        :param delay_step: Delay line step
        """
        fig, ax = plt.subplots()
        n, bins, patches = plt.hist(x=self.raw_data[delay_step].reshape((-1)), bins=40, rwidth=.85, color='b')

        # set labels
        ax.set_xlabel('Distance [mm]')
        ax.set_ylabel('Count [-]')
        ax.set_title(f'{self.chip}, DLL = {delay_step + 1}')

        out_path = Path(self.output_path).joinpath('histograms',
                                                   self.chip + '_histogram_DLL-' + f'{delay_step + 1:02d}' + '.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

    def plot_hist_fit_params(self):
        """
        Plots histograms of all fitted parameters
        """
        n_params = 5
        fit_params = self.fit_params_all.reshape((n_params, -1))

        for i in range(n_params):
            fig, ax = plt.subplots()
            n, bins, patches = plt.hist(x=fit_params[i], bins=40, rwidth=.85, color='m')

            # set labels
            ax.set_xlabel(f'p{i}')
            ax.set_ylabel('Count [-]')
            ax.set_title(f'{self.chip}, distribution of fit parameter p{i}\n' +
                         r'y = p0.sin(p1.x + p2) + p3 + p4.x')

            out_path = Path(self.output_path).joinpath('histograms', self.chip + f'_histogram_fit_par-{i}.png')
            plt.savefig(out_path, dpi=150)
            plt.close()

    def plot_all(self, plot_type):
        """
        Plots plot_type charts for all delay line steps.
        :param plot_type: 'heat' or 'hist'
        """
        avail_plots = ['heat', 'discr', 'hist']

        print(f'Creating {plot_type} plots...')

        # pick the plot type
        if plot_type == 'heat':
            plot_func = self.plot_distance
        elif plot_type == 'discr':
            plot_func = self.plot_discr
        elif plot_type == 'hist':
            plot_func = self.plot_hist_meas_dist
        else:
            raise Exception('plot_type must be ' + ' or '.join(avail_plots))

        # run the according plotting function in a loop
        for dll in range(self.n_delay_steps):
            plot_func(dll)

        print('done\n')

    def plot_mean_std(self):
        """
        Plots mean value and standard deviation for the whole frame for each DL step.
        Creates two plots, for mean and std dev.
        """
        print('Plotting mean value and standard deviations vs. DL step...')

        # plot std dev of one frame vs. DL step
        plt.plot(self.dll_to_mm(range(self.n_delay_steps)), self.stdev_vs_dll, 'bo-', linewidth=0.6, markersize=3)
        plt.grid(True)
        plt.xlabel('True distance [mm]')
        plt.ylabel('Measured distance std. deviation [mm]')
        plt.title(f'{self.chip}')

        out_path = Path(self.output_path).joinpath(self.chip + '_stdev-vs-dll' + '.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

        # plot mean value of one frame vs. DL step
        plt.plot(self.dll_to_mm(range(self.n_delay_steps)), self.mean_vs_dll, 'rs', markersize=3)
        plt.grid(True)
        plt.xlabel('True distance [mm]')
        plt.ylabel('Measured distance mean [mm]')
        plt.title(f'{self.chip}')

        out_path = Path(self.output_path).joinpath(self.chip + '_mean-vs-dll' + '.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

        print('done')

    def plot_pixel_err(self, x, y, sub_folder=''):
        """
        Plot measured value minus mean for all DL steps for one pixel
        :param x: pixel x coordinate
        :param y: pixel y coordinate
        """
        print(f'Plotting deviation from mean for pixel [{x}, {y}]...')

        err = np.array([self.raw_data[dll, x, y] for dll in range(self.n_delay_steps)])
        err = err - self.mean_vs_dll
        plt.plot(self.dll_to_mm(range(self.n_delay_steps)), err, 'cd-', linewidth=0.6, markersize=5)

        plt.grid(True)
        plt.xlabel('True distance [mm]')
        plt.ylabel('Measured distance -  mean [mm]')
        plt.title(f'{self.chip}, pixel [{x}, {y}]')

        out_path = Path(self.output_path).joinpath(sub_folder, self.chip + f'_pixel-dev_{x:03d}-{y:03d}' + '.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

        print('done')

    def plot_extreme_pix(self, dll, n_pix, ext='max', plot=True):
        """
        Picks a given number of pixels with extreme value for a given DL and plots the respective pix_x error plots
        :param n_pix: Number of extreme pixels to be found.
        :param dll: DL step.
        :param ext: Extreme, 'min' or 'max' (default).
        :param plot: Plotting flag. If False, no plots are created (just pixel coordinates returned).
        :return: Two arrays for x and y pixel coordinate.
        """
        if ext == 'max':
            # get indices of extremal values of flattened array
            ext_inds = np.argsort(self.raw_data[dll], axis=None)[-n_pix:]
        elif ext == 'min':
            ext_inds = np.argsort(self.raw_data[dll], axis=None)[:n_pix]
        else:
            raise Exception('The requested extreme must be min or max.')

        # get indices of extremal values within a 2D array
        extreme_pixels = np.unravel_index(ext_inds, self.raw_data[dll].shape)

        #  plot pixel error graphs
        if plot:
            for x, y in zip(extreme_pixels[0], extreme_pixels[1]):
                reader.plot_pixel_err(x, y, ext + '_pix')

        return extreme_pixels

    def plot_sigma_map(self):
        """
        Plots sigma heat map.
        Method create_sigma_map has to be run prior to this one.
        """
        fig, ax = plt.subplots()
        im = ax.pcolorfast(self.sigma_map)

        # set labels
        ax.set_xlabel('Pixel [-]')
        ax.set_ylabel('Pixel [-]')
        ax.set_title(self.chip)

        # Create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('Std. deviation [mm]')
        # issue - colorbar title outside of plottable area

        out_path = Path(self.output_path).joinpath(self.chip + '_sigma_map.png')
        plt.savefig(out_path, dpi=200)
        plt.close()

    def create_discr_map(self, n_sigma=2):
        """
        Create heat maps without showing pixels above a threshold given by a number of sigma (st.dev).
        :param n_sigma: Number of std dev to exclude
        """
        threshold_low = self.mean_vs_dll - n_sigma * self.stdev_vs_dll
        threshold_high = self.mean_vs_dll + n_sigma * self.stdev_vs_dll

        for dll in range(self.n_delay_steps):
            # pixel outside of +/- n_sigma
            self.discr_map[dll] = np.where(self.raw_data[dll] < threshold_low[dll], self.raw_data[dll], 0)
            self.discr_map[dll] += np.where(self.raw_data[dll] >= threshold_high[dll], self.raw_data[dll], 0)
            self.discr_map[dll] = np.where(self.discr_map[dll] == 0, np.nan, self.discr_map[dll])

            # pixel inside of +/- n_sigma
            # self.discr_map[dll] = np.where(self.raw_data[dll] > threshold_low[dll], self.raw_data[dll], np.nan)
            # self.discr_map[dll] += np.where(self.discr_map[dll] <= threshold_high[dll], self.discr_map[dll], np.nan)

    def create_sigma_map(self):
        """
        Creates a map of std. devs for each pixel based on measured values ove all DL steps.
        """
        temp = np.array([self.raw_data[dll] - self.mean_vs_dll[dll] for dll in range(self.n_delay_steps)])
        self.sigma_map = np.std(temp, axis=0)

    @staticmethod
    def diff(x, a, pix_x, pix_y, calib_params):
        """
        Helper function for minimization in calculation of inverse function.
        It has to be a scalar function due to requirements of sklearn.optimimize.minimize.
        :param x: Value to be iterated.
        :param a: Value to which the fit function result shall get close to.
        :param pix_x: Pixel x coordinate.
        :param pix_y: Pixel y coordinate.
        :param calib_params: Full set of calibration parameters for the sin function.
        :return: Quadratic deviation of fit function result and value a.
        """

        yt = CalibReader.fit_funct(x,
                                   calib_params[0, pix_x, pix_y],
                                   calib_params[1, pix_x, pix_y],
                                   calib_params[2, pix_x, pix_y],
                                   calib_params[3, pix_x, pix_y],
                                   calib_params[4, pix_x, pix_y],
                                   )
        return (yt - a)**2

    def fit_funct_inverse(self, x, pix_x, pix_y, calib_params, plot_test=False):
        """
        Inverse function to the fit function. To be used for calibrated sensor output. Uses the self.diff function.
        :param x: Input values (1D array), 2D array won't work.
        :param pix_x: Pixel x coordinate.
        :param pix_y: Pixel y coordinate.
        :param calib_params: Full set of calibration parameters for the sin function.
        :param plot_test: Plot flag, test purposes.
        :return: Measured distance after applying the calibration.
        """
        y = np.zeros_like(x)

        # Minimize difference the 'measurement result'(x value) and calibration function result.
        # The resulting input parameter of the calibration function is the actual distance.
        for idx, x_value in enumerate(x):
            res = minimize(self.diff, 7000, args=(x_value, pix_x, pix_y, calib_params),
                           method='Nelder-Mead', tol=1e-0)
            y[idx] = res.x[0]

        if plot_test:
            plt.plot(x, y)
            plt.show()

        return y

    @staticmethod
    def fit_funct(x, p0, p1, p2, p3, p4):
        """
        function to be used for fitting the DRNU data
        :param x: argument
        :return: function result
        """
        return p0 * np.sin(p1 * x + p2) + p3 + p4 * x

    @staticmethod
    def fit_pix(self, y, x, check_plot=False, remove_lin_part=False):
        """
        Fit output y of one pixel in all delay steps x with function fit_funct.
        :param x: x-axis values for the fit (delay steps)
        :param y: y-axis data to be fitted (pixel output)
        :param check_plot: Flag, create check plot for test purposes.
        :param remove_lin_part: Flag, subtract linear part from fit in check plot.
        :return: fit parameters
        """
        prior = np.array([150, .5, 1.6, 3500, 1])
        bounds = ([50, 0, 0, 0, 0], [250, 1, np.pi, 6000, 5])
        param, _ = curve_fit(CalibReader.fit_funct, x, y, p0=prior, bounds=bounds)

        if check_plot:
            if remove_lin_part:
                lin_part = param[3] + param[4] * x
            else:
                lin_part = 0

            plt.plot(x, y - lin_part, 'rd-')
            plt.plot(x, CalibReader.fit_funct(x, *param) - lin_part, 'b-')
            plt.show()

        return param

    def fit_all_pixels_calib(self, n_fit_points=50, save_pickl=True):
        """
        Fits all pixels on the chip with fit_funct formula
        :param n_fit_points: Number of DL steps to be used for the fit.
        :param n_fit_params: Number of parameters the fit function does have.
        :return: Array with shape (n_fit_params, chip_dim[0], chip_dim[1])
        :param save_pickl: Flag, save fit parameters to pickle file.
        """
        print(f'Fitting {self.chip_dim[0] * self.chip_dim[1]} pixels with calibration curve...')
        start_time = time.time()

        # navigate through data using ordinary for loops
        # self.fit_params_all = np.zeros((n_fit_params, self.chip_dim[0], self.chip_dim[1]))
        # for row in range(self.chip_dim[0]):
        #     for col in range(self.chip_dim[1]):
        #         self.fit_params_all[:, row, col] = CalibReader.fit_pix(self.raw_data[:n_fit_points, row, col],
        #                                                            range(n_fit_points))

        # navigate through data using the np.apply_along_axis method
        self.fit_params_all = np.apply_along_axis(CalibReader.fit_pix, 0,
                                                  self.raw_data[:n_fit_points, :, :],
                                                  CalibReader.dll_to_mm(range(n_fit_points)),
                                                  # check_plot=False
                                                  )
        # self.fit_params_all = np.apply_along_axis(CalibReader.fit_pix, 0, self.raw_data[:n_fit_points, :, :],
        #                                       x=range(n_fit_points))

        print(f'done, it took {time.time() - start_time:.1f} seconds')

        if save_pickl:
            with open(self.output_path / 'pickl' / (self.chip + '_all_pix_fit_params.pkl'), 'wb') as file:
                pkl.dump(self.fit_params_all, file)

            print('Fit parameters saved to a file')

    def apply_calibration(self, calib_params, save_pickl=True, pickl_id=''):
        """
        Applies sin calibration on the measured data.
        :param calib_params: Set of fit parameters for fit_funct.
        :param save_pickl: Save to pickle flag.
        :param pickl_id: Pickle file extension.
        :return: Measured data with applied calibration.
        """
        print(f'Applying calibration for {self.chip_dim[0] * self.chip_dim[1]} pixels ...')
        start_time = time.time()

        calibrated_data = np.zeros((self.n_delay_steps, *self.chip_dim))
        for pix_x in range(self.chip_dim[0]):
            print(f'\tProcessing pixel row {pix_x} / {self.chip_dim[0]} ...')
            for pix_y in range(self.chip_dim[1]):
                calibrated_data[:, pix_x, pix_y] = self.fit_funct_inverse(self.raw_data[:, pix_x, pix_y],
                                                                          pix_x, pix_y, calib_params)

        print(f'done, it took {time.time() - start_time:.1f} seconds')

        if save_pickl:
            with open(self.output_path / 'pickl' / f'{self.chip}_calibr_data_{pickl_id}.pkl', 'wb') as file:
                pkl.dump(calibrated_data, file)

            print('Data with applied calibration saved to a file')

        return calibrated_data

    @staticmethod
    def reduce_fit_params(fit_params, par_list):
        """
        Replaces parameters in list with their mean.
        :param fit_params: Set of fit parameters.
        :param par_list: Indices of parameters to be replaced with their mean.
        :return: New set of fit parameters with the same shape as fit_params
        """
        red_params = fit_params

        for i in par_list:
            red_params[i] = np.full(reader.chip_dim, np.mean(red_params[i]))

        return red_params

    def plot_degr(self):
        """
        Plots distance measurement degradation in each DL step for all sets of calibrated data.
        The degradation is defined as std. dev. of measured distance in one full frame.
        :return:
        """
        print('Plotting standard deviations of calibrated frame vs. DL step...')

        colors = ['c', 'm', 'y', 'g']
        labels = ['basic sin calib', '2 params fixed', '3 params fixed']
        for i in range(len(self.calibrated_data)):
            # stdev_vs_dll = np.array([self.calibrated_data[i][dll].std() for dll in range(self.n_delay_steps)])
            # stdev_vs_dll = np.array([np.amin(self.calibrated_data[i][dll]) for dll in range(self.n_delay_steps)])
            # stdev_vs_dll = np.array([np.amax(self.calibrated_data[i][dll]) for dll in range(self.n_delay_steps)])
            mean_vs_dll = np.array([self.calibrated_data[i][dll].mean() for dll in range(self.n_delay_steps)])

            stdev_vs_dll = mean_vs_dll - np.array([CalibReader.dll_to_mm(dl) for dl in range(50)])

            # plot std dev of one frame vs. DL step
            plt.plot(self.dll_to_mm(range(self.n_delay_steps)), stdev_vs_dll,
                     f'{colors[i]}o-', linewidth=0.6, markersize=3, label=labels[i])

        plt.grid(True)
        plt.legend()
        plt.xlabel('True distance [mm]')
        plt.ylabel('Measured distance std. deviation [mm]')
        plt.title(f'{self.chip}, calibration applied')

        out_path = Path(self.output_path).joinpath(f'{self.chip}_stdev_calib-vs-dll.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

        print('done')


if __name__ == '__main__':
    calib_file_path = r'C:\Data\01_NFL\calib_data\W455_C266\W455_C266_10000_drnu_images.bin'
    output_path = r'C:\Data\01_NFL\calib_data\Analysis\DRNU'

    reader = CalibReader(calib_file_path, output_path)
    reader.load_raw_file()
    reader.compensate_rollover()

    # fit calibration curve for all pixels
    # reader.fit_all_pixels_calib()

    # load calibration curve parameters from external file
    reader.load_fit_params_ext()

    # reduce number of fit parameters
    # fit_par_red_2 = reader.reduce_fit_params(reader.fit_params_all, [1, 2])
    # fit_par_red_3 = reader.reduce_fit_params(reader.fit_params_all, [1, 2, 4])

    # apply calibration on raw data based on the above fit parameters
    # reader.apply_calibration(reader.fit_params_all, True, 'sin_full') # already started
    # reader.apply_calibration(fit_par_red_2, True, 'sin_red_2')
    # reader.apply_calibration(fit_par_red_3, True, 'sin_red_3')

    # load calibrated data from pickle files
    reader.load_calib_data_ext('sin_full')
    reader.load_calib_data_ext('sin_red_2')
    reader.load_calib_data_ext('sin_red_3')

    # reader.raw_data = reader.calibrated_data[0]
    # reader.plot_all('hist')

    # PLOT THINGS
    # reader.plot_mean_std()

    # reader.plot_extreme_pix(3, 5, 'max')
    # reader.plot_extreme_pix(3, 5, 'min')

    # reader.plot_all('heat')

    # reader.plot_all('hist')

    # reader.create_discr_map(2)
    # reader.plot_all('discr')

    # reader.create_sigma_map()
    # reader.plot_sigma_map()

    # reader.plot_hist_fit_params()

    reader.plot_degr()

    # ---- TESTING SPACE ----

    # plt.plot(CalibReader.dll_to_mm(range(50)), reader.calibrated_data[0][:, 0, 4], 'd')
    # plt.show()

    # try f(f^-1(x))
    # n_pix_x = 100
    # n_pix_y = 100
    # src = reader.raw_data[0, :n_pix_y, :n_pix_y]
    # out = np.zeros_like(src)
    # for pix_x in range(n_pix_x):
    #     print(pix_x)
    #     for pix_y in range(n_pix_y):
    #         out[pix_x, pix_y] = reader.fit_funct_inverse([src[pix_x, pix_y]], pix_x, pix_y, reader.fit_params_all)
    #         out[pix_x, pix_y] = CalibReader.fit_funct(out[pix_x, pix_y], *reader.fit_params_all[:, pix_x, pix_y])
    # out -= src
    # out = out.astype('int')
    # print(np.amin(out), np.amax(out), np.mean(out), np.std(out))


    # print(reader.fit_funct_inverse(np.arange(3600, 15000, 100), 20, 30, reader.fit_params_all, True))

    # n_points = 50
    # pix_coord = (0, 4)
    #
    # params = reader.fit_pix(
    #                         reader.raw_data[:n_points, pix_coord[0], pix_coord[1]],
    #                         CalibReader.dll_to_mm(range(n_points)),
    #                         True)

    # print('synt', CalibReader.fit_funct(345, *params), '\norig', reader.raw_data[0, 0, 3])



    # TODO heatmapy parametru
