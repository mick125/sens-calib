import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit


class CalibReader:
    """
    Reader for DRNU calibration file. Additional features like plotting are available, too.
    """

    def __init__(self, file_path, output_path):
        self.chip_dim = (240, 320)
        self.n_delay_steps = 50
        self.file_path = Path(file_path)
        self.output_path = Path(output_path)
        self.calib_data = np.zeros((1, 1, 1))
        self.sigma_map = np.zeros((1, 1, 1))
        self.discr_map = np.zeros((self.n_delay_steps, self.chip_dim[0], self.chip_dim[1]))
        # self.discr_map = np.ma.masked_array(np.zeros((self.n_delay_steps, self.chip_dim[0], self.chip_dim[1])))
        self.mod_frequency = int(self.file_path.parts[-1].split('_')[2]) * 1000
        self.chip = '_'.join(self.file_path.parts[-1].split('_')[:2])
        self.mean_vs_dll = np.zeros((1))
        self.stdev_vs_dll = np.zeros((1))
        self.fit_params = np.zeros((1))

    def load_calib_file(self):
        """
        Parses and loads binary DRNU calibration file into variable self.calib_data.
        Calculates mean and std. deviation for each DL step.
        """
        with open(self.file_path, 'rb') as file:
            self.calib_data = self.lsb_to_mm(np.fromfile(file, dtype=np.uint16), self.mod_frequency)
            # 50 delay line steps, 240 x 320 pixels
            self.calib_data = self.calib_data.reshape((self.n_delay_steps, self.chip_dim[0], self.chip_dim[1]))

        self.mean_vs_dll = np.array([self.calib_data[dll].mean() for dll in range(self.n_delay_steps)])
        self.stdev_vs_dll = np.array([self.calib_data[dll].std() for dll in range(self.n_delay_steps)])

        print('Calibration data loaded,', np.count_nonzero(self.calib_data), 'non-zero elements found.\n')

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
        self.plot_heatmap(delay_step, self.calib_data, sub_folder='distance', name='distance')

    def plot_discr(self, delay_step):
        """
        Wrapper for plotting discriminated heat map.
        Method create_discr_map must be run prior to this method to create the discriminated plots.
        :param delay_step: DL step
        """
        self.plot_heatmap(delay_step, self.discr_map, sub_folder='discr_maps', name='discr-map')

    def plot_hist(self, delay_step):
        """
        Plot histogram of one frame for one DL step.
        :param delay_step: Delay line step
        """
        fig, ax = plt.subplots()
        n, bins, patches = plt.hist(x=self.calib_data[delay_step].reshape((-1)), bins=40, rwidth=.85, color='b')

        # set labels
        ax.set_xlabel('Distance [mm]')
        ax.set_ylabel('Count [-]')
        ax.set_title(f'{self.chip}, DLL = {delay_step + 1}')

        out_path = Path(self.output_path).joinpath('histograms',
                                                   self.chip + '_histogram_DLL-' + f'{delay_step + 1:02d}' + '.png')
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
            plot_func = self.plot_hist
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
        plt.plot(self.stdev_vs_dll, 'bo-', linewidth=0.6, markersize=3)
        plt.grid(True)
        plt.xlabel('Delay line step [-]')
        plt.ylabel('Measured distance std. deviation [mm]')
        plt.title(f'{self.chip}')

        out_path = Path(self.output_path).joinpath(self.chip + '_stdev-vs-dll' + '.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

        # plot mean value of one frame vs. DL step
        plt.plot(self.mean_vs_dll, 'rs', markersize=3)
        plt.grid(True)
        plt.xlabel('Delay line step [-]')
        plt.ylabel('Measured distance mean [mm]')
        plt.title(f'{self.chip}')

        out_path = Path(self.output_path).joinpath(self.chip + '_mean-vs-dll' + '.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

    def plot_pixel_err(self, x, y, sub_folder=''):
        """
        Plot measured value minus mean for all DL steps
        :param x: pixel x coordinate
        :param y: pixel y coordinate
        """
        print(f'Plotting deviation from mean for pixel [{x}, {y}]...')

        err = np.array([self.calib_data[dll, x, y] for dll in range(self.n_delay_steps)])
        err = err - self.mean_vs_dll
        plt.plot(err, 'cd-', linewidth=0.6, markersize=5)

        plt.grid(True)
        plt.xlabel('Delay line step [-]')
        plt.ylabel('Measured distance -  mean [mm]')
        plt.title(f'{self.chip}, pixel [{x}, {y}]')

        out_path = Path(self.output_path).joinpath(sub_folder, self.chip + f'_pixel-dev_{x:03d}-{y:03d}' + '.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

        print('done')

    def plot_extreme_pix(self, dll, n_pix, ext='max', plot=True):
        """
        Picks a given number of pixels with extreme value for a given DL and plots the respective pix error plots
        :param n_pix: Number of extreme pixels to be found.
        :param dll: DL step.
        :param ext: Extreme, 'min' or 'max' (default).
        :param plot: Plotting flag. If False, no plots are created (just pixel coordinates returned).
        :return: Two arrays for x and y pixel coordinate.
        """
        if ext == 'max':
            # get indices of extremal values of flattened array
            ext_inds = np.argsort(self.calib_data[dll], axis=None)[-n_pix:]
        elif ext == 'min':
            ext_inds = np.argsort(self.calib_data[dll], axis=None)[:n_pix]
        else:
            raise Exception('The requested extreme must be min or max.')

        # get indices of extremal values within a 2D array
        extreme_pixels = np.unravel_index(ext_inds, self.calib_data[dll].shape)

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
            self.discr_map[dll] = np.where(self.calib_data[dll] < threshold_low[dll], self.calib_data[dll], 0)
            self.discr_map[dll] += np.where(self.calib_data[dll] >= threshold_high[dll], self.calib_data[dll], 0)
            self.discr_map[dll] = np.where(self.discr_map[dll] == 0, np.nan, self.discr_map[dll])

            # pixel inside of +/- n_sigma
            # self.discr_map[dll] = np.where(self.calib_data[dll] > threshold_low[dll], self.calib_data[dll], np.nan)
            # self.discr_map[dll] += np.where(self.discr_map[dll] <= threshold_high[dll], self.discr_map[dll], np.nan)

    def create_sigma_map(self):
        """
        Creates a map of std. devs for each pixel based on measured values ove all DL steps.
        """
        temp = np.array([self.calib_data[dll] - self.mean_vs_dll[dll] for dll in range(self.n_delay_steps)])
        self.sigma_map = np.std(temp, axis=0)

    @staticmethod
    def fit_funct_sin(x, p0, p1, p2, p3, p4):
        """
        function to be used for fitting the DRNU data
        :param x: argument
        :return: function result
        """
        return p0 * np.sin(p1 * x + p2) + p3 + p4 * x

    @staticmethod
    def fit_pix(y, x):
        """
        Fit output y of one pixel in all delay steps x with function fit_funct_sin.
        :param x: x-axis values for the fit (delay steps)
        :param y: y-axis data to be fitted (pixel output)
        :return: fit parameters
        """
        # x = range(40)
        prior = np.array([150, .5, 1.6, 3500, 300])
        bounds = ([50, 0, 0, 300, 150], [250, 1, np.pi, 5000, 450])
        param, _ = curve_fit(CalibReader.fit_funct_sin, x, y, p0=prior, bounds=bounds)
        return param

    def fit_all_pixels_calib(self, n_fit_points=40, n_fit_params=5):
        """
        Fits all pixels on the chip with fit_funct_sin formula
        :param n_fit_points: Number of DL steps to be used for the fit.
        :param n_fit_params: Number of parameters the fit function does have.
        :return: Array with shape (n_fit_params, chip_dim[0], chip_dim[1])
        """
        print(f'Fitting {self.chip_dim[0] * self.chip_dim[1]} pixels with calibration curve...')
        start_time = time.time()

        # navigate through data using ordinary for loops
        # self.fit_params = np.zeros((n_fit_params, self.chip_dim[0], self.chip_dim[1]))
        # for row in range(self.chip_dim[0]):
        #     for col in range(self.chip_dim[1]):
        #         self.fit_params[:, row, col] = CalibReader.fit_pix(self.calib_data[:n_fit_points, row, col],
        #                                                            range(n_fit_points))

        # navigate through data using the np.apply_along_axis method
        self.fit_params = np.apply_along_axis(CalibReader.fit_pix, 0, self.calib_data[:n_fit_points, :, :],
                                              x=range(n_fit_points))

        print(f'done, it took {time.time() - start_time:.1f} seconds')


if __name__ == '__main__':
    calib_file_path = r'C:\Data\01_NFL\calib_data\W455_C266\W455_C266_10000_drnu_images.bin'
    output_path = r'C:\Data\01_NFL\calib_data\Analysis\DRNU'

    reader = CalibReader(calib_file_path, output_path)
    reader.load_calib_file()

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

    # TESTING SPACE
    # n_points = 40
    # pix_coord = (84, 242)
    # params, _ = reader.fit_pix(range(n_points), reader.calib_data[:n_points, pix_coord[0], pix_coord[1]])
    # tmp = [f'{par:.2f}' for par in params]
    # print(*tmp, sep='\n')
    # plt.plot(CalibReader.fit_funct_sin(range(n_points), *params), 'b.')
    # plt.plot(reader.calib_data[:n_points, pix_coord[0], pix_coord[1]], 'r+')
    # plt.show()
    reader.fit_all_pixels_calib()
    # TODO histogramy pro nafitovane parametry
    # TODO heatmapy parametru
    # TODO ukladani a nacitani fit parametru
