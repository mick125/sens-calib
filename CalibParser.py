from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt


class CalibReader:
    """
    Reader for DRNU calibration file. Additional features like plotting are available, too.
    """

    def __init__(self, file_path, output_path):
        self.file_path = Path(file_path)
        self.output_path = Path(output_path)
        self.calib_data = np.zeros((1, 1, 1))
        self.n_delay_steps = 50
        self.mod_frequency = int(self.file_path.parts[-1].split('_')[2]) * 1000
        self.chip = '_'.join(self.file_path.parts[-1].split('_')[:2])
        self.mean_vs_dll = np.zeros((1))
        self.stdev_vs_dll = np.zeros((1))

    def load_calib_file(self):
        """
        Parses and loads binary DRNU calibration file into variable self.calib_data.
        Calculates mean and std. deviation for each DL step.
        """
        with open(self.file_path, 'rb') as file:
            self.calib_data = self.lsb_to_mm(np.fromfile(file, dtype=np.uint16), self.mod_frequency)
            # 50 delay line steps, 240 x 320 pixels
            self.calib_data = self.calib_data.reshape((self.n_delay_steps, 240, 320))

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

    def plot_heatmap(self, delay_step):
        """
        Plot one frame at given DL step.
        :param delay_step: Delay line step
        :param output_path: base output path, sub folder and file name are added automatically
        """
        fig, ax = plt.subplots()
        im = ax.imshow(self.calib_data[delay_step], origin='lower')

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

        out_path = Path(self.output_path).joinpath('heatmaps',
                                                   self.chip + '_heatmap_DLL-' + f'{delay_step + 1:02d}' + '.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

    def plot_hist(self, delay_step):
        """
        Plot histogram of one frame for one DL step.
        :param delay_step: Delay line step
        :param output_path: base output path, sub folder and file name are added automatically
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

    def plot_all(self, plot_type, output_path):
        """
        Plots plot_type charts for all delay line steps.
        :param plot_type: 'heat' or 'hist'
        :param output_path: base output path, sub folder and file name are added automatically
        """
        avail_plots = ['heat', 'hist']

        print(f'Creating {plot_type} plots...')

        # pick the plot type
        if plot_type == 'heat':
            plot_func = self.plot_heatmap
        elif plot_type == 'hist':
            plot_func = self.plot_hist
        else:
            raise Exception('plot_type must be ' + ' or '.join(avail_plots))

        # run the according plotting function in a loop
        for dll in range(self.n_delay_steps):
            plot_func(dll, output_path)

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

    def plot_pixel_err(self, x, y, folder=''):
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

        out_path = Path(self.output_path).joinpath(folder, self.chip + f'_pixel-dev_{x:03d}-{y:03d}' + '.png')
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


if __name__ == '__main__':
    calib_file_path = r'C:\Data\01_NFL\calib_data\W455_C266\W455_C266_10000_drnu_images.bin'
    output_path = r'C:\Data\01_NFL\calib_data\Analysis\DRNU'

    reader = CalibReader(calib_file_path, output_path)
    reader.load_calib_file()

    # reader.plot_pixel_err(10, 21)

    reader.plot_extreme_pix(3, 5, 'max')
    reader.plot_extreme_pix(3, 5, 'min')

    # reader.plot_mean_std()
    # reader.plot_all('heat')
