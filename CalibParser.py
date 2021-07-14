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

    def load_calib_file(self):
        """
        Parses and loads binary DRNU calibration file into variable self.calib_data.
        """
        with open(self.file_path, 'rb') as file:
            self.calib_data = self.lsb_to_mm(np.fromfile(file, dtype=np.uint16), self.mod_frequency)
            # 50 delay line steps, 240 x 320 pixels
            self.calib_data = self.calib_data.reshape((self.n_delay_steps, 240, 320))

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
        Calculates and plots mean value and standard deviation for the whole frame for each DL step.
        Creates two plots, for mean and std dev.
        """
        mean = [self.calib_data[dll].mean() for dll in range(self.n_delay_steps)]
        stdev = [self.calib_data[dll].std() for dll in range(self.n_delay_steps)]

        print('Plotting mean value and standard deviations vs. DL step...')

        # plot std dev of one frame vs. DL step
        plt.plot(stdev, 'bo-', linewidth=0.6, markersize=3)
        plt.grid(True)
        plt.xlabel('Delay line step [-]')
        plt.ylabel('Measured distance std. deviation [mm]')
        plt.title(f'{self.chip}')

        out_path = Path(self.output_path).joinpath(self.chip + '_stdev-vs-dll' + '.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

        # plot mean value of one frame vs. DL step
        plt.plot(mean, 'rs', markersize=3)
        plt.grid(True)
        plt.xlabel('Delay line step [-]')
        plt.ylabel('Measured distance mean [mm]')
        plt.title(f'{self.chip}')

        out_path = Path(self.output_path).joinpath(self.chip + '_mean-vs-dll' + '.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

        print('done')


if __name__ == '__main__':
    calib_file_path = r'C:\Data\01_NFL\calib_data\W455_C266\W455_C266_10000_drnu_images.bin'
    output_path = r'C:\Data\01_NFL\calib_data\Analysis\DRNU'

    reader = CalibReader(calib_file_path, output_path)
    reader.load_calib_file()

    reader.plot_mean_std()

    # reader.plot_all('heat')
