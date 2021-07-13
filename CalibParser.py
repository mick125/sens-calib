from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt


class CalibReader:
    """
    Reader for DRNU calibration file. Additional features like plotting are available, too.
    """

    def __init__(self, file_path):
        self.file_path = Path(file_path)
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
            self.calib_data = self.calib_data.reshape((self.n_delay_steps, 240, 320))  # 50 delay line steps, 240 x 320 pixels

        print('Calibration data loaded,', np.count_nonzero(self.calib_data), 'non-zero elements found.')

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

    def plot_frame(self, delay_step, output_path):
        """
        Plot one frame at given DL step.
        :param delay_step: Delay line step
        :param output_path: file name prefix of output image, DL step and suffix will be added
        """
        fig, ax = plt.subplots()
        # plot heat map
        im = ax.imshow(self.calib_data[delay_step], origin='lower')

        # set labels
        ax.set_xlabel('Pixel [-]')
        ax.set_ylabel('Pixel [-]')
        ax.set_title(f'{self.chip}, DLL = {delay_step+1}')

        # Create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('Distance [mm]')
        # issue - colorbar title outside of plottable area

        # plt.show()
        plt.savefig(Path(output_path).joinpath('heatmaps',
                    self.chip + '_heatmap_DLL-' + f'{delay_step+1:02d}' + '.png'), dpi=150)

    def plot_hist(self, delay_step, output_file_name_prefix):
        """
        Plot histogram of one frame for one DL step.
        :param delay_step: Delay line step
        :param output_file_name_prefix: file name prefix of output image, DL step and suffix will be added
        """
        n, bins, patches = plt.hist(x=self.calib_data[delay_step])
        plt.show()


if __name__ == '__main__':
    calib_file_path = r'C:\Data\01_NFL\calib_data\W455_C266\W455_C266_10000_drnu_images.bin'
    output_file = r'C:\Data\01_NFL\calib_data\Analysis\DRNU'

    reader = CalibReader(calib_file_path)
    reader.load_calib_file()

    delat = False
    # delat = True
    if delat:
        for dll in range(reader.n_delay_steps):
            # plot heat maps
            reader.plot_frame(dll, output_file)

            # plot histograms
            reader.plot_hist(1, output_file)

    reader.plot_frame(40, output_file)
    # reader.plot_hist(1, output_file)

    # print(reader.chip)
