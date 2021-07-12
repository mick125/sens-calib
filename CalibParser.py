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

    def load_calib_file(self):
        """
        Parses and loads binary DRNU calibration file into variable self.calib_data.
        """
        with open(self.file_path, 'rb') as file:
            self.calib_data = np.fromfile(file, dtype=np.uint16)
            self.calib_data = self.calib_data.reshape((self.n_delay_steps, 240, 320))  # 50 delay line steps, 240 x 320 pixels

        print('Calibration data loaded,', np.count_nonzero(self.calib_data), 'non-zero elements found.')

    def plot_frame(self, delay_step, output_file_name_prefix):
        """
        Plot one frame at given DL step.
        :param delay_step: Delay line step <0, 49>
        :param output_file_name_prefix: file name prefix of output image, DL step and suffix will be added
        """
        fig, ax = plt.subplots()
        # plot heatmap
        im = ax.imshow(self.calib_data[delay_step], origin='lower')

        # set labels
        ax.set_xlabel('Pixel [-]')
        ax.set_ylabel('Pixel [-]')
        ax.set_title(f'{Path(output_file_name_prefix).parts[-1]}, DLL = {delay_step+1}')

        # Create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('Distance [?]')
        # issue - colorbar title outside of plottable area

        # plt.show()
        plt.savefig(Path(output_file_name_prefix + '_DLL-' + f'{delay_step+1:02d}' + '.png'), dpi=150)


if __name__ == '__main__':
    calib_file_path = r'C:\Data\01_NFL\calib_data\W455_C266\W455_C266_10000_drnu_images.bin'
    output_file = r'C:\Data\01_NFL\calib_data\Analysis\DRNU_images\W455_C266'

    reader = CalibReader(calib_file_path)
    reader.load_calib_file()
    for dll in range(reader.n_delay_steps):
        reader.plot_frame(dll, output_file)
    # reader.plot_frame(40, output_file)

