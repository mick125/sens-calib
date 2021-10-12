import h5py
import time
# import math
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.constants as sc
from threading import Thread
from CalibDataProcessor import CalibDataProcessor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class RawDataProcessor(CalibDataProcessor):
    """
    Reading and processing of raw data.
    Raw means single frames without averaging and DCS values.
    """
    def __init__(self, input_file_path, output_path):
        super().__init__(input_file_path, output_path)
        self.raw_frames = np.zeros((1, ))
        self.stdev_pixel = np.zeros((1, ))

        self.dcs_frames = np.zeros((1, ))

        # set class variables in order to set the class to single measurement mode
        self.data_type = 'single_meas'
        self.n_recorded_frames = 0
        self.n_x_steps = 0
        self.x_ax_data = range(self.n_x_steps)

    def get_file_info(self):
        """
        Print it out hierarchical structure of the input h5 file.
        """
        def printname(name):
            """ helper function for Group.visit() method"""
            print(name)

        with h5py.File(self.input_file_path) as file:
            file.visit(printname)
            pass

    def load_raw_file(self):
        """
        Load following data from input file:
            - distance
        """
        with h5py.File(self.input_file_path) as file:
            self.raw_frames = self.lsb_to_mm(np.array(file['DRNU/distances']), frequency=self.mod_frequency)
            # example: 50 delay line steps, 25 frames, 240 x 320 pixels

        self.n_recorded_frames = self.raw_frames.shape[1]
        self.n_x_steps = self.n_recorded_frames

        print('Raw measured data loaded')

    def load_raw_file_DCS(self):
        """
        Load following data from input file:
            - imagesDCS
        """
        with h5py.File(self.input_file_path) as file:
            self.dcs_frames = np.array(file['DRNU/imagesDCS'], dtype='float64')
            # example: 50 delay line steps, 25 frames, 240 x 320 pixels, 4 DCS'

        self.n_recorded_frames = self.dcs_frames.shape[1]
        self.n_x_steps = self.n_recorded_frames

        print('Raw measured DCS data loaded')

    def load_mm_from_dcs_ext(self):
        """
        Loads distance measurement directly calculated from DCS from pickle
        """
        with open(self.output_path / 'pickl' / f'{self.chip}_mm_from_DCS.pkl', 'rb') as file:
            self.raw_frames = pkl.load(file)

        self.n_recorded_frames = self.raw_frames.shape[1]
        self.n_x_steps = self.n_recorded_frames

        print('Distance measurement directly calculated from DCS loaded from pickle')

    def convert_dcs_to_mm(self):
        """
        Takes DCS frames, coverts to mm and saves to self.raw_frames
        """
        print('Converting DCS signals to mm...', end=' ')

        data = [self.dcs_frames[:, :, :, :, i] for i in range(4)]
        self.raw_frames = RawDataProcessor.dcs_to_mm(data, self.mod_frequency)

        with open(self.output_path / 'pickl' / (self.chip + '_mm_from_DCS.pkl'), 'wb') as file:
            pkl.dump(self.raw_frames, file)

        print('done')

    @staticmethod
    def dcs_to_mm(dcs, frequency=1e7):
        """
        Formula which converts DCS to mm
        """
        return sc.c / 2 / (2 * sc.pi * frequency) * (sc.pi + np.arctan2(dcs[3] - dcs[1], dcs[2] - dcs[0])) * 1000

    def calc_mean_std(self, dll):
        """
        Calculate mean and std dev for each recorded frame for a given delay line.
        :param dll: Delay line step
        """
        self.mean_vs_dll = np.array([self.raw_frames[dll, frame, :, :].mean() for frame in range(self.n_recorded_frames)])
        self.stdev_vs_dll = np.array([self.raw_frames[dll, frame, :, :].std() for frame in range(self.n_recorded_frames)])

    def pick_dll(self, dll):
        """
        Save measured frames to self.raw_data variable for given delay line.
        :param dll: Delay line.
        """
        self.raw_data = self.raw_frames[dll]

    def calc_calib_data(self):
        """
        Calculates 'calibration data' out of single frame data by averaging of each measure frame.
        In other words, it averages all measurements at one delay line for each pixel.
        """
        print('Calculating calibration data...', end=' ')
        self.raw_data = np.mean(self.raw_frames, 1)
        print('done')

    def plot_hist_pix_std(self, delay_step):
        """
         Plot histogram of statistical std. dev. for all pixels in one DL step.
         :param delay_step: Delay line step
         """
        print(f'Histogram with statistical deviation per pixel for DL step {delay_step} will be plotted...', end=' ')
        self.stdev_pixel = np.std(self.raw_frames, 1)

        plot_title = f'{self.chip}, DLL = {delay_step + 1}'

        fig, ax = plt.subplots()
        plt.hist(x=self.stdev_pixel[delay_step].flatten(), bins=320, rwidth=1., color='b', range=[0, 100])

        # set labels
        ax.set_xlabel('Statistical deviation of one pixel [mm]')
        ax.set_ylabel('Count [-]')
        ax.set_title(plot_title)

        out_path = Path(self.output_path).joinpath('histograms',
                                                   f'{self.chip}_hist_pix-stat-dev_step-{delay_step:02d}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()

        print('done')


if __name__ == '__main__':

    # EXAMPLE USAGE

    input_file_path = \
        r'C:\Data\01_NFL\NFL_data\raw_data\W578_C132\W578_C132_10000_RawData_DRNU_27082021_170551.hdf5' # 75 frames per DLL, just DCS
        # r'C:\Data\01_NFL\NFL_data\raw_data\W603_C096_41\GrayCalibData_All_W603_C096_09062021_125503.hdf5' # greyscale
        # r'C:\Data\01_NFL\NFL_data\raw_data\W603_C096_41\W603_C096_10000_CalibData_DRNU_09062021_131457.hdf5'
        # r'C:\Data\01_NFL\NFL_data\raw_data\W578_C132\W578_C132_10000_RawData_DRNU_27082021_162711.hdf5' # 50 frames per DLL, just DCS
    output_path = r'C:\Data\01_NFL\NFL_data\Analysis\Raw_recordings'

    reader = RawDataProcessor(input_file_path, output_path)

    # reader.create_folders()
    reader.get_file_info()
    # reader.load_raw_file()        # load distance data from h5 file
    reader.load_raw_file_DCS()    # load DCS data from h5 file
    reader.convert_dcs_to_mm()
    # reader.compensate_rollover()

    # reader.load_mm_from_dcs_ext()   # load DCS data from pickle file

    reader.calc_calib_data()        # average frames to get calibration data

    reader.pick_dll(2)
    reader.plot_all('heat')
    reader.plot_all('hist')

    # plot mean and stdev as if calibration data was loaded
    reader.calc_mean_std_all_dll()  # calculate mean and stdev for each DLL
    reader.data_type = 'dl_step'
    # reader.data_type = 'single_meas'
    reader.plot_mean_std()

    # plot std. dev. histograms for all delay line steps
    [reader.plot_hist_pix_std(i) for i in range(reader.n_delay_steps)]

    # plot mean and stdev for all delay lines
    for dll in range(reader.n_delay_steps):
        reader.calc_mean_std(dll)
        reader.plot_mean_std(file_suff=f'dll-{dll:02d}')

    # ====== TESTING SPACE =======
