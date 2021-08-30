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
    """
    def __init__(self, input_file_path, output_path, n_recorded_frames=25):
        super().__init__(input_file_path, output_path)
        self.n_recorded_frames = n_recorded_frames
        self.raw_frames = np.zeros((1, ))
        self.stdev_pixel = np.zeros((1, ))

        self.dcs_frames = np.zeros((1, ))
        self.mm_from_DCS = np.zeros((1, ))

        # set class to single measurement mode
        self.data_type = 'single_meas'
        self.n_x_steps = self.n_recorded_frames
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
        Load following data from file:
            - distance
        """
        with h5py.File(self.input_file_path) as file:
            self.raw_frames = self.lsb_to_mm(np.array(file['DRNU/distances']), frequency=self.mod_frequency)
            # example: 50 delay line steps, 25 frames, 240 x 320 pixels

        print('Raw measured data loaded')

    def load_raw_file_DCS(self):
        """
        Load following data from file:
            - imagesDCS
        """
        with h5py.File(self.input_file_path) as file:
            self.dcs_frames = np.array(file['DRNU/imagesDCS'])
            # example: 50 delay line steps, 25 frames, 240 x 320 pixels, 4 DCS'

        print('Raw measured DCS data loaded')

    def load_mm_from_dcs(self):
        """
        Loads distance measurement directly calculated from DCS from pickle
        """
        with open(self.output_path / 'pickl' / f'{self.chip}_mm_from_DCS.pkl', 'rb') as file:
            self.raw_frames = pkl.load(file)

        print('Distance measurement directly calculated from DCS loaded from pickle')

    def convert_dcs(self):
        """
        Takes DCS frames, coverts to mm and saves to self.raw_frames
        """
        print('Converting DCS signals to mm...')

        self.mm_from_DCS = np.zeros((self.n_delay_steps, self.n_recorded_frames, self.chip_dim[0], self.chip_dim[1]))

        for dl in range(self.n_delay_steps):
            print(f'\tDelay line {dl + 1}/{self.n_delay_steps}')
            for meas in range(self.n_recorded_frames):
                for pix_x in range(self.chip_dim[0]):
                    for pix_y in range(self.chip_dim[1]):
                        self.mm_from_DCS[dl, meas, pix_x, pix_y] = \
                            RawDataProcessor.dcs_to_mm(self.dcs_frames[dl, meas, pix_x, pix_y, :], self.mod_frequency)

        with open(self.output_path / 'pickl' / (self.chip + '_mm_from_DCS.pkl'), 'wb') as file:
            pkl.dump(self.mm_from_DCS, file)

        print('done')

    @staticmethod
    def dcs_to_mm(dcs, frequency=1e6):
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

    def plot_hist_pix_std(self, delay_step):
        """
         Plot histogram of statistical std. dev. for all pixels in one DL step.
         :param delay_step: Delay line step
         """
        print(f'Histogram with statistical deviation per pixel for DL step {delay_step} will be plotted...', end=' ')
        self.stdev_pixel = np.std(self.raw_frames, 1)

        plot_title = f'{self.chip}, DLL = {delay_step + 1}'

        fig, ax = plt.subplots()
        n, bins, patches = plt.hist(x=self.stdev_pixel[delay_step].flatten(), bins=320, rwidth=1., color='b', range=[0, 100])

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
    input_file_path = \
        r'C:\Data\01_NFL\NFL_data\raw_data\W578_C132\W578_C132_10000_RawData_DRNU_27082021_170551.hdf5' # 75 frames per DLL, just DCS
        # r'C:\Data\01_NFL\NFL_data\raw_data\W603_C096_41\W603_C096_10000_CalibData_DRNU_09062021_131457.hdf5'
        # r'C:\Data\01_NFL\NFL_data\raw_data\W578_C132\W578_C132_10000_RawData_DRNU_27082021_162711.hdf5' # 50 frames per DLL, just DCS
    n_frames = 75
    output_path = r'C:\Data\01_NFL\NFL_data\Analysis\Raw_recordings'

    reader = RawDataProcessor(input_file_path, output_path, n_frames)

    reader.create_folders()
    # reader.get_file_info()
    # reader.load_raw_file()
    reader.load_raw_file_DCS()

    start = time.time()
    reader.convert_dcs()
    print(f'It took {time.time() - start:.2f}')

    # reader.load_mm_from_dcs()

    reader.pick_dll(2)
    reader.plot_all('heat')
    reader.plot_all('hist')

    [reader.plot_hist_pix_std(i) for i in range(reader.n_delay_steps)]
    #
    # # plot mean and stdev for all delay lines
    # start = time.time()

    # Plot mean std
    for dll in range(reader.n_delay_steps):
        reader.calc_mean_std(dll)
        reader.plot_mean_std(file_suff=f'dll-{dll:02d}')

    # print(f'It took {time.time() - start:.2f}')

    # ====== TESTING SPACE =======

    #  THREADING
    # start = time.time()
    # threads = []
    # # for dll in range(reader.n_delay_steps):
    # for dll in range(2):
    #     reader.calc_mean_std(dll)
    #     params = ('single_meas', reader.n_recorded_frames, f'dll-{dll:02d}')
    #     threads.append(Thread(target=reader.plot_mean_std, args=params))
    #
    # [t.start() for t in threads]
    # [t.join() for t in threads]
    # print(f'It took {time.time() - start:.2f}')
    #  ---------------

    # start = time.time()

    # THREADING POOL
    # with ThreadPoolExecutor() as ex:
    #     for dll in range(reader.n_delay_steps):
    #         reader.calc_mean_std(dll)
    #         ex.submit(reader.plot_mean_std, f'dll-{dll:02d}')

    # PROCESSING POOL
    # with ProcessPoolExecutor() as ex:
    #     for dll in range(reader.n_delay_steps):
    #         reader.calc_mean_std(dll)
    #         ex.submit(reader.plot_mean_std, f'dll-{dll:02d}')
    #
    # print(f'It took {time.time() - start:.2f}')

