import h5py
import time
import numpy as np
from pathlib import Path
from threading import Thread
from CalibDataProcessor import CalibDataProcessor


class RawDataProcessor(CalibDataProcessor):
    """
    Reading and processing of raw data.
    """
    def __init__(self, input_file_path, output_path):
        super().__init__(input_file_path, output_path)
        self.n_recorded_frames = 25
        self.raw_frames = np.zeros((1))

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

    def load_raw_file(self):
        """
        Load folowing data from file:
            - distance
        """
        with h5py.File(self.input_file_path) as file:
            self.raw_frames = np.array(file['DRNU/distances'])
            # 50 delay line steps, 25 frames, 240 x 320 pixels

        print('Raw measured data loaded')

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


if __name__ == '__main__':
    input_file_path = \
        r'C:\Data\01_NFL\NFL_data\raw_data\W603_C096_41\W603_C096_10000_CalibData_DRNU_09062021_131457.hdf5'
    output_path = r'C:\Data\01_NFL\NFL_data\Analysis\Raw_recordings'

    reader = RawDataProcessor(input_file_path, output_path)

    # reader.create_folders()
    # reader.get_file_info()
    reader.load_raw_file()

    reader.pick_dll(2)
    reader.plot_all('heat')
    # reader.plot_all('hist')

    #  plot mean and stdev for all delay lines
    # for dll in range(reader.n_delay_steps):
    #     reader.calc_mean_std(dll)
    #     reader.plot_mean_std(file_suff=f'dll-{dll:02d}')

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


