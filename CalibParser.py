from pathlib import Path
import numpy as np

class CalibReader:
    """
    Parser for DRNU calibration file
    """

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.calib_data = np.zeros((1, 1, 1))

    def load_calib_file(self):
        """
        Parses and loads binary DRNU calibration file into variable self.calib_data.
        """
        with open(self.file_path, 'rb') as file:
            self.calib_data = np.fromfile(file, dtype=np.uint16)
            self.calib_data = self.calib_data.reshape((50, 240, 320))  # 50 delay line steps, 240 x 320 pixels

        print('Calibration data loaded,', np.count_nonzero(self.calib_data), 'non-zero elements found.')

if __name__ == '__main__':
    file_path = r'C:\Data\01_NFL\calib_data\W455_C266\W455_C266_10000_drnu_images.bin'

    reader = CalibReader(file_path)
    reader.load_calib_file()

