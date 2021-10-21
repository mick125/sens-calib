from pathlib import Path
from CalibDataProcessor import CalibDataProcessor
from RawDataProcessor import RawDataProcessor
import matplotlib.pyplot as plt

"""
This script can open any number of files with measured data and plot the calibration curve in a single plot.
Depending on the data format, CalibDataProcessor or RawDataProcessor class shall be used.
Respective modifications in the code must be done in order to switch these classes.
"""

input_file_path = [
                    r'<file_1>>',
                    r'<file_2>>',
                    r'<file_3>>',
                    r'<file_4>>'
                  ]

output_path = r'<output path>'

file_title = 'anything you like'

n_delay_steps = 50

# x-axis data
x_ax_data = CalibDataProcessor.dll_to_mm(range(n_delay_steps))

readers = []

# predefined markers and their color
markers = ['rs', 'bo', 'g+', 'ch', 'mx', 'yp']

# loop through files and create a class for each
for i in range(len(input_file_path)):
    # readers.append(CalibDataProcessor(input_file_path[i], output_path))
    readers.append(RawDataProcessor(input_file_path[i], output_path))

    # readers[i].load_raw_file()
    readers[i].create_folders()
    readers[i].load_raw_file_DCS()
    readers[i].convert_dcs_to_mm()
    readers[i].calc_calib_data()
    readers[i].calc_mean_std_all_dll()
    readers[i].compensate_rollover()

    print('Plotting mean value of multiple measurement...', end=' ')

    plt.plot(x_ax_data, readers[i].mean_vs_dll, markers[i], markersize=3, label=f'{readers[i].chip}, {readers[i].mod_frequency/1e6:.1f} MHz')

# plot settings
plt.grid(True)
plt.xlabel('True distance [mm]')
plt.ylabel('Measured distance mean [mm]')
plt.title(f'{readers[i].chip}')
plt.legend()

out_path = Path(output_path).joinpath(f'{readers[0].chip}_{readers[0].mod_frequency/1e6:.1f}MHz', f'mean_std_{file_title}.png')
plt.savefig(out_path, dpi=150)
print(f'Figure saved:\n{out_path}')
plt.close()

print('done')
