from time import sleep
from datetime import date
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import pyvisa

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--delay", type=float, default=0.3, 
                    help="delay between measurements in seconds")
parser.add_argument("-i", "--int_time", type=int, default=200000,
                    help="integration time in microseconds")
parser.add_argument("-a", "--avg", type=int, default=1,
                    help="no. of measures for spectrometer internal averaging ")
#parser.add_argument("--raw", action="store_true",
#                    help="reads a raw spectrum")                    
                 
args = parser.parse_args()

delay = args.delay  # s

rm = pyvisa.ResourceManager('C:\WINDOWS\system32\\visa64.dll')
#rm.list_resources()
hera = rm.open_resource('USB0::0x23CF::0x1023::00368::INSTR')
print(hera.query(':*TST'))
print(hera.query(':*IDN?'))

"""
# wavelength
hera.write(':get:wave')
wl = np.frombuffer(hera.read_raw(), dtype='>f')

# spectrum
hera.write(':meas:spec')
s = np.frombuffer(hera.read_raw(), dtype='>f')
s[1:] # cos'è il primo punto??

# raw spectrum
hera.write(':meas:rawspec')
s = np.frombuffer(hera.read_raw(), dtype='>f')
"""

init_commands = {
    ':SENS:INT': str(args.int_time), 
    ':SENS:SP:AVERAGE': str(args.avg), 
    ':SENS:CAL': '0',
    ':SENS:SP:SBW':  'off',
    ':SENS:AUTORANGE': '0',
    ':SENS:RES': '1'
}

# Initialize spectrometer
for command, value in init_commands.items():
    hera.write(command + ' ' + value)
    
# Read back configuration
# for command in init_commands:
#     print(command, hera.query(command+'?'))

# Sensor temperature
init_commands['sensor_temp'] = hera.query(':MEAS:TEMP')

# Get wavelengths
hera.write(':get:wave')
wl = np.frombuffer(hera.read_raw(), dtype='>f')

def hera_read_raw():
    hera.write(':meas:rawspec')
    spectrum = np.frombuffer(hera.read_raw(), dtype='>f').astype(int)
    return savgol_filter(spectrum, 51, 3)

fig, ax = plt.subplots()
max_spectrum = hera_read_raw()
line, = plt.plot(range(len(max_spectrum)), max_spectrum)
max_line, = plt.plot(range(len(max_spectrum)), max_spectrum, 'r')

def update(frame):
    global max_spectrum
    spectrum = hera_read_raw()
    if np.mean(max_spectrum) < np.mean(spectrum):
        max_line.set_data(range(len(spectrum)), spectrum)
        max_spectrum = spectrum

    line.set_data(range(len(spectrum)), spectrum)
    mean_spectrum = np.mean(spectrum)
    mean_max_spectrum = np.mean(max_spectrum)
    print(f"{mean_spectrum:.1f} \t {mean_max_spectrum:.1f} \t {mean_spectrum/mean_max_spectrum*100:.1f}", end='\r')
    return line, max_line

print("Avg \t Max \t %") 
animation = FuncAnimation(fig, update, interval=delay, blit=True)

plt.show()
    