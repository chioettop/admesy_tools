from time import sleep
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pyvisa

parser = argparse.ArgumentParser()
parser.add_argument("name", 
                    help="name of file")
parser.add_argument("-n", "--n_meas", type=int, default= 30,
                    help="number of measurements to make")
parser.add_argument("-d", "--delay", type=float, default=0.3, 
                    help="delay between measurements in seconds")
parser.add_argument("-i", "--int_time", type=int, default=200000,
                    help="integration time in microseconds")
parser.add_argument("-a", "--avg", type=int, default=1,
                    help="no. of measures for spectrometer internal averaging ")
parser.add_argument("--raw", action="store_true",
                    help="reads a raw spectrum")                    
                 
args = parser.parse_args()

n_meas = args.n_meas
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

if args.raw:
    spectra = []
else:
    spectra = [wl]
    
clipping = 0
# Make measurements
for n in range(n_meas):
    if args.raw:
        hera.write(':meas:rawspec')
        spectra.append(np.frombuffer(hera.read_raw(), dtype='>f').astype(int))
        print(f"Measurement {n}/{n_meas}")
    else:    
        hera.write(':meas:spec')
        s = np.frombuffer(hera.read_raw(), dtype='>f')
        print(f"Measurement {n}/{n_meas}, clipping {s[0]:.2f}") 
        clipping += s[0]
        spectra.append(s[1:])
    sleep(delay)
        
spectra = np.column_stack(spectra)
datestr = date.today().strftime("%Y%m%d")
filename = datestr + ' ' + args.name + ('_raw' if args.raw else '') + '.csv'
init_commands['clipping'] = clipping/n_meas
np.savetxt(filename, spectra, header=str(init_commands)+str(args))

if args.raw:
    plt.plot(spectra.mean(axis=1))
else:
    plt.plot(wl, spectra[:,1:].mean(axis=1))
plt.grid()
plt.show()