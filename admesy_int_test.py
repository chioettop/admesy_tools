import sys
from time import sleep
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pyvisa

parser = argparse.ArgumentParser()
parser.add_argument("name", 
                    help="name of file")
parser.add_argument("start", type=int, 
                    help="start integration time in microseconds")
parser.add_argument("stop", type=int, 
                    help="stop integration time in microseconds")                 
parser.add_argument("-s", "--step", type=int, default=1000, 
                    help="integration time step in microseconds")
parser.add_argument("-d", "--delay", type=float, default=0.1, 
                    help="delay between measurements in seconds")
parser.add_argument("-m", "--n_meas", type=int, default= 30,
                    help="number of measurements to make")
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
    ':SENS:INT': str(args.start), 
    ':SENS:SP:AVERAGE': '1', 
    ':SENS:CAL': '0',
    ':SENS:SP:SBW':  'off',
    ':SENS:AUTORANGE': '0',
    ':SENS:RES': '1'
}

# Initialize spectrometer
for command, value in init_commands.items():
    hera.write(command + ' ' + value)
    
# Read back configuration
#for command in init_commands:
#    print(command, hera.query(command+'?'))

# Sensor temperature
init_commands['sensor_temp'] = hera.query(':MEAS:TEMP')

# Get wavelengths
hera.write(':get:wave')
wl = np.frombuffer(hera.read_raw(), dtype='>f')

spectra = [wl]
fig = plt.figure()
plt.xlim((450, 1000))
plt.ylim((.9, 1.1))
plt.grid()

# Make measurements
int_range = range(args.start, args.stop+1, args.step)
for it in int_range:
    hera.write(f':SENS:INT {it}')
    hera.write(':meas:spec')
    s_temp = []
    clipping = 0
    for n in range(args.n_meas):
        hera.write(':meas:spec')
        s = np.frombuffer(hera.read_raw(), dtype='>f')
        clipping += s[0]
        s_temp.append(s[1:])
        sleep(delay)

    print(f"Int time {it}, clipping {s[0]:.2f}")
    s_temp = np.mean(np.column_stack(s_temp), axis=1)
    spectra.append(s_temp)
    plt.plot(wl, s_temp/spectra[1])
    plt.pause(1)
    
plt.ioff()
spectra = np.column_stack(spectra)
datestr = date.today().strftime("%Y%m%d")
filename = datestr + ' ' + args.name + '_it test.csv'
init_commands['it_range'] = int_range
np.savetxt(filename, spectra, header=str(init_commands))
plt.show()