import sys
from time import sleep
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import pyvisa

assert len(sys.argv) == 2

n_meas = 30
delay = 0.2  # s

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
    ':SENS:INT': '15000', 
    ':SENS:SP:AVERAGE': '10', 
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

# Get wavelengths
hera.write(':get:wave')
wl = np.frombuffer(hera.read_raw(), dtype='>f')

spectra = wl
# Make measurements
int_range = range(16000, 300000, 5000)
for n in int_range:
    print(f"Int time {n}")
    hera.write(f':SENS:INT {n}')
    hera.write(':meas:spec')
    sleep(n*10/1e6)
    s = np.frombuffer(hera.read_raw(), dtype='>f')
    spectra = np.column_stack((spectra, s[1:]))  # cosa c'è nel primo byte??
    sleep(delay)

datestr = date.today().strftime("%Y%m%d")
filename = datestr + ' ' + sys.argv[1] + '.csv'
np.savetxt(filename, spectra, header=str(init_commands))

lo = plt.plot(wl, spectra[:,1:])
#plt.legend(lo, int_range)
plt.grid()
plt.show()