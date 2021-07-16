import sys
from time import sleep
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import pyvisa

assert len(sys.argv) == 2

n_meas = 30
delay = 0.3  # s

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
    ':SENS:INT': '200000', 
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
# for command in init_commands:
#     print(command, hera.query(command+'?'))

# Get wavelengths
hera.write(':get:wave')
wl = np.frombuffer(hera.read_raw(), dtype='>f')

spectra = wl
# Make measurements
for n in range(n_meas):
    print(f"Measurement {n}/{n_meas}") 
    hera.write(':meas:spec')
    s = np.frombuffer(hera.read_raw(), dtype='>f')
    print(s[0])
    spectra = np.column_stack((spectra, s[1:]))  # cosa c'è nel primo byte??
    sleep(delay)

datestr = date.today().strftime("%Y%m%d")
filename = datestr + ' ' + sys.argv[1] + '.csv'
np.savetxt(filename, spectra, header=str(init_commands))

plt.plot(wl, spectra[:,1:].mean(axis=1))
plt.grid()
plt.show()