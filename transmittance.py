import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# parameters nor sav'ol_filt-

windowlength = 41
polyorder = 3

# load data from a multiple measurement file and returns the average measurement and statistics
def load_avg_spectra(filename, plot=False):
    data = np.loadtxt(filename, skiprows=2, delimiter=',')
    n_measures = data.shape[1] // 2 
    # add: check that x are all equal
    # wavelengths are in the first column
    x = data[:,0] 
    # intensities in each even number column
    intensities = data[:,1:-1:2]
    I = np.mean(intensities, axis=1)
    std = np.std(intensities, axis=1)
    
    if plot:
        plt.plot(data[:,0], intensities)
        plt.grid()
        plt.show()
        
    return x, I, std, n_measures

# load two multiple-measurement files and calculate transmissivity
def transmissivity(direct, transmitted):
    x_d, d, std_d, n_d = load_avg_spectra(direct)
    x_t, t, std_t, n_t = load_avg_spectra(transmitted)
    assert(np.array_equal(x_d, x_t)) # wavelengths must coincide
    return x_d, t / d, std_d + std_t # sicuro che si sommano le std? Non sarà RSS?
    
# plot spectrum
def plot_spectrum(x, spectrum, window=slice(None)):
    if type(window) is tuple:
        a, b = window
        window = np.where((x >= a) & (x <= b))
    plt.plot(x[window], spectrum[window])
    #plt.legend()
    plt.grid()
    plt.show()
    
def plot_sigma(direct, transmitted, window=slice(None)):
    x, d, std_d, n_d = load_avg_spectra(direct)
    x_t, t, std_t, n_t = load_avg_spectra(transmitted)

    if type(window) is tuple:
        a, b = window
        window = np.where((x >= a) & (x <= b))

    assert(np.array_equal(x, x_t)) # wavelengths must coincide
    plt.plot(x[window], std_d[window], label='Direct')
    plt.plot(x[window], std_t[window], label='Transmitted')
    plt.grid()
    plt.legend()
    plt.show()

path = "Misure Riflettività 01 08 2019\\"
direct = path+'Finestra diretto 1836.csv'
transmitted =  path+'Finestra trasmesso 1834.csv'
x, T, std = transmissivity(direct, transmitted)
plot_spectrum(x, T, (400, 1000))

    
#direct1 = 'Oro 8g diretto vicino setup 2 20190801 1715.csv'
#transmitted1 =  'Oro 8g diretto lontano setup 2 20190801 1654.csv'
#x1, T1, std1 = transmissivity(direct1, transmitted1)
#plt.plot(x1, T1, label='setup 2')
#
#direct2 = 'Oro 8g diretto vicino 20190801 1621.csv'
#transmitted2 =  'Oro 8g diretto lontano 20190801 1624.csv'
#x2, T2, std2 = transmissivity(direct2, transmitted2)
#plt.plot(x2, T2, label='setup 1')
#
#plt.grid()
#plt.legend()

#plot_spectrum(x, T)
#plot_sigma(direct, transmitted)