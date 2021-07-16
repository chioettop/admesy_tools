import logging
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.stats as st

# parameters for Savgol filter
windowlength = 41
polyorder = 3

# Matplotlib default plot size
# plt.rcParams['figure.figsize'] = [12, 9]

logger = logging.getLogger(__name__)

# from https://stackoverflow.com/questions/431684/how-do-i-change-the-working-directory-in-python/24176022#24176022
from contextlib import contextmanager
import os

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

# Silicon reference reflectivity for Si with .2 nm of SiO2
from SiSiO2_theor import model_x, model_r, model_std

# hot pixels
hp = [652, 653]

def load_spectra(files, plot=False):
    if type(files) is not list:
        files = [files]
    data = np.hstack([np.loadtxt(file, skiprows=2, delimiter=',') for file in files])
    n_measures = data.shape[1] // 2 
    # add: check that x are all equal
    # wavelengths are in the first column
    x = data[:,0] 
    # intensities in each even number column
    intensities = data[:,1::2]
    
    return x, intensities, n_measures

def load_avg_spectra(filename, plot=False):
    """ 
    Loads data from a multiple measurements file and returns the average measurement and
    statistics.
    """
    x, intensities, n_measures = load_spectra(filename)
    
    I = np.mean(intensities, axis=1)
    std = np.std(intensities, axis=1)
    
    t, p = st.normaltest(I, axis=0)
    logger.info('test')
    logger.info(f"Avg t: {t.flatten().mean():.2f}, % of p>5%: \
                {np.count_nonzero(p.flatten()>=.05)/p.size:.2f}")
    
    if plot:
        plt.plot(x, intensities)
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':')
        plt.grid(which='major')
        plt.show()
        
    return x, I, std, n_measures

def load_avg_measurement(dirs, refls, plot=False):
    # take every pair of dir/ref in turn and return the average measurement
    
    # loads spectra into a list of tuples (x, r, std)
    spectra = [calc_spectrum(s, r) for s, r in zip(dirs, refls)]
   
    x = spectra[0][0] # take x from the first measurement, assuming they are all equal
    r_avg = np.mean([r for x,r,s,n in spectra], axis=0)
    s_avg = np.sqrt(np.mean([s**2 for x,r,s,n in spectra], axis=0))
    # remove hot pixels
    r_avg[hp] = np.NaN
            
    return spectra[0][0], r_avg, s_avg


# Load and plot a series of measurements (one plot for each file)
def compare_spectra(files, window=slice(None), ax=None, diff=False, ci=.95, **kwargs):
    if type(files) is str:
        files = glob.glob(files)
    
    assert files # assert files list is not empty
    
    if diff:
        # compute average spectrum
        x, I_avg, std_avg, n_measures = load_avg_spectra(files)
    
    for file in files:
        x, I, std, n_measures = load_avg_spectra(file)
        if diff:
            I = I/I_avg

        critical_value = st.t.interval(ci, df=n_measures-1)[1]
        s = critical_value * std / np.sqrt(n_measures)

        ax = plot_spectrum(x, I, window=window, std=s, ax=ax, label=file + ', ' + str(n_measures) + ' meas.', ylabel='Intensity', **kwargs)
    
    if diff:
        ax.set_title('Intensity/Avg. of Intensities')
        
    ax.legend()
    
    return ax


def snr_spectra(files, window=slice(None), ax=None, **kwargs):
    if type(files) is str:
        files = glob.glob(files)
    
    assert files # assert files list is not empty
    
    for file in files:
        x, I, std, n_measures = load_avg_spectra(file)
        ax = plot_spectrum(x, I/std, window=window, ax=ax, label=file + ', ' + str(n_measures) + ' meas.', ylabel='SNR', **kwargs)
    
    ax.set_title('SNR')
        
    ax.legend()
    
    return ax


def compare_measurements(refs, meas, labels=None, window=slice(None), ax=None, std=False,
                         ci=0.95, diff=False, ylabel=None, calib=1, **kwargs):
    """
    Load and plot a series of measurements (one plot for each file)
    

    Parameters
    ----------
    refs : list
        list of reference measurements. 
        If only one element, it is used for all measurements.
    meas : list
        list of measurements.
    labels : TYPE, optional
        DESCRIPTION. The default is None.
    window : TYPE, optional
        DESCRIPTION. The default is slice(None).
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    std : TYPE, optional
        DESCRIPTION. The default is False.
    ci : float, optional
        confidence interval percentage to plot. Default is 95% (two sided).
    diff : TYPE, optional
        DESCRIPTION. The default is False.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.
    calib : ndarray, optional
        Calibration coefficients. The default is 1.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    Matplotlib axes.

    """
    # loads spectra into a list of tuples (x, r, std, n_samples)
    if len(refs) == 1:
        # if there is only one reference, use it for all measurements
        spectra = [calc_spectrum(s, refs) for s in meas]
    elif len(refs) == len(meas):
        spectra = [calc_spectrum(s, r) for s, r in zip(refs, meas)]
    else:
        raise ValueError("Different number of references and measurements") 
    
    if labels is None:
        labels = [f'{s} / {r}' for s, r in zip(refs, meas)]
    
    if diff:
        r_avg = np.mean([r for x,r,s in spectra], axis=0)
           
    for (x, r, s, n), label in zip(spectra, labels):
        if diff:
            r = r / r_avg
        else:
            r = r / calib
            
        # remove hot pixels
        r[hp] = np.NaN    
        if std:
            critical_value = st.t.interval(ci, df=n-1)[1]
            s = critical_value * s / np.sqrt(n)
        else:
            s = None

        ax = plot_spectrum(x, r, window, ax=ax, std=s, label=label, **kwargs)

    if ylabel is None:
        ylabel = 'Reflectance' if not diff else 'Diff. with average'
    ax.set_ylabel(ylabel)

    leg = ax.legend()
    
    handles, _ = ax.get_legend_handles_labels()
    
    lined = dict()
    
    for legline, origline in zip(leg.get_lines(), handles):
        legline.set_pickradius(15)  # 15 pts tolerance
        lined[legline] = origline
    
    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        ax.figure.canvas.draw()
        
    ax.figure.canvas.mpl_connect('pick_event', onpick)
    
    return ax

def plot_meas_wref(mref, msample, cref=1, labels=None, window=slice(None), ax=None,
                   ci=0.95, diff=False, label=None, ylabel=None, **kwargs):
    """
    Load and plot a series of measurements (one plot for each file)
    

    Parameters
    ----------
    mref : list
        list of measurements of the reflectivity reference. 
    msample : list
        list of measurements of the sample.
    cref : array
        nominal reflectivity of the reference.
    labels : TYPE, optional
        DESCRIPTION. The default is None.
    window : TYPE, optional
        DESCRIPTION. The default is slice(None).
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    ci : float, optional
        confidence interval percentage to plot. Default is 95% (two sided).
    diff : TYPE, optional
        DESCRIPTION. The default is False.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    Matplotlib axes.

    """
    # loads spectra into list of tuples (x, y, std, n_measurements)
    sp_ref = [load_avg_spectra(m) for m in mref]
    sp_sample = [load_avg_spectra(m) for m in msample]
    
    # check that x's are the same and cref is same length
    sp_ref_x = np.array([x for x,y,s,n in sp_ref]).T
    x = sp_ref_x[:,0]
    assert(all([np.array_equal(x, sp_ref_x[:,t]) for t in range(1, len(sp_ref))]))

    sp_sample_x = np.array([x for x,y,s,n in sp_sample]).T
    assert(all([np.array_equal(x, sp_sample_x[:,t]) for t in range(len(sp_sample))])) 
    
    if not np.isscalar(cref):
        assert(len(x) == len(cref))
    
    # calculate reflectivity and statistics
    sp_ref_y = np.array([y for x,y,s,n in sp_ref]).T
    ref_mean = np.mean(sp_ref_y, axis=1)
    ref_stderr = np.std(sp_ref_y, axis=1)/np.sqrt(len(sp_ref))
    
    sp_sample_y = np.array([y for x,y,s,n in sp_sample]).T
    sample_mean = np.mean(sp_sample_y, axis=1)
    sample_stderr = np.std(sp_sample_y, axis=1)/np.sqrt(len(sp_sample))
        
    R = sample_mean / ref_mean * cref
    std = np.abs(R) * np.sqrt(
        (sample_stderr/sample_mean)**2 + (ref_stderr/ref_mean)**2
    )  # error propagation (propagating standard error instead of std)
    n = min(len(sp_ref), len(sp_sample))  # sample size
    critical_value = st.t.ppf(.5 + ci/2, df=n-1)  # ppf returns the one-sided value
    ci_hw = critical_value * std / np.sqrt(n)  # confidence interval half width
    logger.info(f"ref n {len(sp_ref)}, sample n {len(sp_sample)}, \
                 critical value {critical_value:.3f}")
    
    # remove hot pixels
    R[hp] = np.NaN
    ci_hw[hp] = np.NaN
    
    ax = plot_spectrum(x, R, window, ax=ax, std=ci_hw, label=label, **kwargs)
    ax.set_title(f"({ci*100:.1f}% confidence interval)", fontsize=10)

    if ylabel is None:
        ylabel = 'Reflectance'
    ax.set_ylabel(ylabel)

    return ax


def compare_si_meas(dirs, refls, labels=None, window=slice(None), **kwargs):
    
    ax = plot_spectrum(model_x, model_r, window=window, std=model_std*1.5, label='SiSiO2 Theor. (±1.5%)')
    
    compare_measurements(dirs, refls, labels=labels, window=window, ax=ax, **kwargs)
    
    return ax


def calc_spectrum(direct, indirect):
    # load two multiple-measurement files and calculate transmissivity/reflectivity
    x_d, d, std_d, n_d = load_avg_spectra(direct)
    x_i, i, std_i, n_i = load_avg_spectra(indirect)
    assert(np.array_equal(x_d, x_i)) # wavelengths must coincide
    # calculated spectrum
    sp = i / d
    
    std = np.abs(sp) * np.sqrt((std_d/d)**2 + (std_i/i)**2) 
    
    # some spectra have a wavelength step of 0.5 nm: resample
    if (x_d[1] - x_d[0]) < 1:
        return x_d[::2], sp[::2], std[::2]
    else:
        return x_d, sp, std, min(n_d, n_i) 

# alias
transmissivity = calc_spectrum        

# plot spectrum
def plot_spectrum(x, sp, window=slice(None), ax=None, std=None, 
                  smooth=False, ylabel='Reflectance', **kwargs):
    # std: half-width of the error bar to plot
    if type(window) is tuple:
        a, b = window
        window = np.where((x >= a) & (x <= b))
    
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':')
    ax.grid(True, which='major')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(ylabel)    
    
    s = sp
    
    if smooth:
        if np.any(np.isnan(s)):
            s[hp] = np.interp(hp, x, s)  # if there are NaNs, then hot pixels have been filtered, so interpolate them
        s = savgol_filter(s, window_length=windowlength, polyorder=polyorder)
    
    p = ax.plot(x[window], s[window], **kwargs)
    
    if std is not None:
        if smooth:
            std = savgol_filter(std, window_length=windowlength, polyorder=polyorder)
        ax.fill_between(x[window], s[window] - std[window], s[window] + std[window],
                        color=p[0].get_color(), alpha=0.2)
    
    return ax
    
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

def reproducibility(measures):
    '''
    Calculates reproducibility of a set of measurements, as max variation / mean measurement
    '''
    
    r_max = np.nanmax(measures, axis=0)
    r_min = np.nanmin(measures, axis=0)
    r_mean = np.nanmean(measures, axis=0)
    r_var = np.nanmax([np.abs(r_max - r_mean), np.abs(r_mean - r_min)], axis=0)

    return r_var/r_mean

#transmitted =  path+'Finestra trasmesso 1834.csv'
#x, T, std = transmissivity(direct, transmitted)
#plot_spectrum(x, T, (400, 1000))

    
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