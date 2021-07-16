import logging
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.stats as st

# parameters for Savgol filter
windowlength = 21
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
import SiSiO2_theor 

# hot pixels
hp = [852]  # wavelength in nm

def load_avg_spectrum(filename, plot=False):
    """ 
    Loads data from a multiple measurements file and returns the average measurement and
    statistics.
    """
    data = np.loadtxt(filename)
    x = data[:,0]  # x's in first column
    intensities = data[:,1:]
    
    I = np.mean(intensities, axis=1)
    std = np.std(intensities, axis=1)
    n_measures = intensities.shape[1]
    
    t, p = st.normaltest(I, axis=0)
    logger.info(f"Avg t: {t.flatten().mean():.2f}, % of p>5%: \
                {np.count_nonzero(p.flatten()>=.05)/p.size:.2f}")
    
    if plot:
        plt.plot(x, intensities)
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':')
        plt.grid(which='major')
        plt.show()
        
    return {'x': x, 'i': I, 'std': std, 'n': n_measures}

def pickable_legend(ax):

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

def interp_nans(x, y):
    # linear interpolation of NaN values in y array
    isnan = np.isnan(y)
    if isnan.any():
        y[isnan] = np.interp(x[isnan], x[~isnan], y[~isnan]).flatten()
    return y

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
        s = savgol_filter(interp_nans(x, s), window_length=windowlength, polyorder=polyorder)
    
    p = ax.plot(x[window], s[window], **kwargs)
    
    if std is not None:
        if smooth:
            std = savgol_filter(interp_nans(x, std), window_length=windowlength, polyorder=polyorder)
        ax.fill_between(x[window], s[window] - std[window], s[window] + std[window],
                        color=p[0].get_color(), alpha=0.2)
    
    return ax

def plot_measurement(refms, samplems, alref, window=slice(None), ax=None,
                   ci=0.95, diff=False, label=None, ylabel=None, **kwargs):
    """
    Calculates and plots a reflectivity measurement made with a reflectivity reference sample.
    

    Parameters
    ----------
    refms : list
        list of measurements of the reflectivity reference. 
    samplems : list
        list of measurements of the sample to measure.
    alref : array
        nominal reflectivity data of the reference (two columns array with wavelentght in nm and reflectance).
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
    # loads spectra
    if type(refms) is not list:
        refms = [refms]
    ref = [load_avg_spectrum(m) for m in refms]
    if type(samplems) is not list:
        samplems = [samplems]
    sample = [load_avg_spectrum(m) for m in samplems]
    
    # check that x's are the same and cref is same length
    xs = np.array([m['x'] for m in ref + sample])
    x = xs[0]
    assert (xs == x).all()

    # filter reference nominal reflectivity by measured wavelengths
    wvls, indices_ref, indices_meas = np.intersect1d(alref[:,0], x, return_indices=True)
    assert np.array_equal(x, wvls)  # only handle the case where x is a subset of alref_x
    nref_r = alref[indices_ref][:,1]
    
    # calculate reflectivity and statistics
    ref_i = [m['i'] for m in ref]
    ref_mean = np.mean(ref_i, axis=0)
    ref_stderr = np.std(ref_i, axis=0) / np.sqrt(len(ref))
    
    sample_i = [m['i'] for m in sample]
    sample_mean = np.mean(sample_i, axis=0)
    sample_stderr = np.std(sample_i, axis=0) / np.sqrt(len(sample))
        
    R = nref_r * sample_mean / ref_mean
    # remove hot pixels
    hp_in_x = np.isin(x, hp)
    R[hp_in_x] = np.NaN

    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(f"Reflectance measurement with ref {refms[0]}")
    
    n = min(len(ref), len(sample))  # sample size

    if n > 1:
        # more than one sample: we can calculate t statistic
        std = np.abs(R) * np.sqrt(
            (sample_stderr/sample_mean)**2 + (ref_stderr/ref_mean)**2
        )  # error propagation (propagating standard error instead of std)
        critical_value = st.t.ppf(.5 + ci/2, df=n-1)  # ppf returns the one-sided value
        ci_hw = critical_value * std / np.sqrt(n)  # confidence interval half width
        logger.info(f"ref n {len(ref)}, sample n {len(sample)}, \
                     critical value {critical_value:.3f}")
        ci_hw[hp_in_x] = np.NaN  # remove hot pixels
        ax.set_title(f"({ci*100:.1f}% confidence interval)", fontsize=10)

    else: 
        ci_hw = None
        
    if label is None:
        label = f"{samplems[0]}, {n} measurements"
    
    plot_spectrum(x, R, window, ax=ax, std=ci_hw, label=label, **kwargs)

    if ylabel is None:
        ylabel = 'Reflectance'
    ax.set_ylabel(ylabel)

    return ax


def compare_si_meas(dirs, refls, labels=None, window=slice(None), **kwargs):
    
    ax = plot_spectrum(model_x, model_r, window=window, std=model_std*1.5, label='SiSiO2 Theor. (±1.5%)')
    
    compare_measurements(dirs, refls, labels=labels, window=window, ax=ax, **kwargs)
    
    return ax