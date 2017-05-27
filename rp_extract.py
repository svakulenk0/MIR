'''

RP_extract: Rhythm Patterns Audio Feature Extractor

@author: 2014-2015 Alexander Schindler, Thomas Lidy


Re-implementation by Alexander Schindler of RP_extract for Matlab
Matlab version originally by Thomas Lidy, based on Musik Analysis Toolbox by Elias Pampalk
( see http://ifs.tuwien.ac.at/mir/downloads.html )

Main function is rp_extract. See function definition and description for more information,
or example usage in main function.

Note: All required functions are provided by the two main scientific libraries numpy and scipy.

Note: In case you alter the code to use transform2mel, librosa needs to be installed: pip install librosa
'''


import numpy as np

from scipy import stats
from scipy.fftpack import fft
#from scipy.fftpack import rfft #  	Discrete Fourier transform of a real sequence.
from scipy import interpolate

# suppress numpy warnings (divide by 0 etc.)
np.set_printoptions(suppress=True)

# required for debugging
np.set_printoptions(precision=8,
                    threshold=10,
                    suppress=True,
                    linewidth=200,
                    edgeitems=10)


# INITIALIZATION: Constants & Mappings

# Bark Scale
bark = [100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]
n_bark_bands = len(bark)

# copy the bark vector (using [:]) and add a 0 in front (to make calculations below easier)
barks = bark[:]
barks.insert(0,0)

# Phone Scale
phon = [3, 20, 40, 60, 80, 100, 101]

# copy the bark vector (using [:]) and add a 0 in front (to make calculations below easier)
phons     = phon[:]
phons.insert(0,0)
phons     = np.asarray(phons)


# Loudness Curves

eq_loudness = np.array([[55,  40, 32, 24, 19, 14, 10,  6,  4,  3,  2,   2, 0,-2,-5,-4, 0,  5, 10, 14, 25, 35], 
                        [66,  52, 43, 37, 32, 27, 23, 21, 20, 20, 20,  20,19,16,13,13,18, 22, 25, 30, 40, 50], 
                        [76,  64, 57, 51, 47, 43, 41, 41, 40, 40, 40,39.5,38,35,33,33,35, 41, 46, 50, 60, 70], 
                        [89,  79, 74, 70, 66, 63, 61, 60, 60, 60, 60,  59,56,53,52,53,56, 61, 65, 70, 80, 90], 
                        [103, 96, 92, 88, 85, 83, 81, 80, 80, 80, 80,  79,76,72,70,70,75, 79, 83, 87, 95,105], 
                        [118,110,107,105,103,102,101,100,100,100,100,  99,97,94,90,90,95,100,103,105,108,115]])

loudn_freq = np.array([31.62, 50, 70.7, 100, 141.4, 200, 316.2, 500, 707.1, 1000, 1414, 1682, 2000, 2515, 3162, 3976, 5000, 7071, 10000, 11890, 14140, 15500])

# We have the loudness values for the frequencies in loudn_freq
# now we calculate in loudn_bark a matrix of loudness sensation values for the bark bands margins

i = 0
j = 0

loudn_bark = np.zeros((eq_loudness.shape[0], len(bark)))

for bsi in bark:

    while j < len(loudn_freq) and bsi > loudn_freq[j]:
        j += 1
    
    j -= 1
    
    if np.where(loudn_freq == bsi)[0].size != 0: # loudness value for this frequency already exists
        loudn_bark[:,i] = eq_loudness[:,np.where(loudn_freq == bsi)][:,0,0]
    else:
        w1 = 1 / np.abs(loudn_freq[j] - bsi)
        w2 = 1 / np.abs(loudn_freq[j + 1] - bsi)
        loudn_bark[:,i] = (eq_loudness[:,j]*w1 + eq_loudness[:,j+1]*w2) / (w1 + w2)
    
    i += 1



# SPECTRAL MASKING Spreading Function
# CONST_spread contains matrix of spectral frequency masking factors

CONST_spread = np.zeros((n_bark_bands,n_bark_bands))

for i in range(n_bark_bands):
    CONST_spread[i,:] = 10**((15.81+7.5*((i-np.arange(n_bark_bands))+0.474)-17.5*(1+((i-np.arange(n_bark_bands))+0.474)**2)**0.5)/10)

# UTILITY FUNCTIONS

def nextpow2(num):
    '''NextPow2

    find the next highest number to the power of 2 to a given number
    and return the exponent to 2
    (analogously to Matlab's nextpow2() function)
    '''

    n = 2
    i = 1
    while n < num:
        n *= 2 
        i += 1
    return i



# FFT FUNCTIONS

def periodogram(x,win,Fs=None,nfft=1024):
    ''' Periodogram

    Periodogram power spectral density estimate
    Note: this function was written with 1:1 Matlab compatibility in mind.

    The number of points, nfft, in the discrete Fourier transform (DFT) is the maximum of 256 or the next power of two greater than the signal length.

    :param x: time series data (e.g. audio signal), ideally length matches nfft
    :param win: window function to be applied (e.g. Hanning window). in this case win expects already data points of the window to be provided.
    :param Fs: sampling frequency (unused)
    :param nfft: number of bins for FFT (ideally matches length of x)
    :return: Periodogram power spectrum (np.array)
    '''


    #if Fs == None:
    #    Fs = 2 * np.pi         # commented out because unused
   
    U  = np.dot(win.conj().transpose(), win) # compensates for the power of the window.
    Xx = fft((x * win),nfft) # verified
    P  = Xx*np.conjugate(Xx)/U
    
    # Compute the 1-sided or 2-sided PSD [Power/freq] or mean-square [Power].
    # Also, compute the corresponding freq vector & freq units.
    
    # Generate the one-sided spectrum [Power] if so wanted
    if nfft % 2 != 0:
        select = np.arange((nfft+1)/2)  # ODD
        P = P[select,:] # Take only [0,pi] or [0,pi)
        P[1:-1] = P[1:-1] * 2 # Only DC is a unique point and doesn't get doubled
    else:
        #select = np.arange(nfft/2+1);    # EVEN
        #P = P[select,:]         # Take only [0,pi] or [0,pi) # TODO: why commented out?
        P[1:-2] = P[1:-2] * 2

    P = P / (2 * np.pi)

    return P




def calc_spectrogram(wavsegment,fft_window_size,fft_overlap = 0.5,real_values=True):

    ''' Calc_Spectrogram

    calculate spectrogram using periodogram function (which performs FFT) to convert wave signal data
    from time to frequency domain (applying a Hanning window and (by default) 50 % window overlap)

    :param wavsegment: audio wave file data for a segment to be analyzed (mono (i.e. 1-dimensional vector) only
    :param fft_window_size: windows size to apply FFT to
    :param fft_overlap: overlap to apply during FFT analysis in % fraction (e.g. default = 0.5, means 50% overlap)
    :param real_values: if True, return real values by taking abs(spectrogram), if False return complex values
    :return: spectrogram matrix as numpy array (fft_window_size, n_frames)
    '''

    # hop_size (increment step in samples, determined by fft_window_size and fft_overlap)
    hop_size = int(fft_window_size*(1-fft_overlap))

    # this would compute the segment length, but it's pre-defined above ...
    # segment_size = fft_window_size + (frames-1) * hop_size
    # ... therefore we convert the formula to give the number of frames needed to iterate over the segment:
    n_frames = (wavsegment.shape[0] - fft_window_size) / hop_size + 1
    # n_frames_old = wavsegment.shape[0] / fft_window_size * 2 - 1  # number of iterations with 50% overlap

    # TODO: provide this as parameter for better caching?
    han_window = np.hanning(fft_window_size) # verified

    # initialize result matrix for spectrogram
    spectrogram  = np.zeros((fft_window_size, n_frames), dtype=np.complex128)

    # start index for frame-wise iteration
    ix = 0

    for i in range(n_frames): # stepping through the wave segment, building spectrum for each window
        spectrogram[:,i] = periodogram(wavsegment[ix:ix+fft_window_size], win=han_window,nfft=fft_window_size)
        ix = ix + hop_size

        # NOTE: tested scipy periodogram BUT it delivers totally different values AND takes 2x the time of our periodogram function (0.13 sec vs. 0.06 sec)
        # from scipy.signal import periodogram # move on top
        #f,  spec = periodogram(x=wavsegment[idx],fs=samplerate,window='hann',nfft=fft_window_size,scaling='spectrum',return_onesided=True)

    if real_values: spectrogram = np.abs(spectrogram)

    return (spectrogram)


# FEATURE FUNCTIONS

def calc_statistical_features(matrix):

    result = np.zeros((matrix.shape[0],7))
    
    result[:,0] = np.mean(matrix, axis=1)
    result[:,1] = np.var(matrix, axis=1, dtype=np.float64) # the values for variance differ between MATLAB and Numpy!
    result[:,2] = stats.skew(matrix, axis=1)
    result[:,3] = stats.kurtosis(matrix, axis=1, fisher=False) # Matlab calculates Pearson's Kurtosis
    result[:,4] = np.median(matrix, axis=1)
    result[:,5] = np.min(matrix, axis=1)
    result[:,6] = np.max(matrix, axis=1)
    
    result[np.where(np.isnan(result))] = 0
    
    return result


# PSYCHO-ACOUSTIC TRANSFORMS as individual functions


# Transform 2 Mel Scale: NOT USED by rp_extract, but included for testing purposes or for import into other programs

def transform2mel(spectrogram,samplerate,fft_window_size,n_mel_bands = 80,freq_min = 0,freq_max = None):
    '''Transform to Mel

    convert a spectrogram to a Mel scale spectrogram by grouping original frequency bins
    to Mel frequency bands (using Mel filter from Librosa)

    Parameters
    spectrogram: input spectrogram
    samplerate: samplerate of audio signal
    fft_window_size: number of time window / frequency bins in the FFT analysis
    n_mel_bands: number of desired Mel bands, typically 20, 40, 80 (max. 128 which is default when 'None' is provided)
    freq_min: minimum frequency (Mel filters will be applied >= this frequency, but still return n_meld_bands number of bands)
    freq_max: cut-off frequency (Mel filters will be applied <= this frequency, but still return n_meld_bands number of bands)

    Returns:
    mel_spectrogram: Mel spectrogram: np.array of shape(n_mel_bands,frames) maintaining the number of frames in the original spectrogram
    '''

    from librosa.filters import mel

    # Syntax: librosa.filters.mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False)
    mel_basis = mel(samplerate,fft_window_size, n_mels=n_mel_bands,fmin=freq_min,fmax=freq_max)

    freq_bin_max = mel_basis.shape[1] # will be fft_window_size / 2 + 1

    # IMPLEMENTATION WITH FOR LOOP
    # initialize Mel Spectrogram matrix
    #n_mel_bands = mel_basis.shape[0]  # get the number of bands from result in case 'None' was specified as parameter
    #mel_spectrogram = np.empty((n_mel_bands, frames))

    #for i in range(frames): # stepping through the wave segment, building spectrum for each window
    #    mel_spectrogram[:,i] = np.dot(mel_basis,spectrogram[0:freq_bin_max,i])

    # IMPLEMENTATION WITH DOT PRODUCT (15% faster)
    # multiply the mel filter of each band with the spectogram frame (dot product executes it on all frames)
    # filter will be adapted in a way so that frequencies beyond freq_max will be discarded
    mel_spectrogram = np.dot(mel_basis,spectrogram[0:freq_bin_max,:])
    return (mel_spectrogram)




# Bark Transform: Convert Spectrogram to Bark Scale
# matrix: Spectrogram values as returned from periodogram function
# freq_axis: array of frequency values along the frequency axis
# max_bands: limit number of Bark bands (1...24) (counting from lowest band)
def transform2bark(matrix, freq_axis, max_bands=None):

    # barks and n_bark_bands have been initialized globally above

    if max_bands == None:
        max_band = n_bark_bands
    else:
        max_band = min(n_bark_bands,max_bands)

    matrix_out = np.zeros((max_band,matrix.shape[1]),dtype=matrix.dtype)

    for b in range(max_band-1):
        # TODO: VisibleDeprecationWarning: boolean index did not match indexed array along dimension 0; dimension is 1024 but corresponding boolean dimension is 513
        matrix_out[b] = np.sum(matrix[((freq_axis >= barks[b]) & (freq_axis < barks[b+1]))], axis=0)

    return(matrix_out)

# Spectral Masking (assumes values are arranged in <=24 Bark bands)
def do_spectral_masking(matrix):

    n_bands = matrix.shape[0]

    # CONST_spread has been initialized globally above
    spread = CONST_spread[0:n_bands,0:n_bands] # not sure if column limitation is right here; was originally written for n_bark_bands = 24 only
    matrix = np.dot(spread, matrix)
    return(matrix)

# Map to Decibel Scale
def transform2db(matrix):
    '''Map to Decibel Scale'''
    matrix[np.where(matrix < 1)] = 1
    matrix = 10 * np.log10(matrix)
    return(matrix)

# Transform to Phon (assumes matrix is in dB scale)
def transform2phon(matrix):

    old_npsetting = np.seterr(invalid='ignore') # avoid 'RuntimeWarning: invalid value encountered in divide' at ifac division below

    # number of bark bands, matrix length in time dim
    n_bands = matrix.shape[0]
    t       = matrix.shape[1]

    # DB-TO-PHON BARK-SCALE-LIMIT TABLE
    # introducing 1 level more with level(1) being infinite
    # to avoid (levels - 1) producing errors like division by 0

    #%%table_dim = size(CONST_loudn_bark,2);
    table_dim = n_bands; # OK
    cbv       = np.concatenate((np.tile(np.inf,(table_dim,1)), loudn_bark[:,0:n_bands].transpose()),1) # OK

    # init lowest level = 2
    levels = np.tile(2,(n_bands,t)) # OK

    for lev in range(1,6): # OK
        db_thislev = np.tile(np.asarray([cbv[:,lev]]).transpose(),(1,t))
        levels[np.where(matrix > db_thislev)] = lev + 2

    # the matrix 'levels' stores the correct Phon level for each data point
    cbv_ind_hi = np.ravel_multi_index(dims=(table_dim,7), multi_index=np.array([np.tile(np.array([range(0,table_dim)]).transpose(),(1,t)), levels-1]), order='F')
    cbv_ind_lo = np.ravel_multi_index(dims=(table_dim,7), multi_index=np.array([np.tile(np.array([range(0,table_dim)]).transpose(),(1,t)), levels-2]), order='F')

    # interpolation factor % OPT: pre-calc diff
    ifac = (matrix[:,0:t] - cbv.transpose().ravel()[cbv_ind_lo]) / (cbv.transpose().ravel()[cbv_ind_hi] - cbv.transpose().ravel()[cbv_ind_lo])

    ifac[np.where(levels==2)] = 1 # keeps the upper phon value;
    ifac[np.where(levels==8)] = 1 # keeps the upper phon value;

    # phons has been initialized globally above

    matrix[:,0:t] = phons.transpose().ravel()[levels - 2] + (ifac * (phons.transpose().ravel()[levels - 1] - phons.transpose().ravel()[levels - 2])) # OPT: pre-calc diff

    np.seterr(invalid=old_npsetting['invalid']) # restore RuntimeWarning setting for np division error

    return(matrix)


# Transform to Sone scale (assumes matrix is in Phon scale)
def transform2sone(matrix):
    idx     = np.where(matrix >= 40)
    not_idx = np.where(matrix < 40)

    matrix[idx]     =  2**((matrix[idx]-40)/10)    #
    matrix[not_idx] =  (matrix[not_idx]/40)**2.642 # max => 438.53
    return(matrix)


# MAIN Rhythm Pattern Extraction Function

def rp_extract( wavedata,                          # pcm (wav) signal data normalized to (-1,1)
                samplerate,                    # signal sampling rate

                # which features to extract
                extract_rp   = False,          # extract Rhythm Patterns features
                extract_ssd  = False,          # extract Statistical Spectrum Descriptor
                extract_tssd = False,          # extract temporal Statistical Spectrum Descriptor
                extract_rh   = False,          # extract Rhythm Histogram features
                extract_rh2  = False,          # extract Rhythm Histogram features including Fluctuation Strength Weighting
                extract_trh  = False,          # extract temporal Rhythm Histogram features
                extract_mvd  = False,          # extract modulation variance descriptor

                # processing options
                skip_leadin_fadeout =  1,      # >=0  how many sample windows to skip at the beginning and the end
                step_width          =  1,      # >=1  each step_width'th sample window is analyzed
                n_bark_bands        = 24,      # 2..24 number of desired Bark bands (from low frequencies to high) (e.g. 15 or 20 or 24 for 11, 22 and 44 kHz audio respectively) (1 delivers undefined output)
                mod_ampl_limit      = 60,      # 2..257 number of modulation frequencies on x-axis
                
                # enable/disable parts of feature extraction
                transform_bark                 = True,  # [S2] transform to Bark scale
                spectral_masking               = True,  # [S3] compute Spectral Masking
                transform_db                   = True,  # [S4] transfrom to dB: advisable only to turn off when [S5] and [S6] are turned off too
                transform_phon                 = True,  # [S5] transform to Phon: if disabled, Sone_transform will be disabled too
                transform_sone                 = True,  # [S6] transform to Sone scale (only applies if transform_phon = True)
                fluctuation_strength_weighting = True,  # [R2] apply Fluctuation Strength weighting curve
                #blurring                       = True  # [R3] Gradient+Gauss filter # TODO: not yet implemented

                return_segment_features = False,     # this will return features per each analyzed segment instead of aggregated ones
                verbose = True                       # print messages whats going on
                ):

    '''Rhythm Pattern Feature Extraction

    performs segment-wise audio feature extraction from provided audio wave (PCM) data
    and extracts the following features:

        Rhythm Pattern
        Statistical Spectrum Descriptor
        Statistical Histogram
        temporal Statistical Spectrum Descriptor
        Rhythm Histogram
        temporal Rhythm Histogram features
        Modulation Variance Descriptor

    Examples:
    >>> from audiofile_read import *
    >>> samplerate, samplewidth, wavedata = audiofile_read("music/BoxCat_Games_-_10_-_Epic_Song.mp3") #doctest: +ELLIPSIS
    Decoded .mp3 with: mpg123 -q -w /....wav music/BoxCat_Games_-_10_-_Epic_Song.mp3
    >>> feat = rp_extract(wavedata, samplerate, extract_rp=True, extract_ssd=True, extract_rh=True)
    Analyzing 7 segments
    >>> for k in feat.keys():
    ...     print k.upper() +  ":", feat[k].shape[0], "dimensions"
    SSD: 168 dimensions
    RH: 60 dimensions
    RP: 1440 dimensions
    >>> print feat["rp"]
    [ 0.01599218  0.01979605  0.01564305  0.01674175  0.00959912  0.00931604  0.00937831  0.00709122  0.00929631  0.00754473 ...,  0.02998088  0.03602739  0.03633861  0.03664331  0.02589753  0.02110256
      0.01457744  0.01221825  0.0073788   0.00164668]
    >>> print feat["rh"]
    [  7.11614842  12.58303013   6.96717295   5.24244146   6.49677561   4.21249659  12.43844045   4.19672357   5.30714983   6.1674115  ...,   1.55870044   2.69988854   2.75075831   3.67269877  13.0351257
      11.7871738    3.76106713   2.45225195   2.20457928   2.06494926]
    >>> print feat["ssd"]
    [  3.7783279    5.84444695   5.58439197   4.87849697   4.14983056   4.09638223   4.04971225   3.96152261   3.65551062   3.2857232  ...,  14.45953191  14.6088727   14.03351539  12.84783095  10.81735946
       9.04121124   7.13804008   5.6633501    3.09678286   0.52076428]

    '''


    # PARAMETER INITIALIZATION
    # non-exhibited parameters
    include_DC = False
    FLATTEN_ORDER = 'F' # order how matrices are flattened to vector: 'F' for Matlab/Fortran, 'C' for C order (IMPORTANT TO USE THE SAME WHEN reading+reshaping the features)

    # segment_size should always be ~6 sec, fft_window_size should always be ~ 23ms

    if (samplerate == 11025):
        segment_size    = 2**16
        fft_window_size = 256
    elif (samplerate == 22050):
        segment_size    = 2**17
        fft_window_size = 512
    elif (samplerate == 44100):
        segment_size    = 2**18
        fft_window_size = 1024
    else:
        # throw error not supported
        raise ValueError('A sample rate of ' + str(samplerate) + " is not supported (only 11, 22 and 44 kHz).")
    
    # calculate frequency values on y-axis (for Bark scale calculation):
    # freq_axis = float(samplerate)/fft_window_size * np.arange(0,(fft_window_size/2) + 1)
    # linear space from 0 to samplerate/2 in (fft_window_size/2+1) steps
    freq_axis = np.linspace(0, float(samplerate)/2, int(fft_window_size//2) + 1, endpoint=True)


    # CONVERT STEREO TO MONO: Average the channels
    if wavedata.ndim > 1:                    # if we have more than 1 dimension
        if wavedata.shape[1] == 1:           # check if 2nd dimension is just 1
            wavedata = wavedata[:,0]         # then we take first and only channel
        else:
            wavedata = np.mean(wavedata, 1)  # otherwise we average the signals over the channels


    # SEGMENT INITIALIZATION
    # find positions of wave segments
    
    skip_seg = skip_leadin_fadeout
    seg_pos  = np.array([1, segment_size]) # array with 2 entries: start and end position of selected segment

    seg_pos_list = []  # list to store all the individual segment positions (only when return_segment_features == True)

    # if file is too small, don't skip leadin/fadeout and set step_width to 1
    if ((skip_leadin_fadeout > 0) or (step_width > 1)):

        duration =  wavedata.shape[0]/samplerate

        if (duration < 45):
            step_width = 1
            skip_seg   = 0
            # TODO: do this as a warning?
            if verbose: print "Duration < 45 seconds: setting step_width to 1 and skip_leadin_fadeout to 0."

        else:
            # advance by number of skip_seg segments (i.e. skip lead_in)
            seg_pos = seg_pos + segment_size * skip_seg
    
    # calculate number of segments
    n_segments = int(np.floor( (np.floor( (wavedata.shape[0] - (skip_seg*2*segment_size)) / segment_size ) - 1 ) / step_width ) + 1)
    if verbose: print "Analyzing", n_segments, "segments"

    if n_segments == 0:
        raise ValueError("Not enough data to analyze! Minimum sample length needs to be " +
                         str(segment_size) + " (5.94 seconds) but it is " + str(wavedata.shape[0]) +
                         " (" + str(round(wavedata.shape[0] * 1.0 / samplerate,2)) + " seconds)")

    # initialize output
    features = {}

    ssd_list = []
    rh_list  = []
    rh2_list = []
    rp_list  = []
    mvd_list = []

    hearing_threshold_factor = 0.0875 * (2**15)

    # SEGMENT ITERATION

    for seg_id in range(n_segments):

        # keep track of segment position
        if return_segment_features:
            seg_pos_list.append(seg_pos)
        
        # EXTRACT WAVE SEGMENT that will be processed
        # data is assumed to be mono waveform
        wavsegment = wavedata[seg_pos[0]-1:seg_pos[1]] # verified

        # adjust hearing threshold # TODO: move after stereo-mono conversion above?
        wavsegment = wavsegment * hearing_threshold_factor

        matrix = calc_spectrogram(wavsegment,fft_window_size)

        # PSYCHO-ACOUSTIC TRANSFORMS

        # Map to Bark Scale
        if transform_bark:
            matrix = transform2bark(matrix,freq_axis,n_bark_bands)

        # Spectral Masking
        if spectral_masking:
            matrix = do_spectral_masking(matrix)

        # Map to Decibel Scale
        if transform_db:
            matrix = transform2db(matrix)
        
        # Transform Phon
        if transform_phon:
            matrix = transform2phon(matrix)

        # Transform Sone
        if transform_sone:
            matrix = transform2sone(matrix)

        # FEATURES: now we got a Sonogram and extract statistical features
    
        # SSD: Statistical Spectrum Descriptors
        if (extract_ssd or extract_tssd):
            ssd = calc_statistical_features(matrix)
            ssd_list.append(ssd.flatten(FLATTEN_ORDER))

        # RP: RHYTHM PATTERNS
        feature_part_xaxis1 = range(0,mod_ampl_limit)    # take first (opts.mod_ampl_limit) values of fft result including DC component
        feature_part_xaxis2 = range(1,mod_ampl_limit+1)  # leave DC component and take next (opts.mod_ampl_limit) values of fft result

        if (include_DC):
            feature_part_xaxis_rp = feature_part_xaxis1
        else:
            feature_part_xaxis_rp = feature_part_xaxis2

        # 2nd FFT
        fft_size = 2**(nextpow2(matrix.shape[1]))

        if (mod_ampl_limit >= fft_size):
            raise(ValueError("mod_ampl_limit option must be smaller than FFT window size (" + str(fft_size) +  ")."))
            # NOTE: in fact only half of it (256) makes sense due to the symmetry of the FFT result
        
        rhythm_patterns = np.zeros((matrix.shape[0], fft_size), dtype=np.complex128)
        #rhythm_patterns = np.zeros((matrix.shape[0], fft_size), dtype=np.float64)

        # real_matrix = abs(matrix)

        for b in range(0,matrix.shape[0]):
        
            rhythm_patterns[b,:] = fft(matrix[b,:], fft_size)

            # tried this instead, but ...
            #rhythm_patterns[b,:] = fft(real_matrix[b,:], fft_size)   # ... no performance improvement
            #rhythm_patterns[b,:] = rfft(real_matrix[b,:], fft_size)  # ... different output values
        
        rhythm_patterns = rhythm_patterns / 256  # why 256?

        # convert from complex128 to float64 (real)
        rp = np.abs(rhythm_patterns[:,feature_part_xaxis_rp]) # verified

        # MVD: Modulation Variance Descriptors
        if extract_mvd:
            mvd = calc_statistical_features(rp.transpose()) # verified
            mvd_list.append(mvd.flatten(FLATTEN_ORDER))

        # RH: Rhythm Histograms - OPTION 1: before fluctuation_strength_weighting (as in Matlab)
        if extract_rh:
            rh = np.sum(np.abs(rhythm_patterns[:,feature_part_xaxis2]),axis=0) #without DC component # verified
            rh_list.append(rh.flatten(FLATTEN_ORDER))

        # final steps for RP:

        # Fluctuation Strength weighting curve
        if fluctuation_strength_weighting:

            # modulation frequency x-axis (after 2nd FFT)
            # mod_freq_res = resolution of modulation frequency axis (0.17 Hz)
            mod_freq_res  = 1 / (float(segment_size) / samplerate)

            #  modulation frequencies along x-axis from index 0 to 256)
            mod_freq_axis = mod_freq_res * np.array(feature_part_xaxis_rp)

            #  fluctuation strength curve
            fluct_curve = 1 / (mod_freq_axis/4 + 4/mod_freq_axis)

            for b in range(rp.shape[0]):
                rp[b,:] = rp[b,:] * fluct_curve #[feature_part_xaxis_rp]

        #values verified


        # RH: Rhythm Histograms - OPTION 2 (after Fluctuation weighting)
        if extract_rh2:
            rh2 = np.sum(rp,axis=0) #TODO: adapt to do always without DC component
            rh2_list.append(rh2.flatten(FLATTEN_ORDER))


        # Gradient+Gauss filter
        #if extract_rp:
            # TODO Gradient+Gauss filter

            #for i in range(1,rp.shape[1]):
            #    rp[:,i-1] = np.abs(rp[:,i] - rp[:,i-1]);
            #
            #rp = blur1 * rp * blur2;

        rp_list.append(rp.flatten(FLATTEN_ORDER))

        seg_pos = seg_pos + segment_size * step_width


    if extract_rp:
        if return_segment_features:
            features["rp"] = np.array(rp_list)
        else:
            features["rp"] = np.median(np.asarray(rp_list), axis=0)

    if extract_ssd:
        if return_segment_features:
            features["ssd"] = np.array(ssd_list)
        else:
            features["ssd"]  = np.mean(np.asarray(ssd_list), axis=0)
        
    if extract_rh:
        if return_segment_features:
            features["rh"] = np.array(rh_list)
        else:
            features["rh"] = np.median(np.asarray(rh_list), axis=0)

    if extract_mvd:
        if return_segment_features:
            features["mvd"] = np.array(mvd_list)
        else:
            features["mvd"]  = np.mean(np.asarray(mvd_list), axis=0)

    # NOTE: no return_segment_features for temporal features as they measure variation of features over time

    if extract_tssd:
        features["tssd"] = calc_statistical_features(np.asarray(ssd_list).transpose()).flatten(FLATTEN_ORDER)

    if extract_trh:
        features["trh"]  = calc_statistical_features(np.asarray(rh_list).transpose()).flatten(FLATTEN_ORDER)

    if return_segment_features:
        # also include the segment positions in the result
        features["segpos"] = np.array(seg_pos_list)
        features["timepos"] = features["segpos"] / (samplerate * 1.0)

    return features



# function to self test rp_extract if working properly
def self_test():
    import doctest
    #doctest.testmod()
    doctest.run_docstring_examples(rp_extract, globals(), verbose=True)



if __name__ == '__main__':

    import sys
    from audiofile_read import *       # import our library for reading wav and mp3 files

    # process file given on command line or default song (included)
    if len(sys.argv) > 1:
        if sys.argv[1] == '-test': # RUN DOCSTRING SELF TEST
            print "Doing self test. If nothing is printed, it is ok."
            import doctest
            doctest.run_docstring_examples(rp_extract, globals()) #, verbose=True)
            exit()   # Note: no output means that everything went fine
        else:
            audiofile = sys.argv[1]
    else:
        audiofile = "music/BoxCat_Games_-_10_-_Epic_Song.mp3"

    # Read audio file and extract features
    try:

        samplerate, samplewidth, wavedata = audiofile_read(audiofile)

        np.set_printoptions(suppress=True)

        bark_bands = 24  # choose the number of Bark bands (2..24)
        mod_ampl_limit = 60 # number modulation frequencies on x-axis

        feat = rp_extract(wavedata,
                          samplerate,
                          extract_rp=True,
                          extract_ssd=True,
                          extract_tssd=False,
                          extract_rh=True,
                          n_bark_bands=bark_bands,
                          spectral_masking=True,
                          transform_db=True,
                          transform_phon=True,
                          transform_sone=True,
                          fluctuation_strength_weighting=True,
                          skip_leadin_fadeout=1,
                          step_width=1,
                          mod_ampl_limit=mod_ampl_limit)

        # feat is a dict containing arrays for different feature sets
        print "Successfully extracted features:" , feat.keys()

    except ValueError, e:
        print e
        exit()


    print "Rhythm Histogram feature vector:"
    print feat["rh"]

    # EXAMPLE on how to plot the features
    do_plots = False

    if do_plots:
        from rp_plot import *

        plotrp(feat["rp"],rows=bark_bands,cols=mod_ampl_limit)
        plotrh(feat["rh"])
        plotssd(feat["ssd"],rows=bark_bands)

    # EXAMPLE on how to store RP features in CSV file
    # import pandas as pd
    # filename = "features.rp.csv"
    # rp = pd.DataFrame(feat["rp"].reshape([1,feat["rp"].shape[0]]))
    # rp.to_csv(filename)