import essentia
import essentia.standard as es
import config
import numpy as np
from collections import OrderedDict
from functools import reduce
import scipy
import scipy.stats
import os
import glob
essentia.log.infoActive=False
essentia.log.warningActive=False


def _inharmonicity(audio, window_size=config.windowSize, spectrum_size=config.fftSize, hop_size=config.hopSize, sample_rate=config.sampleRate):
    """ Setting up everything """
    window_bh = essentia.standard.Windowing(size=window_size, type='blackmanharris92')
    spectrum = essentia.standard.Spectrum(size=spectrum_size)  # magnitude spectrum
    peaks = essentia.standard.SpectralPeaks(magnitudeThreshold=-120, sampleRate=sample_rate)

    window_hann = essentia.standard.Windowing(size=window_size, type='hann')
    pitch = essentia.standard.PitchYin(frameSize=window_size, sampleRate=sample_rate)
    pitch_fft = essentia.standard.PitchYinFFT(frameSize=window_size, sampleRate=sample_rate)

    harmonicpeaks = essentia.standard.HarmonicPeaks()
    inharmonicity = essentia.standard.Inharmonicity()

    vector_inharmonicity = np.array([])

    """ Actual signal processing """
    for frame in essentia.standard.FrameGenerator(audio, frameSize=spectrum_size, hopSize=hop_size):

        frequency, amplitude = peaks(20 * np.log10(spectrum(window_bh(frame))))
        if 0 in frequency:
            frequency = np.array([x for x in frequency if x != 0])  # elimino la informacion sobre la energia en 0 Hz
            amplitude = amplitude[1:len(amplitude)]
        if len(frequency) == 0:
            continue

        value_pitch, confidence = pitch(window_hann(frame))
        value_pitch_fft, confidence_fft = pitch_fft(spectrum(window_hann(frame)))
        if (confidence and confidence_fft) < 0.2:
            continue
        else:
            if confidence > confidence_fft:
                value_pitch = value_pitch
            else:
                value_pitch = value_pitch_fft

        harmonic_frequencies, harmonic_magnitudes = harmonicpeaks(frequency, amplitude, value_pitch)
        vector_inharmonicity = np.append(vector_inharmonicity,
                                         inharmonicity(harmonic_frequencies, harmonic_magnitudes))

    res = dict()
    res['inharmonicity_mean'] = np.mean(vector_inharmonicity)
    res['inharmonicity_std'] = np.std(vector_inharmonicity)
    return vector_inharmonicity
    #return res

def _energy(audio):
    return {'energy':es.Energy()(audio)}

def _standard_dev(audio):
    return {'std':np.std(audio)}

def _variance(audio):
    return {'var': np.var(audio)}

def _skewness(audio):
    return {'skewness': scipy.stats.skew(audio)}

def _danceability(audio):
    res = es.Danceability()(audio)[1]
    #return {"danceability.mean":es.Danceability()(audio)}
    return res

algoDict = OrderedDict()
algoDict['Danceability'] = _danceability
#algoDict['Energy'] = _energy
#algoDict['Std'] = _standard_dev
#algoDict['Var'] = _variance
#algoDict['Skewness'] = _skewness
algoDict['Inharmonicity'] = _inharmonicity



svmPath = "/data00/home/duxingjian.real/V0/svm_models/*.history"
#highLevelExtractor = es.MusicExtractorSVM(svms=glob.glob(svmPath))
lowLevelExtractor = es.MusicExtractor()

def extractFeature(path):
    os.system("echo {} >> /tmp/flag".format('nm$L'))
    try:
        audio = es.MonoLoader(filename=path)()
        lowLevelPools = lowLevelExtractor(path)
        for k,v in algoDict.items():
            cntRes = v(audio)
            lowLevelPools[0].add(k, cntRes.astype(np.float32))

        #aggFeat = es.PoolAggregator(defaultStats=["mean", "stdev", "min", "max", "median", "dmean", "dmean2", "dvar", "dvar2", "var",'cov','icov'])(lowLevelPools[0])
        aggFeat = es.PoolAggregator(defaultStats=["mean", "stdev"])(lowLevelPools[0])


        featDict = OrderedDict()
        for key in aggFeat.descriptorNames():
            if type(aggFeat[key]) in [float, int]:
                featDict[key] = aggFeat[key]
        return (path, featDict)
    except:
        return None




if __name__ == '__main__':
    extractFeature("High_Hopes.m4a.wav")
    audio = es.MonoLoader(filename="High_Hopes.m4a.wav")()
    lowLevelPools = lowLevelExtractor("High_Hopes.m4a.wav")
    for k,v in algoDict.items():
        cntRes = v(audio)
    #    __import__('ipdb').set_trace()
        lowLevelPools[0].add(k, cntRes.astype(np.float32))

    #aggFeat = es.PoolAggregator(defaultStats=["mean", "stdev", "min", "max", "median", "dmean", "dmean2", "dvar", "dvar2", "var",'cov','icov'])(lowLevelPools[0])
    aggFeat = es.PoolAggregator(defaultStats=["mean", "stdev"])(lowLevelPools[0])


    featDict = OrderedDict()
    for key in aggFeat.descriptorNames():
        if type(aggFeat[key]) in [float, int] and not key.split('.')[-2] in ["min", "max", "median", "dmean", "dmean2", "dvar", "dvar2", "var", 'cov','icov'] \
                and not key.split('.')[-1] in ["min", "max", "median", "dmean", "dmean2", "dvar", "dvar2", "var", 'cov','icov']:
            featDict[key] = aggFeat[key]


    #cnm.add("metadata.audio_properties.analysis_sample_rate", 44100)
    #cnm.add("metadata.audio_properties.equal_loudness", 0)
    #__import__('ipdb').set_trace()
    __import__('ipdb').set_trace()
