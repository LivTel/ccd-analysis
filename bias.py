
import optparse
import logging

import numpy as np
import scipy.stats
import pyfits
import pylab as plt

from utility import readSettingsFile as rsf
from utility import processDataPathParameter as pdpp
import fft

class bias_errors:
    def __init__(self, logging):
        self._errorCode		= 0
	self._errorCodeDict	= {0	: "no errors encountered",
				   -1	: "(__main__) no dataPath specified", 
				   -2	: "(__main__) invalid settings file", 
				   -3	: "(__main__) failed to find instrument setup from settings file",
				   -4	: "(bias.__init__) no data found at specified dataPath"
                                  } 

    def setError(self, newErrorCode):
        '''
        Set internal error code.
        '''
        self._errorCode = newErrorCode
        return True

    def handleError(self):
        '''
        Handle internal error code.
        '''
        errorMsg = self._errorCodeDict.get(self._errorCode)
        if self._errorCode is 0:
            logging.info(errorMsg)
        elif self._errorCode < 0:
            logging.critical(errorMsg)
            exit(0)
        elif self._errorCode > 0:
            logging.warning(errorMsg)

class bias:  
    def __init__(self, dataPath, settingsFile, instName, makePlots, diagnosticMode, err, logging):
        self.files = pdpp(dataPath)
        if len(self.files) == 0:
            err.setError(-4)
            err.handleError()

        self.settings = rsf(settingsFile)[instName][0]
        self.makePlots = makePlots
        self.diagnosticMode = diagnosticMode
        self.err = err
        self.logging = logging

    def calculateSpatialReadNoise(self, data, hdr):
        return np.mean(data), np.std(data)

    def calculateTemporalReadNoise(self, data, hdr):
        return np.mean(data, axis=0), np.std(data, axis=0)       
        return 

    def run(self, doSpatialAnalysis=True, doTemporalAnalysis=True, doFFTAnalysis=False):
        files_data = []
        files_dummy = []
        files_hdr = []
        for f in self.files:
            logging.info("(bias.run) caching file " + f)
            ff = pyfits.open(f)
            this_data = ff[self.settings['data_hdu']].data-np.mean(ff[self.settings['data_hdu']].data)
            this_dummy = ff[self.settings['dummy_hdu']].data-np.mean(ff[self.settings['dummy_hdu']].data)
            this_hdr = ff[self.settings['data_hdu']].header
            if self.settings['do_dummy_subtraction']:
                files_data.append(this_data-this_dummy)
            else:
                files_data.append(this_data)
            files_hdr.append(this_hdr)
            ff.close()

        if doSpatialAnalysis:
            '''
            returns a list as [file][quadrant].
            '''
            logging.info("(bias.run) performing spatial analysis of read noise") 
            res = []
            idx_f = 0
            for data, hdr in zip(files_data, files_hdr):
                res_thisfile = []
                for q in self.settings['quadrants']:
                    qid = q['id']
                    pos = q['pos']
                    x_lo = int(q['x_lo'])
                    x_hi = int(q['x_hi'])
                    y_lo = int(q['y_lo'])
                    y_hi = int(q['y_hi'])
                    is_defective = bool(q['is_defective'])
                    if is_defective:
                        logging.info("(bias.run) omitting defective quadrant " + str(qid) + " with position \"" + str(pos) + "\"")
                        res_thisfile.append((None, None))
                        continue
                    logging.info("(bias.run) processing frame " + str(idx_f) + " of quadrant " + str(qid) + " with position \"" + str(pos) + "\"")
                    logging.debug("(bias.run) x range of quadrant is defined by " + str(x_lo) + " < x < " + str(x_hi))
                    logging.debug("(bias.run) y range of quadrant is defined by " + str(y_lo) + " < y < " + str(y_hi))
                    this_q_mean, this_q_std = self.calculateSpatialReadNoise(data[y_lo:y_hi, x_lo:x_hi], hdr)
                    res_thisfile.append((this_q_mean, this_q_std))
                    logging.info("(bias.run) SPATIAL READ NOISE:\t" + str(round(this_q_mean, 2)) + " +/- " + str(round(this_q_std, 2)) + " ADU")
                res.append(res_thisfile)
                idx_f = idx_f + 1
            if self.makePlots:
                if len(files_data) > 2: 
                    q_means = np.asarray(res).transpose()[0]		# list of means for each quadrant
                    q_stds = np.asarray(res).transpose()[1]		# list of stds for each quadrant
                    for idx_q, thisq_stds in enumerate(q_stds):
                        plt.subplot(2, 2, idx_q+1)
                        pos = self.settings['quadrants'][idx_q]['pos']
                        is_defective = bool(self.settings['quadrants'][idx_q]['is_defective'])
                        if is_defective:
                            plt.plot([], label=self.settings['quadrants'][idx_q]['pos'], color='white')
                            plt.legend()
                            continue
                        thisq_max = np.percentile(thisq_stds, 99.5)
                        thisq_min = np.percentile(thisq_stds, 0.5)
                        nbins = 20
                        bins = np.arange(thisq_min, thisq_max, (thisq_max-thisq_min)/nbins)
                        plt.hist(thisq_stds, label=self.settings['quadrants'][idx_q]['pos'], bins=bins, color="white")
                        plt.legend()
                    plt.show()
                else:
                    logging.info("(bias.run) only one file in dataPath found, omitting histogram plot in spatial analysis")

        if doTemporalAnalysis:
            '''
            returns a list as [quadrant][pixel].
            '''
            if len(files_data) > 2:
                logging.info("(bias.run) performing temporal analysis of read noise")
                res = []
                for q in self.settings['quadrants']:
                    qid = q['id']
                    pos = q['pos']
                    x_lo = int(q['x_lo'])
                    x_hi = int(q['x_hi'])
                    y_lo = int(q['y_lo'])
                    y_hi = int(q['y_hi'])
                    is_defective = bool(q['is_defective'])
                    if is_defective:
                        logging.info("(bias.run) omitting defective quadrant " + str(qid) + " with position \"" + str(pos) + "\"")
                        res.append((None, None))
                        continue
                    logging.info("(bias.run) processing quadrant " + str(qid) + " with position \"" + str(pos) + "\"")
                    logging.debug("(bias.run) x range of quadrant is defined by " + str(x_lo) + " < x < " + str(x_hi))
                    logging.debug("(bias.run) y range of quadrant is defined by " + str(y_lo) + " < y < " + str(y_hi))
                    data_thisq = []
                    for data, hdr in zip(files_data, files_hdr):
                        data_thisq.append(data[y_lo:y_hi, x_lo:x_hi])
                    this_f_mean, this_f_std = self.calculateTemporalReadNoise(data_thisq, hdr)
                    res.append((this_f_mean, this_f_std))
                    logging.info("(bias.run) MEAN TEMPORAL READ NOISE:\t" + str(round(np.mean(this_f_mean), 2)) + " +/- " + str(round(np.mean(this_f_std), 2)) + " ADU")
                if self.makePlots:
                    for idx_q, q in enumerate(res):
                        plt.subplot(2, 2, idx_q+1)
                        pos = self.settings['quadrants'][idx_q]['pos']
                        is_defective = bool(self.settings['quadrants'][idx_q]['is_defective'])
                        if is_defective:
                            plt.plot([], label=self.settings['quadrants'][idx_q]['pos'], color='white')
                            plt.legend()
                            continue
                        thisq_stds = q[1].flatten()
                        thisq_max = np.percentile(thisq_stds, 99.5)
                        thisq_min = np.min(thisq_stds, 0.5)
                        nbins = 20
                        bins = np.arange(thisq_min, thisq_max, (thisq_max-thisq_min)/nbins)
                        plt.hist(thisq_stds, label=self.settings['quadrants'][idx_q]['pos'], bins=bins, color='white')
                        plt.legend()
                    plt.show()

            else:  
                logging.info("(bias.run) only one file in dataPath found, omitting temporal analysis")

        if doFFTAnalysis: 
            for data, hdr in zip(files_data, files_hdr):
                for q in self.settings['quadrants']:
                    qid = q['id']
                    pos = q['pos']
                    x_lo = int(q['x_lo'])
                    x_hi = int(q['x_hi'])
                    y_lo = int(q['y_lo'])
                    y_hi = int(q['y_hi'])
                    is_defective = bool(q['is_defective'])
                    if is_defective:
                        logging.info("(bias.run) omitting defective quadrant " + str(qid) + " with position \"" + str(pos) + "\"")
                        continue
                    logging.info("(bias.run) processing quadrant " + str(qid) + " with position \"" + str(pos) + "\"")
                    logging.debug("(bias.run) x range of quadrant is defined by " + str(x_lo) + " < x < " + str(x_hi))
                    logging.debug("(bias.run) y range of quadrant is defined by " + str(y_lo) + " < y < " + str(y_hi))
                 
                    n = fft.fft()
                    freq, psdx = n.do1DReadoutFFT(data[y_lo:y_hi,x_lo:x_hi], int(self.settings['readout_speed']))

                    if self.makePlots:
                        plt.figure()
                        plt.clf()
                        plt.loglog(freq, psdx)
                        plt.yticks([])
                        plt.xlabel('Frequency (Hz)')
                        plt.ylabel('Power')
                        plt.show()
  
if __name__  == "__main__":
    parser = optparse.OptionParser()
    group1 = optparse.OptionGroup(parser, "General") 
    group1.add_option('--l', action='store', default='INFO', dest='logLevel', help='logging level (DEBUG|INFO|WARNING|ERROR|CRITICAL)')    
    group1.add_option('--f', action='store', default="settings.json", type=str, dest='settingsFile', help='Path to settings file')
    group1.add_option('--d', action='store', default='/mnt/NAS/devel/WEAVE/ccd_analysis/bias_test/tmp', type=str, dest='dataPath', help='Path to data files (either directory or single file)')
    group1.add_option('--p', action='store_true', dest='makePlots', help='Make plots?')
    group1.add_option('--dm', action='store_true', dest='diagnosticMode', help='Diagnostic mode')
    parser.add_option_group(group1)

    group2 = optparse.OptionGroup(parser, "Instrument Setup") 
    group2.add_option('--s', action='store', default="WEAVEPROTO", type=str, dest='instName', help='Instrument name from settings file')
    parser.add_option_group(group2)

    args = parser.parse_args()
    options, args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=getattr(logging, options.logLevel.upper()))
    err = bias_errors(logging)

    ##
    ## Input checks.
    ##
    if options.dataPath is None:
        err.setError(-1)
        err.handleError() 
   
    try:
        rsf(options.settingsFile)[options.instName][0]
    except IOError:
        err.setError(-2)
        err.handleError()
    except KeyError:
        err.setError(-3)
        err.handleError()

    b = bias(options.dataPath, options.settingsFile, options.instName, options.makePlots, options.diagnosticMode, err, logging)
    b.run()
    
    



