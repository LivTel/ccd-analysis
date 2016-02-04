import optparse
import logging
import collections

import numpy as np
import scipy.stats
import pyfits
import pylab as plt
from scipy import interpolate

from utility import readSettingsFile as rsf
from utility import processDataPathParameter as pdpp

class ptc_errors:
    def __init__(self, logging):
        self._errorCode		= 0
	self._errorCodeDict	= { 2	: "(ptc.run) pair has negative mean or std, ignoring",
				    1	: "(ptc.run) two frames not found for this exposure time, ignoring",
				    0	: "no errors encountered",
				   -1	: "(__main__) no dataPath specified", 
				   -2	: "(__main__) invalid settings file", 
				   -3	: "(__main__) failed to find instrument setup from settings file",
				   -4	: "(ptc.__init__) no data found at specified dataPath"
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

class ptc:  
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

    def run(self):
        logging.info("(ptc.doPTC) executing")
        res = []
        ##
        ## for each quadrant
        ##
        for q in self.settings['quadrants']:
            ##
            ## read this quadrant's attributes from settings file 
            ##
            qid = q['id']
            pos = q['pos']
            x_lo = int(q['x_lo'])
            x_hi = int(q['x_hi'])
            y_lo = int(q['y_lo'])
            y_hi = int(q['y_hi'])
            overscan_x_lo = int(q['overscan_x_lo'])
            overscan_x_hi = int(q['overscan_x_hi'])
            overscan_y_lo = int(q['overscan_y_lo'])
            overscan_y_hi = int(q['overscan_y_hi'])
            is_defective = bool(q['is_defective'])

            if is_defective:
                logging.info("(ptc.run) omitting defective quadrant " + str(qid) + " with position \"" + str(pos) + "\"")
                res.append(None)
                continue

            logging.info("(ptc.run) processing quadrant " + str(qid) + " with position \"" + str(pos) + "\"")
            logging.debug("(ptc.run) x range of quadrant is defined by " + str(x_lo) + " < x < " + str(x_hi))
            logging.debug("(ptc.run) y range of quadrant is defined by " + str(y_lo) + " < y < " + str(y_hi))
            logging.debug("(ptc.run) overscan x range of quadrant is defined by " + str(overscan_x_lo) + " < x < " + str(overscan_x_hi))
            logging.debug("(ptc.run) overscan y range of quadrant is defined by " + str(overscan_y_lo) + " < y < " + str(overscan_y_hi))

            ##
            ## read this quadrant's data and remove bias
            ## 
            files_data = {}
            files_hdr = {}
            for f in self.files:
                logging.info("(ptc.run) caching file " + f)
                ff = pyfits.open(f)
                this_data = ff[self.settings['data_hdu']].data[y_lo:y_hi, x_lo:x_hi]-np.mean(ff[self.settings['data_hdu']].data[overscan_y_lo:overscan_y_hi, overscan_x_lo:overscan_x_hi])
                this_dummy = ff[self.settings['dummy_hdu']].data[y_lo:y_hi, x_lo:x_hi]-np.mean(ff[self.settings['dummy_hdu']].data[overscan_y_lo:overscan_y_hi, overscan_x_lo:overscan_x_hi])
                this_hdr = ff[self.settings['data_hdu']].header
                exptime = this_hdr['EXPTIME']
                if exptime not in files_data:
                    files_data[exptime] = []
                    files_hdr[exptime] = []
                if self.settings['do_dummy_subtraction']:
                    files_data[exptime].append(this_data-this_dummy)
                else:
                    files_data[exptime].append(this_data)
                files_hdr[exptime].append(this_hdr)
                ff.close()

            ##
            ## order quadrant data by exptime
            ##
            files_data_od =  collections.OrderedDict(sorted(files_data.items()))

            ##
            ## for each exposure time, take the mean of the signal and calculate the noise of the difference frame
            ## 
            res_thisq = []		# this keeps track of (mean, noise) tuples
            diff_stk = []		# this generates a difference stack which, when plotted, is useful for diagnosing which regions of the quadrant are suitable
            for exposure_time, data in files_data_od.iteritems():
                if len(data) != 2:	# check we have two frames for this exposure time
                    err.setError(1)
                    err.handleError()
                    continue

                diff = data[1]-data[0]						# make diff frame
                diff_stk.append(diff)						# append to stack

                thisq_mean = np.mean(data, axis=0)				# mean of frames
                thisq_std_diff = np.std(diff, axis=0)				# error on diff frame
              	this_shot_and_read_noise = thisq_std_diff/(pow(2, 0.5))		# NOTE: THIS WON'T BE TRUE FOR DUMMY SUBTRACTION

                if np.mean(thisq_mean) < 0 or np.mean(thisq_std_diff) < 0:	# we have a duff pair here
                    err.setError(2)
                    err.handleError()
                    continue

                logging.debug("(ptc.run) exposure time of " + str(exposure_time) + " has mean signal level of " + str(round(np.mean(thisq_mean),2)) + "ADU +/- " + str(round(np.mean(thisq_std_diff),2)) + "ADU")
 
                res_thisq.append((thisq_mean, this_shot_and_read_noise))
 
            if self.diagnosticMode:
                plt.imshow(np.mean(diff_stk, axis=0), vmax=np.percentile(np.mean(diff_stk, axis=0), 95), vmin=np.percentile(np.mean(diff_stk, axis=0), 5))
                plt.colorbar()
                plt.show()

            res.append(res_thisq)
    
        for idx_q, q in enumerate(res): 
            pos = self.settings['quadrants'][idx_q]['pos']
            is_defective = bool(self.settings['quadrants'][idx_q]['is_defective'])
            if is_defective:
                if self.makePlots:
                    plt.subplot(2, 2, idx_q+1)
                    plt.plot([], label='data for quadrant: ' + str(self.settings['quadrants'][idx_q]['pos']), color='white')
                    plt.legend(loc='upper left')
                    continue
            thisq_mean_all = []
            thisq_std_all = []
            for p in q:
                thisq_mean_all.append(np.mean(p[0]))
                thisq_std_all.append(np.mean(p[1]))
 
            x = np.asarray(thisq_mean_all)
            y = np.asarray(thisq_std_all)

            # calculate gradients to find suspect entries
            x_gradients = np.asarray([x1+(x2-x1)/2 for x1, x2 in zip(x[:-1], x[1:])])
            y_gradients = np.asarray([y1+(y2-y1)/2 for y1, y2 in zip(y[:-1], y[1:])])
            g_gradients = np.asarray([(y2-y1)/(x2-x1) for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:])])

            truth = []
            for idx_g in range(2, len(g_gradients)-1):
                lt = [True for gi in g_gradients[idx_g-2:idx_g] if np.logical_and(gi>0, gi<1.2)]
                gt = [True for gi in g_gradients[idx_g:idx_g+2] if np.logical_or(gi<0, gi>1.2)]
                print idx_g, np.sum(lt), np.sum(gt)
                n_true = np.sum(lt) + np.sum(gt)
                truth.append(n_true)	
            g_gradients_to_idx = np.argmax(truth) + 3	# +3 for offsets incurred from taking gradient and cycling through truth array	
		
            print truth
            exit(0)

            s = interpolate.UnivariateSpline(x[:g_gradients_to_idx+1], y[:g_gradients_to_idx+1], k=3, s=1)	# apply smoothing cubic bspline
            x_sm = x[:g_gradients_to_idx+1]
            y_sm = s(x_sm)

            x_sm_log = np.log10(x_sm)
            y_sm_log = np.log10(y_sm)

            # calculate read noise (ADU)
            ## i) fit a second order polynomial to start of smoothed data
            ## ii) find gradient = 0
            ## iii) find y-intercept
            f_co = np.polyfit(x_sm_log[0:4], y_sm_log[0:4], 2)		# take a few points around the start and fit second order poly
            f_xmin = f_co[1]/-(2*f_co[0])				# find minimum
            rn_yi_log = np.polyval(f_co, f_xmin)			# y-intercept (rn in ADU)		
             
            # calculate gain (e-/ADU)
            ## i) find gradients for each index in smoothed data array using a linear fit
            ## ii) find index with gradient of ~0.5 (shot regime for loglog) by assessing truth array for adjacent indices of <0.5 and >0.5
            ### iii) fit second order polynomial to this index, and find x coordinate at which gradient is exactly 0.5
            ### iv) find x-intercept
            #f_co_nearest = None
            #idx_x_nearest = None
            gradients_sm_log = []
            for idx_x in range(1, len(x_sm_log)):
                gradients_sm_log.append(np.polyfit(x_sm_log[idx_x-1:idx_x+1], y_sm_log[idx_x-1:idx_x+1], 1)[0])

            truth = []
            for idx_g in range(1, len(gradients_sm_log)):
                lt = [True for gi in gradients_sm_log[:idx_g] if gi<0.5]
                gt = [True for gi in gradients_sm_log[idx_g:] if gi>0.5]
                n_true = np.sum(lt) + np.sum(gt)
                truth.append(n_true)
            idx_x_nearest = np.argmax(truth)+2	# +2 for offsets incurred from taking gradient and cycling through truth array

            f_co = np.polyfit(x_sm_log[idx_x_nearest-2:idx_x_nearest+2], y_sm_log[idx_x_nearest-2:idx_x_nearest+2], 2)
 
            x_g_of_0p5 = (0.5-f_co[1])/(2*f_co[0]) 
            y_g_of_0p5 = np.polyval(f_co, x_g_of_0p5) 
            yi_g_of_0p5 = y_g_of_0p5 - (0.5*x_g_of_0p5)
            xi_g_of_0p5 = -yi_g_of_0p5/0.5

            if self.makePlots:
                plt.subplot(2, 2, idx_q+1)
                plt.plot(x, y, 'k.', label='data for quadrant: ' + str(self.settings['quadrants'][idx_q]['pos']))
                plt.plot(x_sm, y_sm, 'r-', label='smoothed fit')
                plt.plot(10**f_xmin, 10**rn_yi_log, 'ko')
                plt.plot([10**0, np.max(x)], [10**rn_yi_log, 10**rn_yi_log], 'k--', label="read noise: " + str(round(10**rn_yi_log, 2)) + " ADU")
                plt.plot([10**xi_g_of_0p5, 10**x_g_of_0p5], [10**0, 10**y_g_of_0p5], 'k--', label="gain: " + str(round(10**xi_g_of_0p5, 2)) + " e-/ADU")
                plt.yscale('log')
                plt.xscale('log')
                plt.legend(loc='upper left')

        if self.makePlots: 
            plt.show()
  
if __name__  == "__main__":
    parser = optparse.OptionParser()
    group1 = optparse.OptionGroup(parser, "General") 
    group1.add_option('--l', action='store', default='INFO', dest='logLevel', help='logging level (DEBUG|INFO|WARNING|ERROR|CRITICAL)')    
    group1.add_option('--f', action='store', default="settings.json", type=str, dest='settingsFile', help='Path to settings file')
    group1.add_option('--d', action='store', default=None, type=str, dest='dataPath', help='Path to data files (either directory or single file)')
    group1.add_option('--p', action='store_true', dest='makePlots', help='Make plots?')
    group1.add_option('--dm', action='store_true', dest='diagnosticMode', help='Diagnostic mode')
    parser.add_option_group(group1)

    group2 = optparse.OptionGroup(parser, "Instrument Setup") 
    group2.add_option('--s', action='store', default="WEAVEPROTO", type=str, dest='instName', help='Instrument name from settings file')
    parser.add_option_group(group2)

    args = parser.parse_args()
    options, args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=getattr(logging, options.logLevel.upper()))
    err = ptc_errors(logging)

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

    p = ptc(options.dataPath, options.settingsFile, options.instName, options.makePlots, options.diagnosticMode, err, logging)
    p.run()

