import optparse
import logging
import collections

import numpy as np
import scipy.stats
import pyfits
import pylab as plt
from scipy import interpolate
from scipy.stats import sigmaclip

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
				   -4	: "(ptc.run) no data found at specified dataPath"
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

            logging.info("(ptc.run) processing quadrant " + str(qid+1) + " with position \"" + str(pos) + "\"")
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

                diff = (data[1]-data[0])					# make diff frame
                diff_stk.append(diff)						# append to stack

                thisq_mean = np.mean(data)				        # mean of frames
                thisq_std_diff = np.std(diff)				        # error on diff frame
              	this_shot_and_read_noise = thisq_std_diff/(pow(2, 0.5))		# NOTE: THIS ISN'T BE TRUE FOR DUMMY SUBTRACTION

                if np.mean(thisq_mean) < 0 or np.mean(thisq_std_diff) < 0:	# we have a duff pair here
                    err.setError(2)
                    err.handleError()
                    continue

                logging.debug("(ptc.run) exposure time of " + str(exposure_time) + " has mean signal level of " + str(round(np.mean(thisq_mean),2)) + "ADU +/- " + str(round(np.mean(thisq_std_diff),2)) + "ADU")
 
                res_thisq.append((thisq_mean, this_shot_and_read_noise, exposure_time))
 
            if self.diagnosticMode:
                plt.imshow(np.mean(diff_stk, axis=0), vmax=np.percentile(np.mean(diff_stk, axis=0), 95), vmin=np.percentile(np.mean(diff_stk, axis=0), 5))
                plt.colorbar()
                plt.show()

            res.append(res_thisq)
            
        rn = []
        gain = []
        qx = []
        qy = []
        for idx_q, q in enumerate(res): 
            pos = self.settings['quadrants'][idx_q]['pos']
            is_defective = bool(self.settings['quadrants'][idx_q]['is_defective'])
            if is_defective:
                rn.append(None)
                gain.append(None)
                qx.append(None)
                qy.append(None)
                continue
                  
            thisq_mean_all = []
            thisq_std_all = []
            thisq_exptimes = []
            for p in q:   # p == pair.
                thisq_mean_all.append(p[0])
                thisq_std_all.append(p[1])
                thisq_exptimes.append(p[2])
            thisq_rates = [c/e for c, e in zip(thisq_mean_all, thisq_exptimes)]
                
            x = np.asarray(thisq_mean_all)
            y = np.asarray(thisq_std_all)
            
            # hazard a guess at read regime by:
            ## i) finding gradients for each index in data array using a linear fit
            ## ii) find index with gradient of ~0.2 (shot regime for loglog) by assessing truth array for adjacent indices of <0.2 and >0.2
            gradients_log = []
            for idx_x in range(1, len(x)):
                gradients_log.append(np.polyfit(np.log10(x[idx_x-1:idx_x+1]), np.log10(y[idx_x-1:idx_x+1]), 1)[0])
            truth = []
            for idx_g in range(1, len(gradients_log)):
                lt = [True for gi in gradients_log[:idx_g] if gi<0.2]
                gt = [True for gi in gradients_log[idx_g:] if gi>0.2]
                n_true = np.sum(lt) + np.sum(gt)
                truth.append(n_true)
            idx_x_nearest = np.argmax(truth)+2	# +2 for offsets incurred from taking gradient and cycling through truth array
            read_guess = range(0,idx_x_nearest)
            if idx_x_nearest > len(x)-1 or idx_x_nearest < 0:
                read_guess = []
                         
            # hazard a guess at shot regime by:
            ## i) finding gradients for each index in data array using a linear fit
            ## ii) find index with gradient of ~0.5 (shot regime for loglog) by assessing truth array for adjacent indices of <0.5 and >0.5
            gradients_log = []
            for idx_x in range(1, len(x)):
                gradients_log.append(np.polyfit(np.log10(x[idx_x-1:idx_x+1]), np.log10(y[idx_x-1:idx_x+1]), 1)[0])
            truth = []
            for idx_g in range(1, len(gradients_log)):
                lt = [True for gi in gradients_log[:idx_g] if gi<0.5]
                gt = [True for gi in gradients_log[idx_g:] if gi>0.5]
                n_true = np.sum(lt) + np.sum(gt)
                truth.append(n_true)
            idx_x_nearest = np.argmax(truth)+2	# +2 for offsets incurred from taking gradient and cycling through truth array
            shot_guess = [idx_x_nearest-1, idx_x_nearest, idx_x_nearest+1] 
            if idx_x_nearest+1 > len(x)-1 or idx_x_nearest-1 < 0:
                shot_guess = []
            
            # interactive selection of PTC
            # - follow on screen prompts
            logging.info("(ptc.run) interactive selection of PTC regions")
            print 
            print "\t\tCOMMAND SET"
            print
            print "\tq: define point for read noise"
            print "\tw: define point for shot noise"
            print "\te: define full well"
            print "\ta: smooth data with cubic spline"
            print "\tx: clear point"
            print "\tm: clear all"
            print
            
            class define_PTC_regions(object):
                def __init__(self, ax, x, y, read_guess=read_guess, shot_guess=shot_guess, fwd_guess=None):
                    self.ax = ax
                    self.x = x                                         # data array
                    self.y = y                                         # data array
                    self.c_idx = None                                  # current cursor idx
                    
                    self.read = read_guess                             # (idx_1, idx_2 ... idx_n) 
                    self.shot = shot_guess                             # (idx_1, idx_2 ... idx_n)
                    self.fwd = fwd_guess                               # idx
                    
                    self.rn = None
                    self.gain = None
 
                def calculate_nearby_gradient(self, idx):
                    if idx != 0 and idx != len(self.x):
                        c = np.polyfit(self.x[idx-1:idx+2], self.y[idx-1:idx+2], 1)
                        return c[0]
                    else:
                        return None
                      
                def calculate_read_noise(self):
                    # calculate read noise (ADU)
                    ## i) fit a second order polynomial to read data array
                    ## ii) find gradient = 0
                    ## iii) find y-intercept
                    if len(self.read) < 2:
                        self.rn = None
                        print "e: need more than two points for read regime"
                        return 
                    f_co = np.polyfit(self.x[self.read], self.y[self.read], 2)	
                    f_xmin = f_co[1]/-(2*f_co[0])				# find minimum
                    rn_yi_log = np.polyval(f_co, f_xmin)			# y-intercept (rn in ADU)
                    print "i: read noise calculated as " + str(round(10**rn_yi_log, 2)) + "ADU"
                    self.rn = rn_yi_log
                  
                def calculate_gain(self):
                    # calculate gain (e-/ADU)
                    ### i) fit second order polynomial to shot data array, and find x coordinate at which gradient is exactly 0.5
                    ### ii) calculate corresponding x-intercept
                    if len(self.shot) < 2:
                        self.gain = None
                        print "e: need more than two points for shot regime"  
                        return
                    f_co = np.polyfit(self.x[self.shot], self.y[self.shot], 2)
                    x_g_of_0p5 = (0.5-f_co[1])/(2*f_co[0]) 
                    y_g_of_0p5 = np.polyval(f_co, x_g_of_0p5) 
                    yi_g_of_0p5 = y_g_of_0p5 - (0.5*x_g_of_0p5)
                    xi_g_of_0p5 = -yi_g_of_0p5/0.5
                    print "i: gain calculated as " + str(round(10**xi_g_of_0p5, 2)) + "e-/ADU"
                    self.gain = (xi_g_of_0p5, x_g_of_0p5, y_g_of_0p5)
  
                def draw(self):
                    self.ax.cla()
                    
                    plt.title("PTC") 
                    plt.xlabel("Log10 (Signal, ADU)")
                    plt.ylabel("Log10 (Noise, ADU)")
                    
                    # text location in axes coords
                    self.txt = ax.text(0.1, 0.9, '', transform=ax.transAxes)
                    
                    plt.plot(self.x, self.y, 'kx-')
                    plt.xlim([0, np.max(self.x)])
                    plt.ylim([0, np.max(self.y)])
                    if self.c_idx is not None:
                        # update line positions
                        lx = ax.axhline(color='k')                    # horiz line (cursor)
                        ly = ax.axvline(color='k')                    # vert line (cursor)
                        lx.set_ydata(self.y[self.c_idx])
                        ly.set_xdata(self.x[self.c_idx])
                        
                        # show gradient at point
                        m = self.calculate_nearby_gradient(self.c_idx)
                        if m is not None:
                            self.txt.set_text('nearby_m=%1.2f' % (m))
                            
                    if self.read is not None and self.rn is not None:
                        # update line positions
                        lx = ax.axhline(color='k', linestyle='--')     # horiz line (read noise)
                        lx.set_ydata(self.rn)    
                        
                    if self.shot is not None and self.gain is not None:
                        # update line positions
                        plt.plot([self.gain[0], self.gain[1]], [0, self.gain[2]], 'k--')
                        
                    # update regime points
                    self.ax.plot(self.x[self.read], self.y[self.read], 'ro')   
                    self.ax.plot(self.x[self.shot], self.y[self.shot], 'bo')  
                    
                    if self.fwd is not None:
                        lyf = ax.axvline(color='k', linestyle='--')   # the vert line (fwd)
                        lyf.set_xdata(self.x[self.fwd])
                        
                    # draw
                    plt.draw()
                    
                def find_closest_point(self, xc, yc, x, y):
                    '''
                      xc/yc are the cursor input coords
                      x/y are the data arrays
                    '''
                    delta_x = ([xc]*len(x))-x
                    delta_y = ([yc]*len(y))-y
                    r = ((delta_x**2)+(delta_y**2))**0.5
                    return int(np.argmin(r)), np.min(r)

                def key_press(self, event):
                    if not event.inaxes:
                        return
                      
                    x, y = event.xdata, event.ydata
                    if event.key == 'q':
                        idx, val = self.find_closest_point(x, y, self.x, self.y)
                        if idx not in self.read:
                            self.read.append(idx)
                            self.calculate_read_noise() 
                            print "i: added read regime point"
                    if event.key == 'w':
                        idx, val = self.find_closest_point(x, y, self.x, self.y)
                        if idx not in self.shot:
                            self.shot.append(idx)                        
                            self.calculate_gain()   
                            print "i: added shot regime point"
                    if event.key == 'e':
                        idx, val = self.find_closest_point(x, y, self.x, self.y)
                        self.fwd = idx       
                        print "i: added fwd line"
                    elif event.key == 'x':
                        idx, val = self.find_closest_point(x, y, self.x, self.y)
                        if idx in self.read:
                            idx_to_pop = self.read.index(idx)
                            self.read.pop(idx_to_pop)
                            self.calculate_read_noise() 
                            print "i: cleared read regime point"
                        if idx in self.shot:
                            idx_to_pop = self.shot.index(idx)
                            self.shot.pop(idx_to_pop)
                            self.calculate_gain() 
                            print "i: cleared shot regime point"
                        if idx == self.fwd:
                            self.fwd = None
                            print "i: cleared fwd line"
                    elif event.key == 'm':
                        self.read = []
                        self.shot = []
                        self.fwd = None
                        self.rn = None
                        self.gain = None
                        print "i: cleared all points" 
                    elif event.key == 'a':
                        self.smooth_data()
                        self.calculate_gain() 
                        self.calculate_read_noise() 
                        print "i: smoothed data"
                    self.draw()
                    
                def mouse_move(self, event):
                    if not event.inaxes:
                        return

                    x, y = event.xdata, event.ydata
                    idx, val = self.find_closest_point(x, y, self.x, self.y)
                    self.c_idx = idx

                    self.draw()
                
                def smooth_data(self):
                    to_idx = len(self.x)-1
                    
                    if self.fwd is not None:                                                # use FWD if it's been applied
                        to_idx = self.fwd
                        
                    to_rev_idx = [x2-x1 < 0 for x1, x2 in zip(self.x[:-1], self.x[1:])]     # catch for reverse turnover (occurs in some data after full well)
                    if True in to_rev_idx and to_rev_idx < self.fwd:
                        to_idx = [idx for idx, xi in enumerate(to_rev_idx) if xi is True]
                        
                    s = interpolate.UnivariateSpline(self.x[:to_idx+1], self.y[:to_idx+1], k=3, s=10)	# apply smoothing cubic bspline
                    self.x = self.x[:to_idx+1]
                    self.y = s(self.x)

            fig = plt.figure()
            ax = plt.gca()
            reg = define_PTC_regions(ax, np.log10(x), np.log10(y))
            reg.calculate_read_noise()
            reg.calculate_gain()
            reg.draw()
            plt.connect('motion_notify_event', reg.mouse_move)
            plt.connect('key_press_event', reg.key_press)
            plt.show()
            
            rn.append(reg.rn)
            gain.append(reg.gain)
            qx.append(reg.x)
            qy.append(reg.y)
 
        if self.makePlots: 
            for idx_q in range(len(qx)):
                this_rn = rn[idx_q]
                this_gain = gain[idx_q]
                this_q_x = qx[idx_q]
                this_q_y = qy[idx_q]
                
                pos = self.settings['quadrants'][idx_q]['pos']
                is_defective = bool(self.settings['quadrants'][idx_q]['is_defective'])
                plt.subplot(2, 2, idx_q+1)
                plt.yscale('log')
                plt.xscale('log')
                plt.xlabel("Signal (ADU)")
                plt.ylabel("Noise (ADU)")
                if is_defective:
                    plt.plot([], label='data for quadrant: ' + str(self.settings['quadrants'][idx_q]['pos']), color='white')
                    plt.legend(loc='upper left')
                    continue
                plt.plot(10**this_q_x, 10**this_q_y, 'k.', label='data for quadrant: ' + str(self.settings['quadrants'][idx_q]['pos']))
                if this_rn is not None:
                    plt.plot([10**0, np.max(10**this_q_x)], [10**this_rn, 10**this_rn], 'k--', label="read noise: " + str(round(10**this_rn, 2)) + " ADU")
                if this_gain is not None:                
                    plt.plot([10**this_gain[0], 10**this_gain[1]], [10**0, 10**this_gain[2]], 'k--', label="gain: " + str(round(10**this_gain[0], 2)) + " e-/ADU")
                plt.legend(loc='upper left')
            plt.tight_layout()
            plt.show()    
  
if __name__  == "__main__":
    parser = optparse.OptionParser()
    group1 = optparse.OptionGroup(parser, "General") 
    group1.add_option('--l', action='store', default='INFO', dest='logLevel', help='logging level (DEBUG|INFO|WARNING|ERROR|CRITICAL)')    
    group1.add_option('--f', action='store', default="settings.json", type=str, dest='settingsFile', help='Path to settings file')
    group1.add_option('--d', action='store', default=None, type=str, dest='dataPath', help='Path to data files (either directory or single file)')
    group1.add_option('--p', action='store_true', dest='makePlots', help='Make plots?')
    group1.add_option('--dm', action='store_true', dest='diagnosticMode', help='Diagnostic mode?')
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

