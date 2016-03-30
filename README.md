weave-ccd-analysis
=============

# Overview

This package provides a set of Python2 tools to visualise and analyse frames taken with a CCD. 

Currently it includes:

* `bias.py` - measure spatial and temporal read noise
* `ptc.py` - interactively generate a ptc curve, calculating corresponding read noise and gain
* `fft.py` - construct a noise power spectrum for a given frame (*not complete*)

# Getting Started

Create a new instrument setting in `settings.json`. **WEAVE_PROTO** illustrates a sample entry, 
formatted as a key:value pair, where the key specifies the instrument setting name, and the value 
is an array of options defining this configuration. This name will need to be specified with the --f 
flag on the command line. Most of the configuration items should be easily understandable from their 
nomenclature alone, except for the:

* **[id]** keys. These are arbitrary.
* **[pos]** keys. These provide labels to identify each quadrant, used in plots.
* **[is\_defective]** keys. These can be used to omit analysis on a particular quadrant. 

Analysis regions are defined by the [**[xy?]\_\***] parameters in the settings file.

Debiasing is performed on each frame using the region defined by the [**overscan\_\***] 
parameters in the settings file.

Make sure you specify the correct HDU ([**data\_hdu**] and [**dummy\_hdu**]) for your data and dummy. 
Dummy subtraction can be turned off/on using the [**do\_dummy\_subtraction**] flag in the settings 
file.

Both `bias.py` and `ptc.py` have help options. These can be found by invoking each 
with the --h flag, e.g.

`[rmb@rmb-tower weave-ccd-analysis]$ python bias.py --h`

# bias.py

This tool provides the ability to measure spatial and temporal noise. The former is achieved by 
measuring read noise within a section on a frame by a frame basis, and the latter by measuring 
the read noise across each frame for each pixel within a given section.

More than one frame is required for temporal read noise measurement.

# ptc.py

This is an interactive tool to measure gain and read noise from an "appropriate" set of frames. Do not 
use this tool with dummy subtraction enabled.

An appropriate set of frames is defined as being pairs of frames with the same exposure time (deduced 
from the *EXPTIME* key in the header of each file), taken in a sequence of increasing (or decreasing) 
exposure times. If only one frame is found for a particular exposure time, it will be ignored. If insufficient 
pairs are found (defined by the global [**MINIMUM\_FRAMES\_REQUIRED**]), then the program will exit.

A log-log plot of mean signal against noise is displayed on success. At this point, the user is required 
to define points that will be used to deduce the read and shot regimes of the curve. Each time a point is 
deleted or added, the read noise and gain will be recalculated. Based on the gradient of the slope, a first 
guess at each regime is done automagically.

The read noise floor is found by extrapolation of a second order polynomial fitted to the read noise regime 
points, where the corresponding y value corresponding to a gradient of 0 for this polynomial is calculated, 
giving the read noise in ADU.

The gain is found by fitting a second order polynomial to the shot noise regime points, calculating the x, y 
corresponding ot a gradient of 0.5 for this fitted polynomial, and extrapolating a line back to y=0 
using these parameters. The resulting x-intercept then defines the gain in e-/ADU.

The interactive process has a number of other options (shown in console):

| Key            | Function                                           | 
| :------------- | :------------------------------------------------- |
| q              | define a point for read noise regime               |
| w              | define a point for shot noise regime               |
| e              | define full well                                   |
| a              | smooth data first using cubic spline interpolation |
| x              | clear single point definition                      |
| m              | clear all point definitions                        |
| r              | remove point from dataset entirely                 |








