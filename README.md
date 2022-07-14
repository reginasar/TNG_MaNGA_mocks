# MaNGA mocks from TNG50 galaxies

This is code is based partially based on https://github.com/hjibarram/mock_ifu, a code to emulate a MaNGA observation from a numerically simulated galaxy.

The code has been updated to Python3 and some libraries have been replaced with more standard ones (see requirements below). The steps of the procedure are separated to provide more versatility to the user. Additionally, one part of the code has been adapted to run in parallel.

This is the code used for the data release "". Where the sample selection was done with -  mk_mangalike_tng_sample.py  - and each mock observation was produced following steps described in the next section. 

Running times heavily depend on the number of particles in the simulation's field of view and bundle size choice. It is possible to parallelize the "fiber obsevation" (step 3), which reduces the computing time.

## Steps to create a MaNGA mock:

1 - Make particle files from a simulation (adapted to the TNG format and TNG50 cosmological assumptions).  -  mk_particle_files.py  -

        Output: star particles and gas cells properties in two separated files.
      
      
2 - Produce a row stacked spectrum (RSS) from the particle file.
	    Output: RSS file.
      
      
3 - Add noise and degrade spectral resolution. Recombine the n-fibres to obtain a MaNGA DRP-like cube.

	    Output: MaNGA DRP-like file (a.k.a. the datacube).
      
      
4 - Stellar population analysis is performed using pyPipe3d. The configuration file used in the paper is -   -.

	    Output: Stellar populations post-processed maps.

### Requirements

Python3 (Numpy, Astropy, Scipy)

For step 1 access to TNG simulations is required, as well as installing illustris_python module.



