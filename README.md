# MaNGA mocks from TNG50 galaxies

This is the code used for the data release "". In this article we present the mocking procedure to mimic the galaxy integral spectroscopic data from the MaNGA survey (~10,000 galaxies) from the TNG50 hydro-dynamical cosmological simulation. 

![spec_at_R_examples](https://user-images.githubusercontent.com/50836927/178962418-71b3ca6a-c501-48a9-8876-1b69394e5595.png)

The sample selection was done with -  mk_mangalike_tng_sample.py  - and each mock observation was produced following steps described in the next section. This second part of the code is based partially based on https://github.com/hjibarram/mock_ifu, a code to emulate a MaNGA observation from a numerically simulated galaxy. The code has been updated to Python3 and some libraries have been replaced with more standard ones (see requirements below). The steps of the procedure are separated to provide more versatility to the user. Additionally, one part of the code has been adapted to run in parallel. Running times heavily depend on the number of particles in the simulation's field of view and bundle size choice. It is possible to parallelize the "fiber obsevation" (step 2), which reduces the computing time.

## Steps to create a MaNGA mock:


1 - Make particle files from a simulation (adapted to the TNG format and TNG50 cosmological assumptions).

        Output: star particles and gas cells properties in two separated files.
      
      
2 - Produce a row stacked spectrum (RSS) from the particle file.

	    Output: RSS file.
      
      
3 - Add noise and degrade spectral resolution. Recombine the n-fibres to obtain a MaNGA DRP-like cube.

	    Output: MaNGA DRP-like file (a.k.a. the datacube).
      
      
4 - Stellar population analysis is performed using pyPipe3d. The configuration file used in the paper is -   -.

	    Output: Stellar populations post-processed maps.

### Requirements

Python3: Numpy, Astropy, Scipy

For step 1 access to TNG simulations is required, as well as installing illustris_python module. Publicly available at https://www.tng-project.org/data/ 

### References


# Fast Docs

mk_particle_files(subhalo_id, snap, basePath, ex=[1,0,0], FOV=19, overwrite=True, view=0, outdir=''):

    Makes the stellar particle and gas cells files for a subhalo living in
    snapshot snap, observed in ex direction covering a field of view 
    of 2FOV of diameter.

    Arguments:
    ----------
    subhalo_id: subhalo index in the snapshot. (integer)
    snap: snapshot number. (integer with value in [1;99]) 
    basePath: path to simulation data. (string) 
    ex (=[1,0,0]): unitary 3-D vector indicating the direction in which
                   the observer is placed. (2-sized float array)
    FOV (=19): radius of the circular field of view in arcsec. (float)
    view (=0): view identifier. (integer)
    overwrite: (bool)
    outdir: path where the output files are saved. (string)

    Returns:
    -------
    -

    Outputs:
    -------
    Stellar particles and gas cells information in two individual 
    files:

    'snap'+snap+'_shalo'+subhalo_id+'_'+view+'_stars.dat'
    Contains as many rows as stellar particles in the FOV, columns are defined as:
         - x, y, z coordinates relative to the galaxy centre in physical kpcs
           (z in the direction of the observer).
         - vx, vy, vz velocity components relative to the volume in km/s.
         - age in Gyrs.
         - metallicity in Z/H.
         - mass in solar masses.

    'snap'+snap+'_shalo'+subhalo_id+'_'+view+'_gas.dat'
    Contains as many rows as gas cells in the FOV, columns are defined as:
         - x, y, z coordinates relative to the galaxy centre in physical kpcs
           (z in the direction of the observer).
         - vx, vy, vz velocity components relative to the volume in km/s.
         - metallicity in Z/H.
         - volume in kpc**3.
         - density in solar masses/ kpc**3.
         - star formation rate in solar masses per yr.
         - temperature in K.
         - Av extinction index.
         - mass in solar masses.

mk_mock_RSS(snap, subhalo, view, sp_samp=1., template_SSP_control, template_SSP, template_gas, outdir='', fib_n=7, cpu_count=2\
                psf=0, nl=110, fov=30.0, fov1=0.2, sig=2.5, thet=0.0, rx=[0,0.5,1.0,2.0], ifutype='MaNGA', red_0=0.01, indir=''):

    Reads particle/cell files and feeds it to mk_the_light function.
    
mk_the_light(outf, x, y, z, vx, vy, vz, x_g, y_g, z_g, vx_g, vy_g, vz_g, age_s, met_s, mass_s, met_g, vol, dens, sfri, temp_g,\
              Av_g, mass_g, template_SSP_control, template_SSP, template_gas, dir_o='', sp_samp=1.25, psfi=0,\
              red_0=0.01, nl=7, cpu_count=8, fov=30.0, sig=2.5, thet=0.0, pdf=2, rx=[0,0.5,1.0,2.0],\
              ifutype='MaNGA'):
	      
    Given the particle/cell properties, SSP template and the IFU type produces the fiber spectra.
    
regrid(rss_file, template_SSP_control, dir_r='', dir_o='', n_fib=7, thet=0.0, R_eff=None, include_gas=False):


