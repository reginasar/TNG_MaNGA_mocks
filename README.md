[MaNGIA_prototype.pdf](https://github.com/reginasar/TNG_MaNGA_mocks/files/9609465/MaNGIA_prototype.pdf)

# 10,000 MaNGA mocks from TNG50 galaxies

This is the code used for the data release "". In this article we present the mocking procedure to mimic the galaxy integral spectroscopic data from the MaNGA survey (~10,000 galaxies) from the TNG50 hydro-dynamical cosmological simulation. 

![spec_at_R_examples](https://user-images.githubusercontent.com/50836927/178962418-71b3ca6a-c501-48a9-8876-1b69394e5595.png)

The sample selection was done with -  mk_mangalike_tng_sample.py  - and each mock observation was produced following steps described in the next section. This second part of the code is based on https://github.com/hjibarram/mock_ifu, a code to emulate a MaNGA observation from a numerically simulated galaxy. The code has been updated to Python3 and some libraries have been replaced with more standard ones (see requirements below). The steps of the procedure are separated to provide more versatility to the user. Additionally, one part of the code has been adapted to run in parallel. Running times heavily depend on the number of particles in the simulation's field of view and bundle size choice. It is possible to parallelize the "fiber obsevation" (step 2), which reduces the computing time.

## Steps to create a MaNGA mock:


1 - Make particle files from a simulation (adapted to the TNG format and TNG50 cosmological assumptions).

        Output: star particles and gas cells properties in two separated files.
      
      
2 - Produce a row stacked spectrum (RSS) from the particle file.

	    Output: RSS file.
      
      
3 - Add noise and degrade spectral resolution. Recombine the n-fibres to obtain a MaNGA DRP-like cube.

	    Output: MaNGA DRP-like file (a.k.a. the datacube).
      
      

### Requirements

Python3: Numpy, Astropy, Scipy

For step 1 access to TNG simulations is required, as well as installing illustris_python module. Publicly available at https://www.tng-project.org/data/ 

SSP templates in pyPipe3D format: http://ifs.astroscu.unam.mx/pyPipe3D/templates/

### References


# Fast Docs

Key arguments together with the outputs of the main functions are defined below. If you want more, feel free to dig into the code :)

<strong>mk_particle_files</strong>():

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
         - x, y, z coordinates relative to the observer in physical kpcs
           (z in the direction of the observer).
         - vx, vy, vz velocity components relative to the volume in km/s.
         - age in Gyrs.
         - metallicity in Z/H.
         - mass in solar masses.

    'snap'+snap+'_shalo'+subhalo_id+'_'+view+'_gas.dat'
    Contains as many rows as gas cells in the FOV, columns are defined as:
         - x, y, z coordinates relative to the observer in physical kpcs
           (z in the direction of the observer).
         - vx, vy, vz velocity components relative to the volume in km/s.
         - metallicity in Z/H.
         - volume in kpc**3.
         - density in solar masses/ kpc**3.
         - star formation rate in solar masses per yr.
         - temperature in K.
         - Av extinction index.
         - mass in solar masses.

<strong>mk_mock_RSS</strong>():
		
    Reads particle/cell files and feeds it to mk_the_light() function.

    Arguments:
    ----------
    star_file: file with stellar particles info. (string)
    gas_file: file with gas cells info. (string)
    template_SSP_control: SSP template file name for control. (string)
    template_SSP: SSP template file name to produce the stellar 
                  spectra. (string)
    template_gas: gas template file name to produce the emission lines.
                  (string)
    sp_samp (=1.): spectral sampling in Ang. (float)
    fib_n (=7): fiber number radius, defines de IFU size. (integer in [3,4,5,6,7])
    cpu_count (=2): number of CPUs to be used in parallel. (integer)
                    Ignored if environment variable 'SLURM_CPUS_PER_TASK' 
                    is defined.
    psfi (=0): instantaneous point spread function (PSF as FWHM). (float)
    thet (=0.0): angular offset. (float)
    ifutype (='MaNGA'): IFU options fixed to IFS instrument. 
                       (string in ['MaNGA', 'CALIFA', 'MUSE'])
    red_0 (=0.01):redshift at which the galaxy is placed. (float)
    outdir (=''): output directory path. (string)
    indir (=''): input directory path. (string)
    rssf (=''): RSS output file name. (string)

    Returns:
    -------
    -

    Outputs:
    RSS file produced by mk_the_light() funtion.


    
<strong>mk_the_light</strong>():
	      
    Given the particle/cell properties, SSP template and the IFU type
    produces the fiber spectra.

    Arguments:
    ----------
    outf: output file name. (string)
    stellar particle and gas cells properties (each a float array):
      Stellar
         - x, y, z coordinates relative to the observer in physical kpcs
           (z in the direction of the observer).
         - vx, vy, vz velocity components relative to the volume in km/s.
         - age in Gyrs.
         - metallicity in Z/H.
         - mass in solar masses.
      Gas
         - x, y, z coordinates relative to the observer in physical kpcs
           (z in the direction of the observer).
         - vx, vy, vz velocity components relative to the volume in km/s.
         - metallicity in Z/H.
         - volume in kpc**3.
         - density in solar masses/ kpc**3.
         - star formation rate in solar masses per yr.
         - temperature in K.
         - Av extinction index.
         - mass in solar masses.    
    template_SSP_control: SSP template file name for control. (string)
    template_SSP: SSP template file name to produce the stellar 
                  spectra. (string)
    template_gas: gas template file name to produce the emission lines.
                  (string)
    sp_samp (=1.25): spectral sampling in Ang. (float)
    dir_o (=''): output directory. (string)
    psfi (=0): instantaneous point spread function (PSF as FWHM). (float)
    red_0 (=0.01): redshift at which the galaxy is placed. (float)
    nl (=7): fiber number radius, defines de IFU size. (integer in [3,4,5,6,7])
    cpu_count (=8): number of CPUs to be used in parallel. (integer)
                    Ignored if environment variable 'SLURM_CPUS_PER_TASK' 
                    is defined.
    thet (=0.0): angular offset. (float)
    ifutype (='MaNGA'): IFU options fixed to IFS instrument. 
                       (string in ['MaNGA', 'CALIFA', 'MUSE'])

    Returns:
    -------
    -

    Outputs:
    -------
    FITS file containing the row stacked spectrum (RSS).

    
<strong>regrid</strong>():

    Computes the spectral data cube from the row stacked spectra. 
    Downgrades the gas emission spectral resolution and adds noise to 
    each spectrum before combining the fiber spectra. 

    Arguments:
    ----------
    rss_file: Input RSS file name. (string)
    outf: Output file name, without extensions. (string)
    template_SSP_control:  SSP template file name for control. (string)
    dir_r (=''): Input directory, where the RSS file is stored.(string)
    dir_o (=''): Output directory. (string)
    n_fib (=7): 
    thet (=0.0):
    R_eff (=None): Effective raius of the galaxy, to define S/N. (float)
    include_gas (=False): include emission lines in the output cube. (bool)


    Returns:
    -------
    -

    Outputs:
    -------
    Two FITS files:
    
    outf + '.cube.fits.gz'
    File comprising a Primary HDU and three extensions:
       Primary - Spatial-spectral datacube.
       ERROR - Error per spatial-wavelength unit.
       MASK - Valid mask array, necessary for pyPipe3D processing.
       GAS - Gas only spatial-spectral datacube.
    
    outf + '.cube_val.fits.gz'
    File comprising a Primary HDU and three extensions:
       Primary - Spatial real values from simulations.
       MASS_PER_AGE - Total mass assigned to each age in the template.
       MASS_PER_AGE_MET - Total mass assigned to each age and 
                          metallicityin the template.
       LUM_PER_AGE_MET - Total luminosity assigned to each age and 
                         metallicity in the template.


