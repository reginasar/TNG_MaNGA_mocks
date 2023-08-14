# This example generates the RSS file, together with the cube_val and 3D datacube
# We use reduced stellar particle and gas cell files (to 10%) called:
# snap98_shalo323653_stars.dat and snap98_shalo323653_gas.dat (both in the repo)
# You will need an SSP template in the libs folder, we use:
# MaStar_CB19.slog_1_5.fits.gz (in the repo)
# And an additional emission line template from (e.g. produced with cloudy):
# templete_gas.fits 

#### For Python versions > 3.5 import module as
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('sin_ifu_clean', 'sin_ifu_clean.py')
sifu = importlib.util.module_from_spec(spec)
sys.modules['sin_ifu_clean'] = sifu
spec.loader.exec_module(sifu)
#### For older Python versions use the following 2 lines
#import imp
#sifu = imp.load_source('sin_ifu_clean', 'sin_ifu_clean.py')

snap = 98 # snapshot number
subhalo = 323653 # subhalo ID within the snapshot
re_kpc = 1.37177 # Petrossian effective radius (predicted from fit)
redshift = 0.035 # redshift of snapshot
fib_n = 7 # largest bundle size
ns = 3 * fib_n * (fib_n-1) + 1 #total number of fibres in bundle

#######################################################################
###### Step 1: Create particle files ##################################
#######################################################################

# For testing, you can use the particle files saved in the libs folder,
# which contains randomly selected 10% of particles and cells of a TNG50 galaxy
#sifu.mk_particle_files(subhalo, snap, basePath, ex=[1,0,0], FOV=19, overwrite=True, \
#             outdir='libs')

#######################################################################
###### Step 2: Generate row stacked spectrum (RSS) file ###############
#######################################################################

# For the next step template gas file must be dowloaded separately and saved in libs directory templete_gas.fits.gz
# mk_mock_RSS outputs a RSS file

sifu.mk_mock_RSS('snap98_shalo323653_stars.dat', 'snap98_shalo323653_gas.dat', \
             indir='../', outdir='../', red_0=redshift,\
             rssf='TNG50-'+str(snap)+'-'+str(subhalo),\
             template_SSP_control='libs/MaStar_CB19.slog_1_5.fits.gz',\
             template_SSP='libs/MaStar_CB19.slog_1_5.fits.gz', \
             template_gas='libs/templete_gas.fits', fib_n=fib_n, cpu_count=4)


#######################################################################
###### Step 3: Recombine fiber spectrum to create mock data cube ######
#######################################################################

sifu.regrid('TNG50-'+str(snap)+'-'+str(subhalo)+'-'+str(ns)+'.cube_RSS.fits.gz',\
            outf='TNG50-'+str(snap)+'-'+str(subhalo),
            dir_r='../', dir_o='../', n_fib=7, thet=0.0, R_eff=re_kpc, \
            template_SSP_control='libs/MaStar_CB19.slog_1_5.fits.gz', \
            include_gas=False)


