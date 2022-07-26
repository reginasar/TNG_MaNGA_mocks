
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

snap = 98
subhalo = 323653
view = 0
re_kpc = 1.37177
redshift = 0.035
basePath = 'path/to/TNG/Illustris/data/'

#######################################################################
###### Step 1: Create particle files ##################################
#######################################################################

sifu.mk_particle_files(subhalo, snap, basePath, ex=[1,0,0], FOV=19, overwrite=True)

#######################################################################
###### Step 2: Generate row stacked spectrum (RSS) file ###############
#######################################################################

#For the next step template gas file must be dowloaded separately and saved in libs directory templete_gas.fits.gz
# mk_mock_RSS outputs a RSS file called
rss_file = 'ilust-'+str(snap)+'-'+str(subahlo)+'-0-127.cube_RSS.fits.gz'

sifu.mk_mock_RSS(star_file='/snap'+str(snap)+'_shalo'+str(subhalo)+'_'+str(view)+'_stars.dat',\
                 gas_file='/snap'+str(snap)+'_shalo'+str(subhalo)+'_'+str(view)+'_gas.dat',\
                 sp_samp=1., indir='', red_0=redshift,\
                 rssf='ilust-'+str(snap)+'-'+str(subhalo)+'-'+str(view),\
                 template_SSP_control='libs/MaStar_CB19.slog_1_5.fits.gz',\
                 template_SSP='libs/MaStar_CB19.slog_1_5.fits.gz', \
                 template_gas='libs/templete_gas.fits', fib_n=7, cpu_count=4)

#######################################################################
###### Step 3: Recombine fiber spectrum to create mock data cube ######
#######################################################################

sifu.regrid(rss_file, template_SSP_control='libs/MaStar_CB19.slog_1_5.fits.gz', \
            n_fib=7, thet=0.0, R_eff=re_kpc, include_gas=False)


