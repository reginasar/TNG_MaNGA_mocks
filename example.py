import imp
sifu = imp.load_source('sin_ifu_clean', 'sin_ifu_clean.py')

snap = 98
subhalo = 323653
re_kpc = 1.37177
basePath = 'path/to/TNG/Illustris/data/'

########## Step 1 ###########
mk_particle_files(subhalo, snap, basePath, ex=[1,0,0], FOV=19, overwrite=True)

########## Step 2 ###########
#For the next step template gas file must be dowloaded separately and saved in libs directory templete_gas.fits.gz

mk_mock_RSS(snap, subhalo, view=0, sp_samp=1., template_SSP_control='libs/MaStar_CB19.slog_1_5.fits.gz',\
                template_SSP='libs/MaStar_CB19.slog_1_5.fits.gz', template_gas='libs/templete_gas.fits.gz', fib_n=7)
# mk_mock_RSS outputs
rss_file = 'ilust-'+str(snap)+'-'+str(subahlo)+'-0-127.cube_RSS.fits.gz'

########## Step 3 ###########
regrid(rss_file, template_SSP_control='libs/MaStar_CB19.slog_1_5.fits.gz', \
            n_fib=7, thet=0.0, R_eff=re_kpc, include_gas=False)

########## Step 4 ###########



