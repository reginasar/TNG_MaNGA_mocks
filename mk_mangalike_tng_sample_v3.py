import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
from utils_real.helpers import load_tng_maps, cross_catalog_indices#, load_labels
from scipy.spatial.distance import cdist
from scipy.stats import binned_statistic_2d, binned_statistic
from astropy.cosmology import FlatLambdaCDM
import seaborn as sn
import pandas as pd
from matplotlib.gridspec import GridSpec
from astropy.cosmology import Planck15 as cosmo
from collections import Counter
from scipy.optimize import curve_fit
from astropy.table import Table

max_step=50
mangapath = '/Volumes/Personal/Datos/Beckys_sim/manga/'

def load_labels(y_title):
    hdu_cat = fits.open('/Volumes/Personal/Datos/Beckys_sim/catalogs/drpall-v3_1_1.fits')
    plate_ifu = hdu_cat[1].data['plateifu']
    n_files_end = len(plate_ifu)
    #y = np.zeros((n_files_end, len(y_title)), dtype=np.float32)
    y = []
    evt_data = Table(hdu_cat[1].data)
    for column in y_title:
        if column in evt_data.colnames:
            #y[:, y_title.index(column)] = hdu_cat[1].data[column]
            column_data = np.array(hdu_cat[1].data[column])
            #column_data = np.delete(column_data, hdu_cat[1].data['z']!=-999., axis=0)
            if column_data.size>n_files_end:
                for ii in range(column_data.shape[1]):
                    y.append(column_data[:,ii])
            else:
                y.append(column_data)
        else:
            print('Warning: No column with name '+column+' in catalog.')
    #y = y[hdu_cat[1].data['z']!=-999., :]
    hdu_cat.close()
    y = np.array(y)
    #print(y.shape)
    return y#[:,valid_pipe]

def get_tng_params(tng_sim='tng50-1', resolved=[0,3]):
    #tng_snap_z = np.array([0., 0.012, 0.023, 0.035, 0.048, 0.06, 0.073, 0.086, 0.1, 0.11, 0.13, 0.14, 0.15, 0.17, 0.18])
    #tng_snap = range(99, 84, -1)
    mass_bins_ = np.array([8, 10, 10.5, 11, 11.5, 20.])
    tng_snap_z = np.array([0.012, 0.023, 0.035, 0.048, 0.06, 0.073, 0.086, 0.1, 0.11, 0.13, 0.14, 0.15])
    tng_snap = range(98, 86, -1)
    stellar_mass_tng = {}
    re_kpc_tng = {}
    log_re_arc_tng = {}
    tng_id = {}
    g_r_tng = {}
    star_part = {}
    gas_part = {}
    SFRT = {}
    SFR05 = {}
    SFR1 = {}
    cpetro_mass = {}
    m_pop = np.array([0.9331183649420131, 1.0314710924885302, 1.051059901121417, 1.1064691388558692, 1.3324744673090678])
    a_pop = 10.**(np.array([0.17797250689766253, 0.13534594258671162, 0.06858537287119049, -0.019359946206312095, -0.38161267215327976]))
    #m_pop_cpetro = np.array([0.8589682917072374, 1.016953713556076, 1.0705398818394671, 1.0712989817357226, 0.9487941283577627])
    #a_pop_cpetro = 10.**(np.array([0.4525467835762671, 0.3799795037376916, 0.3013600972184495, 0.2645208659754392, 0.3078198437677128])
    mass_bins = np.array([8.5, 9.5, 10, 10.5, 11, 11.5, 14.])
    for ii in range(len(tng_snap)):
        #labels_tng = np.load('/Users/reginasarmiento/Documents/machineGALearning/prog/TNG/plots/'+tng_sim+'_prop_list/'+str(tng_snap[ii])+'_info_'+tng_sim+'_no_particle_lim.npy')
        labels_tng = np.load('/Users/reginasarmiento/Documents/machineGALearning/prog/TNG/setup_vera/plots/'+tng_sim+'_prop_list/'+str(tng_snap[ii])+'_info_'+tng_sim+'_8.5mass.npy')
        stellar_mass_tng[tng_snap[ii]] = np.log10(labels_tng[:,-5])#mass within 30pkpc radius
        #re_kpc_tng[tng_snap[ii]] = 1.325*labels_tng[:,1] #factor to aprox reff 1.17*
        mass_dig = np.digitize(labels_tng[:,2], mass_bins_)
        re_kpc_tng[tng_snap[ii]] = np.array([a_pop[mass_dig[rr]-1]*labels_tng[rr,1]**m_pop[mass_dig[rr]-1] for rr in range(len(labels_tng[:,1]))])
        log_re_arc_tng[tng_snap[ii]] = np.log10(re_kpc_tng[tng_snap[ii]] / (cosmo.kpc_proper_per_arcmin(z=tng_snap_z[ii]).value/60.))
        tng_id[tng_snap[ii]] = np.array(labels_tng[:,0], dtype=np.int32)    
        g_r_tng[tng_snap[ii]] = labels_tng[:,6] - labels_tng[:,7] #g-r +3.66#g-i SDSS + 3.76 made up dust
        star_part[tng_snap[ii]] = labels_tng[:,15]
        gas_part[tng_snap[ii]] = labels_tng[:,11]
        SFRT[tng_snap[ii]] = labels_tng[:,-5]
        SFR05[tng_snap[ii]] = labels_tng[:,-4]
        SFR1[tng_snap[ii]] = labels_tng[:,-3]
        cpetro_mass[tng_snap[ii]] = np.log10(labels_tng[:,-1])

    return tng_snap, tng_snap_z, stellar_mass_tng, re_kpc_tng, tng_id, g_r_tng, star_part, gas_part, SFRT, SFR05, SFR1, log_re_arc_tng, cpetro_mass


### TNG snaps
snapz_dict = {99:0., 98:0.012, 97:0.023, 96:0.035, 95:0.048, 94:0.06, 
              93:0.073, 92:0.086, 91:0.1, 90:0.11, 89:0.13, 88:0.14, 
              87:0.15, 86:0.17, 85:0.18}

#tng_snap_zlimit = np.array([0., 0.012, 0.023, 0.035, 0.048, 0.06, 0.073, 0.086, 0.1, 0.11, 0.13, 0.14, 0.15, 0.17, 0.18])
#tng_snap = range(99, 84, -1)
tng_snap_zlimit = np.array([0.023, 0.035, 0.048, 0.06, 0.073, 0.086, 0.1, 0.11, 0.13, 0.14, 0.15])
tng_snap = range(98, 86, -1)

### MaNGA
#labels_manga, discard = load_labels(['ifu_size', 'log_Mass', 're_kpc', 'spa_cov', 'redshift'])
labels_manga = load_labels(['ifudesignsize', 'nsa_elpetro_mass', 'nsa_elpetro_th50_r','z', 'pweight', 'sweight', 'eweight', 'nsa_elpetro_absmag', 'esweight'])#, catalog='drpall', drogon=False)
plateifu_manga_drp = load_labels(['plateifu'])#, catalog='drpall', drogon=False)
manga_id = load_labels(['mangaid'])#, catalog='drpall', drogon=False)

pweight = labels_manga[4]
sweight = labels_manga[5]
eweight = labels_manga[6]
my_sample_div = -np.ones(pweight.size, dtype=np.int16)
my_sample_div[pweight!=-999.] = 1
my_sample_div[sweight!=-999.] = 2
my_sample_div[(eweight!=-999.)&(pweight==-999.)] = 0

##for ii in range(len(labels_manga)):
##    labels_manga[ii] = labels_manga[ii][my_sample_div!=-1]
#labels_manga = labels_manga[:,my_sample_div!=-1]
plateifu_manga_drp = plateifu_manga_drp[0]#,my_sample_div!=-1]
manga_id = manga_id[0]#,my_sample_div!=-1]
_, unique_manga, _ = np.intersect1d(manga_id, manga_id, return_indices=True)
#my_sample_div = my_sample_div[my_sample_div!=-1]
wrong_rows = np.nonzero((labels_manga[2]>0.)&(labels_manga[3]>0.)&(labels_manga[1]>10.**8.5))[0]#np.where((labels_manga[2]>0.)&(labels_manga[3]>0.), True, False)
wrong_rows = np.intersect1d(wrong_rows, unique_manga)
#for ii in range(len(labels_manga)):
    #labels_manga[ii] = labels_manga[ii][labels_manga[2]!=-9999.]
#    labels_manga[ii] = labels_manga[ii][wrong_rows]
my_sample_div = my_sample_div[wrong_rows]
labels_manga = labels_manga[:,wrong_rows]
plateifu_manga_drp = plateifu_manga_drp[wrong_rows]
plateifu_manga_drp = np.array(plateifu_manga_drp, dtype='<U11')
manga_id = manga_id[wrong_rows]
ifu_size_manga = labels_manga[0]
stellar_mass_manga = np.log10(labels_manga[1])
re_arc_manga = labels_manga[2]
redshift_manga = labels_manga[3]
#g_r_manga = labels_manga[7][:,4]-labels_manga[7][:,5] #far-UV, near-UV, UV, g, r, i, z
color_manga = labels_manga[11,:]-labels_manga[12,:] #g-r#g-i SDSS
esweight = labels_manga[-1,:]

re_kpc_manga = np.zeros_like(re_arc_manga)
##cosmo=FlatLambdaCDM(H0=67.4,Om0=0.315)
##from astropy.cosmology import Plank15 as cosmo
#cosmo_f=FlatLambdaCDM(H0=100,Om0=0.315)
for ii in range(re_arc_manga.size):
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z=redshift_manga[ii]).value/60.
    re_kpc_manga[ii] = re_arc_manga[ii]*kpc_per_arcsec

manga_mzr = np.stack((stellar_mass_manga, redshift_manga, np.log10(re_arc_manga))).T

tng_snap, tng_snap_z, stellar_mass_tng, re_kpc_tng, \
tng_id, g_r_tng, star_part, gas_part, SFRT, SFR05, SFR1, log_re_arc_tng, cpetro_mass = get_tng_params()



def get_subsample_grid(mass_m, r_m, mass_bins, r_bins, n_gal=100, z_assigned=None):
    #assert n_gal<=mass_m.size
    if z_assigned==None:
        Ndens_grid,_,_,_ = binned_statistic_2d(mass_m, r_m, r_m, \
                            bins=[mass_bins, r_bins], statistic='count')
        subsamp_grid = np.round_(Ndens_grid * n_gal / mass_m.size)
        if subsamp_grid.sum() < (n_gal-1):
            subsamp_grid = np.round_(Ndens_grid * (n_gal+n_gal-subsamp_grid.sum()) / mass_m.size)
        return np.array(subsamp_grid, dtype=np.int32)
    else:
        Ndens_volume = np.zeros((mass_bins.size-1, r_bins.size-1, 12))
        for ii in range(87,99):
            try:
                Ndens_grid,_,_,_ = binned_statistic_2d(mass_m[z_assigned==ii], \
                        r_m[z_assigned==ii], r_m[z_assigned==ii], bins=[mass_bins, r_bins],\
                        statistic='count')
                Ndens_volume[:,:,ii-87] = Ndens_grid
            except ValueError:
                continue
        subsamp_volume = np.round_(Ndens_volume * n_gal / mass_m.size)
        if subsamp_volume.sum() < (n_gal-1):
            subsamp_volume = np.round_(Ndens_volume * (n_gal+n_gal-subsamp_volume.sum()) / mass_m.size)
        return np.array(subsamp_volume, dtype=np.int32)


def digitize_2d(x, y, x_bins, y_bins, param):
    param_list = []
    x_digit = np.digitize(x, x_bins, right=False)
    y_digit = np.digitize(y, y_bins, right=False)
    for ii in range(len(x_bins)):
        param_list.append([param[(x_digit==(ii+1))&(y_digit==(jj+1))] for jj in range(len(y_bins))])
    return param_list

def assign_in_grid(subsamp_grid, id_in_bin, return_list=True, repeat=None, random=True):
    assigned_id = []
    if random==True:
        if return_list:
            if repeat==None:
                for ii in range(subsamp_grid.shape[0]):
                    for jj in range(subsamp_grid.shape[1]):
                        limit_for_bin = np.min([id_in_bin[ii][jj].size, subsamp_grid[ii, jj]])
                        assigned_id.extend(np.random.permutation(id_in_bin[ii][jj])[:limit_for_bin])
                return assigned_id
            else:
                #assert type(repeat) is int, ''
                for ii in range(subsamp_grid.shape[0]):
                    for jj in range(subsamp_grid.shape[1]):
                        id_in_bin_max = np.repeat(id_in_bin[ii][jj], repeat)
                        assigned_id.extend(np.random.permutation(id_in_bin_max)[:subsamp_grid[ii, jj]])
                return assigned_id
        else:
            if repeat==None:
                for ii in range(subsamp_grid.shape[0]):
                    for jj in range(subsamp_grid.shape[1]):
                        limit_for_bin = np.min([id_in_bin[ii][jj].size, subsamp_grid[ii, jj]])
                        assigned_id.append(np.random.permutation(id_in_bin[ii][jj])[:limit_for_bin])
                return assigned_id
            else:
                #assert type(repeat) is int, ''
                for ii in range(subsamp_grid.shape[0]):
                    for jj in range(subsamp_grid.shape[1]):
                        id_in_bin_max = np.repeat(id_in_bin[ii][jj], repeat)
                        assigned_id.append(np.random.permutation(id_in_bin_max)[:subsamp_grid[ii, jj]])
                return assigned_id
    else:
        if return_list:
            if repeat==None:
                for ii in range(subsamp_grid.shape[0]):
                    for jj in range(subsamp_grid.shape[1]):
                        limit_for_bin = np.min([id_in_bin[ii][jj].size, subsamp_grid[ii, jj]])
                        assigned_id.extend(id_in_bin[ii][jj][:limit_for_bin])
                return assigned_id
            else:
                #assert type(repeat) is int, ''
                for ii in range(subsamp_grid.shape[0]):
                    for jj in range(subsamp_grid.shape[1]):
                        id_in_bin_max = np.repeat(id_in_bin[ii][jj], repeat)
                        assigned_id.extend(id_in_bin_max[:subsamp_grid[ii, jj]])
                return assigned_id
        else:
            if repeat==None:
                for ii in range(subsamp_grid.shape[0]):
                    for jj in range(subsamp_grid.shape[1]):
                        limit_for_bin = np.min([id_in_bin[ii][jj].size, subsamp_grid[ii, jj]])
                        assigned_id.append(id_in_bin[ii][jj][:limit_for_bin])
                return assigned_id
            else:
                #assert type(repeat) is int, ''
                for ii in range(subsamp_grid.shape[0]):
                    for jj in range(subsamp_grid.shape[1]):
                        id_in_bin_max = np.repeat(id_in_bin[ii][jj], repeat)
                        assigned_id.append(id_in_bin_max[:subsamp_grid[ii, jj]])
                return assigned_id


def pick_tng_subsamp(subsamp_grid, mass_bins, r_bins, snap=None, seed=0, repeat=None, tng_pack=None):
    if tng_pack!=None:
        tng_snap, tng_snap_z, stellar_mass_tng, re_kpc_tng, \
        tng_id, g_r_tng, star_part, gas_part, SFRT, SFR05, SFR1, log_re_arc_tng, cpetro_mass = np.copy(tng_pack)
    else:
        tng_snap, tng_snap_z, stellar_mass_tng, re_kpc_tng, \
        tng_id, g_r_tng, star_part, gas_part, SFRT, SFR05, SFR1, log_re_arc_tng, cpetro_mass = get_tng_params()
    

    if snap!=None:
        re_kpc_tng[snap] = np.log10(re_kpc_tng[snap])
        Ndens_grid_tng, _, _, _ = binned_statistic_2d(cpetro_mass[snap], \
                         re_kpc_tng[snap], re_kpc_tng[snap], bins=[mass_bins, r_bins], statistic='count')
        print(np.nonzero(Ndens_grid_tng<subsamp_grid)[0].size, ' bins require repeating TNG galaxies')
        print('Max number of repetitions found: ', np.where((Ndens_grid_tng<subsamp_grid)&(subsamp_grid/Ndens_grid_tng!=np.inf),\
                         subsamp_grid/Ndens_grid_tng, 0.).max())
        tng_id_in_bin = digitize_2d(cpetro_mass[snap], re_kpc_tng[snap], mass_bins, r_bins, np.arange(cpetro_mass[snap].size))
        ind_in_snap = assign_in_grid(subsamp_grid, tng_id_in_bin, repeat=repeat)
        return tng_id[snap][ind_in_snap], ind_in_snap
    else:
        ind_in_snap = {}
        tng_id_sub = {}
        for ii in range(87,99):
            re_kpc_tng[ii] = np.log10(re_kpc_tng[ii])
            Ndens_grid_tng, _, _, _ = binned_statistic_2d(cpetro_mass[ii], \
                             re_kpc_tng[ii], re_kpc_tng[ii], bins=[mass_bins, r_bins], statistic='count')
            print(np.nonzero(Ndens_grid_tng<subsamp_grid[:,:,ii-87])[0].size, ' bins require repeating TNG galaxies')
            print('Max number of repetitions found: ', np.where((Ndens_grid_tng<subsamp_grid[:,:,ii-87])&(subsamp_grid[:,:,ii-87]/Ndens_grid_tng!=np.inf),\
                         subsamp_grid[:,:,ii-87]/Ndens_grid_tng, 0.).max())
            tng_id_in_bin = digitize_2d(cpetro_mass[ii], re_kpc_tng[ii], \
                             mass_bins, r_bins, np.arange(cpetro_mass[ii].size))
            ind_in_snap[ii] = assign_in_grid(subsamp_grid[:,:,ii-87], tng_id_in_bin, repeat=repeat)
            tng_id_sub[ii] = tng_id[ii][ind_in_snap[ii]]
        return tng_id_sub, ind_in_snap


def pick_manga_subsamp(subsamp_grid, mass_bins, r_bins, mass, reff_kpc, seed=0, repeat=None):  
    reff_kpc = np.log10(np.copy(reff_kpc))
    Ndens_grid_tng, _, _, _ = binned_statistic_2d(mass, \
                     reff_kpc, reff_kpc, bins=[mass_bins, r_bins], statistic='count')
    #print(np.nonzero(Ndens_grid_tng<subsamp_grid)[0].size, ' bins require repeating TNG galaxies')
    #print('Max number of repetitions found: ', np.where((Ndens_grid_tng<subsamp_grid)&(subsamp_grid/Ndens_grid_tng!=np.inf),\
    #                 subsamp_grid/Ndens_grid_tng, 0.).max())
    manga_id_in_bin = digitize_2d(mass, reff_kpc, mass_bins, r_bins, np.arange(mass.size))
    ind_in_manga = assign_in_grid(subsamp_grid, manga_id_in_bin, repeat=repeat, random=False)
    return ind_in_manga

def assign_snap(tng_z, manga_z):
    tng_z = np.array([0.012, 0.023, 0.035, 0.048, 0.06, 0.073, 0.086, 0.1, 0.11, 0.13, 0.14, 0.15])
    return 98 - np.argmin(np.abs(np.tile(tng_z, (manga_z.size, 1))-np.tile(manga_z, (tng_z.size, 1)).T), axis=1)

def sample_contained(subh1, subh2):
    """checks if sample 1 is contained in sample 2
    returns T/F for each index in sample 1"""
    cont_sample = []
    cont_sample_check = []
    cont_sample_ = {}
    for snap in range(87,99):
        cont_sample.append(np.array([np.nonzero(subhalo1==subh2[snap])[0].sum() for subhalo1 in subh1[snap]], dtype=bool))
        cont_sample_check.extend(np.array([np.nonzero(subhalo1==subh2[snap])[0].sum() for subhalo1 in subh1[snap]], dtype=bool))
        cont_sample_[snap] = np.intersect1d(subh1[snap], subh2[snap])
    percen_content = 100. * np.array(cont_sample_check).sum() / len(cont_sample_check)
    print('First sample is ', percen_content, '% contained in the second')
    return cont_sample_

def mk_tng_sub_pack(tng_id_, tng_pack):
    tng_snap, tng_snap_z, stellar_mass_tng, re_kpc_tng, tng_id, g_r_tng, star_part, gas_part, SFRT, SFR05, SFR1, log_re_arc_tng, cpetro_mass = tng_pack
    tng_snap_ = range(98,86,-1)
    tng_snap_z_ = np.array([0.012, 0.023, 0.035, 0.048, 0.06, 0.073, 0.086, 0.1, 0.11, 0.13, 0.14, 0.15])
    stellar_mass_tng_ = {}
    re_kpc_tng_ = {}
    re_arc_tng_ = {}
    g_r_tng_ = {}
    star_part_ = {}
    gas_part_ = {}
    SFRT_ = {}
    SFR05_ = {}
    SFR1_ = {}
    log_re_arc_tng_ = {}
    cpetro_mass_ = {}
    for ii in tng_snap_:
        stellar_mass_tng_[ii] = np.array([stellar_mass_tng[ii][tng_id[ii]==jj] for jj in tng_id_[ii]])[:,0]#mass within 30pkpc radius
        re_kpc_tng_[ii] = np.array([re_kpc_tng[ii][tng_id[ii]==jj] for jj in tng_id_[ii]])[:,0]
        #re_arc_tng_[ii] = np.array([re_arc_tng[ii][tng_id[ii]==jj] for jj in tng_id_[ii]])[:,0]
        g_r_tng_[ii] = np.array([g_r_tng[ii][tng_id[ii]==jj] for jj in tng_id_[ii]])[:,0]
        star_part_[ii] = np.array([star_part[ii][tng_id[ii]==jj] for jj in tng_id_[ii]])[:,0]
        gas_part_[ii] = np.array([gas_part[ii][tng_id[ii]==jj] for jj in tng_id_[ii]])[:,0]
        SFRT_[ii] = np.array([SFRT[ii][tng_id[ii]==jj] for jj in tng_id_[ii]])[:,0]
        SFR05_[ii] = np.array([SFR05[ii][tng_id[ii]==jj] for jj in tng_id_[ii]])[:,0]
        SFR1_[ii] = np.array([SFR1[ii][tng_id[ii]==jj] for jj in tng_id_[ii]])[:,0]
        log_re_arc_tng_[ii] = np.array([log_re_arc_tng[ii][tng_id[ii]==jj] for jj in tng_id_[ii]])[:,0]
        cpetro_mass_[ii] = np.array([cpetro_mass[ii][tng_id[ii]==jj] for jj in tng_id_[ii]])[:,0]
    return tng_snap_, tng_snap_z_, stellar_mass_tng_, re_kpc_tng_, tng_id_, g_r_tng_, star_part_, gas_part_, SFRT_, SFR05_, SFR1_,log_re_arc_tng_, cpetro_mass_

def iterative_bubble_match_fixedZ(manga_mzr, tng_sim='tng50-1', max_step=40, m_lim=0.2, \
                       r_lim=0.15, random=False, seed=0, radius=1., r_increase=0.25, more_than=1, subsample=False):
    non_rep = []
    yes_rep = []
    unas = []
    tng_pack = get_tng_params(tng_sim=tng_sim)
    if subsample: #subsample has the tng galaxies that we dont want to include
        tng_pack = get_tng_pack_rest(tng_pack, subsample)
    sample_match, sample_match_labels = bubble_sample_match_fixedZ(manga_mzr, tng_pack,\
                m_lim=m_lim, r_lim=r_lim, random=random, seed=seed, radius=radius)
    #print(manga_mzr.shape[0])
    non_rep.extend(np.array([np.intersect1d(sample_match[sample_match[:,0]==ii,1],\
            sample_match[sample_match[:,0]==ii,1]).size for ii in range(87,99)]))
    subh = [np.intersect1d(sample_match[sample_match[:,0]==ii,1],\
        sample_match[sample_match[:,0]==ii,1]) for ii in range(87,99)]
    print(subh)
    yes_rep_snap = np.zeros(12, dtype=np.int16)
    for jj,ii in enumerate(range(87,99)):
        try:
            yes_rep_snap[jj] = np.array([np.where(sample_match[sample_match[:,0]==ii,1]==subid)[0].size for subid in subh[jj]]).max()
        except ValueError:
            print('Idk what it is')   
            continue 
    yes_rep.append(yes_rep_snap.max())


    iterate = True
#    while(iterate==True):
    rep_tng_all = []
    for kk in range(max_step):
        rep_tng, rep_manga_ind = repeated_subhalos_in_manga(sample_match[:,0], \
                sample_match[:,1], sample_match_labels[:,2], more_than=more_than)
        rep_manga_ind = np.concatenate((rep_manga_ind, np.nonzero(sample_match[:,0]==0)[0]))
        unas.append(rep_manga_ind.size)
        if kk==0:
            print(rep_tng.shape, rep_manga_ind.shape, sample_match.shape)
        rep_tng_all.extend(rep_tng)
        tng_pack = get_tng_params(tng_sim=tng_sim)
        tng_pack = get_tng_pack_rest(tng_pack, np.array(rep_tng_all))
        try:
            sample_match_2, sample_match_labels_2 = bubble_sample_match_fixedZ(manga_mzr[rep_manga_ind],\
                tng_pack, m_lim=m_lim, r_lim=r_lim, random=random,\
                seed=seed, radius=radius*(1+r_increase*kk)) #z_lim*(0.05*kk+1), r_lim*(0.02*kk+1)
            non_rep.extend(np.array([np.intersect1d(sample_match_2[sample_match_2[:,0]==ii,1],\
                    sample_match_2[sample_match_2[:,0]==ii,1]).size for ii in range(87,99)]))
            subh = [np.intersect1d(sample_match_2[sample_match_2[:,0]==ii,1],\
                    sample_match_2[sample_match_2[:,0]==ii,1]) for ii in range(87,99)]
            yes_rep_snap = np.zeros(12, dtype=np.int16)
            for jj,ii in enumerate(range(87,99)):
                try:
                    yes_rep_snap[jj] = np.array([np.where(sample_match_2[sample_match_2[:,0]==ii,1]==subid)[0].size for subid in subh[jj]]).max()
                except ValueError:
                    continue
            yes_rep.append(yes_rep_snap.max())
            sample_match[rep_manga_ind,:] = sample_match_2
            sample_match_labels[rep_manga_ind,:] = sample_match_labels_2
        except IndexError:
            print('No more TNG galaxies in range')


    rep_tng, rep_manga_ind = repeated_subhalos_in_manga(sample_match[:,0], \
            sample_match[:,1], sample_match_labels[:,-1], more_than=more_than)
    rep_manga_ind = np.array(np.concatenate((rep_manga_ind, np.nonzero(sample_match[:,0]==0)[0])), dtype=np.int16)
    sample_match[rep_manga_ind,:] = np.zeros((rep_manga_ind.size,2), dtype=np.int16)
    sample_match_labels[rep_manga_ind,:] = np.zeros((rep_manga_ind.size,3), dtype=np.float32)

    return sample_match, sample_match_labels, non_rep, yes_rep, unas

def bubble_sample_match_fixedZ(manga_mzr, tng_pack, m_lim=0.2, \
                    r_lim=0.1, random=False, seed=0, radius=1.):
    tng_snap, tng_snap_z, stellar_mass_tng, re_kpc_tng, tng_id, color_tng,_,_,_,_,_, log_re_arc_tng, cpetro_mass = tng_pack
    total_gal = manga_mzr.shape[0]
    print(total_gal)
    sample_match = np.zeros((total_gal,2), dtype=np.int32)#snapshot, subhalo_id
    sample_match_labels = np.zeros((total_gal,3), dtype=np.float32) #mass, tng_z, manga_z, re_arc_tng, distance_selec

    manga_mzr_normed = np.zeros_like(manga_mzr)
    manga_mzr_normed[:,0] = manga_mzr[:,0]/(m_lim)
    manga_mzr_normed[:,2] = manga_mzr[:,2]/r_lim

    for ii in range(len(tng_snap)):
        #stellar_mass_tng[tng_snap[ii]] = stellar_mass_tng[tng_snap[ii]]/m_lim
        cpetro_mass[tng_snap[ii]] = cpetro_mass[tng_snap[ii]]/m_lim
        log_re_arc_tng[tng_snap[ii]] = log_re_arc_tng[tng_snap[ii]]/r_lim
        #print(cpetro_mass[tng_snap[ii]][:10])
    rng = np.random.default_rng(seed)

    from_snap = assign_snap(tng_snap_z, manga_mzr[:,1])
    for ii in range(total_gal):
        #print(np.where(np.abs(log_re_arc_tng[from_snap[ii]]-manga_mzr_normed[ii,2]) < (radius))[0].size)
        #print(np.nonzero((np.abs(cpetro_mass[from_snap[ii]]-manga_mzr_normed[ii,0]) < radius))[0].size)
        box_indices = np.nonzero((np.abs(cpetro_mass[from_snap[ii]]-manga_mzr_normed[ii,0]) < radius) &\
                       (np.abs(log_re_arc_tng[from_snap[ii]]-manga_mzr_normed[ii,2]) < radius))[0]
        if box_indices.size>0:
            tng_mini_mr = np.stack((cpetro_mass[from_snap[ii]][box_indices],\
                        log_re_arc_tng[from_snap[ii]][box_indices])).T
            tng_mini_mr_id = np.stack((from_snap[ii]*np.ones_like(box_indices), tng_id[from_snap[ii]][box_indices])).T
            #print(tng_mini_mr_id.shape)
            distances_box = cdist(tng_mini_mr, np.array([manga_mzr_normed[ii,[0,2]]]), 'euclidean')[:,0]
            #tng_snaps_in_box = np.nonzero(np.abs(tng_snap_z-manga_mzr_normed[ii,1])<radius)[0]
            if distances_box.min()<radius:
                if not random:
                    #print(np.argmin(distances_box), tng_mini_mr_id)
                    selected_gal_ind = np.argmin(distances_box)
                    sample_match[ii,:] = np.array(tng_mini_mr_id)[selected_gal_ind]
                else:
                    bubble_restrict = np.nonzero(distances_box<1.)[0]
                    selected_gal_ind = rng.choice(bubble_restrict)
                    sample_match[ii,:] = tng_mini_mr_id[selected_gal_ind]

                sample_match_labels[ii,:] = tng_mini_mr[selected_gal_ind,0]*m_lim,\
                        tng_mini_mr[selected_gal_ind,1]*r_lim,\
                        distances_box[selected_gal_ind]#, tng_mini_g_r[np.argmin(distances_box)]
            else:
                print('no bubble, No matches found near manga galaxy n', ii)
        else:
            print('no box, No matches found near manga galaxy n', ii)

    return sample_match, sample_match_labels

def repeated_subhalos_in_manga(snap, subhalo, distances, more_than=1):
    """ Returns 
        manga index: non prioritary galaxies in the match
        snap, subhalo: tng galaxies that have "more_than"-times
    """
    tng_snap = np.arange(99, 84, -1)
    rep_tng = []
    rep_manga_ind = []
    for snap_ii in tng_snap:
        in_snap = np.nonzero(snap==snap_ii)[0]
        rep_subhalos = np.array([item for item, count in Counter(subhalo[in_snap]).items() if count > more_than])
        #print(rep_subhalos)
        for jj in range(rep_subhalos.size):
#            print('holis')
            index_multi_subhalo = np.nonzero((rep_subhalos[jj]==subhalo[:])&(snap_ii==snap))[0]
            #print(index_multi_subhalo.shape)
            rep_manga_ind.extend(index_multi_subhalo[np.argsort(distances[index_multi_subhalo])[more_than:]])
            rep_tng.append(np.array([snap_ii, rep_subhalos[jj]]))

    return np.array(rep_tng), np.array(rep_manga_ind)


def get_tng_pack_rest(tng_pack, rep_tng):
    tng_snap, tng_snap_z, stellar_mass_tng, re_kpc_tng, tng_id, g_r_tng, star_part, gas_part, SFRT, SFR05, SFR1, log_re_arc_tng, cpetro_mass = tng_pack
    for ii in range(rep_tng.shape[0]):
        stellar_mass_tng[rep_tng[ii,0]] = np.delete(stellar_mass_tng[rep_tng[ii,0]],\
                                 np.nonzero(tng_id[rep_tng[ii,0]]==rep_tng[ii,1])[0])
        log_re_arc_tng[rep_tng[ii,0]] = np.delete(log_re_arc_tng[rep_tng[ii,0]], \
                                 np.nonzero(tng_id[rep_tng[ii,0]]==rep_tng[ii,1])[0])
        g_r_tng[rep_tng[ii,0]] = np.delete(g_r_tng[rep_tng[ii,0]], \
                                 np.nonzero(tng_id[rep_tng[ii,0]]==rep_tng[ii,1])[0])
        cpetro_mass[rep_tng[ii,0]] = np.delete(cpetro_mass[rep_tng[ii,0]],\
                                 np.nonzero(tng_id[rep_tng[ii,0]]==rep_tng[ii,1])[0])      
        tng_id[rep_tng[ii,0]] = np.delete(tng_id[rep_tng[ii,0]], \
                                 np.nonzero(tng_id[rep_tng[ii,0]]==rep_tng[ii,1])[0])


    return tng_snap, tng_snap_z, stellar_mass_tng, re_kpc_tng, tng_id, g_r_tng, star_part, gas_part, SFRT, SFR05, SFR1, log_re_arc_tng, cpetro_mass


allocated_manga_1 = []
sample_match, sample_match_labels, a, b, c = iterative_bubble_match_fixedZ(manga_mzr, \
                            tng_sim='tng50-1', max_step=max_step, m_lim=0.25, \
                            r_lim=0.25, random=False, seed=0, radius=1, \
                            r_increase=0., more_than=1) 
subh1 = [np.intersect1d(sample_match[sample_match[:,0]==ii,1],\
        sample_match[sample_match[:,0]==ii,1]) for ii in range(87,99)]
    
allocated_manga_1.append(np.nonzero(sample_match[:,0]!=0)[0].size)

for n_loop in range(10):
    manga_mzr_2 = manga_mzr[sample_match[:,0]==0]
    sample_match_2, sample_match_labels_2, a, b, c = iterative_bubble_match_fixedZ(manga_mzr_2, \
                                tng_sim='tng50-1', max_step=max_step, m_lim=0.25, \
                                r_lim=0.25, random=False, seed=0, radius=1, \
                                r_increase=0., more_than=1) 
    #subh2 = [np.intersect1d(sample_match_2[sample_match_2[:,0]==ii,1],\
    #        sample_match_2[sample_match_2[:,0]==ii,1]) for ii in range(87,99)]
    print('expectation ',sample_match_2.shape[0])
    print('result ', np.nonzero(sample_match_2[:,0]!=0)[0].size)
    sample_match_labels[sample_match[:,0]==0,:] = sample_match_labels_2
    sample_match[sample_match[:,0]==0,:] = sample_match_2
    
    allocated_manga_1.append(np.nonzero(sample_match[:,0]!=0)[0])



subh = {ii:sample_match[sample_match[:,0]==ii,1] for ii in range(87,99)}
tng_subpack = mk_tng_sub_pack(subh, get_tng_params())
mass_bins = np.linspace(stellar_mass_manga.min(), stellar_mass_manga.max(), 15)
r_bins = np.linspace(np.log10(re_kpc_manga).min(), np.log10(re_kpc_manga).max(), 13)

subsamp_sizes = [100, 1000, 5000]
rep_layer = np.ones_like(stellar_mass_manga)*len(subsamp_sizes)
for ii, subsamp_size in enumerate(subsamp_sizes):
    from_snap = assign_snap(tng_snap_z, redshift_manga)
    subsamp_grid = get_subsample_grid(stellar_mass_manga, np.log10(re_kpc_manga), mass_bins, r_bins, n_gal=subsamp_size)
    manga_index = pick_manga_subsamp(subsamp_grid, mass_bins, r_bins, stellar_mass_manga, re_kpc_manga)
    rep_layer[manga_index] = rep_layer[manga_index]-1
    #tng_subsamp, ind_in_snap = pick_tng_subsamp(subsamp_grid, mass_bins, r_bins, tng_pack=tng_subpack)

views = np.zeros_like(my_sample_div)
tng_snap = sample_match[:,0]
subhalo_id = sample_match[:,1]
re_kpc = np.zeros_like(my_sample_div, dtype=np.float32)
log_re_arc = np.zeros_like(my_sample_div, dtype=np.float32)
n_star = np.zeros_like(my_sample_div, dtype=np.int32)
n_gas = np.zeros_like(my_sample_div, dtype=np.int32)
color = np.zeros_like(my_sample_div, dtype=np.float32)
tng_ifu = np.zeros_like(my_sample_div)
for snap_ii in range(87,99):
    in_snap = np.nonzero(tng_snap==snap_ii)[0]
    rep_subhalos = [item for item, count in Counter(subhalo_id[in_snap]).items() if count > 1]
    #print(rep_subhalos)
    for subhal in rep_subhalos:
        repeated_single = np.nonzero(subhalo_id[in_snap]==subhal)[0]
        n_times = repeated_single.size
        for ii in range(n_times):
            views[in_snap[repeated_single[ii]]]=ii
    for ii in subhalo_id[in_snap]:
        re_kpc[(tng_snap==snap_ii) & (subhalo_id==ii)] = re_kpc_tng[snap_ii][tng_id[snap_ii]==ii]
        n_star[(tng_snap==snap_ii) & (subhalo_id==ii)] = star_part[snap_ii][tng_id[snap_ii]==ii]
        n_gas[(tng_snap==snap_ii) & (subhalo_id==ii)] = gas_part[snap_ii][tng_id[snap_ii]==ii]
        color[(tng_snap==snap_ii) & (subhalo_id==ii)] = g_r_tng[snap_ii][tng_id[snap_ii]==ii]
        log_re_arc[(tng_snap==snap_ii) & (subhalo_id==ii)] = log_re_arc_tng[snap_ii][tng_id[snap_ii]==ii]
        
samp = np.where(my_sample_div==0, 1, my_sample_div)
ifu_size_arcsec=10**log_re_arc*2*(samp+0.5)
for ii in range(len(ifu_size_arcsec)):
    tng_ifu[ii] = [19,37,61,91,127][np.argmin(np.abs(ifu_size_arcsec[ii]-np.array([12.5,17.5,22.5,27.5,32.5])))]
distance_selec = sample_match_labels[:,2]
stellar_mass = sample_match_labels[:,0]

#plt.scatter(stellar_mass_manga, np.log10(re_kpc_manga), c=1/(rep_layer+1), cmap='rainbow', s=20/(rep_layer+1), alpha=0.4)
#plt.show()

col_8 = fits.Column(name='sample', format='I', array=my_sample_div)
col_7 = fits.Column(name='manga_g-r', format='E', array=color_manga)
col_6 = fits.Column(name='manga_ifu_dsn', format='J', array=ifu_size_manga)
col_5 = fits.Column(name='manga_nsa_elpetro_mass', format='E', array=stellar_mass_manga)
col_4 = fits.Column(name='manga_nsa_z', format='E', array=redshift_manga)
col_3 = fits.Column(name='manga_nsa_elpetro_th50_r', format='E', array=re_arc_manga)
col_2 = fits.Column(name='manga_plateifu', format='A11', array=plateifu_manga_drp)
col_1 = fits.Column(name='mangaid', format='A11', array=manga_id)
col0 = fits.Column(name='snapshot', format='J', array=tng_snap)
col1 = fits.Column(name='subhalo_id', format='K', array=subhalo_id)
col2 = fits.Column(name='stellar_mass_30kpc', format='E', array=stellar_mass)
col3 = fits.Column(name='re_kpc', format='E', array=re_kpc)
col4 = fits.Column(name='ideal_ifu', format='J', array=tng_ifu)
col5 = fits.Column(name='view', format='I', array=views)
col6 = fits.Column(name='distances_selec', format='E', array=distance_selec)
col7 = fits.Column(name='tng_g-r', format='E', array=color)
col8 = fits.Column(name='n_star_part', format='K', array=n_star)
col9 = fits.Column(name='n_gas_cell', format='K', array=n_gas)
col10 = fits.Column(name='rep_subsamp', format='J', array=rep_layer)
coldefs = fits.ColDefs([col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10])
hdu = fits.BinTableHDU.from_columns(coldefs)
#hdu.writeto('/scratch/reginas/projects/real_SimCLR/'+tng_sim+'_'+method+'_'+fraction_sample_name+'_match_MaNGA_mass_size_z_seed'+str(seed)+'.fits', overwrite=1)
#hdu.writeto('TNG50_MaNGA_v3_3view.fits', overwrite=1)
hdu.writeto('/Users/reginasarmiento/Documents/machineGALearning/prog/TNG/setup_vera/TNG50_MaNGA_v13_Nview.fits', overwrite=1)



###########################################################################
#subsamp_size = stellar_mass_manga.size
snap=96
subsamp_size = 10000
mass_bins = np.linspace(stellar_mass_manga.min(), stellar_mass_manga.max(), 15)
r_bins = np.linspace(np.log10(re_kpc_manga).min(), np.log10(re_kpc_manga).max(), 13)
subsamp_grid = get_subsample_grid(stellar_mass_manga, np.log10(re_kpc_manga), mass_bins, r_bins, n_gal=subsamp_size)
tng_subsamp, ind_in_snap = pick_tng_subsamp(subsamp_grid, mass_bins, r_bins, snap=snap)

plt.figure(figsize=(10, 5))
plt.suptitle('subsample size: '+str(subsamp_size))
plt.subplot(1,2,1)
plt.title('N MaNGA galaxies per bin')
plt.xlabel(r'log(M/M${_\odot}$)')
plt.ylabel(r'log(R$_e$ [kpc])')
plt.imshow(np.swapaxes(subsamp_grid,1,0), origin='lower', extent=(mass_bins[0], mass_bins[-1], r_bins[0], r_bins[-1]))
plt.colorbar()

tng_snap, tng_snap_z, stellar_mass_tng, re_kpc_tng, \
tng_id, g_r_tng, star_part, gas_part, SFRT, SFR05, SFR1, log_re_arc_tng, cpetro_mass = get_tng_params()
re_kpc_tng[snap] = np.log10(re_kpc_tng[snap])


pipe_cat = fits.open('/Users/reginasarmiento/Documents/machineGALearning/prog/manga_SimCLR2/catalogues/SDSS15Pipe3D_clean_v3_0_1_MaStar2.fits')
plateifu_pipe = np.array([jj.strip() for jj in pipe_cat[1].data['plateifu']])
_,ind_pipe, ind_drp = np.intersect1d(plateifu_pipe, plateifu_manga_drp, return_indices=True)

plt.subplot(1,2,2)
plt.title('TNG subsample')
plt.xlabel(r'log(M/M${_\odot}$)')
plt.ylabel(r'log(R$_e$ [kpc])')
plt.plot(stellar_mass_tng[snap][ind_in_snap], re_kpc_tng[snap][ind_in_snap], '.', ls='', alpha=0.3)
for x_l in mass_bins:
    plt.axvline(x_l, ls='--', color='grey')
for y_l in r_bins:
    plt.plot([mass_bins[0], mass_bins[-1]], np.ones(2)*y_l, ls='--', color='grey')

#plt.show()
plt.savefig('plots/subsample_check_'+str(subsamp_size)+'.png', bbox_inches='tight')

########################

subsamp_size = 10000
from_snap = assign_snap(tng_snap_z, redshift_manga)
subsamp_grid = get_subsample_grid(stellar_mass_manga, np.log10(re_kpc_manga), mass_bins, r_bins, n_gal=subsamp_size, z_assigned=from_snap)
tng_subsamp, ind_in_snap = pick_tng_subsamp(subsamp_grid, mass_bins, r_bins)

tng_snap, tng_snap_z, stellar_mass_tng, re_kpc_tng, \
tng_id, g_r_tng, star_part, gas_part, SFRT, SFR05, SFR1, log_re_arc_tng, cpetro_mass = get_tng_params()

plt.figure(figsize=(10, 7))
plt.suptitle('subsample size: '+str(subsamp_size))
plt.subplot(2,2,1)
plt.title('N MaNGA galaxies per bin')
plt.xlabel(r'log(M/M${_\odot}$)')
plt.ylabel(r'log(R$_e$ [kpc])')
subsamp_grid_ = np.sum(subsamp_grid, axis=2)
plt.imshow(np.swapaxes(subsamp_grid_,1,0), origin='lower', extent=(mass_bins[0], mass_bins[-1], r_bins[0], r_bins[-1]), aspect='auto')
plt.colorbar()

plt.subplot(2,2,2)
plt.title('TNG subsample')
plt.suptitle('TNG subsample size: '+str(subsamp_size))
plt.xlabel(r'log(M/M${_\odot}$)')
plt.ylabel(r'log(R$_e$ [kpc])')
for ii, snap in enumerate(range(87,99)):
    plt.plot(stellar_mass_tng[snap][ind_in_snap[snap]], np.log10(re_kpc_tng[snap][ind_in_snap[snap]]), '.', ls='', alpha=0.4)
for x_l in mass_bins:
    plt.axvline(x_l, ls='--', color='grey')
for y_l in r_bins:
    plt.plot([mass_bins[0], mass_bins[-1]], np.ones(2)*y_l, ls='--', color='grey')

plt.subplot(2,2,3)
plt.xlabel(r'log(M/M${_\odot}$)')
plt.ylabel(r'log(SFR [M$_\odot$/yr])')
sfr_bins = np.linspace(-4, 2, 10)
subsamp_grid2 = get_subsample_grid(stellar_mass_manga[ind_drp], pipe_cat[1].data['log_SFR_Ha'][ind_pipe], mass_bins, sfr_bins, n_gal=subsamp_size, z_assigned=from_snap[ind_drp])
subsamp_grid2_ = np.sum(subsamp_grid2, axis=2)
plt.imshow(np.swapaxes(subsamp_grid2_,1,0), origin='lower', extent=(mass_bins[0], mass_bins[-1], sfr_bins[0], sfr_bins[-1]), aspect='auto')
plt.colorbar()

plt.subplot(2,2,4)
plt.suptitle('TNG subsample size: '+str(subsamp_size))
plt.xlabel(r'log(M/M${_\odot}$)')
plt.ylabel(r'log(SFR [M$_\odot$/yr])')
for ii, snap in enumerate(range(87,99)):
    plt.plot(stellar_mass_tng[snap][ind_in_snap[snap]], np.log10(SFRT[snap][ind_in_snap[snap]])+0.213, '.', ls='', alpha=0.4)
for x_l in mass_bins:
    plt.axvline(x_l, ls='--', color='grey')
for y_l in sfr_bins:
    plt.plot([mass_bins[0], mass_bins[-1]], np.ones(2)*y_l, ls='--', color='grey')

#plt.show()
plt.savefig('plots/subsample_check_multisnap_'+str(subsamp_size)+'.png', bbox_inches='tight')




SFRT = np.log(SFRT)+0.213

pipe_cat = fits.open('/Users/reginasarmiento/Documents/machineGALearning/prog/manga_SimCLR2/catalogues/SDSS15Pipe3D_clean_v3_0_1_MaStar2.fits')
plateifu_pipe = np.array([jj.strip() for jj in pipe_cat[1].data['plateifu']])
_,ind_pipe, ind_drp = np.intersect1d(plateifu_pipe, plateifu_manga_drp, return_indices=True)
_,ind_pipe1, ind_drp1 = np.intersect1d(plateifu_pipe, plateifu_manga_drp[my_sample_div==1], return_indices=True)
_,ind_pipe2, ind_drp2 = np.intersect1d(plateifu_pipe, plateifu_manga_drp[my_sample_div==2], return_indices=True)
_,ind_pipe3, ind_drp3 = np.intersect1d(plateifu_pipe, plateifu_manga_drp[my_sample_div==0], return_indices=True)

a = stellar_mass_manga[ind_drp]
b = pipe_cat[1].data['log_SFR_Ha'][ind_pipe]
#c = get_volume_limited_manga(esrweight[ind_drp], np.linspace(0,1,5))
#c = get_volume_limited_manga2(esrweight[ind_drp], np.linspace(8.5,11.5,10), a)

def recta(x,m,b):
    return m*x+b

def recta_fix(x,b):
    return 0.6884556791424935*x+b


data_path = '/Volumes/Personal/Datos/TNG50-1/'
snap = 96
SFR_file = h5py.File(data_path + 'star_formation_rates.hdf5', 'r')
SFR_id = SFR_file['Snapshot_'+str(snap)]['SubfindID']
SFR_200myr = np.log10(SFR_file['Snapshot_'+str(snap)]['SFR_MsunPerYrs_in_r5pkpc_200Myrs'])
_, ind_sfr, ind_mass = np.intersect1d(SFR_id, tng_id[snap], return_indices=True)

plt.xlabel('log(Mass)')
plt.ylabel('log(SFR Ha)')
#
plt.ylim(-6,3)
plt.xlim(8.5,11.5)
#plt.scatter(stellar_mass_tng[96], np.log10(SFR1[96]), alpha=0.2)
#plt.scatter(stellar_mass_tng[96], np.log10(SFR05[96]), alpha=0.2)
plt.scatter(stellar_mass_manga[ind_drp], pipe_cat[1].data['log_SFR_Ha'][ind_pipe], alpha=0.3, label='MaNGA-complete', s=3)
#plt.scatter(stellar_mass_manga[ind_drp3], pipe_cat[1].data['log_SFR_Ha'][ind_pipe3], alpha=0.3, label='MaNGA-color enhanced', s=3, color='g')
plt.scatter(stellar_mass_tng[96], np.log10(SFRT[96]), alpha=0.3, label='TNG snap 96', s=3)

plt.scatter(stellar_mass_tng[96][ind_mass], SFR_200myr[ind_sfr], s=3, alpha=0.3, label='SFR 200 Myrs')
#popt, pcov = curve_fit(recta, stellar_mass_tng[96][SFRT[96]>0], np.log10(SFRT[96][SFRT[96]>0]))
#plt.plot(stellar_mass_tng[96], popt[0]*stellar_mass_tng[96]+popt[1], 'r')
#
#popt_, pcov = curve_fit(recta_fix, a[popt[0]*a+popt[1]-1<b], b[popt[0]*a+popt[1]-1<b])
#plt.scatter(a[popt[0]*a+popt[1]-1<b], b[popt[0]*a+popt[1]-1<b], alpha=0.2)
#plt.plot(a[popt[0]*a+popt[1]-1<b], popt[0]*a[popt[0]*a+popt[1]-1<b]+popt_[0], 'g')

#popt, pcov = curve_fit(recta, stellar_mass_tng[96][SFR1[96]>0], np.log10(SFR1[96][SFR1[96]>0]))
#plt.plot(stellar_mass_tng[96], popt[0]*stellar_mass_tng[96]+popt[1])
plt.legend()

#plt.show()
plt.savefig('plots/SFR_Mass_comparison.png')

def get_volume_limited_manga(weights, bins):
    id_num = np.arange(weights.size)
    weighted_samp = []
    hist, bin_edges, bin_num = binned_statistic(weights, weights, bins=bins, statistic='count')
    n_bins = bin_edges.size-1
    for ii in range(n_bins):
        weighted_samp.extend(np.random.choice(id_num[bin_num==ii+1], size=int(np.round_((ii+1)*hist[-1]/n_bins)), replace=False))
    return weighted_samp


def get_volume_limited_manga2(weights, bins, mass, max_bin=10):
    id_num = np.arange(weights.size)
    weighted_samp = []
    weights = np.where(weights<0, 0, weights)
    hist, bin_edges, bin_num = binned_statistic(mass, mass, bins=bins, statistic='count')
    n_bins = bin_edges.size-1
    for ii in range(n_bins):
        weights_ii = weights[bin_num==ii+1] / weights[bin_num==ii+1].sum()
        weighted_samp.extend(np.random.choice(id_num[bin_num==ii+1], size=max_bin, replace=False, p=weights_ii))
    return weighted_samp


