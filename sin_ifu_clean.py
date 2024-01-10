import os
import gzip
import shutil
import warnings
import numpy as np
import multiprocessing as mp
import illustris_python as il
from os import remove
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from astropy.cosmology import Planck15 as cosmo


warnings.filterwarnings("ignore")

M_proton = 1.67262192e-24 # Proton mass [gr]
X_h = 0.76 # hydrogen mass fraction
k_B = 1.3807e-16 # Boltzmann constant in cgs [cm**2 g s**-2 K**-1]
gamma = 5/3 # adiabatic index
M_unit__E_unit =  1e10 #UnitMass/UnitEnergy in cgs, 
                       #UnitEnergy = UnitMass * UnitLength^2 / UnitTime^2 
                       #(UnitLength is 1 kpc, UnitTime is 1 Gyr), so their ratio is 10**10


def compress_gzip(file, compresslevel=6):
    with open(file, 'rb') as f_in:
        with gzip.open(f'{file}.gz', mode='wb', compresslevel=compresslevel) as f_comp:
            shutil.copyfileobj(f_in, f_comp)
    remove(file)

def periodicfix(x, boxsize=35000):
    if (np.min(x) < boxsize/10) & (np.max(x) > boxsize - boxsize/10):
        x = x + boxsize/2
        for j in range(3):
            for i in range(len(x)):
                if x[i,j] > boxsize:
                    x[i,j] = x[i,j]- boxsize
        x = x - boxsize/2
    return(x)

def periodicfix_cm(x, cm, boxsize=35000):    
    if (np.min(x) < boxsize/10) & (np.max(x) > boxsize - boxsize/10):
        cm = cm + boxsize/2
        for i in range(3):
            if cm[i] > boxsize:
                cm[i] = cm[i] - boxsize
        cm = cm - boxsize/2
    return(cm)

def mk_particle_files(subhalo_id, snap, basePath, ex=[1,0,0], FOV=19, overwrite=True, view=0, outdir=''):
    """ 
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
    """

    vx_ = [1,0,0]
    vy_ = [0,1,0]
    vz_ = [0,0,1]
    R3_ = np.array([[1,0,0],[0,1,0],[0,0,1]])
    Ev = np.transpose(np.array([vx_,vy_,vz_]))
    snapz_dict = {99:0., 98:0.012, 97:0.023, 96:0.035, 95:0.048, 94:0.06, 
                  93:0.073, 92:0.086, 91:0.1, 90:0.11, 89:0.13, 88:0.14, 87:0.15,
                  86:0.17, 85:0.18}
    snapa_dict = {99:1., 98:0.9885, 97:0.9771, 96:0.9657, 95:0.9545, 94:0.9433,
                  93:0.9322, 92:0.9212, 91:0.9091, 90:0.8993, 89:0.8885, 88:0.8778, 
                  87:0.8671, 86:0.8564, 85:0.8459}
    snapt_dict = {87:0.8674917093619414, 88:0.8757516765159388, 89:0.8882893272206793, 
                  90:0.9010064725120812, 91:0.9095855560793044, 92:0.9226075876597282, 
                  93:0.9313923498515226, 94:0.9447265771954694, 95:0.9537219490392906, 
                  96:0.9673758568617342, 97:0.9765868876036025, 98:0.99056814006128}
    h = cosmo.H0.value * 1e-2
    fields_s = ['GFM_StellarFormationTime', 'Masses', 'GFM_Metallicity', \
                'Coordinates', 'Velocities']
    fields_g = ['StarFormationRate', 'Coordinates', 'Velocities', 'Density', \
                'GFM_Metallicity', 'Masses', 'InternalEnergy', 'ElectronAbundance']

    halo_all = il.groupcat.loadSubhalos(basePath, snap, fields=['SubhaloGrNr'])
    halo_id = halo_all[subhalo_id]
    redshift = snapz_dict[snap]
    a = snapa_dict[snap]
    box_side_half_kpc = FOV * cosmo.kpc_proper_per_arcmin(redshift).value/60.
    dist = cosmo.comoving_distance(redshift).value * 1e3

    Halo_stars = il.snapshot.loadHalo(basePath, snap, halo_id, partType=4, fields=fields_s)
    #Halo_stars = il.subhalo.subfind(basePath, snap, subhalo, partType=4, fields=fields_s)
    Halo_stars_age_orig = Halo_stars['GFM_StellarFormationTime'][:]
    Halo_stars_mass = Halo_stars['Masses'][Halo_stars_age_orig>=0] * 1e10 / h
    Halo_stars_coord = periodicfix(Halo_stars['Coordinates'][Halo_stars_age_orig>=0, :]) * a / h
    Halo_stars_vel = Halo_stars['Velocities'][Halo_stars_age_orig>=0, :] * np.sqrt(a)
    vx, vy, vz = Halo_stars_vel[:,0], Halo_stars_vel[:,1], Halo_stars_vel[:,2]
    Halo_stars_met = Halo_stars['GFM_Metallicity'][Halo_stars_age_orig>=0]
    Halo_stars_age = (1-Halo_stars_age_orig[Halo_stars_age_orig>=0])*cosmo.age(1/snapt_dict[snap]-1).value
        
    Halo_gas = il.snapshot.loadHalo(basePath, snap, halo_id, partType=0, fields=fields_g)
    Halo_gas_mass = Halo_gas['Masses'][:] * 1e10 / h
    Halo_gas_coord = periodicfix(Halo_gas['Coordinates'][:, :]) * a / h
    Halo_gas_vel = Halo_gas['Velocities'][:, :] * np.sqrt(a)
    vx_g, vy_g, vz_g = Halo_gas_vel[:,0], Halo_gas_vel[:,1], Halo_gas_vel[:,2]
    Halo_gas_met = Halo_gas['GFM_Metallicity'][:]
    Halo_gas_dens = Halo_gas['Density'][:] * 1e10 * h**2 / a**3
    Halo_gas_vol = Halo_gas_mass / Halo_gas_dens
    Halo_gas_vol = (Halo_gas_vol/(4.0*np.pi/3.0))**(1./3.0)*(3.08567758e19*100)
    Halo_gas_dens = Halo_gas_dens/(3.08567758e19*100)**3.0*1.9891e30/1.67262178e-27
    Halo_sfri = Halo_gas['StarFormationRate'][:]
    TE = Halo_gas['InternalEnergy'][:]
    EA = Halo_gas['ElectronAbundance'][:]
    MMW = M_proton * 4. / (1. + 3.*X_h + 4.*X_h*EA)
    temp_g = (gamma-1.) * TE * MMW * M_unit__E_unit / k_B
    Halo_Av_g = Halo_gas_met*(3.0*1.67262e-24*np.pi*Halo_gas_dens*Halo_gas_vol)/(4.0*np.log(10.)*3.0*5494e-8)
    Halo_Av_g = np.where(Halo_gas_met > (10.0**(-0.59)*0.0127), \
                            Halo_Av_g/(10.0**(2.21-1.0)), \
                            Halo_Av_g/(10.0**(2.21-1.0)/(Halo_gas_met/0.0127)**(3.1-1.0)))
        
    cm = periodicfix_cm(Halo_stars['Coordinates'][Halo_stars_age_orig>=0, :], il.groupcat.loadSingle(basePath,snap,subhaloID=subhalo_id)["SubhaloPos"]) * a / h
    xyz = (Halo_stars_coord - cm)
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    xyz_g = (Halo_gas_coord - cm)
    x_g = xyz_g[:,0]
    y_g = xyz_g[:,1]
    z_g = xyz_g[:,2] 
    if not overwrite:
        if os.path.exists(outdir + '/snap_'+str(snap)+'_shalo'+str(subhalo_id)+'_0_stars.dat') &\
            os.path.exists(outdir + '/snap_'+str(snap)+'shalo'+str(subhalo_id)+'_0_gas.dat'):
            print('Output files already exist and overwrite option is off.')
    else:
        try:
            print('For snap-'+str(snap)+' running subhalo '+str(subhalo_id)+' in halo '+str(halo_id))
            obs = np.dot(np.dot(Ev,R3_),ex)*dist
            xyz0 = obs
            x0 = xyz0[0]
            y0 = xyz0[1]
            z0 = xyz0[2]
            Rc = dist #np.sqrt(np.sum(xyz0**2))
            red_0 = reds_cos(Rc/1e3)
            Ra = Rc #/(1+red_0)
            A1 = np.arctan2(y0,z0)
            A2 = np.arcsin(x0/Ra)
            R1 = np.array([[1,0,0],[0,np.cos(A1),-np.sin(A1)],[0,np.sin(A1),np.cos(A1)]])
            R2 = np.array([[np.cos(A2),0,-np.sin(A2)],[0,1,0],[np.sin(A2),0,np.cos(A2)]])
            R3 = np.array([[np.cos(A2),-np.sin(A2),0],[np.sin(A2),np.cos(A2),0],[0,0,1]])
            Ve = np.array([x,y,z])
            Vf = np.dot(np.dot(R2,R1),Ve)
            stars_in_brick = np.nonzero((np.abs(Vf[0])<box_side_half_kpc) & (np.abs(Vf[1])<box_side_half_kpc))[0]
            stars_in_tube = stars_in_brick[(Vf[0][stars_in_brick]**2+Vf[1][stars_in_brick]**2)<box_side_half_kpc**2]
            mass_b = Halo_stars_mass[stars_in_tube]
            meta_b = Halo_stars_met[stars_in_tube]
            age_s_b = Halo_stars_age[stars_in_tube]
            x_b = Vf[0][stars_in_tube]
            y_b = Vf[1][stars_in_tube]
            z_b = Vf[2][stars_in_tube]
            Ve = np.array([vx,vy,vz])
            Vf = np.dot(np.dot(R2,R1),Ve)
            vx_b = Vf[0][stars_in_tube]
            vy_b = Vf[1][stars_in_tube]
            vz_b = Vf[2][stars_in_tube]
            Ve = np.array([x_g,y_g,z_g])
            Vf = np.dot(np.dot(R2,R1),Ve)
            gas_in_brick = np.nonzero((np.abs(Vf[0])<box_side_half_kpc) & (np.abs(Vf[1])<box_side_half_kpc))[0]
            gas_in_tube = gas_in_brick[(Vf[0][gas_in_brick]**2+Vf[1][gas_in_brick]**2)<box_side_half_kpc**2]
            mass_g_b = Halo_gas_mass[gas_in_tube]
            meta_g_b = Halo_gas_met[gas_in_tube]
            temp_g_b = temp_g[gas_in_tube]
            volm_b = Halo_gas_vol[gas_in_tube]
            dens_b = Halo_gas_dens[gas_in_tube]
            sfri_b = Halo_sfri[gas_in_tube]
            Av_g_b = Halo_Av_g[gas_in_tube]
            x_g_b = Vf[0][gas_in_tube]
            y_g_b = Vf[1][gas_in_tube]
            z_g_b = Vf[2][gas_in_tube]
            Ve = np.array([vx_g,vy_g,vz_g])
            Vf = np.dot(np.dot(R2,R1),Ve)
            vx_g_b = Vf[0][gas_in_tube]
            vy_g_b = Vf[1][gas_in_tube]
            vz_g_b = Vf[2][gas_in_tube]        
                        
            total = np.array(np.column_stack((x_b,y_b,z_b,vx_b,vy_b,vz_b,age_s_b,meta_b,mass_b)), dtype=np.float32)
            np.savetxt(outdir + '/snap'+str(snap)+'_shalo'+str(subhalo_id)+'_'+str(view)+'_stars.dat', total, delimiter = ' ')
                    
            totalg = np.array(np.column_stack((x_g_b,y_g_b,z_g_b,vx_g_b,vy_g_b,vz_g_b,meta_g_b,volm_b,dens_b,sfri_b,temp_g_b,Av_g_b,mass_g_b)), dtype=np.float32)
            np.savetxt(outdir + '/snap'+str(snap)+'_shalo'+str(subhalo_id)+'_'+str(view)+'_gas.dat', totalg, delimiter = ' ')

        except OSError:
            print('Disk quota exceeded')
        
            
def get_F0(RSS_127, R_eff_kpc, kpc_per_arcsec, wl, n_radii=2.):
    """
    Calculates the average flux in the wavelngth range 5415-6989 Ang
    at a n_radii*R_eff_kpc distance from the centre of the IFU.
    Calculated of the fiber spectra.

    Arguments:
    ----------
    RSS_127: Raw stacked spectrum for the largest IFU. ((381,w)-sized float array)
    R_eff_kpc: Effective radius in kpc. (float)
    kpc_per_arcsec: kpc per arcsec scale. (float)
    wl: wavelength array. (w-sized float array)
    n_radii: multiplicative factor to determine at what distrance from the centre F0 
             should be calculated. (float)

    Returns:
    -------
    F0: average flux in the wavelngth range 5415-6989 Ang at a n_radii*R_eff_kpc 
        distance from the centre of the IFU in the same units as RSS_127.
    """

    fib_index= {#2: np.array([5,7,18,19,30,31]),
                3: np.array([4,8,17,20,29,32,41,42,43,52,53,54]),
                4: np.array([3,9,16,21,28,33,40,44,51,55,62,63,64,65,72,73,74,75]),
                5: np.array([2,10,15,22,27,34,39,45,50,56,61,66,71,76,81,82,83,84,85,90,91,92,93,94]),
                6: np.array([1,11,14,23,26,35,38,46,49,57,60,67,70,77,80,86,89,95,98,99,100,101,102,103,106,107,108,109,110,111]),
                7: np.array([0,12,13,24,25,36,37,47,48,58,59,68,69,78,79,87,88,96,97,104,105]+list(np.arange(112,127)))}
    manga_ifu_designs = np.arange(3,8)
    manga_ifu_diameters_arcsec = np.array([12.5,17.5,22.5,27.5,32.5])
    R_arcsec = n_radii * R_eff_kpc / kpc_per_arcsec
    n_fib_arg = np.argmin(np.abs(2*R_arcsec-manga_ifu_diameters_arcsec))
    n_fib = manga_ifu_designs[n_fib_arg]
    print(n_fib)
    fib_ind_donut = np.concatenate([fib_index[n_fib], fib_index[n_fib]+127, fib_index[n_fib]+254])
    min_r = np.searchsorted(wl, 5415.)
    max_r = np.searchsorted(wl, 6989.)
    RSS_127_ = np.where(RSS_127==0, np.nan, RSS_127)
    F0 = np.nanmean(RSS_127_[min_r:max_r, fib_ind_donut])
    return F0

def noise_sig(wave):
    """

    Source: https://github.com/hjibarram/mock_ifu
    """

    w1=3900
    w2=10100
    dw2=100
    dw1=100
    s=1./(1.+np.exp(-(wave-w1)/dw1))/(1.+np.exp((wave-w2)/dw2))+1.0
    return s

def get_noise(wl, F0, nspec, SN=5., realization=True):
    """

    """
    s = noise_sig(wl)
    SN = SN/1.7361 #factor due to recombination enhancement of SN
    if realization:
        return 2. * F0 * 1.5 / SN * np.random.randn(nspec, wl.size) / np.tile(s, (nspec,1))
    else:
        return 2. * F0 * 1.5 / SN / np.tile(s, (nspec,1))

def arg_ages(age_s):
    """

    Source: https://github.com/hjibarram/mock_ifu
    """
    age_a=[]
    nssp=len(age_s)
    ban=0
    age_t=sorted(age_s)
    age_a.extend([age_t[0]])
    for i in range(1, nssp):
        if age_t[i-1] > age_t[i]:
            ban =1
        if age_t[i-1] < age_t[i] and ban == 0:
            age_a.extend([age_t[i]])
    return age_a[::-1]

def num_ages(age_s):
    """
    Source: https://github.com/hjibarram/mock_ifu
    """
    age_a=[]
    nssp=len(age_s)
    ban=0
    age_t=sorted(age_s)
    age_a.extend([age_t[0]])
    for i in range(1, nssp):
        if age_t[i-1] > age_t[i]:
            ban =1
        if age_t[i-1] < age_t[i] and ban == 0:
            age_a.extend([age_t[i]])
    n_age=len(age_a)
    return n_age

def shifts(spect_s, wave, dlam):
    """ 
    Doppler shifts a given spectrum.

    Arguments:
    ----------
    spect_s: spectrum to shift. (w-sized float array)
    wave: wavelengths associeted to the spec. (w-sized float array)
    dlam: shift in wavelength. (float)

    Returns:
    -------
    spect_f: shifted spectrum. (w-sized float array)

    Source: https://github.com/hjibarram/mock_ifu
    """

    wave_f = wave * dlam
    spect_f = interp1d(wave_f, spect_s, bounds_error=False, fill_value=0.)(wave)
    nt = np.where(spect_f == 0)[0]
    return  spect_f 

def cube_conv_lsf(wavelengths, spec, resolution, delta_wl=100):
    """
    Calculates the convolution between a spectrum and a light spread
    function that is wavelength dependent.
    
    Arguments
    ----------
    wavelengths: float array
        Set of wavelengths associated with the input spectrum 
        []
    
    spec: float array 
        The spectrum.
        
    resolution: 
        'MaNGA', manga median resolution
        np.single, fixed resolution per wl 
        float array, same shape as wavelengths
        Defines the light spread function resolution as FWHM.
        [-]
        
    Returns
    -------  
    convolution: float array
        The convolved spectrum.
    """

    if spec.size!=wavelengths.size:
        print('Spec size ',spec.size,' does not match wavelengths size ', wavelengths.size)

    if resolution=='MaNGA':
        manga_wave = np.load('libs/MaNGA_wl_LIN.npy')
        median_resolution = np.load('libs/MaNGA_median_spec_res_LIN.npy')
        resolution = np.ones_like(wavelengths) * 2000. # 2000 is aprox the mean resolution, will be used for edges
        int_func_res = interp1d(manga_wave, median_resolution)
        min_index = np.searchsorted(wavelengths, manga_wave[0], side='left')
        max_index = np.searchsorted(wavelengths, manga_wave[-1], side='right')
        int_res = int_func_res(wavelengths[min_index:max_index-1])
        resolution[min_index:max_index-1] = int_res
    elif np.array(resolution).size==1:
        resolution = resolution * np.ones_like(wavelengths)

    convolution = np.zeros_like(spec)
    for ii in range(len(wavelengths)-2):
        initial_ind = ii - np.min([ii, delta_wl])
        final_ind = np.min([ii+delta_wl, len(wavelengths)-1])
        lsf = line_spread_function(wavelengths[initial_ind:final_ind], resolution[ii], wavelengths[ii])
        convolution[ii] = simpson_r(lsf * spec[initial_ind:final_ind], wavelengths[initial_ind:final_ind], 0, wavelengths[initial_ind:final_ind].size-2) 
    return convolution

def line_spread_function(x, res, x0):
    """
    Calculates the LSF as a gaussian at xo for a range of x,
    given the resolution.

    Arguments
    ----------
    x: (float array) wavelengths
    res: (float) resolution as FWHM at x0
    x0: (float) central wavelength

    Returns
    -------
    lsf: (float array) line spread function at each x contributed at x0
    """ 

    sigma = x0 / res # if res is FWHM add this * 2. * np.sqrt(2.*np.log(2.)))
    lsf = np.exp(-(x-x0)**2 / (2.*sigma**2)) #/ (sigma * np.sqrt(np.pi*2.))
    lsf = lsf / simpson_r(lsf, x, 0, x.size-2)
    return lsf

def simpson_r(f,x,i1,i2,typ=0):
    """ 
    Integral calculation with Simpson method.

    Arguments:
    ----------
    f: function to integrate. (n-sized float array)
    x: domain of f. (n-sized float array)
    i1: initial integration index. (integer)
    i2: final integration index. (integer)

    Returns:
    -------
    Area under the function f with Simpson's numerical approximation.

    Source: https://github.com/hjibarram/mock_ifu
    """

    n = (i2-i1) * 1.0
    if n % 2:
        n = n + 1.0
        i2 = i2 + 1
    b = x[i2]
    a = x[i1]
    h = (b-a) / n
    s = f[i1] + f[i2]
    n = np.int32(n)
    dx = b - a
    for ii in range(1, n, 2):
        s += 4 * f[i1+ii]
    for i in range(2, n-1, 2):
        s += 2 * f[i1+ii]
    if typ == 0:
        return s * h / 3.0
    if typ == 1:
        return s * h / 3.0 / dx

def A_l(Rv,l):
    """

    Source: https://github.com/hjibarram/mock_ifu
    based in Cardelli+()
    """
    l = l/10000.; #Amstrongs to Microns
    x = 1.0/l
    Arat = np.zeros(len(x))
    for i in range(0, len(x)):
        if x[i] > 1.1 and x[i] <= 3.3:
            y = (x[i]-1.82)
            ax = 1+0.17699*y-0.50447*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7
            bx = 1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.09002*y**7
        if x[i] <= 1.1 and x[i] > 0.3:
            ax = 0.574*x[i]**1.61
            bx = -0.527*x[i]**1.61
        if x[i] > 3.3 and x[i] <= 8.0:
            if x[i] > 5.9 and x[i] <= 8.0:
                Fa = -0.04473*(x[i]-5.9)**2.0-0.009779*(x[i]-5.9)**3.0
                Fb = 0.2130*(x[i]-5.9)**2.0+0.1207*(x[i]-5.9)**3.0
            else:
                Fa = 0.0
                Fb = 0.0
            ax = 1.752-0.316*x[i]-0.104/((x[i]-4.67)**2.0+0.341)+Fa
            bx = -3.090+1.825*x[i]+1.206/((x[i]-4.62)**2.0+0.263)+Fb
        if x[i] > 8.0:
            ax = -1.073-0.628*(x[i]-8.0)+0.137*(x[i]-8.0)**2.0-0.070*(x[i]-8.0)**3.0
            bx = 13.670+4.257*(x[i]-8.0)-0.420*(x[i]-8.0)**2.0+0.374*(x[i]-8.0)**3.0
        val = ax+bx/Rv
        if val < 0:
            val = 0
        Arat[i] = val

    return Arat

def reds_cos(dis):
    """

    Source: https://github.com/hjibarram/mock_ifu
    """
    red = np.arange(0, 3, .01)
    dist = cosmo.lookback_time(red).value * 1e3 * 0.307
    #dist = cosmo.comoving_distance(red).value
    z = interp1d(dist, red, kind='linear', bounds_error=False)(dis)
    return z

def val_definition_l(val, val_ssp):
    """

    Source: https://github.com/hjibarram/mock_ifu
    """
    val_l = np.array(val)
    n_val = len(val_ssp)
    ind = []
    for i in range(0, n_val):
        if i < n_val-1:
            dval = (val_ssp[i+1]+val_ssp[i])/2.
        else:
            dval = val_ssp[i]+1.0
        if i == 0:
            val1 = 0
        else:
            val1 = val2
        val2 = dval
        nt = np.where((val_l > val1) & (val_l <= val2))
        ind.extend([nt[0]])

    return ind

def associate_ssp(age_s, met_s, age, met): 
    """ 
    Given the n-stellar particles' ages and metallicities, associate to the 
    template's m-ages and metallicities.

    Arguments:
    ----------
    age_s: stellar particles ages in Gyrs. (n-sized float array)
    met_s: stellar particles metallicities in Z/H. (n-sized float array)
    age: template ages in Gyrs. (m-sized float array)
    met: template metallicities in Z/H. (m-sized float array)

    Returns:
    -------
    ind_ssp: indices in the SSP template per particle. (n-sized integer array
             with values in [0; m]) 

    Source: https://github.com/hjibarram/mock_ifu with minor changes
    """

    age_a = []
    met_a = []
    nssp = len(met_s)
    ban = 0
    ban1 = 0
    age_t = sorted(age_s)
    met_t = met_s
    age_a.extend([age_t[0]])
    met_a.extend([met_t[0]])
    ind_ssp = np.zeros((len(age)), dtype=np.int64)
    #ind_ssp[:] = np.nan
    for ii in range(1, nssp):
        if age_t[ii-1] > age_t[ii]:
            ban =1
        if age_t[ii-1] < age_t[ii] and ban == 0:
            age_a.extend([age_t[ii]])
        if met_t[ii-1] > met_t[ii]:
            ban1 =1
        if met_t[ii-1] < met_t[ii] and ban1 == 0:
            met_a.extend([met_t[ii]])

    ind_age = val_definition_l(age,age_a)
    n_age = len(age_a)
    n_met = len(met_a)
    for ii in range(0, n_age):
        if len(ind_age[ii]) > 0:
            ind_met = val_definition_l(met[ind_age[ii]], met_a)
            for jj in range(0, n_met):
                if len(ind_met[jj]) > 0:
                    nt = np.where((age_s == age_a[ii]) & (met_s == met_a[jj]))[0]
                    ind_ssp[ind_age[ii][ind_met[jj]]] = nt[0]

    return ind_ssp

def associate_gas(phot_s, met_s, den_s, tem_s, phot, met, den, tem):
    """ 
    Given the n-gas cells' phot-io, metallicities, densities and temperatures,
    associate to the template's m-phot-io, metallicities, densities and temperatures.

    Arguments:
    ----------
    phot: gas cells photo-ionization factor. (n-sized float array)
    met: gas cells metallicities in Z/H. (n-sized float array)
    den: gas cells densities in 
    tem: gas cells temperatures in K. (n-sized float array)
    phot_s: template photo-ionization factor. (m-sized float array)
    met_s: template metallicities in Z/H. (m-sized float array)
    den_s: template desities
    tem_s: template temperatures in K. (m-sized float array)

    Returns:
    -------
    ind_gas: indices in the gas template per gas cell. (n-sized integer array
             with values in [0; m]) 

    Source: https://github.com/hjibarram/mock_ifu with minor changes
    """

    phot_s = np.array(phot_s)
    met_s= np.array(met_s)
    den_s= np.array(den_s)
    tem_s= np.array(tem_s)
    met_s = met_s*0.02#127
    phot_a = []
    met_a = []
    den_a = []
    tem_a = []
    nssp = len(met_s)
    ban = 0
    ban1 = 0
    ban2 = 0
    ban3 = 0
    phot_t = sorted(phot_s)
    met_t = sorted(met_s)
    den_t = sorted(den_s)
    tem_t = sorted(tem_s)
    phot_a.extend([phot_t[0]])
    met_a.extend([met_t[0]])
    den_a.extend([den_t[0]])
    tem_a.extend([tem_t[0]])
    ind_gas = np.zeros((len(met)), dtype = np.int64)
    ind_gas[:] = -100
    for i in range(1, nssp):
        if phot_t[i-1] > phot_t[i]:
            ban = 1
        if phot_t[i-1] < phot_t[i] and ban == 0:
            phot_a.extend([phot_t[i]])
        if met_t[i-1] > met_t[i]:
                ban1 = 1
        if met_t[i-1] < met_t[i] and ban1 == 0:
            met_a.extend([met_t[i]])
        if den_t[i-1] > den_t[i]:
                ban2 = 1
        if den_t[i-1] < den_t[i] and ban2 == 0:
            den_a.extend([den_t[i]])
        if tem_t[i-1] > tem_t[i]:
                ban3 = 1
        if tem_t[i-1] < tem_t[i] and ban3 == 0:
            tem_a.extend([tem_t[i]])
    n_phot = len(phot_a)
    n_met = len(met_a)
    n_den = len(den_a)
    n_tem = len(tem_a)
    ind_phot = val_definition_l(phot, phot_a)
    for i in range(0, n_phot):
        if len(ind_phot[i]) > 0:
            ind_met = val_definition_l(met[ind_phot[i]],met_a)
            for j in range(0, n_met):
                if len(ind_met[j]) > 0:
                    ind_den = val_definition_l(den[ind_phot[i][ind_met[j]]],den_a)
                    for k in range(0, n_den):
                        if len(ind_den[k]) > 0:
                            ind_tem = val_definition_l(tem[ind_phot[i][ind_met[j][ind_den[k]]]],tem_a)
                            for h in range(0, n_tem):
                                if len(ind_tem[h]) >  0:               
                                    nt = np.nonzero((phot_s == phot_a[i]) & (met_s == met_a[j]) & (den_s == den_a[k]) & (tem_s == tem_a[h]))[0]
                                    ind_gas[ind_phot[i][ind_met[j][ind_den[k][ind_tem[h]]]]]=nt[0]

    return ind_gas

def associate_pho(ssp_temp, wave, age_s, met_s, ml, mass, met, \
                  Rs=1, n_h=1, wl_i=3540.0, wl_f=5717.0):
    """ 
    Given the n-gas cells' ages, metallicities and masses, this function 
    associates to the mass-weighted SSP template's m-ages and metallicities  
    to obtain a spectrum produced by the new-born stars. Calculates the 
    photo-ionization factor from the energy emitted in the blue wavelength 
    range.

    Arguments:
    ----------
    ssp_temp: SSP template flux. ((m,w)-sized float array)
    wave: wavelength associated with the SSP spectra in Ang. (w-sized float array)
    age_s: template ages in Gyrs. (m-sized float array)
    met_s: template metallicities in Z/H. (m-sized float array)
    ml: mass-luminosity weighting factor. (m-sized float array)
    met: star forming gas cells metallicities in Z/H. (n-sized float array)
    mass: star forming gas cells masses in solar mass. (n-sized float array)
    Rs: Stromgren radius (not used).
    n_h: (not used)
    wl_i (=3540 Ang): initial integration wavelength. (float)
    wl_f (=5717 Ang): final integration wavelength. (float)

    Returns:
    -------
    Photo-ionizing factor in log-scale (n-sized float array)

    Source: https://github.com/hjibarram/mock_ifu with minor changes
    """

    vel_light = 299792458.0
    h_p = 6.62607004e-34
    age = np.ones(len(met))*2.5e6/1e9
    age_a = []
    met_a = []
    nssp = len(met_s)
    ban = 0
    ban1 = 0
    age_t = sorted(age_s)
    met_t = met_s
    age_a.extend([age_t[0]])
    met_a.extend([met_t[0]])
    ind_ssp = np.zeros((len(age)), dtype=np.int64)
    photo = np.zeros(len(age))
    ind_ssp[:] = -100
    for ii in range(1, nssp):
        if age_t[ii-1] > age_t[ii]:
            ban = 1
        if age_t[ii-1] < age_t[ii] and ban == 0:
            age_a.extend([age_t[ii]])
        if met_t[ii-1] > met_t[ii]:
                ban1 = 1
        if met_t[ii-1] < met_t[ii] and ban1 == 0:
            met_a.extend([met_t[ii]])
    ind_age = val_definition_l(age,age_a)
    n_age = len(age_a)
    n_met = len(met_a)
    for ii in range(0, n_age):
        if len(ind_age[ii]) > 0:
            ind_met = val_definition_l(met[ind_age[ii]],met_a)
            for jj in range(0, n_met):
                if len(ind_met[jj]) > 0:
                    nt = np.where((age_s == age_a[ii]) & (met_s == met_a[jj]))[0]
                    ind_ssp[ind_age[ii][ind_met[jj]]]=nt[0]
                    flux_0 = ssp_temp[nt[0],:]/ml[nt[0]]/(h_p*vel_light/wave/1e-10/1e-7)*3.846e33
                    j1 = np.searchsorted(wave, wl_i, side='left')#0#int(0.47*n_c)
                    j2 = np.searchsorted(wave, wl_f, side='left')#int(0.63*len(wave))
                    norm = simpson_r(flux_0, wave, j1, j2)
                    photo[ind_age[ii][ind_met[jj]]]=norm*mass[ind_age[ii][ind_met[jj]]]+1#/(4.0*np.pi*Rs[ind_age[i][ind_met[j]]]**2.0*n_h[ind_age[i][ind_met[j]]])+1
    
    return np.log10(photo)

def associate_mets_ages(age_s, met_s, age, met, mass):
    """

    Source: https://github.com/hjibarram/mock_ifu with minor changes
    """
    age_a = []
    met_a = []
    nssp = len(met_s)
    ban = 0
    ban1 = 0
    age_t = sorted(age_s)
    met_t = met_s
    age_a.extend([age_t[0]])
    met_a.extend([met_t[0]])
    for i in range(1, nssp):
        if age_t[i-1] > age_t[i]:
            ban = 1
        if age_t[i-1] < age_t[i] and ban == 0:
            age_a.extend([age_t[i]])
        if met_t[i-1] > met_t[i]:
            ban1 = 1
        if met_t[i-1] < met_t[i] and ban1 == 0:
            met_a.extend([met_t[i]])

    ind_age = val_definition_l(age, age_a)
    n_age = len(age_a)
    n_met = len(met_a)
    mass_f = np.zeros([n_age, n_met])
    for i in range(0, n_age):
        if len(ind_age[i]) > 0:
            ind_met = val_definition_l(met[ind_age[i]], met_a)
            for j in range(0, n_met):
                if len(ind_met[j]) > 0:
                    mass_f[n_age-1-i, j] = np.sum(mass[ind_age[i][ind_met[j]]])  

    return mass_f

def associate_mets_ages_flux(age_s, met_s, ml_s, age, met, mass):
    """
    
    Source: https://github.com/hjibarram/mock_ifu with minor changes
    """
    age_a=[]
    met_a=[]
    nssp=len(met_s)
    ban=0
    ban1=0
    age_t=sorted(age_s)
    met_t=met_s
    age_a.extend([age_t[0]])
    met_a.extend([met_t[0]])
    for i in range(1, nssp):
        if age_t[i-1] > age_t[i]:
            ban =1
        if age_t[i-1] < age_t[i] and ban == 0:
            age_a.extend([age_t[i]])
        if met_t[i-1] > met_t[i]:
            ban1 =1
        if met_t[i-1] < met_t[i] and ban1 == 0:
            met_a.extend([met_t[i]])
    ind_age=val_definition_l(age,age_a)
    n_age=len(age_a)
    n_met=len(met_a)
    light_f=np.zeros([n_age,n_met])
    mass_f = np.zeros([n_age, n_met])
    for i in range(0,n_age):
        if len(ind_age[i]) > 0:
            ind_met = val_definition_l(met[ind_age[i]], met_a)
            for j in range(0, n_met):
                if len(ind_met[j]) > 0:
                    mass_f[n_age-1-i, j] = np.sum(mass[ind_age[i][ind_met[j]]])
                    nt=np.where((age_s == age_a[i]) & (met_s == met_a[j]))[0]
                    light_f[n_age-1-i, j]=np.sum(mass[ind_age[i][ind_met[j]]]/ml_s[nt[0]]) 

    return light_f, mass_f

def associate_ages(age_s,age,mass):
    age_a=[]
    nssp=len(age_s)
    ban=0
    age_t=sorted(age_s)
    age_a.extend([age_t[0]])
    for i in range(1, nssp):
        if age_t[i-1] > age_t[i]:
            ban =1
        if age_t[i-1] < age_t[i] and ban == 0:
            age_a.extend([age_t[i]])
    ind_age=val_definition_l(age,age_a)
    n_age=len(age_a)
    mass_f=np.zeros(n_age)
    for i in range(0, n_age):
        if len(ind_age[i]) > 0:
            mass_f[i]=np.sum(mass[ind_age[i]])
    return mass_f[::-1]

def ssp_extract(template):
    """ 
    Extract from a SSP template the fluxes, corresponding ages and 
    metallicities, wavelength range and mass-luminosity weighting 
    factors.

    Arguments:
    ----------
    template : SSP template name. (string)

    Returns:
    -------
    pdl_flux_c_ini: fluxes. ((m,w)-sized float array)
    wave_c: wavelengths. (w-sized float array)
    age_mod: template ages in Gyrs. (m-sized float array)
    met_mod: template metallicities in Z/H. (m-sized float array)
    Ha: H-alpha mass-luminosity weighting factors per spectrum.
        (m-sized float array)
    crval: Axis1 value from template header. (float)
    cdelt: Axis1 value from template header. (float)
    crpix: Axis1 value from template header. (float)

    Source: https://github.com/hjibarram/mock_ifu with minor changes
    """

    pdl_flux_c_ini, hdr = fits.getdata(template, 0, header=True)
    nf, n_c = pdl_flux_c_ini.shape
    
    coeffs = np.zeros([nf,3])
    crpix = hdr['CRPIX1']
    cdelt = hdr['CDELT1']
    crval = hdr['CRVAL1']
    age_mod = []
    met_mod = []
    ml = []
    name = []
    for iii in range(0, nf):
        header = 'NAME' + str(iii)
        name.extend([hdr[header]]);
        name_min = name[iii]
        name_min = name_min.replace('spec_ssp_','')
        name_min = name_min.replace('.spec','')    
        name_min = name_min.replace('.dat','')
        data = name_min.split('_')
        AGE = data[0]
        MET = data[1]
        if 'Myr' in AGE:
            age = AGE.replace('Myr','')
            age = np.single(age)/1000.
        else:
            age = AGE.replace('Gyr','')
            age = np.single(age)
        met = np.single(MET.replace('z','0.'))
        age_mod.extend([age])
        met_mod.extend([met])
        header = 'NORM' + str(iii)    
        val_ml = np.single(hdr[header])
        if val_ml != 0:
            ml.extend([1/val_ml])
        else:
            ml.extend([1])

    wave_c = []
    dpix_c_val = []
    for jj in range(0, n_c):
        wave_c.extend([(crval+cdelt*(jj+1-crpix))])
        if jj > 0:
            dpix_c_val.extend([wave_c[jj]-wave_c[jj-1]])

    wave_c = np.array(wave_c)
    ml = np.array(ml)
    age_mod = np.array(age_mod)
    met_mod = np.array(met_mod)

    return pdl_flux_c_ini, wave_c, age_mod, met_mod, ml, crval, cdelt, crpix

def gas_extract(template):
    """ 
    Extract from a gas template the fluxes, corresponding ages and 
    metallicities, wavelength range and mass-luminosity weighting 
    factors.

    Arguments:
    ----------
    template : SSP template name. (string)

    Returns:
    -------
    pdl_flux_c_ini: fluxes. ((m,w)-sized float array)
    wave_c: wavelengths. (w-sized float array)
    pht_mod: template photo-ionization values. (m-sized float array)
    met_mod: template metallicity values. (m-sized float array)
    den_mod: template density values. (m-sized float array)
    tem_mod: template temperature values. (m-sized float array)
    ml: mass-luminosity weighting factors per spectrum.(m-sized np.single
        array)
    crval: Axis1 value from template header. (float)
    cdelt: Axis1 value from template header. (float)
    crpix: Axis1 value from template header. (float)

    Source: https://github.com/hjibarram/mock_ifu with minor changes
    """

    pdl_flux_c_ini,hdr = fits.getdata(template, 0, header=True)
    nf,n_c = pdl_flux_c_ini.shape
    coeffs = np.zeros([nf,3])
    crpix = hdr['CRPIX1']
    cdelt = hdr['CDELT1']
    crval = hdr['CRVAL1']
    tem_mod = []
    pht_mod = []
    den_mod = []
    met_mod = []
    Ha = []
    name = []
    for iii in range(0, nf):
        header = 'NAME' + str(iii)
        name.extend([hdr[header]]);
        name_min = name[iii]
        name_min = name_min.replace('spec_gas_', '')
        name_min = name_min.replace('.spec', '')    
        name_min = name_min.replace('.dat', '')
        data = name_min.split('_')
        TEM = data[3]
        PHT = data[2]
        DEN = data[1]
        MET = data[0]
        tem = np.single(TEM.replace('t', ''))
        pht = np.single(PHT.replace('q', ''))
        den = np.single(DEN.replace('n', ''))
        met = np.single(MET.replace('z', ''))
        tem_mod.extend([tem])    
        pht_mod.extend([pht])
        den_mod.extend([den])
        met_mod.extend([met])
        header = 'NORM' + str(iii)    
        val_ml = np.single(hdr[header])
        Ha.extend([val_ml])
    wave_c = []
    dpix_c_val = []
    for j in range(0, n_c):
        wave_c.extend([(crval+cdelt*(j+1-crpix))])
        if j > 0:
            dpix_c_val.extend([wave_c[j]-wave_c[j-1]])
    wave_c = np.array(wave_c)
    return pdl_flux_c_ini, wave_c, pht_mod, met_mod, den_mod, tem_mod,\
            Ha, crval, cdelt, crpix

def thread_dither(args):
    """ 
    Generates one fiber spectrum. 
    """

    seeing, ns, ndt, rad, i, j, xifu, yifu, dit, phi, the, phi_g, the_g, fibB,\
         scalep, nw_s, nw, nw_g, age_ssp3, met_ssp3, ml_ssp3, age_s, met_s,\
         mass_s, facto, d_r, rad_g, Av_g, in_gas, in_ssp, band_g, gas_template, n_ages,\
         n_mets, v_rad, v_rad_g, n_lib_mod, dust_rat_ssp, ssp_template, ml_ssp, radL, wave,\
         dlam, dlam_g, sigma_inst, sp_res, wave_f, sfri, wave_g, dust_rat_gas, ha_gas, \
         radL_g, met_g, age_ssp, met_ssp = args
    con = i * ns + j
    spec_ifu = np.zeros([nw])
    spec_ifu_e = np.zeros([nw])
    spec_val = np.zeros([41])
    spec_ifu_g = np.zeros([nw])
    sim_imag = np.zeros([n_ages])
    sim_imag2 = np.zeros([n_ages, n_mets])
    sim_imag3 = np.zeros([n_ages, n_mets])
    x_ifu = 0.
    y_ifu = 0.   
    seeing2d_s = np.random.multivariate_normal(np.array([0,0]), \
                 np.array([[seeing/2.0/np.sqrt(2*np.log(2)),0], \
                 [0,seeing/2.0/np.sqrt(2*np.log(2))]]), len(rad))
    seeing2d_g = np.random.multivariate_normal(np.array([0,0]), \
                 np.array([[seeing/2.0/np.sqrt(2*np.log(2)),0], \
                 [0,seeing/2.0/np.sqrt(2*np.log(2))]]), len(rad_g)) 
    phie = phi + seeing2d_s[:,0]
    thee = the + seeing2d_s[:,1]
    phieg = phi_g + seeing2d_g[:,0]
    theeg = the_g + seeing2d_g[:,1]      
    dyf = 1.0
    xo = xifu + dit[i,0]
    yo = yifu + dyf*dit[i,1]  
    s_box = np.arange(len(phie))[(np.abs(xo-phie)<=fibB*scalep/2.0) & (np.abs(yo-thee)<=fibB*scalep/2.0)]
    g_box = np.arange(len(phieg))[(np.abs(xo-phieg)<=fibB*scalep/2.0) & (np.abs(yo-theeg)<=fibB*scalep/2.0)]
    r = cdist(np.stack((phie[s_box],thee[s_box])).T, [(xo, yo)])[:, 0]
    r_g = cdist(np.stack((phieg[g_box],theeg[g_box])).T, [(xo, yo)])[:, 0]
    nt = s_box[r <= fibB*scalep/2.0]
    nt_g = g_box[r_g <= fibB*scalep/2.0]
    #r = np.sqrt((xo-phie)**2.0+(yo-thee)**2.0)
    #r_g = np.sqrt((xo-phieg)**2.0+(yo-theeg)**2.0)
    #nt = np.where(r <= fibB*scalep/2.0)[0]
    #nt_g = np.where(r_g <= fibB*scalep/2.0)[0]
    spect_t = np.zeros(nw_s)
    spect = np.zeros(nw)
    spect_g = np.zeros(nw_g)
    spect_gf = np.zeros(nw)
    mass_t = 0
    ml_t = 0
    vel_t = 0
    sfr_t = 0
    Av_s = 0
    Av_sg = 0
    sve_t = 0
    Lt = 0
    Ltg = 0
    ml_ts = 0
    met_ligt = 0
    met_mas = 0
    age_ligt = 0
    age_mas = 0
    Av_ligt = 0
    Av_flux = 0
    Ve_ligt = 0
    Ve_flux = 0
    Avg_ligt = 0
    Avg_flux = 0
    Veg_ligt = 0
    Veg_flux = 0
    Ft = 0
    Ftg = 0
    Sig_flux = 0
    Sig_ligt = 0
    Sig_flux_g = 0
    Sig_ligt_g = 0
    wl_t = []
    wf_t = []
    wl_tg = []
    wf_tg = []
    va_1 = []
    va_1g = []
    Mft = 0
    age_flux = 0
    age_Mflux = 0
    met_ligt_g = 0
    met_flux_g = 0
    met_flux = 0
    met_Mflux = 0
    met_ligt_assig = 0
    met_mas_assig = 0
    age_ligt_assig = 0
    age_mas_assig = 0
    if len(nt) > 0:
        mass_t = np.sum(mass_s[nt])
        vel_t = np.average(v_rad[nt])
        sve_t = np.std(v_rad[nt])
        mass_t_t = associate_ages(age_ssp3, age_s[nt], mass_s[nt])
        #mass_t_t2 = associate_mets_ages(age_ssp3, met_ssp3, age_s[nt], met_s[nt], mass_s[nt])
        ligt_t_t2, mass_t_t2 = associate_mets_ages_flux(age_ssp3, met_ssp3, ml_ssp3,\
                                             age_s[nt], met_s[nt], mass_s[nt])
        sim_imag[:] = mass_t_t * facto
        sim_imag2[:,:] = mass_t_t2 * facto
        sim_imag3[:,:] = ligt_t_t2 * facto
        for k in range(0, len(nt)):
            nt_e=np.where((abs(phi[nt[k]]-phi_g) <= d_r) & (abs(the[nt[k]]-the_g) <= d_r) & (rad_g <= rad[nt[k]]))[0]#DECOMENTAR
            if len(nt_e) > 0:
                Av=np.sum(Av_g[nt_e])
            else:
                Av=0
            Av_s=10**(-0.4*Av)+Av_s
            if np.isnan(in_ssp[nt[k]]):
                spect=spect
            else:
                if in_ssp[nt[k]] > 0 and in_ssp[nt[k]] < n_lib_mod:
                    dust = 10**(-0.4*Av*dust_rat_ssp*0.44)
                    spect_s = ssp_template[in_ssp[nt[k]],:]/ml_ssp[in_ssp[nt[k]]]*\
                              mass_s[nt[k]]*3.846e33/(4.0*np.pi*radL[nt[k]]**2.0)*dust/1e-16
                    spect_sf = shifts(spect_s, wave, dlam[nt[k]])
                    spect_t += spect_sf
                    ml_t += ml_ssp[in_ssp[nt[k]]]
                    Lt += mass_s[nt[k]] / ml_ssp[in_ssp[nt[k]]]
                    Ft += mass_s[nt[k]]/ml_ssp[in_ssp[nt[k]]]*3.846e33/\
                         (4.0*np.pi*radL[nt[k]]**2.0)/1e-16*10**(-0.4*Av*0.44)
                    met_ligt += np.log10(met_s[nt[k]]) * mass_s[nt[k]] / ml_ssp[in_ssp[nt[k]]]
                    met_mas += np.log10(met_s[nt[k]]) * mass_s[nt[k]]
                    age_ligt += np.log10(age_s[nt[k]]) * mass_s[nt[k]] / ml_ssp[in_ssp[nt[k]]]
                    age_mas += np.log10(age_s[nt[k]]) * mass_s[nt[k]]                            
                    ft_w = mass_s[nt[k]]/ml_ssp[in_ssp[nt[k]]]*3.846e33/\
                          (4.0*np.pi*radL[nt[k]]**2.0)/1e-16*10**(-0.4*Av*0.44)
                    lt_w = mass_s[nt[k]] / ml_ssp[in_ssp[nt[k]]]
                    fm_w = mass_s[nt[k]] * 10**(-0.4*Av*0.44)
                    Mft += fm_w
                    age_flux += np.log10(age_s[nt[k]]) * ft_w
                    age_Mflux += np.log10(age_s[nt[k]]) * fm_w
                    met_flux += np.log10(met_s[nt[k]]) * ft_w
                    met_Mflux += np.log10(met_s[nt[k]]) * fm_w
                    Ve_ligt += v_rad[nt[k]] * lt_w
                    Ve_flux += v_rad[nt[k]] * ft_w
                    Av_ligt += 10**(-0.4*Av) * lt_w
                    Av_flux += 10**(-0.4*Av) * ft_w
                    met_ligt_assig += np.log10(met_ssp[in_ssp[nt[k]]]) * mass_s[nt[k]] / ml_ssp[in_ssp[nt[k]]]
                    met_mas_assig += np.log10(met_ssp[in_ssp[nt[k]]]) * mass_s[nt[k]]
                    age_ligt_assig += np.log10(age_ssp[in_ssp[nt[k]]]) * mass_s[nt[k]] / ml_ssp[in_ssp[nt[k]]]
                    age_mas_assig += np.log10(age_ssp[in_ssp[nt[k]]]) * mass_s[nt[k]]   
                    va_1.extend([v_rad[nt[k]]])
                    wf_t.extend([ft_w])
                    wl_t.extend([lt_w])

        ml_t /= len(nt)
        ml_ts = mass_t / Lt
        if Lt > 0:
            met_ligt /= Lt
            met_ligt_assig /= Lt
            age_ligt = 10.0**(age_ligt/Lt)
            age_ligt_assig = 10.0**(age_ligt_assig/Lt)
            age_flux = 10.0**(age_flux/Ft)
            age_Mflux = 10.0**(age_Mflux/Mft)
            met_flux /= Ft
            met_Mflux /= Mft
            Av_ligt /= Lt
            Av_flux /= Ft
            Ve_ligt /= Lt
            Ve_flux /= Ft
            met_mas /= mass_t
            met_mas_assig /= mass_t
            age_mas = 10.0**(age_mas/mass_t)
            age_mas_assig = 10.0**(age_mas_assig/mass_t)
            va_1 = np.array(va_1)
            wf_t = np.array(wf_t)
            wl_t = np.array(wl_t)
            Sig_flux = np.sqrt(np.nansum(np.abs(wf_t)*(Ve_flux-va_1)**2.0)/\
                       (np.nansum(np.abs(wf_t))-np.nansum(wf_t**2.0)/np.nansum(np.abs(wf_t))))
            Sig_ligt = np.sqrt(np.nansum(np.abs(wl_t)*(Ve_ligt-va_1)**2.0)/\
                       (np.nansum(np.abs(wl_t))-np.nansum(wl_t**2.0)/np.nansum(np.abs(wl_t))))
        spect_t[np.isnan(spect_t)] = 0
        spect = interp1d(wave, spect_t, bounds_error=False, fill_value=0.)(wave_f)
        spect[np.isnan(spect)] = 0
        spec_val[0] = Av_s / len(nt)
    if len(nt_g) > 0:
        sfr_t = np.sum(sfri[nt_g])
        for k in range(0, len(nt_g)):
            if band_g[nt_g[k]] > 0:
                nt_e = np.where((abs(phi_g[nt_g[k]]-phi_g) <= d_r) & \
                                (abs(the_g[nt_g[k]]-the_g) <= d_r) & \
                                (rad_g <= rad_g[nt_g[k]]))[0]
                if len(nt_e) > 0:
                    Av = np.sum(Av_g[nt_e])
                else:
                    Av = 0
                Av_sg += Av
                if np.isnan(in_gas[nt_g[k]]):
                    spect_gf = spect_gf
                else:
                    if in_gas[nt_g[k]] > 0 and in_gas[nt_g[k]] < 525:
                        dust = 10**(-0.4*Av*dust_rat_gas)
                        spect_sg = gas_template[in_gas[nt_g[k]],:]/\
                                   ha_gas[in_gas[nt_g[k]]]*3.846e33*band_g[nt_g[k]]/\
                                   (4.0*np.pi*radL_g[nt_g[k]]**2.0)*dust/1e-16*10.0**(7.18)#+0.3+0.6)#*0.01#*mass_g[nt_g[k]]
                        spect_sfg = shifts(spect_sg,wave_g,dlam_g[nt_g[k]])
                        lt_wg = np.nansum(gas_template[in_gas[nt_g[k]],:])*10.0**(7.18)
                        ft_wg = np.nansum(spect_sfg)
                        Ltg += lt_wg
                        Ftg += ft_wg
                        Veg_ligt += v_rad_g[nt_g[k]] * lt_wg
                        Veg_flux += v_rad_g[nt_g[k]] * ft_wg
                        Avg_ligt += 10**(-0.4*Av) * lt_wg
                        Avg_flux += 10**(-0.4*Av) * ft_wg
                        met_ligt_g += np.log10(met_g[nt_g[k]]) * lt_wg
                        met_flux_g += np.log10(met_g[nt_g[k]]) * ft_wg                   
                        spect_g += spect_sfg   
                        va_1g.extend([v_rad_g[nt_g[k]]])
                        wf_tg.extend([ft_wg])
                        wl_tg.extend([lt_wg])                            
        if Ltg > 0:
            Avg_ligt /= Ltg
            Avg_flux /= Ftg
            Veg_ligt /= Ltg
            Veg_flux /= Ftg
            met_ligt_g /= Ltg
            met_flux_g /= Ftg
            va_1g = np.array(va_1g)
            wf_tg = np.array(wf_tg)
            wl_tg = np.array(wl_tg)
            Sig_flux_g = np.sqrt(np.nansum(np.abs(wf_tg)*(Veg_flux-va_1g)**2.0)/\
                (np.nansum(np.abs(wf_tg))-np.nansum(wf_tg**2.0)/np.nansum(np.abs(wf_tg))))
            Sig_ligt_g = np.sqrt(np.nansum(np.abs(wl_tg)*(Veg_ligt-va_1g)**2.0)/\
                (np.nansum(np.abs(wl_tg))-np.nansum(wl_tg**2.0)/np.nansum(np.abs(wl_tg))))    
        spec_val[6] = np.sum(Av_g[nt_g])         
        spec_val[4] = Av_sg / len(nt_g)
        spect_g[np.isnan(spect_g)] = 0
        spect_gf = interp1d(wave_g,spect_g,bounds_error=False,fill_value=0.)(wave_f)
        spect_gf[np.isnan(spect_gf)] = 0

    spec_ifu[:] = facto * spect
    spec_ifu_g[:] = facto * spect_gf
    spec_val[1] = mass_t * facto
    spec_val[2] = vel_t
    spec_val[3] = sfr_t * facto
    spec_val[5] = spec_val[0] * 0 + spec_val[4]
    spec_val[7] = sve_t
    spec_val[8] = ml_t #*facto
    spec_val[10] = Lt * facto
    spec_val[9] = ml_ts
    spec_val[11] = met_ligt
    spec_val[12] = met_mas
    spec_val[13] = age_ligt
    spec_val[14] = age_mas
    spec_val[15] = Av_ligt
    spec_val[16] = Ft * facto
    spec_val[17] = Av_flux
    spec_val[18] = Ve_ligt
    spec_val[19] = Ve_flux
    spec_val[20] = Sig_ligt
    spec_val[21] = Sig_flux
    spec_val[22] = Avg_ligt
    spec_val[23] = Avg_flux
    spec_val[24] = Veg_ligt
    spec_val[25] = Veg_flux
    spec_val[26] = Sig_ligt_g
    spec_val[27] = Sig_flux_g
    spec_val[28] = Ftg * facto
    spec_val[29] = Ltg * facto
    spec_val[30] = Mft * facto
    spec_val[31] = age_flux
    spec_val[32] = age_Mflux
    spec_val[33] = met_ligt_g
    spec_val[34] = met_flux_g
    spec_val[35] = met_flux
    spec_val[36] = met_Mflux
    spec_val[37] = met_ligt_assig
    spec_val[38] = age_ligt_assig
    spec_val[39] = met_mas_assig
    spec_val[40] = age_mas_assig
    x_ifu = xo
    y_ifu = yo
    n_star = nt.size
    n_gas = nt_g.size
    return spec_ifu, spec_val, spec_ifu_g, sim_imag, sim_imag2,\
                 sim_imag3, x_ifu, y_ifu, n_star, n_gas



def mk_the_light(outf, x, y, z, vx, vy, vz, x_g, y_g, z_g, vx_g, vy_g,\
              vz_g, age_s, met_s, mass_s, met_g, vol, dens, sfri, temp_g,\
              Av_g, mass_g, template_SSP_control, template_SSP,\
              template_gas, wave_samp=[3749,10352,1], dir_o='', psfi=0,\
              red_0=0.01, nl=7, cpu_count=8, thet=0.0, ifutype='MaNGA', \
              err_dith=0., ran_seed=12345):
    """ 
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
         - metallicity in Z.
         - mass in solar masses.
      Gas
         - x, y, z coordinates relative to the observer in physical kpcs
           (z in the direction of the observer).
         - vx, vy, vz velocity components relative to the volume in km/s.
         - metallicity in Z.
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
    wave_samp (=[3749, 10352, 1]): spectral sampling array in Ang given by
              [initial wavelength, final wavelength, step]. (float list N=3)
    err_dith (=0.): maximum random shift in the dithering position in arcsec. (float)
    ran_seed (=12345): seed for random dithering shift. (integer)

    Returns:
    -------
    -

    Outputs:
    -------
    FITS file containing the row stacked spectrum (RSS).
    """

    nh = dens#*1e10/(3.08567758e19*100)**3.0*1.9891e30/1.67262178e-27
    fact = nh / 10.0
    sfri = sfri + 1e-6
    mass_gssp = sfri * 2.5e6#100e6
    Rs = vol#np.float_((vol/(4.0*np.pi/3.0))**(1./3.0)*(3.08567758e19*100))
    sup = 4.0 * np.pi * Rs**2.0#4.0*np.pi*(3.0*vol/4.0/np.pi)**(2.0/3.0)*(3.08567758e19*100)**2.0
    vel_light = 299792.458
    no_nan = 0
    if 'MaNGA' in ifutype:
        sp_res = 2000.0
        pix_s = 0.5#arcsec
        scp_s = 60.4#microns per arcsec
        fibA = 150.0
        fibB = 120.0
        sigma_inst = 25.0
        beta_s = 2.0
        if psfi <= 0:
            seeing = 1.43
        else:
            seeing = psfi
        if sp_res <= 0:
            sp_res = 2000.0
    elif 'CALIFA' in ifutype:
        pix_s = 1.0#arcsec
        scp_s = 56.02#microns per arcsec
        fibA = 197.4
        fibB = 150.0
        nl = 11
        sigma_inst = 25.0
        beta_s = 4.7
        if psfi <= 0:
            seeing = 0.7
        else:
            seeing=psfi
        if sp_res <= 0:
            sp_res = 1700.0
    elif 'MUSE' in ifutype:
        pix_s = 0.2#0.1#0.025#arcsec
        scp_s = 300.0#150.0#300.0#1200.0#microns per arcsec
        fibA = 150.0
        fibB = 120.0
        fov = 15.
        nl = np.int32(fov*scp_s/fibA/2)+1
        sigma_inst = 25.0
        beta_s = 4.7
        if psfi <= 0:
            seeing = 0.6
        else:
            seeing = psfi
        if sp_res <= 0:
            sp_res = 4000.0
    else:
        pix_s = 0.5#arcsec
        scp_s = 60.4#microns per arcsec
        fibA = 150.0
        fibB = 120.0     
        sigma_inst = 25.0
        beta_s = 2.0
        if psfi == 0:
            seeing = 1.43
        else:
            seeing = psfi   
        if sp_res <= 0:
            sp_res = 2000.0       
    if wave_samp[2] <= 0:
        wave_samp[2] = 5000. / sp_res / 2.0
    scalep = 1.0 / scp_s
    
    dkpcs = cosmo.kpc_proper_per_arcmin(red_0).value / 60.
    cam = cosmo.lookback_time(red_0).value * 1e6 * 0.307
    #cam = cosmo.comoving_distance(red_0).value*1e3
    #dkpcs = cam*(1./3600.)*(np.pi/180.)
    dap = scalep
    xima = np.zeros(nl)
    yima = np.zeros(nl)
    rad = np.sqrt(x**2.+y**2.+(cam-z)**2.)
    d_r = 0.10/dkpcs
    v_rad = (vx*x+vy*y+vz*(z-cam)) / rad   
    rad_g = np.sqrt(x_g**2.+y_g**2.+(cam-z_g)**2.)
    v_rad_g = (vx_g*x_g+vy_g*y_g+vz_g*(z_g-cam))/rad_g 
    reds = reds_cos(rad/1e3)
    radA = rad/(1+reds)
    radL = np.array(rad*(1+reds)*(3.08567758e19*100))
    reds_g = reds_cos(rad_g/1e3)
    dlam = (1+(v_rad/vel_light+reds))
    dlam_g = (1+(v_rad_g/vel_light+reds_g))
    radA_g = rad_g/(1+reds_g)
    radL_g = np.array(rad_g*(1+reds_g)*(3.08567758e19*100))
    phi = np.arcsin(x/radA)
    the = np.arcsin(y/(radA*np.cos(phi)))
    the = the * 180 / np.pi * 3600#+ran.randn(len(rad))*2.0
    phi = phi * 180 / np.pi * 3600#+ran.randn(len(rad))*2.0
    phi_g = np.arcsin(x_g/radA_g)
    the_g = np.arcsin(y_g/(radA_g*np.cos(phi_g)))
    the_g = the_g * 180 / np.pi * 3600#+ran.randn(len(rad_g))*2.0
    phi_g = phi_g * 180 / np.pi * 3600#+ran.randn(len(rad_g))*2.0
    ns = 3 * nl * (nl-1) + 1
    Dfib = fibA*scalep
    Rifu = Dfib*((2.0*nl-1.0)/2.0-0.5)
    xfib0 = -Rifu
    yfib0 = 0
    dxf = 1.0
    dyf = np.sin(60.*np.pi/180.)
    xifu = np.zeros(ns)
    yifu = np.zeros(ns)
    ini = 0
    for ii in range(0, nl):
        nt = nl*2-1-ii
        yfib = yfib0+ii*dyf*Dfib
        for jj in range(0, nt):
            xfib = xfib0 + (jj*dxf+0.5*ii) * Dfib
            xifu[ini] = xfib
            yifu[ini] = yfib
            ini = ini + 1

        if ii > 0:
            for jj in range(0, nt):
                xfib = xfib0 + (jj*dxf+0.5*ii) * Dfib
                xifu[ini] = xfib
                yifu[ini] = -yfib
                ini = ini+1

    ndt = 35
    dyf = 1.0
    dyt = Dfib/2.0/np.cos(30.0*np.pi/180.0)
    dxt = Dfib/2.0
    ndt = 3
    dit = np.zeros([ndt,2])
    rng = np.random.default_rng(ran_seed)
    ran_dit_shift = rng.random(size=(ndt, 2))
    dit[0,:] = [+0.00,+0.00] + ran_dit_shift[0] * err_dith
    dit[1,:] = [+0.00,+dyt/1.0] + ran_dit_shift[1] * err_dith
    dit[2,:] = [-dxt, +dyt/2.0] + ran_dit_shift[2] * err_dith

    
    ssp_template, wave, age_ssp, met_ssp, ml_ssp, crval_w, cdelt_w, \
                    crpix_w = ssp_extract(template_SSP)
    n_lib_mod = ssp_template.shape[0]
    ssp_template3, wave3, age_ssp3, met_ssp3, ml_ssp3, crval_w3, cdelt_w3,\
                    crpix_w3 = ssp_extract(template_SSP_control)
    gas_template, wave_g, pht_gas, met_gas, den_gas, tem_gas, ha_gas, \
                    crval_g, cdelt_g, crpix_g = gas_extract(template_gas)
    in_ssp = associate_ssp(age_ssp,  met_ssp,  age_s,  met_s)
    pht_g = associate_pho(ssp_template, wave, age_ssp, met_ssp, ml_ssp, \
                          mass_gssp, met_g, Rs, nh)
    in_gas = associate_gas(pht_gas, met_gas, den_gas, tem_gas, \
                           pht_g, met_g, nh, temp_g)
    dust_rat_ssp = A_l(3.1, wave)
    dust_rat_gas = A_l(3.1, wave_g)
    crpix_w = 1
    wave_f = np.arange(wave_samp[0], wave_samp[1], wave_samp[2]) #7501
    band_g = np.ones(len(met_g))
    band_g[np.where((pht_g == 0))[0]] = 1.0 # &(in_gas == -100)
    nw = len(wave_f)
    nw_s = len(wave)
    nw_g = len(wave_g)
    n_ages = num_ages(age_ssp3)
    n_mets = num_ages(met_ssp3)
    ages_r = arg_ages(age_ssp3)
    x_ifu = np.zeros(ndt*ns)
    y_ifu = np.zeros(ndt*ns)
    facto = (pix_s)**2.0 / (np.pi*(fibB*scalep/2.0)**2.0)#*np.pi


    args0 = [(seeing, ns, ndt, rad, 0, j, xifu[j], yifu[j], dit, phi, the, phi_g, the_g, fibB,\
             scalep, nw_s, nw, nw_g, age_ssp3, met_ssp3, ml_ssp3, age_s, met_s,\
             mass_s, facto, d_r, rad_g, Av_g, in_gas, in_ssp, band_g, gas_template, n_ages, \
             n_mets, v_rad, v_rad_g, n_lib_mod, dust_rat_ssp, ssp_template, ml_ssp, radL, wave,\
             dlam, dlam_g, sigma_inst, sp_res, wave_f, sfri, wave_g, dust_rat_gas, ha_gas, radL_g,\
            met_g, age_ssp, met_ssp) for j in range(ns)]
    args1 = [(seeing, ns, ndt, rad, 1, j, xifu[j], yifu[j], dit, phi, the, phi_g, the_g, fibB,\
             scalep, nw_s, nw, nw_g, age_ssp3, met_ssp3, ml_ssp3, age_s, met_s,\
             mass_s, facto, d_r, rad_g, Av_g, in_gas, in_ssp, band_g, gas_template, n_ages, \
             n_mets, v_rad, v_rad_g, n_lib_mod, dust_rat_ssp, ssp_template, ml_ssp, radL, wave,\
            dlam, dlam_g, sigma_inst, sp_res, wave_f, sfri, wave_g, dust_rat_gas, ha_gas, radL_g,\
            met_g, age_ssp, met_ssp) for j in range(ns)]
    args2 = [(seeing, ns, ndt, rad, 2, j, xifu[j], yifu[j], dit, phi, the, phi_g, the_g, fibB,\
             scalep, nw_s, nw, nw_g, age_ssp3, met_ssp3, ml_ssp3, age_s, met_s,\
             mass_s, facto, d_r, rad_g, Av_g, in_gas, in_ssp, band_g, gas_template, n_ages, \
             n_mets, v_rad, v_rad_g, n_lib_mod, dust_rat_ssp, ssp_template, ml_ssp, radL, wave,\
             dlam, dlam_g, sigma_inst, sp_res, wave_f, sfri, wave_g, dust_rat_gas, ha_gas, radL_g,\
              met_g, age_ssp, met_ssp) for j in range(ns)]
    args = args0 + args1 + args2



    if 'SLURM_CPUS_PER_TASK' in os.environ:
        cpu_count = np.int32(os.environ['SLURM_CPUS_PER_TASK'])
        print('Using slurm number of CPUs...')
    print(cpu_count, ' CPUs employed.')
    
    if cpu_count<=1:
        spec_ifu = []
        spec_val = []
        spec_ifu_g = []
        sim_imag = []
        sim_imag2 = []
        sim_imag3 = []
        x_ifu = []
        y_ifu = []
        n_star = []
        n_gas = []
        for ii in range(len(args)):
            spec_ifu_, spec_val_, spec_ifu_g_, sim_imag_, sim_imag2_,\
                 sim_imag3_, x_ifu_, y_ifu_, n_star_, n_gas_ = thread_dither(args[ii])
            spec_ifu.append(spec_ifu_)
            spec_val.append(spec_val_)
            spec_ifu_g.append(spec_ifu_g_)
            sim_imag.append(sim_imag_)
            sim_imag2.append(sim_imag2_)
            sim_imag3.append(sim_imag3_)
            x_ifu.append(x_ifu_)
            y_ifu.append(y_ifu_) 
            n_star.append(n_star_)
            n_gas.append(n_gas_)
        spec_ifu = np.array(spec_ifu)
        spec_val = np.array(spec_val)
        spec_ifu_g = np.array(spec_ifu_g)
        sim_imag = np.array(sim_imag)
        sim_imag2 = np.array(sim_imag2)
        sim_imag3 = np.array(sim_imag3)
        x_ifu = np.array(x_ifu)
        y_ifu = np.array(y_ifu) 
        n_star = np.array(n_star)
        n_gas = np.array(n_gas)

    else:
        pool = mp.Pool(cpu_count)
        spec_ifu, spec_val, spec_ifu_g, sim_imag, sim_imag2, sim_imag3, x_ifu,\
        y_ifu, n_star, n_gas = zip(*pool.map(thread_dither, args))
        spec_ifu = np.asarray(spec_ifu)
        spec_val = np.asarray(spec_val)
        spec_ifu_g = np.asarray(spec_ifu_g)
        sim_imag = np.asarray(sim_imag)
        sim_imag2 = np.asarray(sim_imag2)
        sim_imag3 = np.asarray(sim_imag3)
        x_ifu = np.asarray(x_ifu)
        y_ifu = np.asarray(y_ifu)
        n_star = np.asarray(n_star)
        n_gas = np.asarray(n_gas)
        pool.close()
        pool.join()
        del pool

    spec_ifu = spec_ifu.T
    spec_val = spec_val.T
    spec_ifu_g = spec_ifu_g.T
    sim_imag = sim_imag.T
    sim_imag2 = np.transpose(sim_imag2,(1,2,0))
    sim_imag3 = np.transpose(sim_imag3,(1,2,0))
    #sycall('echo '+ str(sim_imag.shape))
    #sycall('echo '+ str(sim_imag2.shape))    
    h1 = fits.PrimaryHDU(spec_ifu)#.header
    h3 = fits.ImageHDU(spec_ifu_g)
    h4 = fits.ImageHDU(spec_val)
    h5 = fits.ImageHDU(sim_imag)
    h6 = fits.ImageHDU(sim_imag2)
    h7 = fits.ImageHDU(sim_imag3)
    h8 = fits.ImageHDU(x_ifu)
    h9 = fits.ImageHDU(y_ifu)
    h10 = fits.ImageHDU(np.array(np.stack((n_star, n_gas)), dtype=np.int32))

    h = h1.header
    h['NAXIS'] = 3
    h['NAXIS2'] = nw 
    h['NAXIS1'] = spec_ifu.shape[0]
    h['COMMENT'] = 'Mock '+ifutype+' IFU'
    h['CRVAL1'] = 0
    h['CTYPE1'] = 'N fiber'
    h['CDELT2'] = wave_samp[2]
    h['CRPIX2'] = crpix_w
    h['CRVAL2'] = wave_samp[0]
    h['CUNIT2'] = 'Wavelength [A]'
    h['RADECSYS'] = 'ICRS    '
    h['SYSTEM'] = 'FK5     '
    h['EQUINOX'] = 2000.00
    h['PSF'] = seeing
    h['FOV'] = Rifu*2.0
    h['KPCSEC'] = (dkpcs, 'kpc/arcsec')
    h['CAMX'] = 0
    h['CAMY'] = 0
    h['CAMZ'] = cam
    h['REDSHIFT'] = np.single(red_0)
    h['R'] = ('SSP','Spectral Resolution')
    h['COSMO'] = cosmo.name
    h['H0'] = (cosmo.H0.value, cosmo.H0.unit)
    h['Omega_m'] = cosmo.Om0
    h['IFUCON'] = (str(np.int32(ns))+' ','NFibers')
    h['UNITS'] = '1E-16 erg/s/cm^2'
    h['SSPTEMP'] = template_SSP

    h = h3.header
    h['EXTNAME'] = 'Gas emission' 
    h['CRVAL1'] = 0
    h['CTYPE1'] = 'N fiber'
    h['CDELT2'] = wave_samp[2]
    h['CRPIX2'] = crpix_w
    h['CRVAL2'] = wave_samp[0]
    h['CUNIT2'] = 'Wavelength [A]'
    h['GASTEMP'] = template_gas

    h = h4.header
    h['EXTNAME'] = 'Real values' 
    h['Type0'] = ('Av_T    ','Mag')
    h['Type1'] = ('MASS    ','log10(Msun)')
    h['Type2'] = ('VEL     ','km/s')
    h['Type3'] = ('SFR     ','Msun/yr')
    h['Type4'] = ('DUST_G  ','Av BETA')
    h['Type5'] = ('DUST_T  ','Av BETA')
    h['Type6'] = ('DUST_Av ','Av BETA')
    h['Type7'] = ('DISP    ','km/s')
    h['Type8'] = ('aML     ','Msun/Lsun BETA')
    h['Type9'] =  ('tML     ','Msun/Lsun BETA')
    h['Type10'] = ('LUM     ','log10(Lsun)')
    h['Type11'] = ('Z_lw    ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type12'] = ('Z_mw    ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type13'] = ('AGE_lw  ','Gyr')
    h['Type14'] = ('AGE_mw  ','Gyr')
    h['Type15'] = ('Av_lw   ','Mag')
    h['Type16'] = ('FLUX    ','1e-16 ergs/s/cm2')
    h['Type17'] = ('Av_fw   ','Mag')
    h['Type18'] = ('VEL_lw  ','km/s')
    h['Type19'] = ('VEL_fw  ','km/s')
    h['Type20'] = ('DIS_lw   ','km/s BETA')
    h['Type21'] = ('DIS_fw   ','km/s BETA')
    h['Type22'] = ('Av_lw_g  ','Mag')
    h['Type23'] = ('Av_fw_g  ','Mag')
    h['Type24'] = ('VEL_lw_g ','km/s')
    h['Type25'] = ('VEL_fw_g ','km/s')
    h['Type26'] = ('DIS_l_gas','km/s BETA')
    h['Type27'] = ('DIS_f_gas','km/s BETA')
    h['Type28'] = ('FLUX_gas','1e-16 ergs/s/cm2 Bolometric')
    h['Type29'] = ('LUM_gas ','log10(Lsun) Bolometric')
    h['Type30'] = ('MASS_fw ','log10(Msun) BETA')
    h['Type31'] = ('AGE_fw  ','Gyr')
    h['Type32'] = ('AGE_mfw ','Gyr BETA')
    h['Type33'] = ('Z_lw_gas ','log10(Z/H) add 10.46 to convert to 12+log10(O/H) or add 1.77 to convert to log10(Z/Z_sun)')
    h['Type34'] = ('Z_fw_gas ','log10(Z/H) add 10.46 to convert to 12+log10(O/H) or add 1.77 to convert to log10(Z/Z_sun)')#8.69 
    h['Type35'] = ('Z_fw    ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type36'] = ('Z_mfw    ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type37'] = ('Z_lw_assigned ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type38'] = ('AGE_lw_assigned  ','Gyr')
    h['Type39'] = ('Z_mw_assigned ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type40'] = ('AGE_mw_assigned  ','Gyr')
    h['RADECSYS'] = 'ICRS    '
    h['SYSTEM'] = 'FK5     '
    h['EQUINOX'] = 2000.00
    h['PSF'] = seeing

    h = h5.header
    h['EXTNAME'] = 'mass_t' 
    h['UNITS'] = 'Msun'

    h = h6.header
    h['EXTNAME'] = 'mass_full' 
    h['UNITS'] = 'Msun'

    h = h7.header
    h['EXTNAME'] = 'light_full' 
    h['UNITS'] = 'Lsun'

    h = h8.header
    h['EXTNAME'] = 'x_ifu' 

    h = h9.header
    h['EXTNAME'] = 'y_ifu' 

    h = h10.header
    h['EXTNAME'] = 'N_part_fiber'
    h['Type0'] = ('n_stellar    ','number of stellar particles in fiber FOV')
    h['Type1'] = ('n_gas   ','number of gas cells in fiber FOV')

    hlist = fits.HDUList([h1, h3, h4, h5, h6, h7, h8, h9, h10])
    hlist.update_extend()
    out_fit = dir_o+outf+'_RSS.fits'
    hlist.writeto(out_fit, overwrite=1)
    compress_gzip(out_fit)


def mk_mock_RSS(star_file, gas_file, template_SSP_control,\
                template_SSP, template_gas, fib_n=7, cpu_count=2,\
                psfi=0, thet=0.0, ifutype='MaNGA', red_0=0.01, \
                outdir='', indir='', rssf='', wave_samp=[3749,10352,1], err_dith=0.,
                ran_seed=1234): #snap, subhalo, view
    """ 
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
    wave_samp (=[3749, 10352, 1]): spectral sampling array in Ang given by
              [initial wavelength, final wavelength, step]. (float list N=3)
    err_dith (=0.): maximum random shift in the dithering positions in arcsec. (float)
    ran_seed (=12345): seed for random dithering shift. (integer)

    Returns:
    -------
    -

    Outputs:
    RSS file produced by mk_the_light() funtion.

    """

    data = np.genfromtxt(indir + star_file)
    x_b = data[:, 0]
    y_b = data[:, 1]
    z_b = data[:, 2]
    vx_b = data[:, 3]
    vy_b = data[:, 4]
    vz_b = data[:, 5]
    age_s_b = data[:, 6]
    meta_b = data[:, 7]
    mass_b = data[:, 8]
    if os.stat(indir + gas_file).st_size == 0:
        x_g_b = np.array([]) 
        y_g_b = np.array([]) 
        z_g_b = np.array([]) 
        vx_g_b = np.array([]) 
        vy_g_b = np.array([]) 
        vz_g_b = np.array([]) 
        meta_g_b = np.array([]) 
        volm_b = np.array([]) 
        dens_b = np.array([]) 
        sfri_b = np.array([]) 
        temp_g_b = np.array([]) 
        Av_g_b = np.array([]) 
        mass_g_b = np.array([]) 
    else:
        data = np.genfromtxt(indir + gas_file)
        if data.size==13:
            x_g_b = np.array([data[0]])
            y_g_b = np.array([data[1]])
            z_g_b = np.array([data[2]])
            vx_g_b = np.array([data[3]])
            vy_g_b = np.array([data[4]])
            vz_g_b = np.array([data[5]])
            meta_g_b = np.array([data[6]])
            volm_b = np.array([data[7]])
            dens_b = np.array([data[8]])
            sfri_b = np.array([data[9]])
            temp_g_b = np.array([data[10]])
            Av_g_b = np.array([data[11]])
            mass_g_b = np.array([data[12]])
        else:
            x_g_b = data[:, 0]
            y_g_b = data[:, 1]
            z_g_b = data[:, 2]
            vx_g_b = data[:, 3]
            vy_g_b = data[:, 4]
            vz_g_b = data[:, 5]
            meta_g_b = data[:, 6]
            volm_b = data[:, 7]
            dens_b = data[:, 8]
            sfri_b = data[:, 9]
            temp_g_b = data[:, 10]
            Av_g_b = data[:, 11]
            mass_g_b = data[:, 12]
    ns = 3 * fib_n * (fib_n-1) + 1
    idfib = str(ns)
    cubef = rssf + '-' + idfib + '.cube'
    mk_the_light(cubef, x_b, y_b, z_b, vx_b, vy_b, vz_b, x_g_b, y_g_b, z_g_b, \
              vx_g_b, vy_g_b, vz_g_b, age_s_b, meta_b, mass_b, meta_g_b, volm_b, \
              dens_b, sfri_b, temp_g_b, Av_g_b, mass_g_b, \
              wave_samp=wave_samp, template_SSP_control=template_SSP_control, \
              template_SSP=template_SSP, template_gas=template_gas, psfi=psfi, \
              dir_o=outdir, red_0=red_0, nl=fib_n, cpu_count=cpu_count,\
              thet=thet, ifutype=ifutype, err_dith=err_dith, ran_seed=ran_seed)



def regrid(rss_file, outf, template_SSP_control, dir_r='', dir_o='', \
            n_fib=7, thet=0.0, R_eff=None, include_gas=False, noise=[5., 2.]):#snap, subhalo, view
    """
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
    noise (=[5, 2]): noise parameters, first corresponds to S/N desired,
             while the second is the radius [Re] at which the signal S is 
             referenced. (list or float array with size 2) Note that Re must
             be given to add noise.


    Returns:
    -------
    -

    Outputs:
    -------
    Two FITS files:
    
    outf + '.cube.fits.gz'
    File comprising a Primary HDU and three extensions:
       Primary - Spatial-spectral datacube.
       - Error per spatial-wavelength unit.
       - Valid mask array, necessary for pyPipe3D processing.
       - Gas only spatial-spectral datacube.
    
    outf + '.cube_val.fits.gz'
    File comprising a Primary HDU and three extensions:
       - Intrinsic and assigned values from the simulation.
       - Mass decomposition per age in SSP.
       - Mass decomposition per age and metallicity in SSP.
       - Luminosity decomposition per age and metallicity in SSP.

    """
    outf = outf + '.cube'
    pix_s = 0.5 #MaNGA
    fibA = 150.0 #MaNGA
    scp_s = 60.4 #MaNGA
    scalep = 1.0 / scp_s
    sigma_rec  =  0.7 #2.21/2.0/np.sqrt(2*np.log(2))
    ifutype = 'MaNGA'
    Dfib = fibA * scalep
    Rifu = Dfib * ((2.0*n_fib-1.0)/2.0-0.5)
    ssp_template3, wave3, age_ssp3, met_ssp3, ml_ssp3, crval_w3, cdelt_w3, \
                crpix_w3 = ssp_extract(template_SSP_control)
    n_ages = num_ages(age_ssp3)
    n_mets = num_ages(met_ssp3)
    ages_r = arg_ages(age_ssp3)
    ns = 3 * n_fib * (n_fib-1) + 1

    fib_index = {3: np.array([4,5,6,7,8,17,18,19,20,29,30,31,32,41,42,43,52,53,54]),
                4: np.array([3,4,5,6,7,8,9,16,17,18,19,20,21,28,29,30,31,32,33,40,41,42,43,44,51,52,53,54,55,62,63,64,65,72,73,74,75]),
                5: np.array([2,3,4,5,6,7,8,9,10,15,16,17,18,19,20,21,22,27,28,29,30,31,32,33,34,39,40,41,42,43,44,45,50,51,52,53,54,55,56,61,62,63,64,65,66,71,72,73,74,75,76,81,82,83,84,85,90,91,92,93,94]),
                6: np.array([1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,26,27,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,49,50,51,52,53,54,55,56,57,60,61,62,63,64,65,66,67,70,71,72,73,74,75,76,77,80,81,82,83,84,85,86,89,90,91,92,93,94,95,98,99,100,101,102,103,106,107,108,109,110,111]),
                7: np.arange(127)}
    fib_ind_final = np.concatenate([fib_index[n_fib], fib_index[n_fib]+127, fib_index[n_fib]+254])

    rss = fits.open(dir_r+rss_file)
    dkpcs = rss[0].header['KPCSEC']
    red_0 = np.single(rss[0].header['REDSHIFT'])
    cam = rss[0].header['CAMZ']
    seeing = rss[0].header['PSF']
    nw = rss[0].header['NAXIS2']
    cdelt_w = rss[0].header['CDELT2']
    crpix_w = rss[0].header['CRPIX2']
    crval_w = rss[0].header['CRVAL2']
    sp_res = rss[0].header['R']
    if rss[0].header['NAXIS1'] >= ns*3 :
        spec_ifu = rss[0].data[:, fib_ind_final]
        spec_ifu_g = rss[1].data[:, fib_ind_final]
        spec_val = rss[2].data[:, fib_ind_final]
        sim_imag = rss[3].data[:, fib_ind_final]
        sim_imag2 = rss[4].data[:, :, fib_ind_final]
        sim_imag3 = rss[5].data[:, :, fib_ind_final]
        x_ifu = rss[6].data[fib_ind_final]
        y_ifu = rss[7].data[fib_ind_final]
    else:
        spec_ifu = rss[0].data
        spec_ifu_g = rss[1].data
        spec_val = rss[2].data
        sim_imag = rss[3].data
        sim_imag2 = rss[4].data
        sim_imag3 = rss[5].data
        x_ifu = rss[6].data
        y_ifu = rss[7].data
    bad_pix_ = np.nonzero(spec_ifu[:,6]==0)

    wl = np.arange(crval_w, crval_w+(nw*cdelt_w), cdelt_w)
    for ii in range(spec_ifu_g.shape[1]):
        spec_ifu_g[:, ii] = cube_conv_lsf(wl, spec_ifu_g[:, ii], resolution='MaNGA', delta_wl=100)

    if not R_eff == None:
        F0 = get_F0(spec_ifu, R_eff, dkpcs, wl, n_radii=noise[1]) #R_eff must be given in physical kpc
        spec_noise = get_noise(wl, F0, fib_ind_final.size, SN=noise[0])
        spec_ifu += spec_noise.T
        spec_ifu_e = get_noise(wl, F0, fib_ind_final.size, SN=noise[0], realization=False).T#spec_noise.T

    nl = np.int32(round((np.amax([np.amax(x_ifu), -np.amin(x_ifu), np.amax(y_ifu), -np.amin(y_ifu)])+1)*2/pix_s))

    ifu = np.zeros([nw, nl, nl])
    ifu_g = np.zeros([nw, nl, nl])
    ifu_e = np.ones([nw, nl, nl])
    ifu_m = np.zeros([nw, nl, nl])
    ifu_v = np.zeros([spec_val.shape[0], nl, nl])
    ifu_a = np.zeros([n_ages, nl, nl])
    ifu_b = np.zeros([n_ages, n_mets, nl, nl])
    ifu_c = np.zeros([n_ages, n_mets, nl, nl])
    ifu_g = np.zeros([nw, nl, nl])
    xo = -nl/2*pix_s
    yo = -nl/2*pix_s
    xi = xo
    xf = xo

    int_spect = np.zeros(nw)
    for i in range(0, nl):
        xi = xf
        xf = xf+pix_s
        yi = yo
        yf = yo
        for j in range(0, nl):
            yi = yf
            yf += pix_s
            spt_new = np.zeros(nw)
            spt_err = np.zeros(nw)
            spt_val = np.zeros(spec_val.shape[0])
            spt_mas = np.zeros(n_ages)
            spt_mas_2 = np.zeros([n_ages, n_mets])
            spt_mas_3 = np.zeros([n_ages, n_mets])
            spt_gas = np.zeros(nw)
            Wgt = 0
            for k in range(0, len(x_ifu)):
                V1 = np.sqrt((x_ifu[k]-xi)**2.0 + (y_ifu[k]-yf)**2.0)
                V2 = np.sqrt((x_ifu[k]-xf)**2.0 + (y_ifu[k]-yf)**2.0)
                V3 = np.sqrt((x_ifu[k]-xi)**2.0 + (y_ifu[k]-yi)**2.0)
                V4 = np.sqrt((x_ifu[k]-xf)**2.0 + (y_ifu[k]-yi)**2.0)
                Vt = np.array([V1,V2,V3,V4])
                Rsp = np.sqrt((x_ifu[k]-(xf+xi)/2.0)**2.0+(y_ifu[k]-(yf+yi)/2.0)**2.0)
                Vmin = np.amin(Vt)
                Vmax = np.amax(Vt)
                #if Vmin <= fibB*scalep/2.0:
                #    if Vmax <= fibB*scalep/2.0:
                #        Wg=(pix_s)**2.0/(np.pi*(fibB*scalep/2.0)**2.0)
                #    else:
                #        Wg=(1.0-(Vmax-fibB*scalep/2.0)/(np.sqrt(2.0)*pix_s))*(pix_s)**2.0/(np.pi*(fibB*scalep/2.0)**2.0)
                #    spt_new=spec_ifu[:,k]*Wg+spt_new
                #    spt_err=(spec_ifu_e[:,k]*Wg)**2.0+spt_err**2.0
                #    Wgt=Wgt+Wg
                if Rsp <= fibA*scalep*1.4/2.0: #1.6 arcsec
                    #Wg = np.exp(-(Rsp/pix_s)**2.0/2.0)
                    Wg = np.exp(-(Rsp/sigma_rec)**2.0/2.0)
                    #Wg = 1.0
                    spt_new += spec_ifu[:,k] * Wg
                    spt_err += (spec_ifu_e[:,k] * Wg)**2.0
                    spt_val += spec_val[:,k] * Wg
                    spt_mas += sim_imag[:,k] * Wg
                    spt_mas_2 += sim_imag2[:,:,k] * Wg
                    spt_mas_3 += sim_imag3[:,:,k] * Wg
                    spt_gas += spec_ifu_g[:,k] * Wg
                    Wgt += Wg
            if Wgt == 0:
                Wgt = 1
            ifu[:,j,i] = spt_new / Wgt
            ifu_v[:,j,i] = spt_val / Wgt
            ifu_a[:,j,i] = spt_mas / Wgt
            ifu_b[:,:,j,i] = spt_mas_2 / Wgt
            ifu_c[:,:,j,i] = spt_mas_3 / Wgt
            ifu_g[:,j,i] = spt_gas / Wgt
            #ifu_imag[:,j,i] = spt_imag/Wgt
            if np.sum(np.sqrt(spt_err/Wgt**2.0)) == 0:
                ifu_e[:,j,i] = 1.0
            else:
                ifu_e[:,j,i] = np.sqrt(spt_err/Wgt**2.0)
            int_spect += spt_new/Wgt


    ifu_1 = np.where(ifu!=0., 1, 0)
    ifu_1[bad_pix_, :, :] = 0
    if include_gas:
        h1 = fits.PrimaryHDU(np.array(ifu+ifu_g, dtype=np.np.float32))
    else:
        h1 = fits.PrimaryHDU(np.array(ifu, dtype=np.float32))#.header
    h2 = fits.ImageHDU(np.array(ifu_e, dtype=np.float32))
    h3 = fits.ImageHDU(np.array(ifu_1, dtype=np.int16))
    h4 = fits.ImageHDU(np.array(ifu_g, dtype=np.float32))

    h = h1.header
    h['NAXIS'] = 3
    h['NAXIS3'] = nw 
    h['NAXIS1'] = nl
    h['NAXIS2'] = nl
    h['COMMENT'] = 'Mock '+ifutype+' IFU'
    h['CRVAL1'] = xo
    h['CD1_1'] = np.cos(thet*np.pi/180.)*pix_s/3600.
    h['CD1_2'] = np.sin(thet*np.pi/180.)*pix_s/3600.
    h['CRPIX1'] = nl/2
    h['CTYPE1'] = 'RA---TAN'
    h['CRVAL2'] = yo
    h['CD2_1'] = -np.sin(thet*np.pi/180.)*pix_s/3600.
    h['CD2_2'] = np.cos(thet*np.pi/180.)*pix_s/3600.
    h['CRPIX2'] = nl/2
    h['CTYPE2'] = 'DEC--TAN'
    h['CUNIT1'] = 'deg     '                                           
    h['CUNIT2'] = 'deg     '
    h['CDELT3'] = cdelt_w
    h['CRPIX3'] = crpix_w
    h['CRVAL3'] = crval_w
    h['CUNIT3'] = 'Wavelength [A]'
    h['RADECSYS'] = 'ICRS    '
    h['SYSTEM'] = 'FK5     '
    h['EQUINOX'] = 2000.00
    h['PSF'] = seeing
    h['FOV'] = Rifu*2.0
    h['KPCSEC'] = (dkpcs, 'kpc/arcsec')
    h['CAMX'] = 0
    h['CAMY'] = 0
    h['CAMZ'] = cam
    h['REDSHIFT'] = np.single(red_0)
    h['R'] = (sp_res,'Spectral Resolution')
    h['COSMO'] = cosmo.name
    h['H0'] = (cosmo.H0.value, cosmo.H0.unit)
    h['Omega_m'] = cosmo.Om0
    h['IFUCON'] = (str(np.int32(ns))+' ','NFibers')
    h['UNITS'] = '1E-16 erg/s/cm^2'
    h['WGAS'] = include_gas

    h = h2.header
    h['EXTNAME'] = 'ERROR'
    h['UNITS'] = '1E-16 erg/s/cm^2'

    h = h3.header
    h['EXTNAME'] = 'MASK'

    h = h4.header
    h['EXTNAME'] = 'GAS'  
    h['UNITS'] = '1E-16 erg/s/cm^2'

    hlist = fits.HDUList([h1, h2, h3, h4])
    hlist.update_extend()
    out_fit = dir_o + outf + '.fits'
    hlist.writeto(out_fit, overwrite=1)
    #dir_o1 = dir_o.replace(' ','\ ')
    out_fit1 = dir_o+outf+'.fits'
    compress_gzip(out_fit1)
    print('Datacube done.')

    ifu_v[0,:,:] = -2.5*np.log10(ifu_v[0,:,:]+0.0001)
    ifu_v[1,:,:] = np.log10(ifu_v[1,:,:]+1.0)
    ifu_v[30,:,:] = np.log10(ifu_v[30,:,:]+1.0)
    ifu_v[10,:,:] = np.log10(ifu_v[10,:,:]+1.0)
    ifu_v[29,:,:] = np.log10(ifu_v[29,:,:]+1.0)
    ifu_v[15,:,:] = -2.5*np.log10(ifu_v[15,:,:]+0.0001)
    ifu_v[17,:,:] = -2.5*np.log10(ifu_v[17,:,:]+0.0001)
    ifu_v[22,:,:] = -2.5*np.log10(ifu_v[22,:,:]+0.0001)
    ifu_v[23,:,:] = -2.5*np.log10(ifu_v[23,:,:]+0.0001)
    h1t = fits.PrimaryHDU(ifu_v)
    h2t = fits.ImageHDU(ifu_a)
    h3t = fits.ImageHDU(ifu_b)
    h4t = fits.ImageHDU(ifu_c)

    h = h1t.header
    h['NAXIS'] = 3
    h['NAXIS3'] = 35
    h['NAXIS1'] = nl
    h['NAXIS2'] = nl
    h['COMMENT'] = 'Real Values '+ifutype+' IFU'
    h['CRVAL1'] = 0
    h['CD1_1'] = np.cos(thet*np.pi/180.)*pix_s/3600.
    h['CD1_2'] = np.sin(thet*np.pi/180.)*pix_s/3600.
    h['CRPIX1'] = nl/2
    h['CTYPE1'] = 'RA---TAN'
    h['CRVAL2'] = 0
    h['CD2_1'] = -np.sin(thet*np.pi/180.)*pix_s/3600.
    h['CD2_2'] = np.cos(thet*np.pi/180.)*pix_s/3600.
    h['CRPIX2'] = nl/2
    h['CTYPE2'] = 'DEC--TAN'
    h['CUNIT1'] = 'deg     '                                           
    h['CUNIT2'] = 'deg     '
    h['Type0'] = ('Av_T    ','Mag')
    h['Type1'] = ('MASS    ','log10(Msun)')
    h['Type2'] = ('VEL     ','km/s')
    h['Type3'] = ('SFR     ','Msun/yr')
    h['Type4'] = ('DUST_G  ','Av BETA')
    h['Type5'] = ('DUST_T  ','Av BETA')
    h['Type6'] = ('DUST_Av ','Av BETA')
    h['Type7'] = ('DISP    ','km/s')
    h['Type8'] = ('aML     ','Msun/Lsun BETA')
    h['Type9'] =  ('tML     ','Msun/Lsun BETA')
    h['Type10'] = ('LUM     ','log10(Lsun)')
    h['Type11'] = ('Z_lw    ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type12'] = ('Z_mw    ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type13'] = ('AGE_lw  ','Gyr')
    h['Type14'] = ('AGE_mw  ','Gyr')
    h['Type15'] = ('Av_lw   ','Mag')
    h['Type16'] = ('FLUX    ','1e-16 ergs/s/cm2')
    h['Type17'] = ('Av_fw   ','Mag')
    h['Type18'] = ('VEL_lw  ','km/s')
    h['Type19'] = ('VEL_fw  ','km/s')
    h['Type20'] = ('DIS_lw   ','km/s BETA')
    h['Type21'] = ('DIS_fw   ','km/s BETA')
    h['Type22'] = ('Av_lw_g  ','Mag')
    h['Type23'] = ('Av_fw_g  ','Mag')
    h['Type24'] = ('VEL_lw_g ','km/s')
    h['Type25'] = ('VEL_fw_g ','km/s')
    h['Type26'] = ('DIS_l_gas','km/s BETA')
    h['Type27'] = ('DIS_f_gas','km/s BETA')
    h['Type28'] = ('FLUX_gas','1e-16 ergs/s/cm2 Bolometric')
    h['Type29'] = ('LUM_gas ','log10(Lsun) Bolometric')
    h['Type30'] = ('MASS_fw ','log10(Msun) BETA')
    h['Type31'] = ('AGE_fw  ','Gyr')
    h['Type32'] = ('AGE_mfw ','Gyr BETA')
    h['Type33'] = ('Z_lw_gas ','log10(Z/H) add 10.46 to convert to 12+log10(O/H) or add 1.77 to convert to log10(Z/Z_sun)')
    h['Type34'] = ('Z_fw_gas ','log10(Z/H) add 10.46 to convert to 12+log10(O/H) or add 1.77 to convert to log10(Z/Z_sun)')#8.69 
    h['Type35'] = ('Z_fw    ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type36'] = ('Z_mfw    ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type37'] = ('Z_lw_assigned ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type38'] = ('AGE_lw_assigned  ','Gyr')
    h['Type39'] = ('Z_mw_assigned ','log10(Z/H) add 1.77 to convert to log10(Z/Z_sun)')
    h['Type40'] = ('AGE_mw_assigned  ','Gyr')
    h['RADECSYS'] = 'ICRS    '
    h['SYSTEM'] = 'FK5     '
    h['EQUINOX'] = 2000.00
    h['PSF'] = seeing
    h['FOV'] = Rifu*2.0
    h['KPCSEC'] = (dkpcs, 'kpc/arcsec')
    h['CAMX'] = 0
    h['CAMY'] = 0
    h['CAMZ'] = cam
    h['REDSHIFT'] = np.single(red_0)
    h['R'] = (sp_res,'Spectral Resolution')
    h['COSMO'] = cosmo.name
    h['H0'] = (cosmo.H0.value, cosmo.H0.unit)
    h['Omega_m'] = cosmo.Om0

    h = h2t.header
    h['EXTNAME'] = 'MASS_PER_AGE'
    h['UNITS'] = 'Msun'
    h['CRVAL1'] = 0#oap
    h['CD1_1'] = np.cos(thet*np.pi/180.)*pix_s/3600.
    h['CD1_2'] = np.sin(thet*np.pi/180.)*pix_s/3600.
    h['CRPIX1'] = nl/2
    h['CTYPE1'] = 'RA---TAN'
    h['CRVAL2'] = 0#oap
    h['CD2_1'] = -np.sin(thet*np.pi/180.)*pix_s/3600.
    h['CD2_2'] = np.cos(thet*np.pi/180.)*pix_s/3600.
    h['CRPIX2'] = nl/2
    h['CTYPE2'] = 'DEC--TAN' 
    h['CTYPE3'] = 'AGE'
    for kk in range(0, n_ages):
        h['AGE'+str(kk)] = ages_r[kk]

    h = h3t.header
    h['EXTNAME'] = 'MASS_PER_AGE_MET'
    h['UNITS'] = 'Msun'
    h['CRVAL1'] = 0#oap
    h['CD1_1'] = np.cos(thet*np.pi/180.)*pix_s/3600.
    h['CD1_2'] = np.sin(thet*np.pi/180.)*pix_s/3600.
    h['CRPIX1'] = nl/2
    h['CTYPE1'] = 'RA---TAN'
    h['CRVAL2'] = 0#oap
    h['CD2_1'] = -np.sin(thet*np.pi/180.)*pix_s/3600.
    h['CD2_2'] = np.cos(thet*np.pi/180.)*pix_s/3600.
    h['CRPIX2'] = nl/2
    h['CTYPE2'] = 'DEC--TAN' 
    h['CTYPE3'] = 'METALLICITY'
    h['CTYPE4'] = 'AGE'
    for kk in range(0, n_ages):
        h['AGE'+str(kk)] = (ages_r[kk], 'Gyr')
    for jj in range(0, n_mets):
        h['MET'+str(jj)] = (met_ssp3[jj], 'Z/H')

    h = h4t.header
    h['EXTNAME'] = 'LUM_PER_AGE_MET'
    h['UNITS'] = 'Lsun' 
    h['CRVAL1'] = 0#oap
    h['CD1_1'] = np.cos(thet*np.pi/180.)*pix_s/3600.
    h['CD1_2'] = np.sin(thet*np.pi/180.)*pix_s/3600.
    h['CRPIX1'] = nl/2
    h['CTYPE1'] = 'RA---TAN'
    h['CRVAL2'] = 0#oap
    h['CD2_1'] = -np.sin(thet*np.pi/180.)*pix_s/3600.
    h['CD2_2'] = np.cos(thet*np.pi/180.)*pix_s/3600.
    h['CRPIX2'] = nl/2
    h['CTYPE2'] = 'DEC--TAN'   
    h['CTYPE3'] = 'METALLICITY Z/H'
    h['CTYPE4'] = 'AGE Gyr'
    for kk in range(0, n_ages):
        h['AGE'+str(kk)] = (ages_r[kk], 'Gyr')
    for jj in range(0, n_mets):
        h['MET'+str(jj)] = (met_ssp3[jj], 'Z/H')

    hlist1 = fits.HDUList([h1t, h2t, h3t, h4t])
    hlist1.update_extend()
    out_fit = dir_o + outf + '_val.fits'
    hlist1.writeto(out_fit, overwrite=1)
    #dir_o1 = dir_o.replace(' ','\ ')
    out_fit1 = dir_o + outf + '_val.fits'
    compress_gzip(out_fit1)
    print('Cube_val done.')

def mk_intrinsic_assigned_maps(indir, outdir, star_file, template_SSP, fib_n=7, red_0=0.01, ifutype='MaNGA', psfi=0):
    """ 
    Reads stellar particle file and creates LW- and MW-age and -metallicity maps 
    of based on the assigned properties from the stellar template selected.

    Arguments:
    ----------
    indir: directory where the stellar particle file is saved (string).
    outdir: directory where the output file is saved (string).
    star_file: name of the stellar particle file (string). 
    fib_n (=7): fiber number radius, defines de IFU size. (integer in [3,4,5,6,7])
    red_0 (=0.01):redshift at which the galaxy is placed. (float)
    template_SSP: SSP template file name to produce the stellar 
                  spectra. (string)
    ifutype (='MaNGA'): instrument properties (string).
    psfi (=0): instantaneous point spread function (PSF as FWHM [arcsec]). (float)
    
    Returns:
    -------
    -

    Outputs:
    FITS file containing stack of average LW- and MW-age and -metallicity maps 
    of based on the assigned properties from the stellar template selected.

    """
    data = np.genfromtxt(indir + star_file)
    x_b = data[:, 0]
    y_b = data[:, 1]
    z_b = data[:, 2]
    age_b = data[:, 6]
    met_b = data[:, 7]
    mass_b = data[:, 8]

    ns = 3 * fib_n * (fib_n-1) + 1
    idfib = str(ns)
    cubef = 'TNG50-'+str(star_file[5:7])+'-'+str(star_file[13:-12])+'-'+str(star_file[-11:-10])+'-'+idfib +'.cube_val_asign'
    print(ns, 'fibers')

    vel_light = 299792.458 #[km/s]
    no_nan = 0
    if 'MaNGA' in ifutype:
        sp_res = 2000.0
        pix_s = 0.5#arcsec
        scp_s = 60.4#microns per arcsec
        fibA = 150.0
        fibB = 120.0
        sigma_inst = 25.0
        beta_s = 2.0
        if psfi <= 0:
            seeing = 1.43
        else:
            seeing = psfi
        if sp_res <= 0:
            sp_res = 2000.0
    else:
        pix_s = 0.5#arcsec
        scp_s = 60.4#microns per arcsec
        fibA = 150.0
        fibB = 120.0     
        sigma_inst = 25.0
        beta_s = 2.0
        if psfi == 0:
            seeing = 1.43
        else:
            seeing = psfi   
    scalep = 1.0 / scp_s
    
    dkpcs = cosmo.kpc_proper_per_arcmin(red_0).value / 60.
    cam = cosmo.lookback_time(red_0).value * 1e6 * 0.307
    #cam = cosmo.comoving_distance(red_0).value*1e3
    #dkpcs = cam*(1./3600.)*(np.pi/180.)
    dap = scalep
    rad = np.sqrt(x_b**2.+y_b**2.+(cam-z_b)**2.)
    d_r = 0.10/dkpcs
    reds = reds_cos(rad/1e3)
    radA = rad/(1+reds)
    radL = np.array(rad*(1+reds)*(3.08567758e19*100))
    phi = np.arcsin(x_b/radA)
    the = np.arcsin(y_b/(radA*np.cos(phi)))
    the = the * 180 / np.pi * 3600#+ran.randn(len(rad))*2.0
    phi = phi * 180 / np.pi * 3600#+ran.randn(len(rad))*2.0
    Dfib = fibA*scalep
    Rifu = Dfib*((2.0*fib_n-1.0)/2.0-0.5)
    xfib0 = -Rifu
    yfib0 = 0
    dxf = 1.0
    dyf = np.sin(60.*np.pi/180.)
    xifu = np.zeros(ns)
    yifu = np.zeros(ns)
    ini = 0
    for ii in range(0, fib_n):
        nt = fib_n*2-1-ii
        yfib = yfib0+ii*dyf*Dfib
        for jj in range(0, nt):
            xfib = xfib0 + (jj*dxf+0.5*ii) * Dfib
            xifu[ini] = xfib
            yifu[ini] = yfib
            ini = ini + 1

        if ii > 0:
            for jj in range(0, nt):
                xfib = xfib0 + (jj*dxf+0.5*ii) * Dfib
                xifu[ini] = xfib
                yifu[ini] = -yfib
                ini = ini+1

    ndt = 35
    dyf = 1.0
    dyt = Dfib/2.0/np.cos(30.0*np.pi/180.0)
    dxt = Dfib/2.0
    ndt = 3
    dit = np.zeros([ndt,2])
    dit[0,:] = [+0.00,+0.00]#+ran.randn(2)*0.025
    dit[1,:] = [+0.00,+dyt/1.0]#+ran.randn(2)*0.025
    dit[2,:] = [-dxt, +dyt/2.0]#+ran.randn(2)*0.025
    
    ssp_template, wave, age_ssp, met_ssp, ml_ssp, crval_w, cdelt_w, \
                    crpix_w = ssp_extract(template_SSP)
    n_lib_mod = ssp_template.shape[0]
    in_ssp = associate_ssp(age_ssp, met_ssp, age_b, met_b)

    met_ligt_asig = np.zeros(ns*3)
    met_mas_asig = np.zeros(ns*3)
    age_ligt_asig = np.zeros(ns*3)
    age_mas_asig = np.zeros(ns*3)
    xo = np.zeros(ns*3)
    yo = np.zeros(ns*3)
    Lt = np.zeros(ns*3)
    mass_t = np.zeros(ns*3)
    print('starting dither')

    for i in range(3):
        for j in range(ns):
            con = i * ns + j
            x_ifu = 0.
            y_ifu = 0.   
            seeing2d_s = np.random.multivariate_normal(np.array([0,0]), \
                         np.array([[seeing/2.0/np.sqrt(2*np.log(2)),0], \
                         [0,seeing/2.0/np.sqrt(2*np.log(2))]]), len(rad))
            phie = phi + seeing2d_s[:,0]
            thee = the + seeing2d_s[:,1] 
            dyf = 1.0
            xo[con] = xifu[j] + dit[i,0]
            yo[con] = yifu[j] + dyf*dit[i,1]    
            s_box = np.arange(len(phie))[(np.abs(xo[con]-phie)<=fibB*scalep/2.0) & (np.abs(yo[con]-thee)<=fibB*scalep/2.0)]
            r = cdist(np.stack((phie[s_box],thee[s_box])).T, [(xo[con], yo[con])])[:, 0]
            nt = s_box[r <= fibB*scalep/2.0]
            #r = np.sqrt((xo[con]-phie)**2.0+(yo[con]-thee)**2.0)
            #nt = np.where(r <= fibB*scalep/2.0)[0]
            mass_t = np.zeros(ns*3)    

            if len(nt) > 0:
                mass_t[con] = np.sum(mass_b[nt])
                for k in range(0, len(nt)):
                    if not np.isnan(in_ssp[nt[k]]):
                        if in_ssp[nt[k]] > 0 and in_ssp[nt[k]] < n_lib_mod:
                            Lt[con] += mass_b[nt[k]] / ml_ssp[in_ssp[nt[k]]]
                            met_ligt_asig[con] += np.log10(met_ssp[in_ssp[nt[k]]]) * mass_b[nt[k]] / ml_ssp[in_ssp[nt[k]]]
                            met_mas_asig[con] += np.log10(met_ssp[in_ssp[nt[k]]]) * mass_b[nt[k]]
                            age_ligt_asig[con] += np.log10(age_ssp[in_ssp[nt[k]]]) * mass_b[nt[k]] / ml_ssp[in_ssp[nt[k]]]
                            age_mas_asig[con] += np.log10(age_ssp[in_ssp[nt[k]]]) * mass_b[nt[k]]                            
                if Lt[con] > 0:
                    met_ligt_asig[con] /= Lt[con]
                    age_ligt_asig[con] = 10.0**(age_ligt_asig[con]/Lt[con])
                    met_mas_asig[con] /= mass_t[con]
                    age_mas_asig[con] = 10.0**(age_mas_asig[con]/mass_t[con])

    x_ifu = xo
    y_ifu = yo

    rss_asig = np.stack((met_ligt_asig, age_ligt_asig, met_mas_asig, age_mas_asig))
    
    print(rss_asig.shape)
    print('starting regrid')

    nl = np.int32(round((np.amax([np.amax(x_ifu), -np.amin(x_ifu), np.amax(y_ifu), -np.amin(y_ifu)])+1)*2/pix_s))
    ifu_asig = np.ones([4, nl, nl])

    sigma_rec  =  0.7
    xo = -nl/2*pix_s
    yo = -nl/2*pix_s
    xi = xo
    xf = xo
    for i in range(0, nl):
        xi = xf
        xf = xf+pix_s
        yi = yo
        yf = yo
        for j in range(0, nl):
            yi = yf
            yf += pix_s
            prop_val = np.zeros(4)
            Wgt = 0
            for k in range(0, len(x_ifu)):
                V1 = np.sqrt((x_ifu[k]-xi)**2.0 + (y_ifu[k]-yf)**2.0)
                V2 = np.sqrt((x_ifu[k]-xf)**2.0 + (y_ifu[k]-yf)**2.0)
                V3 = np.sqrt((x_ifu[k]-xi)**2.0 + (y_ifu[k]-yi)**2.0)
                V4 = np.sqrt((x_ifu[k]-xf)**2.0 + (y_ifu[k]-yi)**2.0)
                Vt = np.array([V1,V2,V3,V4])
                Rsp = np.sqrt((x_ifu[k]-(xf+xi)/2.0)**2.0+(y_ifu[k]-(yf+yi)/2.0)**2.0)
                Vmin = np.amin(Vt)
                Vmax = np.amax(Vt)
                #if Vmin <= fibB*scalep/2.0:
                #    if Vmax <= fibB*scalep/2.0:
                #        Wg=(pix_s)**2.0/(np.pi*(fibB*scalep/2.0)**2.0)
                #    else:
                #        Wg=(1.0-(Vmax-fibB*scalep/2.0)/(np.sqrt(2.0)*pix_s))*(pix_s)**2.0/(np.pi*(fibB*scalep/2.0)**2.0)
                #    spt_new=spec_ifu[:,k]*Wg+spt_new
                #    spt_err=(spec_ifu_e[:,k]*Wg)**2.0+spt_err**2.0
                #    Wgt=Wgt+Wg
                if Rsp <= fibA*scalep*1.4/2.0: #1.6 arcsec
                    #Wg = np.exp(-(Rsp/pix_s)**2.0/2.0)
                    Wg = np.exp(-(Rsp/sigma_rec)**2.0/2.0)
                    #Wg = 1.0
                    prop_val += rss_asig[:,k] * Wg
                    Wgt += Wg
            if Wgt == 0:
                Wgt = 1
            ifu_asig[:,j,i] = prop_val / Wgt

    h_asig = fits.PrimaryHDU(np.array(ifu_asig, dtype=np.float32))
    h = h_asig.header
    h['TYPE0'] = ('LW metallicity', 'log10(Z) add 1.77 to convert to log10(Z/Z_sun)')
    h['TYPE1'] = ('LW age', 'Gyr')
    h['TYPE2'] = ('MW metallicity', 'log10(Z) add 1.77 to convert to log10(Z/Z_sun)')
    h['TYPE3'] = ('MW age', 'Gyr')
    hlist1 = fits.HDUList([h_asig])
    hlist1.update_extend()
    out_fit = outdir + cubef + '.fits'
    hlist1.writeto(out_fit, overwrite=1)
    #dir_o1 = outdir.replace(' ','\ ')
    out_fit1 = outdir + cubef + '.fits'
    compress_gzip(out_fit1)
    print('Cube_assig done.')

