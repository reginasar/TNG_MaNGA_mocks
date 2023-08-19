# iterative_bubble_match function defines an algorithm to find a matching
# sample for a given sample1. A counterpart is associated to every point 
# in sample1 -if possible-. To minimize the euclidean distance of the 
# points with their counterparts repeated counterparts are optional.

import numpy as np
from astropy.io import fits
from scipy.spatial.distance import cdist
from collections import Counter


def iterative_bubble_match(sample1, sample2, max_step=40, mr_lim=[], \
                    radius=1., r_increase=0., more_than=1, subsample=False):
    orig_samp_ids = np.arange(sample1.shape[0], dtype=np.int32)
    item_ids = np.arange(sample2.shape[0], dtype=np.int32)
    non_rep = []
    unas = []

    sample2_norm = np.zeros_like(sample2)
    sample1_norm = np.zeros_like(sample1)
    n_coord = sample2.shape[1]
    for ii in range(n_coord):
        sample2_norm[:, ii] = np.copy(sample2[:, ii])/mr_lim[ii]
        sample1_norm[:, ii] = np.copy(sample1[:, ii])/mr_lim[ii]

    #sample_match, sample_match_labels = bubble_sample_match_fixedZ(manga_mr, tng_pack_,\
    #            m_lim=m_lim, r_lim=r_lim, random=random, seed=seed, radius=radius)
    sample_match, dist_match = bubble_sample_match(sample1_norm, sample2_norm,\
                item_ids=item_ids, mr_lim=mr_lim, random=False, seed=seed, radius=radius)
    #print(manga_mzr.shape[0])
    #non_rep.extend(np.array([np.intersect1d(sample_match[sample_match[:,0]==ii,1],\
    #        sample_match[sample_match[:,0]==ii,1]).size for ii in range(87,99)]))
    non_rep.extend(np.intersect1d(sample_match, sample_match))

    iterate = True
#    while(iterate==True):
    rep_sample2_all = []
    for kk in range(max_step):
        rep_sample1, rep_sample2 = repeated_items(orig_samp_ids[sample_match!=-1], \
                sample_match[sample_match!=-1], dist_match[sample_match!=-1], more_than=more_than)
        rep_sample1 = np.concatenate((rep_sample1, orig_samp_ids[sample_match==-1]))
        unas.append(rep_sample1.size)

        rep_sample2_all.extend(rep_sample2)
        #tng_pack = get_tng_params(tng_sim=tng_sim)
        nonrep_sample2_ids = get_sample2_rest(rep_sample2_all, item_ids)
        try:
            sample_match_2, dist_match_2 = bubble_sample_match(sample1_norm[rep_sample1],\
                sample2_norm[nonrep_sample2_ids], item_ids=item_ids[nonrep_sample2_ids], \
                mr_lim=mr_lim, random=False, seed=seed, radius=radius*(1+r_increase*kk)) #z_lim*(0.05*kk+1), r_lim*(0.02*kk+1)
            non_rep.extend(np.intersect1d(sample_match_2,sample_match_2))
            sample_match[rep_sample1] = sample_match_2
            dist_match[rep_sample1] = dist_match_2
            #sample_match_labels[rep_sample1,:] = sample_match_labels_2
        except IndexError:
            print('No more items in sample2 within range')


    rep_sample1, rep_sample2 = repeated_items(orig_samp_ids[sample_match!=-1], \
                sample_match[sample_match!=-1], dist_match[sample_match!=-1], more_than=more_than)
    rep_sample1 = np.concatenate((rep_sample1, orig_samp_ids[sample_match==-1]))
    #rep_manga_ind = np.array(np.concatenate((rep_manga_ind, np.nonzero(sample_match[:,0]==0)[0])), dtype=np.int16)
    rep_sample1 = np.array(rep_sample1, dtype=np.int32)
    sample_match[rep_sample1] = -1
    dist_match[rep_sample1] = -1

    return sample_match, dist_match, non_rep, unas


def bubble_sample_match(sample1_norm, sample2_norm, item_ids, mr_lim=[0.1], random=False,\
                        seed=0, radius=1.):
    n_coord = sample2_norm.shape[1]
    total_gal = sample1_norm.shape[0]
    sample_match = - np.ones((total_gal), dtype=np.int32)#like item_ids
    sample_match_dist = - np.ones((total_gal), dtype=np.float32) #mass, tng_z, manga_z, re_arc_tng, distance_selec

    rng = np.random.default_rng(seed)

    for ii in range(total_gal):
        box_indices = np.nonzero(np.abs(sample2_norm[:, 0]-sample1_norm[ii, 0])<=radius)[0]
        for jj in range(1, n_coord):
            box_indices = np.intersect1d(box_indices, np.nonzero(np.abs(sample2_norm[:, jj]-sample1_norm[ii, jj])<=radius)[0])

        if box_indices.size>0:
            sample2_box = sample2_norm[box_indices]
            item_ids_box = item_ids[box_indices]
            distances_box = cdist(sample2_box, sample1_norm[np.newaxis, ii, :], 'euclidean')[:,0]
            if distances_box.min()<radius:
                if not random:
                    selected_gal_ind = np.argmin(distances_box)
                    sample_match[ii] = item_ids_box[selected_gal_ind]
                    sample_match_dist[ii] = distances_box[selected_gal_ind]
                else:
                    bubble_restrict = np.nonzero(distances_box<1.)[0]
                    selected_gal_ind = rng.permutation(bubble_restrict)
                    sample_match[ii] = item_ids_box[bubble_restrict[selected_gal_ind]]
                    sample_match_dist[ii] = distances_box[bubble_restrict[selected_gal_ind]]

            else:
                print('no bubble, No matches found near original item n', ii)
        else:
            print('no box, No matches found near original item n', ii)

    return sample_match, sample_match_dist#, sample_match_labels



def repeated_items(orig_samp_ids, item_ids, distances, more_than=1):
    """ 
    After a bubble selection, this function finds the items that are repeated
    more_than X times in order to include/remove them in the next matching iteration
    Returns 
        manga index: non prioritary galaxies in the match
        snap, subhalo: tng galaxies that have "more_than"-times
    """
    fsample_items_to_remove = []
    ssample_items_to_remove = []
    #for snap_ii in tng_snap:
    #    in_snap = np.nonzero(snap==snap_ii)[0]
    rep_items = np.array([item for item, count in Counter(item_ids).items() if count > more_than])
        #print(rep_subhalos)
    for jj in range(rep_items.size):
#            print('holis')
            index_multi_item = np.nonzero(rep_items[jj]==item_ids)[0]
            #print(index_multi_subhalo.shape)
            fsample_items_to_remove.extend(orig_samp_ids[index_multi_item[np.argsort(distances[index_multi_item])[more_than:]]])
            ssample_items_to_remove.extend([rep_items[jj]])

    return np.array(fsample_items_to_remove), np.array(ssample_items_to_remove)




def get_sample2_rest(rep_sample2_ids, item_ids):
    remaining_ids = np.copy(item_ids)
    for ii in range(len(rep_sample2_ids)):
        remaining_ids = np.delete(remaining_ids,\
            np.nonzero(remaining_ids==rep_sample2_ids[ii])[0])

    return remaining_ids


# Setup example, as defined in the publication
# selected_tng array saves de snapshot and subhalo IDs of the matching 
# counterparts, in the sample order as the original sample (MaNGA in this case)


snap_id = [99, 98, 97, ...] # snapshot numbers
tng_pack = {99: [...], 98: [...]} # dictionary with all possible galaxies 
           # separated by snapshot as keys, each snapshot key should be associated 
           # to an array (N, 3) with N TNG galaxies and their Mass(logscale), 
           # radius(logscale) and subhalo_id

snap_redshift = [] # array with redshift corresponding to snapshot (order as snap_id)
snapz_bins = snap_redshift[:-1] + np.diff(snap_redshift) / 2. #snapshot bins to match the samples
manga_mrz = # array (N, 3) with N manga galaxies and their Mass, radius and redshift
            # same units as tng_pack

selected_tng = - np.ones((manga_mrz.shape[0], 2), dtype=np.int32)
manga_need_pair = np.ones(manga_mrz.shape[0], dtype=bool)
snap_ind_manga = np.digitize(manga_mrz[:,2], snapz_bins)
for ii in range(15):
    for hh, snap in enumerate(snap_id):
        manga_index_to_match = np.array((snap_ind_manga==hh) & manga_need_pair)
        manga_mr = manga_mrz[manga_index_to_match, :2]
        sample_match, dist_match, non_rep, unas =\
            iterative_bubble_match(manga_mr, tng_pack[snap][:, :2], \
              max_step=40, mr_lim=[0.25, 0.25], radius=1., \
              r_increase=0., more_than=1, subsample=False)
        selected_tng[manga_index_to_match, 0] = np.where(sample_match!=-1, snap, -1)
        selected_tng[manga_index_to_match, 1] = \
            np.where(sample_match!=-1, tng_pack[snap][sample_match, 2], -1)
        manga_need_pair = np.where(selected_tng[:, 0]==-1, True, False)



