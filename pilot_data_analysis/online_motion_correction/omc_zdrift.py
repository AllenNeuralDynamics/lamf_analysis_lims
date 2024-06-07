from pathlib import Path
from pystackreg import StackReg
import tifffile
import time
import numpy as np
from glob import glob
import h5py
import cv2
from dask import delayed, compute
from dask.distributed import Client

from argparse import ArgumentParser
parser = ArgumentParser(description='arguments for offline mesoscope data splitting registration and EMF saving')
parser.add_argument(
    '--file_path',
    type=str,
    default='',
    metavar='file_path',
    help='file path to the tiff file'
)
parser.add_argument(
    '--zstack_dir',
    type=str,
    default='',
    metavar='zstack_dir',
    help='directory with local z-stacks'
)


def get_matched_zstack(emf_fn, ops_fn, zstack_dir, num_planes_around=60):
    ''' 
    
    
    Notes
    - Rolling average of z-stacks was not enough.
    '''
    ops = np.load(ops_fn, allow_pickle=True).item()
    y_roll_bottom = np.min(ops['reg_result'][4][0])
    y_roll_top = np.max(ops['reg_result'][4][0])
    x_roll_right = np.min(ops['reg_result'][4][1])
    x_roll_left = np.max(ops['reg_result'][4][1])
    if y_roll_bottom >= 0:
        y_roll_bottom = -1
    if x_roll_right >= 0:
        x_roll_right = -1

    zstack_fn_list = glob(str(zstack_dir /'ophys_experiment_*_local_z_stack.tiff'))
    center_zstacks = []
    for zstack_fn in zstack_fn_list:
        zstack = tifffile.imread(zstack_fn)
        zstack = med_filt_z_stack(zstack)
        new_zstack = rolling_average_zstack(zstack)
        center_ind = int(np.floor(new_zstack.shape[0]/2))
        center_zstack = new_zstack[center_ind - num_planes_around//2 : center_ind + num_planes_around//2+1]
        center_zstack = center_zstack[:, y_roll_top:y_roll_bottom, x_roll_left:x_roll_right]
        center_zstacks.append(center_zstack)
    first_emf = tifffile.imread(emf_fn)[0, y_roll_top:y_roll_bottom, x_roll_left:x_roll_right]    

    corrcoef_zstack_finding = []
    tmat_zstack_finding = []
    for zstack in center_zstacks:
        _, corrcoef, _, best_tmat = get_correlation_after_reg(first_emf, zstack)
        corrcoef_zstack_finding.append(corrcoef)
        tmat_zstack_finding.append(best_tmat)
    corrcoef_zstack_finding = np.vstack(corrcoef_zstack_finding)
    matched_ind_zstack = np.argmax(np.mean(corrcoef_zstack_finding, axis=1))

    return matched_ind_zstack, zstack_fn_list, corrcoef_zstack_finding, tmat_zstack_finding

    # assert first_emf.min() > 0
    # valid_pix_threshold = first_emf.min()/10
    # num_pix_threshold = first_emf.shape[0] * first_emf.shape[1] / 2

    # first_emf_clahe = image_normalization_uint16(first_emf)

    # sr = StackReg(StackReg.AFFINE)
    # corrcoef = np.zeros((len(center_zstacks), center_zstacks[0].shape[0]))
    
    # for i, zstack in enumerate(center_zstacks):
    #     temp_cc = []
    #     tmat_list = []
    #     for j, zstack_plane in enumerate(zstack):
    #         zstack_plane_clahe = image_normalization_uint16(zstack_plane)
    #         tmat = sr.register(zstack_plane_clahe, first_emf_clahe)
    #         emf_reg = sr.transform(first_emf, tmat=tmat)            
    #         valid_y, valid_x = np.where(emf_reg > valid_pix_threshold)
    #         if len(valid_y) > num_pix_threshold:
    #             temp_cc.append(np.corrcoef(zstack_plane.flatten(), emf_reg.flatten())[0,1])
    #             tmat_list.append(tmat)
    #         else:
    #             temp_cc.append(0)
    #             tmat_list.append(np.eye(3))
    #     temp_ind = np.argmax(temp_cc)
    #     best_tmat = tmat_list[temp_ind]
    #     emf_reg = sr.transform(first_emf, tmat=best_tmat)       
    #     for j, zstack_plane in enumerate(zstack):
    #         corrcoef[i,j] = np.corrcoef(zstack_plane.flatten(), emf_reg.flatten())[0,1]
    # matched_ind = np.argmax(np.mean(corrcoef, axis=1))

    # return matched_ind, zstack_fn_list, corrcoef


def get_correlation_after_reg(fov, zstack, use_clahe=True, sr_method='affine', tmat=None):
    if use_clahe:
        fov_for_reg = image_normalization_uint16(fov)
        zstack_for_reg = np.zeros(zstack.shape)
        for zi in range(zstack.shape[0]):
            temp_zplane = image_normalization_uint16(zstack[zi])
            zstack_for_reg[zi] = temp_zplane
    else:
        fov_for_reg = fov.copy()
        zstack_for_reg = zstack.copy()
    
    if sr_method == 'affine':
        sr = StackReg(StackReg.AFFINE)
    elif sr_method == 'rigid_body':
        sr = StackReg(StackReg.RIGID_BODY)
    else:
        raise ValueError('"sr_method" should be either "affine" or "rigid_body"')
    
    assert fov.min() > 0
    # valid_pix_threshold = fov.min()/10
    valid_pix_threshold = 0
    num_pix_threshold = fov.shape[0] * fov.shape[1] / 2
    
    corrcoef = np.zeros(zstack.shape[0])
    
    if tmat is None:
        temp_cc = []
        tmat_list = []
        for zi in range(zstack_for_reg.shape[0]):
            zstack_plane_clahe = zstack_for_reg[zi]
            zstack_plane = zstack[zi]
            tmat = sr.register(zstack_plane_clahe, fov_for_reg)
            fov_reg = sr.transform(fov, tmat=tmat)            
            valid_y, valid_x = np.where(fov_reg > valid_pix_threshold)
            if len(valid_y) > num_pix_threshold:
                temp_cc.append(np.corrcoef(zstack_plane[valid_y, valid_x].flatten(),
                                           fov_reg[valid_y, valid_x].flatten())[0,1])
                tmat_list.append(tmat)
            else:
                temp_cc.append(0)
                tmat_list.append(np.eye(3))
        temp_ind = np.argmax(temp_cc)
        best_tmat = tmat_list[temp_ind]
    else:
        best_tmat = tmat
    fov_reg = sr.transform(fov, tmat=best_tmat)
    valid_y, valid_x = np.where(fov_reg > valid_pix_threshold)
    for zi, zstack_plane in enumerate(zstack):
        corrcoef[zi] = np.corrcoef(zstack_plane[valid_y, valid_x].flatten(),
                                   fov_reg[valid_y, valid_x].flatten())[0,1]
    matched_ind = np.argmax(corrcoef)

    return matched_ind, corrcoef, fov_reg, best_tmat


def calculate_zdrift(emf_fn, ops_fn, zstack_fn, tmat=None):
    ops = np.load(ops_fn, allow_pickle=True).item()
    y_roll_top = np.max(ops['reg_result'][4][0])
    y_roll_bottom = np.min(ops['reg_result'][4][0])    
    x_roll_left = np.max(ops['reg_result'][4][1])
    x_roll_right = np.min(ops['reg_result'][4][1])
    if y_roll_bottom >= 0:
        y_roll_bottom = -1
    if x_roll_right >= 0:
        x_roll_right = -1

    emf = tifffile.imread(emf_fn)[:, y_roll_top:y_roll_bottom, x_roll_left:x_roll_right]
    zstack = tifffile.imread(zstack_fn)[:, y_roll_top:y_roll_bottom, x_roll_left:x_roll_right]
    zstack = med_filt_z_stack(zstack)
    new_zstack = rolling_average_zstack(zstack)
    
    corrcoef = np.zeros((len(emf), zstack.shape[0]))
    tmat_list = []
    fov_reg_list = []
    for i, fov in enumerate(emf):
        _, corrcoef_temp, fov_reg, best_tmat = get_correlation_after_reg(fov, new_zstack, tmat=tmat)
        tmat_list.append(best_tmat)
        corrcoef[i,:] = corrcoef_temp
        fov_reg_list.append(fov_reg)
    matched_inds = np.argmax(corrcoef, axis=1)
    tmat_array = np.array(tmat_list)
    fov_reg_array = np.array(fov_reg_list)

    # assert emf.min() > 0
    # valid_pix_threshold = emf.min()/10
    # num_pix_threshold = emf.shape[1] * emf.shape[2] / 2

    # sr = StackReg(StackReg.AFFINE)
    # tmat_list = []
    # fov_reg_list = []
    # for i, fov in enumerate(emf):
    #     temp_tmat = []
    #     temp_cc = []
    #     for j, zplane in enumerate(new_zstack):
    #         tmat = sr.register(zplane, fov)
    #         fov_reg = sr.transform(fov, tmat=tmat)
    #         valid_y, valid_x = np.where(fov_reg > valid_pix_threshold)
    #         if len(valid_y) > num_pix_threshold:
    #             temp_cc.append(np.corrcoef(zplane[valid_y, valid_x].flatten(),
    #                                     fov_reg[valid_y, valid_x].flatten())[0,1])
    #             temp_tmat.append(tmat)
    #         else:
    #             temp_cc.append(0)
    #             temp_tmat.append(np.eye(3))
    #     temp_ind = np.argmax(temp_cc)
    #     tmat = temp_tmat[temp_ind]
    #     tmat_list.append(tmat)
    #     fov_reg = sr.transform(fov, tmat=tmat)
    #     fov_reg_list.append(fov_reg)
    #     valid_y, valid_x = np.where(fov_reg > valid_pix_threshold)
    
    #     for j, zplane in enumerate(new_zstack):
    #         corrcoef[i,j] = np.corrcoef(zplane[valid_y, valid_x].flatten(),
    #                                     fov_reg[valid_y, valid_x].flatten())[0,1]
    # matched_inds = np.argmax(corrcoef, axis=1)
    # tmat_array = np.array(tmat_list)
    # fov_reg_array = np.array(fov_reg_list)

    return matched_inds, corrcoef, tmat_array, fov_reg_array


def rolling_average_zstack(zstack, rolling_window_flank=2):
    new_zstack = np.zeros(zstack.shape)
    for i in range(zstack.shape[0]):
        new_zstack[i] = np.mean(zstack[max(0, i-rolling_window_flank) : min(zstack.shape[0], i+rolling_window_flank), :, :],
                                axis=0)
    return new_zstack


def med_filt_z_stack(zstack, kernel_size=5):
    """Get z-stack with each plane median-filtered

    Parameters
    ----------
    zstack : np.ndarray
        z-stack to apply median filtering
    kernel_size : int, optional
        kernel size for median filtering, by default 5
        It seems only certain odd numbers work, e.g., 3, 5, 11, ...

    Returns
    -------
    np.ndarray
        median-filtered z-stack
    """
    filtered_z_stack = []
    for image in zstack:
        filtered_z_stack.append(cv2.medianBlur(
            image.astype(np.uint16), kernel_size))
    return np.array(filtered_z_stack)


def image_normalization_uint16(image, im_thresh=0):
    """Normalize 2D image and convert to uint16
    Prevent saturation.

    Args:
        image (np.ndarray): input image (2D)
                            Just works with 3D data as well.
        im_thresh (float, optional): threshold when calculating pixel intensity percentile.
                            0 by default
    Return:
        norm_image (np.ndarray)
    """
    clip_image = np.clip(image, np.percentile(
        image[image > im_thresh], 0.2), np.percentile(image[image > im_thresh], 99.8))
    norm_image = (clip_image - np.amin(clip_image)) / \
        (np.amax(clip_image) - np.amin(clip_image)) * 0.9
    uint16_image = ((norm_image + 0.05) *
                    np.iinfo(np.uint16).max * 0.9).astype(np.uint16)
    return uint16_image


def save_zdrift(emf_fn, zstack_dir, save_dir=None):
    if save_dir is None:
        save_dir = emf_fn.parent
    fn_base = emf_fn.name.split('.')[0]
    ops_fn = emf_fn.parent / (emf_fn.name[:-7] + 'ops.npy')
    save_fn = save_dir / f'{fn_base}_zdrift.h5'
    # if not save_fn.exists():
    matched_ind_zstack, zstack_fn_list, corrcoef_zstack_finding, tmat_zstack_finding = \
        get_matched_zstack(emf_fn, ops_fn, zstack_dir)
    zstack_fn = zstack_fn_list[matched_ind_zstack]
    tmat = tmat_zstack_finding[matched_ind_zstack]
    matched_inds, corrcoef, tmat, fov_reg = calculate_zdrift(emf_fn, ops_fn, zstack_fn, tmat=tmat)
    with h5py.File(save_fn, 'w') as h:
        h.create_dataset(name='matched_inds', data=matched_inds)
        h.create_dataset(name='corrcoef', data=corrcoef)
        h.create_dataset(name='tmat', data=tmat)
        h.create_dataset(name='emf_registered', data=fov_reg)
        h.create_dataset(name='matched_zstack_fn', data=zstack_fn)
        h.create_dataset(name='corrcoef_zstack_finding', data=corrcoef_zstack_finding)
        h.create_dataset(name='tmat_zstack_finding', data=tmat_zstack_finding)
        h.create_dataset(name='zstack_fn_list', data=zstack_fn_list)
        

if __name__ == '__main__':
    args = parser.parse_args()
    filepath = Path(args.file_path)
    zstack_dir = Path(args.zstack_dir)
    t0 = time.time()
    save_zdrift(filepath, zstack_dir)
    t1 = time.time()
    print(f'Done in {(t1-t0)/60:.2f} min.')
