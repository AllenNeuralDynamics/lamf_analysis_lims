from pathlib import Path
from pystackreg import StackReg
import tifffile
import time
import numpy as np
from glob import glob
import h5py
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


def get_matched_zstack(emf_fn, ops_fn, zstack_dir, num_planes_around=40):
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
        new_zstack = rolling_average_zstack(zstack)
        center_ind = int(np.floor(new_zstack.shape[0]/2))
        center_zstack = new_zstack[center_ind - num_planes_around//2 : center_ind + num_planes_around//2+1]
        center_zstack = center_zstack[:, y_roll_top:y_roll_bottom, x_roll_left:x_roll_right]
        center_zstacks.append(center_zstack)
    first_emf = tifffile.imread(emf_fn)[0, y_roll_top:y_roll_bottom, x_roll_left:x_roll_right]
    
    assert first_emf.min() > 0
    valid_pix_threshold = first_emf.min()/10
    num_pix_threshold = first_emf.shape[0] * first_emf.shape[1] / 3

    sr = StackReg(StackReg.AFFINE)
    corrcoef = np.zeros((len(center_zstacks), center_zstacks[0].shape[0]))
    
    for i, zstack in enumerate(center_zstacks):
        temp_cc = []
        tmat_list = []
        for j, zstack_plane in enumerate(zstack):
            tmat = sr.register(zstack_plane, first_emf)
            emf_reg = sr.transform(first_emf, tmat=tmat)            
            valid_y, valid_x = np.where(emf_reg > valid_pix_threshold)
            if len(valid_y) > num_pix_threshold:
                temp_cc.append(np.corrcoef(zstack_plane.flatten(), emf_reg.flatten())[0,1])
                tmat_list.append(tmat)
            else:
                temp_cc.append(0)
                tmat_list.append(np.eye(3))
        temp_ind = np.argmax(temp_cc)
        best_tmat = tmat_list[temp_ind]
        emf_reg = sr.transform(first_emf, tmat=best_tmat)       
        for j, zstack_plane in enumerate(zstack):
            corrcoef[i,j] = np.corrcoef(zstack_plane.flatten(), emf_reg.flatten())[0,1]
    matched_ind = np.argmax(np.mean(corrcoef, axis=1))
    return matched_ind, zstack_fn_list, corrcoef


def calculate_zdrift(emf_fn, ops_fn, zstack_fn):
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
    new_zstack = rolling_average_zstack(zstack)
    
    corrcoef = np.zeros((len(emf), zstack.shape[0]))

    assert emf.min() > 0
    valid_pix_threshold = emf.min()/10
    num_pix_threshold = emf.shape[1] * emf.shape[2] / 3

    sr = StackReg(StackReg.RIGID_BODY)
    tmat_list = []
    fov_reg_list = []
    for i, fov in enumerate(emf):
        temp_tmat = []
        temp_cc = []
        for j, zplane in enumerate(new_zstack):
            tmat = sr.register(zplane, fov)
            fov_reg = sr.transform(fov, tmat=tmat)
            valid_y, valid_x = np.where(fov_reg > valid_pix_threshold)
            if len(valid_y) > num_pix_threshold:
                temp_cc.append(np.corrcoef(zplane[valid_y, valid_x].flatten(),
                                        fov_reg[valid_y, valid_x].flatten())[0,1])
                temp_tmat.append(tmat)
            else:
                temp_cc.append(0)
                temp_tmat.append(np.eye(3))
        temp_ind = np.argmax(temp_cc)
        tmat = temp_tmat[temp_ind]
        tmat_list.append(tmat)
        fov_reg = sr.transform(fov, tmat=tmat)
        fov_reg_list.append(fov_reg)
        valid_y, valid_x = np.where(fov_reg > valid_pix_threshold)
    
        for j, zplane in enumerate(new_zstack):
            corrcoef[i,j] = np.corrcoef(zplane[valid_y, valid_x].flatten(),
                                        fov_reg[valid_y, valid_x].flatten())[0,1]
    matched_inds = np.argmax(corrcoef, axis=1)
    tmat_array = np.array(tmat_list)
    fov_reg_array = np.array(fov_reg_list)

    return matched_inds, corrcoef, tmat_array, fov_reg_array


def rolling_average_zstack(zstack, rolling_window_flank=2):
    new_zstack = np.zeros(zstack.shape)
    for i in range(zstack.shape[0]):
        new_zstack[i] = np.mean(zstack[max(0, i-rolling_window_flank) : min(zstack.shape[0], i+rolling_window_flank), :, :],
                                axis=0)
    return new_zstack


def save_zdrift(emf_fn, zstack_dir, save_dir=None):
    if save_dir is None:
        save_dir = emf_fn.parent
    fn_base = emf_fn.name.split('.')[0]
    ops_fn = emf_fn.parent / (emf_fn.name[:-7] + 'ops.npy')
    save_fn = save_dir / f'{fn_base}_zdrift.h5'
    # if not save_fn.exists():
    matched_ind, zstack_fn_list, corrcoef_zstack_finding = get_matched_zstack(emf_fn, ops_fn, zstack_dir)
    zstack_fn = zstack_fn_list[matched_ind]
    matched_inds, corrcoef, tmat, fov_reg = calculate_zdrift(emf_fn, ops_fn, zstack_fn)
    with h5py.File(save_fn, 'w') as h:
        h.create_dataset(name='matched_inds', data=matched_inds)
        h.create_dataset(name='corrcoef', data=corrcoef)
        h.create_dataset(name='tmat', data=tmat)
        h.create_dataset(name='emf_registered', data=fov_reg)
        h.create_dataset(name='matched_zstack_fn', data=zstack_fn)
        h.create_dataset(name='corrcoef_zstack_finding', data=corrcoef_zstack_finding)
        h.create_dataset(name='zstack_fn_list', data=zstack_fn_list)
        

if __name__ == '__main__':
    args = parser.parse_args()
    filepath = Path(args.file_path)
    zstack_dir = Path(args.zstack_dir)
    t0 = time.time()
    save_zdrift(filepath, zstack_dir)
    t1 = time.time()
    print(f'Done in {(t1-t0)/60:.2f} min.')
