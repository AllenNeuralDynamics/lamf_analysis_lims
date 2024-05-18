from pathlib import Path
import json
from ScanImageTiffReader import ScanImageTiffReader
import tifffile
import h5py
import time
from suite2p import default_ops
from suite2p.registration import register
import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser(description='arguments for offline mesoscope data splitting registration and EMF saving')
parser.add_argument(
    '--file_path',
    type=str,
    default=' ',
    metavar='file_path',
    help='file path to the tiff file'
)

parser.add_argument(
    '--plane_index',
    type=int,
    default=' ',
    metavar='file_path',
    help='file path to the tiff file'
)

parser.add_argument(
    '--epoch_minutes',
    type=int,
    default=5,
    metavar='epoch_minutes',
    help='how long is one epoch duration (in min)'
)

parser.add_argument(
    '--num_planes',
    type=int,
    default=8,
    metavar='num_planes',
    help='number of planes'
)

def create_plane_h5_from_tiff(data_fn, plane_ind, num_pages=None, num_planes=8, rerun=False):
    if type(data_fn) == str:
        data_fn = Path(data_fn)
    base_name = data_fn.name.split('.')[0]
    save_fn = data_fn.parent / f'{base_name}_{plane_ind:02}.h5'
    if (save_fn.exists() == False) and (rerun==False):
        print(f'{base_name} plane index {plane_ind} splitting...')
        t0 = time.time()
        if num_pages is None:
            with tifffile.TiffFile(data_fn) as tif:
                num_pages = (len(tif.pages))
        with tifffile.TiffFile(data_fn) as tif:
            imgs = [tif.pages[ii].asarray() for ii in range(plane_ind,num_pages,num_planes)]
            with h5py.File(save_fn, 'w') as h:
                h.create_dataset('data', data=imgs)
        t1 = time.time()
        print(f'{base_name} plane index {plane_ind} splitting done in {(t1-t0)/60:.2f} min.')
    else:
        print(f'{base_name} plane index {plane_ind} already split.')
    return save_fn


def _create_and_save_emf(reg_movie, epoch_fn, frame_rate, epoch_minutes):
    if type(epoch_fn) == str:
        epoch_fn = Path(epoch_fn)
    num_frames = reg_movie.shape[0]
    epoch_length = int(np.round(frame_rate * 60 * epoch_minutes))
    num_epochs = int(num_frames // epoch_length)
    last_epoch_minutes = epoch_minutes
    if num_frames % epoch_length > epoch_length / 2:
        num_epochs += 1
        last_epoch_minutes = epoch_minutes * ((num_frames % epoch_length) / epoch_length)
    emf = np.zeros((num_epochs, *reg_movie.shape[1:]))
    for ei in range(num_epochs):
        num_frame_start = ei*epoch_length
        num_frame_end = min((ei+1)*epoch_length, num_frames)
        emf[ei] = np.mean(reg_movie[num_frame_start:num_frame_end], axis=0)    
    with h5py.File(epoch_fn, 'w') as h:
        h.create_dataset(name='data', data=emf)
        h.create_dataset(name='num_epochs', data=num_epochs)
        h.create_dataset(name='epoch_minutes', data=epoch_minutes)
        h.create_dataset(name='last_epoch_minutes', data=last_epoch_minutes)
    epoch_base_fn = epoch_fn.name.split('.')[0]
    epoch_tif_fn = epoch_fn.parent / f'{epoch_base_fn}.tif'
    tifffile.imwrite(epoch_tif_fn, emf)


def register_plane_and_save_emf(h5_fn, frame_rate=11, epoch_minutes=1, key='data'):
    if type(h5_fn) == str:
        h5_fn = Path(h5_fn)
    t0 = time.time()
    base_name = h5_fn.name.split('.')[0]
    save_fn_h5 = h5_fn.parent / f'{base_name}_reg.h5'
    save_fn_npy = h5_fn.parent / f'{base_name}_ops.npy'
    epoch_fn = h5_fn.parent / f'{base_name}_emf.h5'

    if save_fn_h5.exists() and save_fn_npy.exists() and epoch_fn.exists():
        print(f'{base_name} already processed. Quit.')

    elif save_fn_h5.exists() and save_fn_npy.exists():
        print(f'{base_name} registration already done. Processing episodic mean FOV...')
        with h5py.File(save_fn_h5, 'r') as h5:
            reg_movie = h5[key][:]
        _create_and_save_emf(reg_movie, epoch_fn, frame_rate, epoch_minutes)

        t1 = time.time()
        print(f'{base_name} EMF saved and done in {(t1-t0)/60:.2f} min.')
    
    else:
        # suite2p options
        print(f'{base_name} suite2p registration running...')
        ops=default_ops()
        ops['batch_size'] = 1000
        ops['maxregshift'] = 0.2
        ops['snr_thresh'] = 1.2 # Default: 1.2 # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        ops['block_size'] = [64, 64]
        ops['maxregshiftNR'] = np.round(ops['block_size'][0]/10) # Default = 5

        # Read data
        with h5py.File(h5_fn, 'r') as h5:
            imgs = h5[key][:]
        reg_movie = np.zeros_like(imgs)

        # Register
        register_result = register.compute_reference_and_register_frames(f_align_in=imgs, f_align_out=reg_movie, ops=ops)
        ops['reg_result'] = register_result

        t1 = time.time()
        print(f'{base_name} registration done in {(t1-t0)/60:.2f} min.')

        # Save registration results
        save_fn_h5 = h5_fn.parent / f'{base_name}_reg.h5'
        save_fn_npy = h5_fn.parent / f'{base_name}_ops.npy'
        with h5py.File(save_fn_h5, 'w') as h:
            h.create_dataset(name='data', data=reg_movie)
        np.save(save_fn_npy, ops)
        
        t2 = time.time()
        print(f'{base_name} registration saved in {(t2-t1)/60:.2f} min.')

        # calculating and saving episodic mean FOVs (EMF)
        _create_and_save_emf(reg_movie, epoch_fn, frame_rate, epoch_minutes)
        
        t3 = time.time()
        print(f'{base_name} EMF saved in {(t3-t2)/60:.2f} min.')
        print(f'{base_name} done in {(t3-t0)/60:.2f} min')


def _extract_dict_from_si_string(string):
    """Parse the 'SI' variables from a scanimage metadata string"""

    lines = string.split('\n')
    data_dict = {}
    for line in lines:
        if line.strip():  # Check if the line is not empty
            key, value = line.split(' = ')
            key = key.strip()
            if value.strip() == 'true':
                value = True
            elif value.strip() == 'false':
                value = False
            else:
                value = value.strip().strip("'")  # Remove leading/trailing whitespace and single quotes
            data_dict[key] = value

    json_data = json.dumps(data_dict, indent=2)
    loaded_data_dict = json.loads(json_data)
    return loaded_data_dict


if __name__ == '__main__':
    args = parser.parse_args()
    filepath = args.file_path
    plane_index = args.plane_index
    num_planes = args.num_planes
    epoch_minutes = args.epoch_minutes

    with ScanImageTiffReader(str(filepath)) as reader:
        md_string = reader.metadata()

    with tifffile.TiffFile(filepath) as tif:
        num_pages = (len(tif.pages))

    # split si & roi groups, prep for seprate parse
    s = md_string.split("\n{")
    rg_str = "{" + s[1]
    si_str = s[0]

    # parse 1: extract keys and values, dump, then load again
    si_metadata = _extract_dict_from_si_string(si_str)
    frame_rate = float(si_metadata['SI.hRoiManager.scanVolumeRate'])

    h5_fn = create_plane_h5_from_tiff(filepath, plane_index, num_pages, num_planes)

    register_plane_and_save_emf(h5_fn, frame_rate, epoch_minutes)
