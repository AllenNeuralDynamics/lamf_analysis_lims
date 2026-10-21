from pathlib import Path
import tifffile
import numpy as np
import pandas as pd
import h5py
from lamf_analysis.ophys import zdrift
from glob import glob
import time
from matplotlib import pyplot as plt

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
parser.add_argument(
    '--fig_title',
    type=str,
    default='',
    metavar='fig_title',
    help='title for the figure, describing the session'
)


FOV_ORDER_DICT = {
    0: '0_reg_ch_1',
    1: '0_reg_ch_2',
    2: '1_reg_ch_1',
    3: '1_reg_ch_2',
    4: '2_reg_ch_1',
    5: '2_reg_ch_2',
    6: '3_reg_ch_1',
    7: '3_reg_ch_2',
}

PLANE_IND_DICT = {0: (3, 2),
                  1: (2, 2),
                  2: (1, 2),
                  3: (0, 2),
                  4: (0, 1),
                  5: (1, 1),
                  6: (2, 1),
                  7: (3, 1)}

PLANE_ORDER = [7, 5, 3, 1, 0, 2, 4, 6]
PLANE_DEPTHS = range(40, 330, 40)


def get_zdrift_results(emf_filenames, ops_filenames, zstack_dir, parallel=True):
    num_planes = len(emf_filenames)
    if parallel:
        with Client() as client:
            tasks = [delayed(wrapper_calculate_zdrift)(plane_ind, emf_fn, ops_fn, zstack_dir) 
                    for plane_ind, emf_fn, ops_fn in zip(range(num_planes), emf_filenames, ops_filenames)]
            results_session = compute(*tasks)
    else:
        results_session = []
        for plane_ind, emf_fn, ops_fn in zip(range(num_planes), emf_filenames, ops_filenames):
            results = wrapper_calculate_zdrift(plane_ind, emf_fn, ops_fn, zstack_dir)
            results_session.append(results)
    return results_session


def wrapper_calculate_zdrift(plane_ind, emf_fn, ops_fn, zstack_dir):
    with h5py.File(emf_fn, 'r') as f:
        emf = f['data'][:]
        epoch_seconds = f['epoch_seconds'][:]
        num_epochs = f['num_epochs'][:]
        assert num_epochs == emf.shape[0], f'num_epochs ({num_epochs}) does not match emf shape ({emf.shape[0]})'
    y_range, x_range = get_motion_range(ops_fn)
    zstack, matched_zstack_fn  = get_zstack(zstack_dir, plane_ind)
    
    results = calculate_zdrift(zstack, emf, y_range, x_range)
    results['plane_ind'] = plane_ind
    results['emf_fn'] = emf_fn
    results['ops_fn'] = ops_fn
    results['zstack_fn'] = matched_zstack_fn
    results['epoch_seconds'] = epoch_seconds
    return results


def get_zstack(zstack_dir, plane_ind):
    matched_zstack_key = FOV_ORDER_DICT[plane_ind]
    matched_zstack_fn = list(zstack_dir.glob(f'*_local_z_stack{matched_zstack_key}.tif'))
    assert len(matched_zstack_fn) == 1
    matched_zstack_fn = matched_zstack_fn[0]
    zstack = tifffile.imread(matched_zstack_fn)
    return zstack, matched_zstack_fn


def get_motion_range(ops_fn):
    ops = np.load(ops_fn, allow_pickle=True).item()
    y_offs = ops['reg_result'][4][0]
    x_offs = ops['reg_result'][4][1]
    assert max(y_offs) > 0
    assert min(y_offs) < 0
    assert max(x_offs) > 0
    assert min(x_offs) < 0
    range_y = [max(y_offs), min(y_offs)]
    range_x = [max(x_offs), min(x_offs)]
    return range_y, range_x


def calculate_zdrift(zstack, emf, range_y, range_x,
                     use_clahe=True, use_valid_pix=True):
    # prepare zstack and crop emf
    zstack_crop = zstack[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]
    stack_pre = zdrift.med_filt_z_stack(zstack_crop)
    stack_pre = zdrift.rolling_average_stack(stack_pre)    
    episodic_mean_fovs_crop = emf[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

    # calculate
    matched_plane_indices = np.zeros(episodic_mean_fovs_crop.shape[0], dtype=int)
    corrcoef = []
    segment_reg_imgs = []
    shift_list = []
    for i in range(episodic_mean_fovs_crop.shape[0]):
        fov_reg_stack, cc, shift = zdrift.fov_stack_register_phase_correlation(
            episodic_mean_fovs_crop[i], stack_pre, use_clahe=use_clahe,
            use_valid_pix=use_valid_pix)
        matched_plane_indices[i] = np.argmax(cc)
        corrcoef.append(cc)
        segment_reg_imgs.append(fov_reg_stack[np.argmax(cc)])
        shift_list.append(shift)
    corrcoef = np.asarray(corrcoef)

    results = {'matched_plane_indices': matched_plane_indices,
                'corrcoef': corrcoef,
                'segment_fov_registered': segment_reg_imgs,
                'ref_zstack_crop': zstack_crop,                   
                'shift': shift_list,
                'use_clahe': use_clahe,
                'use_valid_pix': use_valid_pix}
    return results


def get_filenames(file_path):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    data_dir = file_path.parent
    fn_stem = file_path.stem

    reg_filenames = list(data_dir.glob(f'{fn_stem}_*_reg.h5'))
    num_planes = len(reg_filenames)
    assert num_planes == len(FOV_ORDER_DICT), f'Number of planes ({num_planes}) does not match FOV_ORDER_DICT ({len(FOV_ORDER_DICT)})'
    # check filename format
    reg_filenames_confirm = [data_dir / f'{fn_stem}_{pi:02}_reg.h5' for pi in range(num_planes)]
    assert reg_filenames == reg_filenames_confirm, f'Filename format mismatch: {reg_filenames} != {reg_filenames_confirm}'
    emf_filenames = [fn.parent / f'{fn.name.split("_reg")[0]}_emf.h5' for fn in reg_filenames]
    ops_filenames = [fn.parent / f'{fn.name.split("_reg")[0]}_ops.npy' for fn in reg_filenames]
    split_filenames = [ref_fn.parent / f'{ref_fn.stem.split("_reg")[0]}.h5' for ref_fn in reg_filenames]
    assert np.all([fn.exists() for fn in emf_filenames]), f'Not all emf files exist: {emf_filenames}'
    assert np.all([fn.exists() for fn in ops_filenames]), f'Not all ops files exist: {ops_filenames}'
    assert np.all([fn.exists() for fn in split_filenames]), f'Not all split files exist: {split_filenames}'
    return reg_filenames, emf_filenames, ops_filenames, split_filenames


### Visualization
def plot_and_save_posthoc_zdrift(data_dir, results_session, save_fn, session_title=None):
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    if isinstance(save_fn, str):
        save_fn = Path(save_fn)
    # if save_fn.exists():
    #     print(f'{save_fn} already exists. Quit.')
    #     return
    num_planes = len(PLANE_ORDER)
    assert len(results_session) == num_planes

    # check number of epochs and epoch duration across results, and set timestamps
    num_epochs = [len(r['matched_plane_indices']) for r in results_session]
    assert np.all([ne == num_epochs[0] for ne in num_epochs]), f'Number of epochs mismatch: {num_epochs}'
    epoch_duration = [r['epoch_seconds'] for r in results_session]
    assert np.all([ed == epoch_duration[0] for ed in epoch_duration]), f'Epoch duration mismatch: {epoch_duration}'
    epoch_duration = epoch_duration[0]
    timestamps = np.arange(num_epochs[0]) * epoch_duration / 60  # in minutes

    # reorder results based on the PLANE_ORDER
    results_plane_inds = [r['plane_ind'] for r in results_session]
    results_ordered = []
    for plane_ind in PLANE_ORDER:
        matched_ind = results_plane_inds.index(plane_ind)
        results_ordered.append(results_session[matched_ind])

    # plot
    fig, axes = plt.subplots(2,4, figsize=(15, 7), sharex=True, sharey=True)
    for i, r in enumerate(results_ordered):
        ax = axes.flatten()[i]
        mpi = r['matched_plane_indices']
        max_range = (np.max(r['matched_plane_indices'][1:]) - np.min(r['matched_plane_indices'][1:])) * 0.75
        max_cc = [max(cc) for cc in r['corrcoef']]
        ax.plot(timestamps, mpi, 'k-', label=f'Plane {i}')
        im = ax.scatter(timestamps, mpi, c=max_cc, s=30, cmap='coolwarm', vmin=0.5, vmax=1, zorder=2)
        fov_depth = PLANE_DEPTHS[i]
        ax.set_title(f'Plane {i} ({-fov_depth} um)\nMotion range: {max_range:.2f} um', fontsize=15)
        if i % 4 == 0:
            ax.set_ylabel('Matched Plane Index', fontsize=15)
        if i >= 4:
            ax.set_xlabel('Time (minutes)', fontsize=15)        
        ax.tick_params(axis='both', labelsize=15)
    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.95, 0.2, 0.01, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Max CC', fontsize=15)
    cbar.ax.tick_params(labelsize=15)

    if session_title is not None:
        fig.suptitle(session_title, fontsize=20)
    fig.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust layout to make space for the colorbar
    fig.show()

    # save figure
    fig.savefig(save_fn, dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def get_omc_motion_est(motion_file_path):
    if isinstance(motion_file_path, str):
        motion_file_path = Path(motion_file_path)
    data_dir = motion_file_path.parent
    session_name, file_ind = check_motion_filepath(motion_file_path)

    motion_est_fn = data_dir / f'{session_name}_timeseries_Motion_{file_ind:05}.csv'
    motion_est = pd.read_csv(motion_est_fn)

    def _str_to_float_list(x):
        """
        Convert a string representation of a list to a list of floats.
        """
        return [float(np.round(float(s), 4)) for s in x.split('[')[1].split(']')[0].split(' ')]

    def _reformat_df(df):
        # strip blank spaces from column names
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if isinstance(df[col].values[0], str):
                df[col] = df[col].str.strip()
        df['roiName'] = df['roiName'].str.strip()
        # convert str of list to a list
        
        list_columns = ['drPixel', 'drRef', 'confidence']
        for col in list_columns:
            df[col] = df[col].apply(lambda x: _str_to_float_list(x))
        return df

    motion_est = _reformat_df(motion_est)

    assert len(motion_est) % 2 == 0, f'Number of motion estimates ({len(motion_est)}) is not even'
    channels = ((motion_est.index.values % 2) + 1).astype(int)
    motion_est['channel'] = channels
    zs = np.sort(motion_est.z.unique())
    motion_est['pair_ind'] = motion_est['z'].apply(lambda x: np.where(zs == x)[0][0])
    return motion_est


def plot_and_save_omc_zdrift(motion_file_path, save_fn, session_title=None):
    # if Path(save_fn).exists():
    #     print(f'{save_fn} already exists. Quit.')
    #     return
    
    motion_est = get_omc_motion_est(motion_file_path)
    num_planes = len(PLANE_ORDER)

    # plot
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for pi in range(num_planes):
        ax = axes.flatten()[pi]
        (pair_ind, channel) = PLANE_IND_DICT[pi]
        fov_motion_est = motion_est.query('channel == @channel and pair_ind == @pair_ind')
        fov_timestamps = fov_motion_est.timestamp.values / 60 # in minutes
        fov_zdrift = [p[2] for p in fov_motion_est.drPixel.values]

        ax.plot(fov_timestamps, fov_zdrift, 'k-')
        fov_depth = PLANE_DEPTHS[pi]
        ax.set_title(f'Plane {pi} ({-fov_depth} um)', fontsize=15)
        if pi % 4 == 0:
            ax.set_ylabel('Estimated Z Drift (um)', fontsize=15)
        if pi >= 4:
            ax.set_xlabel('Time (minutes)', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)
    if session_title is not None:
        fig.suptitle(session_title, fontsize=20)
    fig.tight_layout()
    fig.show()

    # save figure
    fig.savefig(save_fn, dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def check_motion_filepath(motion_file_path):
    if isinstance(motion_file_path, str):
        motion_file_path = Path(motion_file_path)
    fn_fragments = motion_file_path.stem.split('_')
    assert len(fn_fragments) == 4
    assert fn_fragments[1] == 'timeseries'
    assert fn_fragments[2] == 'Motion'
    assert fn_fragments[3].isdigit()
    session_name = fn_fragments[0]
    file_ind = int(fn_fragments[3])
    return session_name, file_ind


def check_data_filepath(file_path):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    fn_fragments = file_path.stem.split('_')
    assert len(fn_fragments) == 3
    assert fn_fragments[1] == 'timeseries'
    assert fn_fragments[2].isdigit()
    session_name = fn_fragments[0]
    file_ind = int(fn_fragments[2])
    return session_name, file_ind


def wrapper_compare_mean_fov_zstack(emf_filenames, zstack_dir, im_adjust_percentiles=[0.02, 99.8]):
    data_dir = emf_filenames[0].parent
    file_stem = emf_filenames[0].stem
    fn_segments = file_stem.split('_')
    save_fn_base = '_'.join(fn_segments[:-2])

    # top planes
    plane_nums = [0, 1, 2, 3]
    fig, axes = compare_mean_fov_zstack(emf_filenames, zstack_dir, plane_nums, im_adjust_percentiles)
    save_fn = data_dir / f'{save_fn_base}_mean_fov_zstack_top.png'
    fig.savefig(save_fn, dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # bottom planes
    plane_nums = [4, 5, 6, 7]
    fig, axes = compare_mean_fov_zstack(emf_filenames, zstack_dir, plane_nums, im_adjust_percentiles)
    save_fn = data_dir / f'{save_fn_base}_mean_fov_zstack_bottom.png'
    fig.savefig(save_fn, dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def compare_mean_fov_zstack(emf_filenames,
                            zstack_dir,
                            plane_nums, 
                            im_adjust_percentiles=[0.02, 99.8]):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for pi in range(4):
        plane_num = plane_nums[pi]
        plane_ind = PLANE_ORDER[plane_num]

        mimg = get_mean_image(emf_filenames[plane_ind])
        zstack, matched_zstack_fn = get_zstack(zstack_dir, plane_ind)
        zcenter = zstack[zstack.shape[0] //2]

        axes[0, pi].imshow(mimg, cmap='gray',
                        vmin=np.percentile(mimg, im_adjust_percentiles[0]),
                        vmax=np.percentile(mimg, im_adjust_percentiles[1]))
        title_str = f'Plane {plane_num} ({-PLANE_DEPTHS[plane_num]} um)'
        title_str += '\nFOV mean image'
        axes[0, pi].set_title(title_str, fontsize=15)
        
        axes[1, pi].imshow(zcenter, cmap='gray',
                        vmin=np.percentile(zcenter, im_adjust_percentiles[0]),
                        vmax=np.percentile(zcenter, im_adjust_percentiles[1]))
        axes[1, pi].set_title('Z stack center slice', fontsize=15)
    for ax in axes.flatten():
        ax.axis('off')
    return fig, axes


def get_mean_image(emf_fn):
    with h5py.File(emf_fn, 'r') as h:
        mimg = h['data'][:].mean(axis=0)
    return mimg


def wrapper_compare_single_frame_zstack(split_filenames, zstack_dir, im_adjust_percentiles=[0.02, 99.8]):
    data_dir = split_filenames[0].parent
    file_stem = split_filenames[0].stem
    fn_segments = file_stem.split('_')
    save_fn_base = '_'.join(fn_segments[:-1])

    # top planes
    plane_nums = [0, 1, 2, 3]
    fig, axes = compare_single_frame_zstack(split_filenames, zstack_dir, plane_nums, im_adjust_percentiles)
    save_fn = data_dir / f'{save_fn_base}_single_frame_zstack_top.png'
    fig.savefig(save_fn, dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # bottom planes
    plane_nums = [4, 5, 6, 7]
    fig, axes = compare_single_frame_zstack(split_filenames, zstack_dir, plane_nums, im_adjust_percentiles)
    save_fn = data_dir / f'{save_fn_base}_single_frame_zstack_bottom.png'
    fig.savefig(save_fn, dpi=300, transparent=False, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def compare_single_frame_zstack(split_filenames,
                            zstack_dir,
                            plane_nums, 
                            im_adjust_percentiles=[0.02, 99.8]):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for pi in range(4):
        plane_num = plane_nums[pi]
        plane_ind = PLANE_ORDER[plane_num]

        mimg = get_single_frame(split_filenames[plane_ind])
        zstack, matched_zstack_fn = get_zstack(zstack_dir, plane_ind)
        zcenter = zstack[zstack.shape[0] //2]

        axes[0, pi].imshow(mimg, cmap='gray',
                        vmin=np.percentile(mimg, im_adjust_percentiles[0]),
                        vmax=np.percentile(mimg, im_adjust_percentiles[1]))
        title_str = f'Plane {plane_num} ({-PLANE_DEPTHS[plane_num]} um)'
        title_str += '\nFOV single frame'
        axes[0, pi].set_title(title_str, fontsize=15)
        
        axes[1, pi].imshow(zcenter, cmap='gray',
                        vmin=np.percentile(zcenter, im_adjust_percentiles[0]),
                        vmax=np.percentile(zcenter, im_adjust_percentiles[1]))
        axes[1, pi].set_title('Z stack center slice', fontsize=15)
    for ax in axes.flatten():
        ax.axis('off')
    return fig, axes


def get_single_frame(split_fn, frame_ind=None):
    with h5py.File(split_fn, 'r') as f:
        if frame_ind is None:
            center_ind = f['data'].shape[0] // 2
            single_frame = f['data'][center_ind]
        else:
            single_frame = f['data'][frame_ind]
    return single_frame


if __name__ == '__main__':
    t0 = time.time()
    args = parser.parse_args()
    file_path = Path(args.file_path)
    zstack_dir = Path(args.zstack_dir)
    fig_title = args.fig_title
    if file_path.exists() == False:
        raise FileNotFoundError(f'{file_path} does not exist. Quit.')
    session_name, file_ind = check_data_filepath(file_path)
    if zstack_dir.exists() == False:
        raise FileNotFoundError(f'{zstack_dir} does not exist. Quit.')

    data_dir = file_path.parent
    fn_stem = file_path.stem
    if fig_title == '':
        fig_title = fn_stem
    
    # get filenames
    reg_filenames, emf_filenames, ops_filenames, split_filenames = get_filenames(file_path)

    # calculate zdrift posthoc (using dask)
    save_fn = data_dir / f'{fn_stem}_zdrift_results.npz'
    if save_fn.exists():
        print(f'{save_fn} already exists. Loading...')
        results_session = np.load(save_fn, allow_pickle=True)['data']
    else:
        print(f'Calculating z-drift for {len(emf_filenames)} planes...')
        results_session = get_zdrift_results(emf_filenames, ops_filenames, zstack_dir)

        dur = (time.time() - t0) / 60
        print(f'Z-drift calculation took {dur:.2f} min.')

        # save results
        np.savez(save_fn, data=results_session)
        print(f'Saved z-drift results to {save_fn}')

    # draw and save posthoc zdrift plot
    plot_save_fn = data_dir / f'{fn_stem}_zdrift_posthoc.png'
    plot_and_save_posthoc_zdrift(data_dir, results_session, plot_save_fn, session_title=fig_title)

    # # draw and save omc zdrift plot
    # omc_save_fn = data_dir / f'{fn_stem}_omc_zdrift.png'
    # omc_motion_file_path = data_dir / f'{session_name}_timeseries_Motion_{file_ind:05}.csv'
    # plot_and_save_omc_zdrift(omc_motion_file_path, omc_save_fn, session_title=fig_title)
    
    print(f'Saved z-drift plots')

    # # draw and save images
    # wrapper_compare_mean_fov_zstack(emf_filenames, zstack_dir)
    # wrapper_compare_single_frame_zstack(split_filenames, zstack_dir)

    # print(f'Saved image comparison results.')

    dur = (time.time() - t0) / 60
    print(f'Calculation and plotting took {dur:.2f} min.')