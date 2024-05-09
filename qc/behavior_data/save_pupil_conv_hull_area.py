import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from brain_observatory_analysis.behavior.video_qc import annotation_tools
from brain_observatory_qc.data_access import from_lims
from scipy.spatial import ConvexHull
from matplotlib.path import Path as mplPath
from argparse import ArgumentParser
import time

parser = ArgumentParser(description='arguments for calculating pupil convex hull area')
parser.add_argument(
    '--osid',
    type=int,
    default=0,
    metavar='ophys_session_id',
    help='ophys session id'
)
parser.add_argument(
    '--downsample',
    type=int,
    default=10,
    metavar='downsample',
    help='downsample rate'
)


def _get_pupil_ch_area(df, likelihood_threshold=0.8, num_over_threshold=6):
    assert len(df.frame_number.unique()) == 1
    pupil_df = df[(df.bodyparts.str.contains('pupil'))]
    pupil_points = np.stack(pupil_df.groupby('bodyparts').apply(lambda x: np.array([x[x['coords']=='x'].value.values[0], x[x['coords']=='y'].value.values[0]])).values)
    likelihood = pupil_df[(pupil_df.coords=='likelihood')].value.values
    inds_over_the_threshold = np.where(likelihood>=likelihood_threshold)[0]
    if len(inds_over_the_threshold) < num_over_threshold:
        return np.nan
    hull = ConvexHull(pupil_points[inds_over_the_threshold])
    return hull.area


def get_pupil_ch_area(osid, downsample=10):
    df = annotation_tools.read_DLC_h5file(from_lims.get_deepcut_h5_filepath(osid))
    if downsample is not None:
        df = df[df.frame_number % downsample == 0]
    new_df = df.groupby('frame_number').apply(_get_pupil_ch_area).to_frame()
    curr_col_name = new_df.columns.values[0]
    # change column names
    new_df.rename(columns={curr_col_name: 'pupil_conv_hull_area'}, inplace=True)
    return new_df


def create_save_pupil_ch_area(osid, save_dir, downsample=10):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    if downsample is None:
        save_file = save_dir / f'{osid}_pupil_convhull_area.pkl'
    else:
        save_file = save_dir / f'{osid}_pupil_convhull_area_ds{downsample}.pkl'
    pupil_info_df = get_pupil_ch_area(osid, downsample=downsample)
    pupil_info_df.to_pickle(save_file)
    return save_file


if __name__ == "__main__":
    t0 = time.time()
    args = parser.parse_args()
    osid = args.osid
    downsample = args.downsample
    save_dir = Path(r'\\allen\programs\mindscope\workgroups\learning\qc_pupil\pupil_convhull_area'.replace('\\', '/'))
    create_save_pupil_ch_area(osid, save_dir, downsample=downsample)
    t1 = time.time()
    print(f'total time = {(t1 - t0)/60:.1f} minutes')
