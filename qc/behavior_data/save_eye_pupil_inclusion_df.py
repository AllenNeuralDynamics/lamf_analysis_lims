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

parser = ArgumentParser(description='arguments for multiplane z-drift calculation')
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


def _get_pupil_inside_ratio(df):
    eye_df = df[(df.bodyparts.str.contains('eye')) & ((df.coords=='x') | (df.coords=='y'))]
    eye_points = np.stack(eye_df.groupby('bodyparts').apply(lambda x: np.array([x[x['coords']=='x'].value.values[0], x[x['coords']=='y'].value.values[0]])).values)
    hull = ConvexHull(eye_points)
    hull_path = mplPath(eye_points[hull.vertices])
    
    pupil_df = df[(df.bodyparts.str.contains('pupil')) & ((df.coords=='x') | (df.coords=='y'))]
    pupil_points = np.stack(pupil_df.groupby('bodyparts').apply(lambda x: np.array([x[x['coords']=='x'].value.values[0], x[x['coords']=='y'].value.values[0]])).values)
    inside=0
    for pupil_point in pupil_points:
        inside += hull_path.contains_point(pupil_point)

    area = hull.area
    mean_likelihood = df[(df.bodyparts.str.contains('eye')) & (df.coords=='likelihood')].value.mean()

    return inside/len(pupil_points), area, mean_likelihood


def get_pupil_info(osid, downsample=10):
    df = annotation_tools.read_DLC_h5file(from_lims.get_deepcut_h5_filepath(osid))
    if downsample is not None:
        df = df[df.frame_number % downsample == 0]
    new_df = df.groupby('frame_number').apply(_get_pupil_inside_ratio).to_frame()
    curr_col_name = new_df.columns.values[0]
    # change column names
    new_df.rename(columns={curr_col_name: 'values_tuple'}, inplace=True)
    new_df['inside_ratio'] = new_df.apply(lambda x: x.values_tuple[0], axis=1)
    new_df['area'] = new_df.apply(lambda x: x.values_tuple[1], axis=1)
    new_df['mean_likelihood'] = new_df.apply(lambda x: x.values_tuple[2], axis=1)
    new_df.drop(columns='values_tuple', inplace=True)
    return new_df


def create_save_eye_pupil_inclusion(osid, save_dir, downsample=10):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    if downsample is None:
        save_file = save_dir / f'{osid}_eye_pupil_inclusion.pkl'
    else:
        save_file = save_dir / f'{osid}_eye_pupil_inclusion_ds{downsample}.pkl'
    pupil_info_df = get_pupil_info(osid, downsample=downsample)
    pupil_info_df.to_pickle(save_file)
    return save_file


if __name__ == "__main__":
    t0 = time.time()
    args = parser.parse_args()
    osid = args.osid
    downsample = args.downsample
    save_dir = Path(r'\\allen\programs\mindscope\workgroups\learning\qc_pupil\eye_pupil_inclusion'.replace('\\', '/'))
    create_save_eye_pupil_inclusion(osid, save_dir, downsample=downsample)
    t1 = time.time()
    print(f'total time = {(t1 - t0)/60:.1f} minutes')
