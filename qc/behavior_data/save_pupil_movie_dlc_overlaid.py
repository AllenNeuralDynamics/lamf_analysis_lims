import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from brain_observatory_analysis.behavior.video_qc import annotation_tools
from brain_observatory_qc.data_access import from_lims


def create_save_pupil_dlc_movie(osid, save_dir, out_fps=5, total_dur=60,
                                cmap='cool', radius=1, thickness=4
                                ):
    df = annotation_tools.read_DLC_h5file(from_lims.get_deepcut_h5_filepath(osid))
    pupil_likelihood_df = df[df.bodyparts.str.contains('pupil') & (df.coords=='likelihood')].reset_index(drop=True)
    pupil_x_df = df[df.bodyparts.str.contains('pupil') & (df.coords=='x')].reset_index(drop=True)
    pupil_y_df = df[df.bodyparts.str.contains('pupil') & (df.coords=='y')].reset_index(drop=True)
    
    # save_dir = Path(r'\\allen\programs\mindscope\workgroups\learning\qc_pupil'.replace('\\','/'))

    eye_tracking_movie = from_lims.get_eye_tracking_avi_filepath(osid)
    cap_eye = cv2.VideoCapture(str(eye_tracking_movie))
    width = int(cap_eye.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_eye.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap_eye.get(cv2.CAP_PROP_FRAME_COUNT))
    assert length == len(pupil_likelihood_df.frame_number.unique())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    movie_path = save_dir / f'pupil_osid_{osid}.mp4'

    out_num_frames = total_dur*out_fps
    frame_nums_to_capture = np.linspace(0, length-1, out_num_frames+2).astype(int)[1:-1]
    out = cv2.VideoWriter(str(movie_path), fourcc, out_fps, (width, height))

    colors = plt.cm.get_cmap(cmap)(np.linspace(0,1,100))*255
    failed_frames = []
    for frame_number in frame_nums_to_capture:
        try:
            cap_eye.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
            _, frame_eye = cap_eye.read()

            cv2.rectangle(frame_eye, (0,0), (60, 482), (255,255,255), -1)
            for bodyparts in pupil_likelihood_df.bodyparts.unique():
                likelihood = pupil_likelihood_df[(pupil_likelihood_df.bodyparts==bodyparts) &
                                                (pupil_likelihood_df.frame_number==frame_number)].value.values[0]
                color = colors[max(int(np.round(likelihood*100))-1, 0)]
                center_x = int(pupil_x_df[(pupil_x_df.bodyparts==bodyparts) &
                                        (pupil_x_df.frame_number==frame_number)].value.values[0])
                center_y = int(pupil_y_df[(pupil_y_df.bodyparts==bodyparts) &
                                        (pupil_y_df.frame_number==frame_number)].value.values[0])
                cv2.circle(frame_eye, (center_x, center_y), radius, color, thickness)

            out.write(frame_eye)
        except:
            failed_frames.append(frame_number)
            pass
    cap_eye.release()
    out.release()

    if len(failed_frames) > 0:
        fail_save_file = save_dir / f'failed_frames_osid_{osid}.npy'
        np.save(fail_save_file, failed_frames)
    


