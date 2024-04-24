import os
import argparse
import time
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
from simple_slurm import Slurm
from pathlib import Path
from datetime import datetime
import numpy as np


parser = argparse.ArgumentParser(description='running sbatch for save_pupil_convhull_area.py')  # noqa: E501
parser.add_argument('--env-path', type=str, default='/home/jinho.kim/anaconda3/envs/allenhpc', metavar='path to conda environment to use')  # noqa: E501

if __name__ == '__main__':
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))

    ####################
    # Choose python file 
    python_dir = Path('/home/jinho.kim/Github/lamf_ophys_analysis_dev/qc/behavior_data')  # noqa: E501
    python_file = python_dir / 'save_pupil_conv_hull_area.py'

    ####################
    # Choose job_dir for saving the job records
    job_dir = Path(r'\\allen\programs\mindscope\workgroups\learning\qc_pupil\pupil_convhull_area'.replace('\\', '/'))  # noqa: E501
    stdout_location = job_dir / 'job_records'
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.makedirs(stdout_location)

    #####################
    # osids to run
    cache = bpc.from_lims()
    table = cache.get_ophys_experiment_table(passed_only=False)    
    project_codes = ['VisualBehaviorMultiscope', 'VisualBehaviorMultiscope4areasx2d', 'LearningmFISHTask1A', 'omFISHGad2Meso']
    table = table[table.project_code.isin(project_codes)]
    last_datetime = datetime(2024, 3, 27, 0, 0, 0)
    table = table[table.date_of_acquisition < last_datetime]
    osids = table.ophys_session_id.unique()
    
    # time_since = datetime(2024, 3, 26, 15, 0, 0)
    # # from job_dir get files created after time_since
    # files = [f for f in stdout_location.glob('**/*') if f.is_file() and f.stat().st_ctime > time_since.timestamp()]
    # succeeded_osids = []
    # failed_osids = []
    # success_word = 'total time ='
    # for file in files:
    #     with open(file, 'r') as f:
    #         lines = f.readlines()
    #         if np.array([success_word in l for l in lines]).any():
    #             succeeded_osids.append(int(file.name.split('_')[-1].split('.')[0]))
    #         else:
    #             failed_osids.append(int(file.name.split('_')[-1].split('.')[0]))
    
    # osids_to_run = [osid for osid in osids if osid not in succeeded_osids]
    osids_to_run = osids
        
    job_count = 0

    rerun = False
    for osid in osids_to_run:
        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(osid, job_count))  # noqa: E501
        job_title = 'osid_{}'.format(osid)
        walltime = '0:15:00'
        mem = '10gb'
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location / f'{job_array_id}_{job_id}_{osid}.out'

        # instantiate a SLURM object
        slurm = Slurm(
            cpus_per_task=1,
            job_name=job_title,
            time=walltime,
            mem=mem,
            output=output,
            partition="braintv"
        )

        #####################
        # Argument string - need space between arguments
        args_string = f'--osid {osid}'

        slurm.sbatch('{} {} {}'.format(
                python_executable,
                python_file,
                args_string,
            )
        )
        time.sleep(0.01)
