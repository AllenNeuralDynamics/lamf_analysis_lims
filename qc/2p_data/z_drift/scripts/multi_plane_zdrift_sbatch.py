import os
import argparse
import time
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
from simple_slurm import Slurm
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='running sbatch for multi_plane_zdrift.py')  # noqa: E501
parser.add_argument('--env-path', type=str, default='/home/jinho.kim/anaconda3/envs/allenhpc', metavar='path to conda environment to use')  # noqa: E501

if __name__ == '__main__':
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))

    ####################
    # Choose python file 
    python_dir = Path('/home/jinho.kim/Github/lamf_ophys_analysis_dev/qc/2p_data/z_drift/scripts/')  # noqa: E501
    python_file = python_dir / 'multi_plane_zdrift.py'

    ####################
    # Choose job_dir for saving the job records
    job_dir = Path(r'\allen\programs\mindscope\workgroups\learning\ophys\zdrift'.replace('\\', '/'))  # noqa: E501
    stdout_location = job_dir / 'job_records'
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.makedirs(stdout_location)

    #####################
    # osids to run    
    # osid_table = job_dir / 'multi_plane_zdrift_vb_multiscope.csv'
    # osid_df = pd.read_csv(osid_table)
    # osids = osid_df['ophys_session_id'].values

    # job_records_dir = Path(r'\allen\programs\mindscope\workgroups\learning\ophys\zdrift\job_records'.replace('\\', '/'))
    # time_since = datetime(2024, 3, 26, 0, 0, 0)
    # files = [f for f in job_records_dir.glob('**/*') if f.is_file() and f.stat().st_ctime > time_since.timestamp()]
    # failed_files = []
    # succeeded_osids = []
    # failed_osids = []
    # success_word = 'total time ='
    # for file in files:
    #     with open(file, 'r') as f:
    #         lines = f.readlines()
    #         if np.array([success_word in l for l in lines]).any():
    #             succeeded_osids.append(int(file.name.split('_')[-1].split('.')[0]))
    #         else:
    #             failed_files.append(file)
    #             failed_osids.append(int(file.name.split('_')[-1].split('.')[0]))

    # osid_table = job_dir / 'multiplane_zdrift_osids_240223.csv'
    # osid_df = pd.read_csv(osid_table)
    # osids = osid_df['ophys_session_id'].values

    # cache = bpc.from_lims()
    # table = cache.get_ophys_experiment_table(passed_only=False)
    # lamf_table = table.query('project_code == "LearningmFISHTask1A"')
    # lamf_table = lamf_table[~(lamf_table.session_type.str.contains("TRAINING_0_"))]        
    # zdrift_test_table = lamf_table.groupby('ophys_session_id').apply(lambda x: len(x.targeted_structure.unique())<=2)
    # osids = zdrift_test_table[zdrift_test_table].index.values

    # prev_job_records_dir = Path(r'\allen\programs\mindscope\workgroups\learning\ophys\zdrift\job_records'.replace('\\', '/'))
    # # filter by created time, if needed
    # time_since = datetime(2024, 2, 22, 0, 0, 0)
    # # from job_dir get files created after time_since
    # files = [f for f in prev_job_records_dir.glob('**/*') if f.is_file() and f.stat().st_ctime > time_since.timestamp()]
    # # read and find the ones that failed
    # failed_osids = []
    # success_word = 'total time ='
    # for file in files:
    #     with open(file, 'r') as f:
    #         lines = f.readlines()
    #         if np.array([success_word in l for l in lines]).any():
    #             continue
    #         else:
    #             failed_osids.append(int(file.name.split('_')[-1].split('.')[0]))
    # osids = failed_osids
        
    cache = bpc.from_lims()
    table = cache.get_ophys_experiment_table(passed_only=False)    
    project_codes = ["omFISHSstMeso", "omFISHGad2Meso", "omFISHCux2Meso", "omFISHRbp4Meso"]
    num_planes = [8, 8, 6, 4]
    omfish_osids = []
    for i in range(4):
        code = project_codes[i]
        num_plane = num_planes[i]
        temp_table = table.query(f'project_code == "{code}"')
        temp_table = table.query(f'project_code == "{code}"')
        num_plane_table = temp_table.groupby('ophys_session_id').apply(lambda x: len(x)==num_plane)
        temp_osids = num_plane_table[num_plane_table].index.values
        omfish_osids.extend(temp_osids)
        
    job_count = 0

    rerun = False
    for osid in omfish_osids:
        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(osid, job_count))  # noqa: E501
        job_title = 'osid_{}'.format(osid)
        walltime = '4:00:00'
        mem = '100gb'
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location / f'{job_array_id}_{job_id}_{osid}.out'

        # instantiate a SLURM object
        slurm = Slurm(
            cpus_per_task=8,
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
