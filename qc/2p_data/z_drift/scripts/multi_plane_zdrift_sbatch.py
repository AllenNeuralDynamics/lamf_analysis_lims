import os
import argparse
import time
import pandas as pd
from simple_slurm import Slurm
from pathlib import Path

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
    osid_table = job_dir / 'multiplane_zdrift_osids_240222.csv'
    osid_df = pd.read_csv(osid_table)
    osids = osid_df['ophys_session_id'].values
        
    job_count = 0

    rerun = False
    for osid in osids:
        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(osid, job_count))  # noqa: E501
        job_title = 'osid_{}'.format(osid)
        walltime = '1:00:00'
        mem = '50gb'
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
