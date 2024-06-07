import os
import argparse
import time
from simple_slurm import Slurm
from pathlib import Path
from glob import glob

parser = argparse.ArgumentParser(description='running sbatch for omc_zdrift.py')  # noqa: E501
parser.add_argument('--env-path', type=str, default='/home/jinho.kim/anaconda3/envs/suite2p', metavar='path to conda environment to use')  # noqa: E501

if __name__ == '__main__':
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))

    ####################
    # Choose python file 
    python_dir = Path('/home/jinho.kim/Github/lamf_ophys_analysis_dev/qc/2p_data/online_motion_correction')  # noqa: E501
    python_file = python_dir / 'omc_zdrift.py'

    ####################
    # Choose job_dir for saving the job records
    job_dir = Path(r'\allen\programs\mindscope\workgroups\learning\pilots\online_motion_correction\mouse_726433\test_240531'.replace('\\', '/'))  # noqa: E501
    stdout_location = job_dir / 'job_records'
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.makedirs(stdout_location)

    #####################
    fn_list = glob(str(job_dir / '*_timeseries_*_emf.tif'))
    # fn_list = [str(job_dir / '240515_721291_global_30min_1366658085_timeseries_00006_00_emf.tif')]
    zstack_dir = str(job_dir / 'ophys_session_1369518919')
    job_count = 0

    rerun = False
    for fn in fn_list:
    # fn = fn_list[0]

        job_name = Path(fn).name.split('.')[0]
        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(job_name, job_count))  # noqa: E501
        job_title = 'filename_base_{}'.format(job_name)
        walltime = '1:00:00'
        cpus_per_task = 1
        mem = '20gb'
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location / f'{job_array_id}_{job_id}_{job_name}.out'

        # instantiate a SLURM object
        slurm = Slurm(
            cpus_per_task=cpus_per_task,
            job_name=job_title,
            time=walltime,
            mem=mem,
            output=output,
            partition="braintv"
        )

        #####################
        # Argument string - need space between arguments
        args_string = f'--file_path {fn} --zstack_dir {zstack_dir}'
        slurm.sbatch('{} {} {}'.format(
                    python_executable,
                    python_file,
                    args_string,
                    )
                    )
        time.sleep(0.01)
