import os
import argparse
import time
from simple_slurm import Slurm
from pathlib import Path
from glob import glob

parser = argparse.ArgumentParser(description='running sbatch for split_reg_emf.py')  # noqa: E501
parser.add_argument('--env-path', type=str, default='/home/jinho.kim/anaconda3/envs/suite2p', metavar='path to conda environment to use')  # noqa: E501

if __name__ == '__main__':
    
    #########################
    ##### Important parameter
    epoch_minutes = 3
    num_planes = 8



    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))

    ####################
    # Choose python file 
    python_dir = Path('/home/jinho.kim/Github/lamf_ophys_analysis_dev/qc/2p_data/online_motion_correction')  # noqa: E501
    python_file = python_dir / 'split_reg_emf_slurm.py'

    ####################
    # Choose job_dir for saving the job records
    job_dir = Path(r'\allen\programs\mindscope\workgroups\learning\pilots\online_motion_correction\mouse_721291\test_240515_721291'.replace('\\', '/'))  # noqa: E501
    stdout_location = job_dir / 'job_records'
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.makedirs(stdout_location)

    #####################
    fn_list = glob(str(job_dir / '*_timeseries_*.tif'))
    emf_fn_list = glob(str(job_dir / '*_timeseries_*_emf.tif'))
    fn_list = [fn for fn in fn_list if fn not in emf_fn_list]
    # fn_list = [str(job_dir / '240515_721291_nomotioncorrection_timeseries_30min.tif')]
        
    job_count = 0

    rerun = False
    for fn in fn_list:
        for plane_index in range(num_planes):
            job_name = Path(fn).name.split('.')[0] + f'plane_index_{plane_index}'
            job_count += 1
            print('starting cluster job for {}, job count = {}'.format(job_name, job_count))  # noqa: E501
            job_title = 'filename_base_{}'.format(job_name)
            walltime = '2:00:00'
            cpus_per_task = 20
            mem = '100gb'
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
            args_string = f'--file_path \"{fn}\" --plane_index {plane_index} --epoch_minutes {epoch_minutes}'
            slurm.sbatch('{} {} {}'.format(
                        python_executable,
                        python_file,
                        args_string,
                        )
                        )
            time.sleep(0.01)
