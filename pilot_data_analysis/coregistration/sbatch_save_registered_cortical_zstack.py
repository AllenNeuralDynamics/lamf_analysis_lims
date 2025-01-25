import os
import argparse
import time
from simple_slurm import Slurm
from pathlib import Path
from glob import glob

parser = argparse.ArgumentParser(description='running sbatch for save_registered_cortical_zstack.py')  # noqa: E501
parser.add_argument('--env-path', type=str, default='/home/jinho.kim/anaconda3/envs/allenhpc', metavar='path to conda environment to use')  # noqa: E501

if __name__ == '__main__':
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))

    ####################
    # Choose python file 
    python_dir = Path('/home/jinho.kim/Github/lamf_ophys_analysis_dev/pilot_data_analysis/coregistration')  # noqa: E501
    python_file = python_dir / 'save_registered_cortical_zstack.py'

    ####################
    # Choose job_dir for saving the job records
    job_dir = Path(r'\\allen\programs\mindscope\workgroups\learning\coreg\cortical_zstacks'.replace('\\', '/'))  # noqa: E501
    stdout_location = job_dir / 'job_records'
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.makedirs(stdout_location)

    #####################
    # Parameters
    cortical_zstack_paths = [r"\\allen\programs\mindscope\production\learning\prod1\specimen_1385236652\20240927_743116_visp_week1_retake_redgreen_00001.tif".replace('\\', '/'),
                            r"\\allen\programs\mindscope\production\learning\prod1\specimen_1385236652\20241021_743116_week4_visp_redgreen_00001.tif".replace('\\', '/'),
                            r"\\allen\programs\mindscope\production\learning\prod1\specimen_1381875512\20240930_753014_visp_week1_cortical_redgreen_00001.tif".replace('\\', '/'),
                            r"\\allen\programs\mindscope\production\learning\prod1\specimen_1381875512\20241021_753014_week4_visp_redgreen_00001.tif".replace('\\', '/'),
                            r"\\allen\programs\mindscope\production\learning\prod1\specimen_1384824885\20240930_753016_visp_week1_cortical_redgreen_00001.tif".replace('\\', '/'),
                            r"\\allen\programs\mindscope\production\learning\prod1\specimen_1384824885\20241021_753016_week4_visp_redgreen_00001.tif".replace('\\', '/')]
    mouse_ids = [czp.split('/')[-1].split('_')[1] for czp in cortical_zstack_paths]
    ref_channel = 1

    job_count = 0

    rerun = False
    for fi, fn in enumerate(cortical_zstack_paths):
    # fn = fn_list[0]
        mouse_id = mouse_ids[fi]
        job_name = Path(fn).name.split('.')[0]
        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(job_name, job_count))  # noqa: E501
        job_title = 'filename_base_{}'.format(job_name)
        walltime = '2:30:00'
        cpus_per_task = 20
        mem = '200gb'
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
        args_string = f'--file_path {fn} --mouse_id {mouse_id} --ref_channel {ref_channel}'
        slurm.sbatch('{} {} {}'.format(
                    python_executable,
                    python_file,
                    args_string,
                    )
                    )
        time.sleep(0.01)
