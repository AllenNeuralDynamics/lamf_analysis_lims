import brain_observatory_qc.pipeline_dev.scripts.depth_estimation_module as dem
from pathlib import Path
import numpy as np
import pandas as pd
from brain_observatory_qc.data_access import from_lims
from multiprocessing import Pool
from argparse import ArgumentParser
import time
import os

parser = ArgumentParser(description='arguments for multiplane z-drift calculation')
parser.add_argument(
    '--osid',
    type=int,
    default=0,
    metavar='ophys_session_id',
    help='ophys session id'
)

def save_experiment_zdrift(oeid, save_dir):
    _ = dem.get_experiment_zdrift(oeid, ref_oeid=oeid, save_dir=save_dir)

if __name__ == '__main__':
    t0 = time.time()
    args = parser.parse_args()
    osid = args.osid
    oeids = from_lims.get_ophys_experiment_ids_for_ophys_session_id(osid).ophys_experiment_id.values

    if os.name == 'nt':
        save_dir = Path(r'\\allen\programs\mindscope\workgroups\learning\ophys\zdrift')
    else:
        save_dir = Path(r'/allen/programs/mindscope/workgroups/learning/ophys/zdrift')
    
    num_processes = 8
    with Pool(num_processes) as pool:
        pool.starmap(save_experiment_zdrift, [(oeid, save_dir) for oeid in oeids])
    # oeid = oeids[1]
    # save_experiment_zdrift(oeid, save_dir)
    t1 = time.time()
    print(f'total time = {(t1 - t0)/60:.1f} minutes')
    