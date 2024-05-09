from allensdk.brain_observatory.behavior.behavior_ophys_experiment import \
    BehaviorOphysExperiment
from brain_observatory_qc.data_access import from_lims    
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from pathlib import Path


def get_rewards_df(exp):
    rewards = exp.rewards.copy()
    rewards['acc_volume'] = rewards.volume.cumsum()
    return rewards

def get_running_df(exp):
    running = exp.running_speed.copy()
    running['dt'] = running.timestamps.diff()
    running['run_dist'] = running.dt * running.speed
    running['acc_run_dist'] = running.run_dist.cumsum()
    return running

def save_behavior_df(osid, save_dir):
    oeid = from_lims.get_ophys_experiment_ids_for_ophys_session_id(osid).ophys_experiment_id.values[0]
    exp = BehaviorOphysExperiment.from_lims(oeid)
    rewards_df = get_rewards_df(exp)
    running_df = get_running_df(exp)
    rewards_save_fn = save_dir / f'osid_{osid}_rewards.pkl'
    rewards_df.to_pickle(rewards_save_fn)
    running_save_fn = save_dir / f'osid_{osid}_running.pkl'
    running_df.to_pickle(running_save_fn)
    

if __name__ == '__main__':
    zdrift_dir = Path(r'\\allen\programs\mindscope\workgroups\learning\ophys\zdrift')
    save_dir = zdrift_dir / 'behavior_df'

    cc_threshold = 0.7
    plane_interval = 0.75

    lamf_table_fn = zdrift_dir / 'multiplane_zdrift_lamf.pkl'
    lamf_table = pd.read_pickle(lamf_table_fn)

    lamf_table['min_cc_episodes'] = lamf_table.apply(lambda x: np.min([x['cc_episodes'][0], x['cc_episodes'][-1]]), axis=1)
    lamf_table['min_cc_fl'] = lamf_table.apply(lambda x: np.min(x.cc_fl), axis=1)
    lamf_table['zdrift_episodes'] = lamf_table.apply(lambda x: plane_interval * (x.mpi_episodes[-1] - x.mpi_episodes[0]), axis=1)
    lamf_table['zdrift_fl'] = lamf_table.apply(lambda x: plane_interval * (x.mpi_fl[1] - x.mpi_fl[0]), axis=1)

    passed_osids = lamf_table.groupby('ophys_session_id').apply(lambda x: np.min(x['min_cc_episodes']) >= cc_threshold)
    passed_osids = passed_osids[passed_osids]
    passed_osids = passed_osids.index.values
    lamf_table = lamf_table.query('ophys_session_id in @passed_osids')


    vbm_table_fn = zdrift_dir / 'multiplane_zdrift_vbm.pkl'
    vbm_table = pd.read_pickle(vbm_table_fn)
    bad_imaging_sessions = [977760370, 1081070236, 1075872563, 906968227,
                            1088200327, 884613038, 921636320, 876303107,
                            923705570, 865024413, 958105827, 976167513, 981845703]
    filtered_table = vbm_table[~vbm_table.ophys_session_id.isin(bad_imaging_sessions)]
    vbm_table = filtered_table[~filtered_table.session_type.str.contains('passive') &
                                ~filtered_table.session_type.str.contains('OPHYS_7')].copy()
    # add mouse IDs
    vbm_table.set_index('oeid', inplace=True)
    vbm_table.reset_index(drop=False, inplace=True)

    vbm_table['min_cc_episodes'] = vbm_table.apply(lambda x: np.min([x['cc_episodes'][0], x['cc_episodes'][-1]]), axis=1)
    vbm_table['min_cc_fl'] = vbm_table.apply(lambda x: np.min(x.cc_fl), axis=1)
    vbm_table['zdrift_episodes'] = vbm_table.apply(lambda x: plane_interval * (x.mpi_episodes[-1] - x.mpi_episodes[0]), axis=1)
    vbm_table['zdrift_fl'] = vbm_table.apply(lambda x: plane_interval * (x.mpi_fl[1] - x.mpi_fl[0]), axis=1)

    passed_osids = vbm_table.groupby('osid').apply(lambda x: np.min(x['min_cc_episodes']) >= cc_threshold)
    passed_osids = passed_osids[passed_osids]
    passed_osids = passed_osids.index.values
    vbm_table = vbm_table.query('osid in @passed_osids')

    exc_table = vbm_table[vbm_table.cre_line.str.contains('Slc17a7')].copy()

    osids = np.union1d(lamf_table.ophys_session_id.unique(), exc_table.osid.unique())

    saved_osids = [int(fn.name.split('_')[1]) for fn in list(save_dir.glob('*_rewards.pkl'))]
    failed_osids = np.setdiff1d(osids, saved_osids)

    num_core = cpu_count() - 2
    print(f'using {num_core} cores.')
    print(f'number of osids: {len(osids)}')
    # save_behavior_df(osids[0], save_dir)
    with Pool(num_core) as pool:
        # pool.starmap(save_behavior_df, [(osid, save_dir) for osid in osids])
        pool.starmap(save_behavior_df, [(osid, save_dir) for osid in failed_osids])



    