from ScanImageTiffReader import ScanImageTiffReader
import json
from time import time
from pathlib import Path

def _extract_dict_from_si_string(string):
    """Parse the 'SI' variables from a scanimage metadata string"""

    lines = string.split('\n')
    data_dict = {}
    for line in lines:
        if line.strip():  # Check if the line is not empty
            key, value = line.split(' = ')
            key = key.strip()
            if value.strip() == 'true':
                value = True
            elif value.strip() == 'false':
                value = False
            else:
                value = value.strip().strip("'")  # Remove leading/trailing whitespace and single quotes
            data_dict[key] = value

    json_data = json.dumps(data_dict, indent=2)
    loaded_data_dict = json.loads(json_data)
    return loaded_data_dict


def _str_to_int_list(string):
    return [int(s) for s in string.strip('[]').split()]


def _str_to_bool_list(string):
    return [bool(s) for s in string.strip('[]').split()]

def metadata_from_scanimage_tif(stack_path):
    """Extract metadata from ScanImage tiff stack

    Dev notes:
    Seems awkward to parse this way
    Depends on ScanImageTiffReader

    Parameters
    ----------
    stack_path : str
        Path to tiff stack

    Returns
    -------
    dict
        stack_metadata: important metadata extracted from scanimage tiff header
    dict
        si_metadata: all scanimge metadata. Each value still a string, so convert if needed.
    dict
        roi_groups_dict: 
    """
    with ScanImageTiffReader(str(stack_path)) as reader:
        md_string = reader.metadata()

    # split si & roi groups, prep for seprate parse
    s = md_string.split("\n{")
    rg_str = "{" + s[1]
    si_str = s[0]

    # parse 1: extract keys and values, dump, then load again
    si_metadata = _extract_dict_from_si_string(si_str)
    # parse 2: json loads works hurray
    roi_groups_dict = json.loads(rg_str)

    stack_metadata = {}
    stack_metadata['num_slices'] = int(si_metadata['SI.hStackManager.actualNumSlices'])
    stack_metadata['num_volumes'] = int(si_metadata['SI.hStackManager.actualNumVolumes'])
    stack_metadata['frames_per_slice'] = int(si_metadata['SI.hStackManager.framesPerSlice'])
    stack_metadata['z_steps'] = _str_to_int_list(si_metadata['SI.hStackManager.zs'])
    stack_metadata['actuator'] = si_metadata['SI.hStackManager.stackActuator']
    stack_metadata['num_channels'] = sum(_str_to_bool_list(si_metadata['SI.hPmts.powersOn']))
    stack_metadata['z_step_size'] = int(si_metadata['SI.hStackManager.actualStackZStepSize'])

    return stack_metadata, si_metadata, roi_groups_dict

t0 = time()
data_dir = Path(r'\\allen\programs\mindscope\workgroups\learning\pilots\online_motion_correction\mouse_721291\test_240515_721291'.replace('\\','/'))

si_fn = '240515_721291_global_30min_1366658085_timeseries_00006.tif'
si_fn = data_dir / si_fn
stack_metadata, si_metadata, roi_groups_dict = metadata_from_scanimage_tif(si_fn)
t1 = time()
print(f'{(t1-t0)} s')