from pathlib import Path
from lamf_analysis.ophys import zstack
import time

from argparse import ArgumentParser
parser = ArgumentParser(description='arguments for registering and saving cortical zstacks')
parser.add_argument(
    '--file_path',
    type=str,
    default='',
    metavar='file_path',
    help='file path to the cortical zstack tiff file'
)

parser.add_argument(
    '--mouse_id',
    type=str,
    default='',
    metavar='mouse_id',
    help='mouse id'
)

parser.add_argument(
    '--ref_channel',
    type=int,
    default=None,
    metavar='ref_channel',
    help='reference channel'
)

parser.add_argument(
    '--output_dir_base',
    type=str,
    default=r'\allen\programs\mindscope\workgroups\learning\coreg\cortical_zstacks'.replace('\\', '/'),
    metavar='output_dir_base',
    help='base directory to save the results'
)


if __name__ == '__main__':
    args = parser.parse_args()
    filepath = Path(args.file_path)
    output_dir_base = Path(args.output_dir_base)
    mouse_id = args.mouse_id
    ref_channel = args.ref_channel
    t0 = time.time()

    # saving
    output_dir = output_dir_base / mouse_id
    zstack.register_cortical_stack(filepath, save=True, output_dir=output_dir, qc_plots=True,
                                   ref_channel=ref_channel)

    t1 = time.time()
    print(f'Done in {(t1-t0)/60:.2f} min.')
