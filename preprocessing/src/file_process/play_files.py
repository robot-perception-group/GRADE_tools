import sys
import argparse
import confuse

import sys
sys.path.append(sys.path[0][:-13])
from preprocessing.src.extract_data.extract_data import extract_data
from add_noise import AddNoise

if __name__ == '__main__':
    # Define parser arguments
    parser = argparse.ArgumentParser(description="Isaac bag player")
    parser.add_argument("--path", type=str, help="Directory of Expriment Folder")
    parser.add_argument("--config", type=str, help="Config File for File Process")
    parser.add_argument("--output", type=str, default=None, help="Output Path for Processed Files")
    args, _ = parser.parse_known_args()

    # load configuration file
    config = confuse.Configuration("IsaacBag", __name__)
    config.set_file(args.config)
    config.set_args(args)
        
    model = AddNoise(config)

    # Process the viewport 0 with occulusion
    model.play_files()