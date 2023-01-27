import sys
import argparse
import confuse

sys.path.append(sys.path[0][:-13])
from extract_data import extract_data

if __name__ == '__main__':
    # Define parser arguments
    parser = argparse.ArgumentParser(description="Isaac bag player")
    parser.add_argument("--path", type=str, help="Directory of Expriment Folder")
    parser.add_argument("--config", type=str, help="Config File for File Process")
    args, _ = parser.parse_known_args()

    # load configuration file
    config = confuse.Configuration("IsaacBag", __name__)
    config.set_file(args.config)
    config.set_args(args)
    
    # Extract the reindex/noisy rosbags
    extract_data(config)