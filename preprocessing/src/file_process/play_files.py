import sys
import argparse
import confuse

import sys
sys.path.append(sys.path[0][:-13])
from extract_bag import extract_data
from add_noise import AddNoise

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
    
    if config['extract_bag']['enable'].get():
        # Extract the raw filtered rosbags
        NOISY_FLAG = False
        extract_data(config, NOISY_FLAG)
        # Extract the noisy rosbags
        if config['extract_bag']['noisy'].get():
            NOISY_FLAG = True
            extract_data(config, NOISY_FLAG)
        
        
    model = AddNoise(config)
    
    # Process the viewport 0
    viewport0 = config['path'].get() + '/Viewport0'
    model.play_files(viewport0)
    
    # Process the viewport 0 with occulusion
    viewport1 = config['path'].get() + '/Viewport0_occluded'
    model.play_files(viewport1)