import confuse
import argparse

import sys
sys.path.append(sys.path[0][:-12])
from time_correction import ReIndex
from add_noise import AddNoise


if __name__ == '__main__':
    # Define parser arguments
    parser = argparse.ArgumentParser(description="Isaac bag player")
    parser.add_argument("--path", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--signal_topic", type=str, default="/starting_experiment")
    args, _ = parser.parse_known_args()

    # load configuration file
    config = confuse.Configuration("IsaacBag", __name__)
    config.set_file(args.config)
    config.set_args(args)

    if config['time_correction']['enable'].get() == True:
        bags = ReIndex(config)
        bags = bags.play_bags()
    
    if config['noise'].get() == True:
        bags = AddNoise(config)
        bags.play_bags()
    
