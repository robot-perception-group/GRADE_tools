import os
import rosbag
import confuse
import argparse
import numpy as np
import matplotlib.pyplot as plt

class MsgSequence:
    def __init__(self, config):
        self.config = config

        self.last_ts = {}  # Record the final timestamp
        self.last_seq = {}  # Record the final sequence number
        self.init_ts = {}  # Record the first timestamp
        self.init_seq = {}  # Record the first sequence number
        self.prev_ts = {}  # Record the previous timestamp
        self.msg_intervals = {}

        self.bag_dir = self.config['path'].get()
        self.bags = os.listdir(self.bag_dir)
        self.bags = [bag for bag in self.bags if (
            '.bag' in bag) and '.orig' not in bag]
        self.bags.sort()

        self.topics = []
        for topic_type in self.config['topics'].keys():
            for topic_name in self.config['topics'][topic_type].keys():
                self.topics.append(topic_name)
        
    def play_bags(self):
        for bag in self.bags:

            print("Playing bag", bag)

            # Define the rosbag data
            bag_path = os.path.join(self.bag_dir, bag)
            bag = rosbag.Bag(bag_path)

            # Obtain the Useful Stamp Information
            for topic, msg, _ in bag.read_messages(topics=self.topics):
                                
                if topic == '/starting_experiment':
                    continue

                if topic == '/tf':
                    # Filter the tf transforms that we don't need
                    if len(msg.transforms) == 0 or len(msg.transforms) == 1:
                        continue
                    else:
                        # Define the initial sequence number and timestamp for tf messages
                        if topic not in self.init_ts.keys():
                            self.init_ts[topic] = msg.transforms[0].header.stamp.to_sec(
                            )
                            self.init_seq[topic] = msg.transforms[0].header.seq
                            self.prev_ts[topic] = [0.0]
                            self.msg_intervals[topic] = []
                        # Iterate to obtain the final sequence number and timestamp for tf message
                        curr_ts = msg.transforms[0].header.stamp.to_sec()
                        self.last_ts[topic] = curr_ts
                        self.last_seq[topic] = msg.transforms[0].header.seq
                        self.msg_intervals[topic].append(
                            curr_ts - self.prev_ts[topic][-1])
                        self.prev_ts[topic].append(curr_ts)
                else:
                    # Define the initial sequence number and timestamp for required messages
                    if topic not in self.init_ts.keys():
                        self.init_ts[topic] = msg.header.stamp.to_sec()
                        self.init_seq[topic] = msg.header.seq
                        self.prev_ts[topic] = [0.0]
                        self.msg_intervals[topic] = []
                    # Iterate to obtain the final sequence number and timestamp for tf message
                    curr_ts = msg.header.stamp.to_sec()
                    self.last_ts[topic] = curr_ts
                    self.last_seq[topic] = msg.header.seq
                    self.msg_intervals[topic].append(
                        curr_ts- self.prev_ts[topic][-1])
                    self.prev_ts[topic].append(curr_ts)
            bag.close()

        # Visualize the statistic result
        for key in self.last_seq.keys():
            print("TOPIC \"%s\" has %d messages" %
                  (key, self.last_seq[key]-self.init_seq[key]+1))
            print("TOPIC \"%s\" started from time %.5f" %
                  (key, self.init_ts[key]))
            print("TOPIC \"%s\" ended at time %.5f \n\n" %
                  (key, self.last_ts[key]))
            
            intervals = self.msg_intervals[key]
            plt.scatter(range(len(intervals)), intervals, s=1, label=key)
            plt.title('Timestamp Interval for [%s]' %key)
            plt.legend()
            plt.show()
        

if __name__ == '__main__':
    # Define parser arguments
    parser = argparse.ArgumentParser(description="Isaac bag player")
    parser.add_argument("--path", type=str, help='reindex bags folders')
    parser.add_argument("--config", type=str, default='preprocessing/config/bag_process.yaml', help='path to bag_process.yaml')
    args, _ = parser.parse_known_args()

    # load configuration file
    config = confuse.Configuration("IsaacBag", __name__)
    config.set_file(args.config)
    config.set_args(args)

    sequences = MsgSequence(config)
    sequences.play_bags()
