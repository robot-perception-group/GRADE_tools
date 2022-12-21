import os
import rosbag
import confuse
import argparse
import numpy as np
import matplotlib.pyplot as plt

from quaternion import quaternion

class MsgSequence:
    def __init__(self, config):
        self.config = config

        self.last_ts = {}  # Record the final timestamp
        self.last_seq = {}  # Record the final sequence number
        self.init_ts = {}  # Record the first timestamp
        self.init_seq = {}  # Record the first sequence number
        self.prev_ts = {}  # Record the previrous timestamp
        self.msg_intervals = {}
        
        self.imu_bodys = []
        self.odoms = []

        self.bag_dir = self.config['bag_folder'].get()
        self.bags = os.listdir(self.bag_dir)
        self.bags = [bag for bag in self.bags if (
            '.bag' in bag) and '.orig' not in bag]
        self.bags.sort()

        self.topics = self.config['topics'].get()
        
    def play_bags(self):
        for bag in self.bags:

            print("Playing bag", bag)

            # Define the rosbag data
            bag_path = os.path.join(self.bag_dir, bag)
            bag = rosbag.Bag(bag_path)

            # Obtain the Useful Stamp Information
            for topic, msg, _ in bag.read_messages(topics=self.topics):
                print(topic, _.to_sec())
                
                if topic == '/starting_experiment':
                    continue

                if topic == '/tf':
                    # Filter the tf transforms that we don't need
                    if len(msg.transforms) == 0 or len(msg.transforms) == 1:
                        continue
                    # elif 'reference' in msg.transforms[0].child_frame_id:
                    #     # Define the initial sequence number and timestamp for tf messages
                    #     if 'tf_reference' not in self.init_ts.keys():
                    #         self.init_ts['tf_reference'] = msg.transforms[0].header.stamp.to_sec()
                    #         self.init_seq['tf_reference'] = msg.transforms[0].header.seq
                    #         self.prev_ts['tf_reference'] = 0.0
                    #         self.msg_intervals['tf_reference'] = []
                    #     # Iterate to obtain the final sequence number and timestamp for tf message
                    #     self.last_ts['tf_reference'] = msg.transforms[0].header.stamp.to_sec()
                    #     self.last_seq['tf_reference'] = msg.transforms[0].header.seq
                    #     # Record the interval with the previours data
                    #     self.msg_intervals['tf_reference'].append(
                    #         msg.transforms[0].header.stamp.to_sec() - self.prev_ts['tf_reference'])
                    #     self.prev_ts['tf_reference'] = msg.transforms[0].header.stamp.to_sec()
                    else:
                        # Define the initial sequence number and timestamp for tf messages
                        if topic not in self.init_ts.keys():
                            self.init_ts[topic] = msg.transforms[0].header.stamp.to_sec(
                            )
                            self.init_seq[topic] = msg.transforms[0].header.seq
                            self.prev_ts[topic] = [0.0]
                            self.msg_intervals[topic] = []
                        # Iterate to obtain the final sequence number and timestamp for tf message
                        self.last_ts[topic] = msg.transforms[0].header.stamp.to_sec()
                        self.last_seq[topic] = msg.transforms[0].header.seq
                        self.msg_intervals[topic].append(
                            msg.transforms[0].header.stamp.to_sec() - self.prev_ts[topic][-1])
                        self.prev_ts[topic].append(msg.transforms[0].header.stamp.to_sec())
                else:
                    # Define the initial sequence number and timestamp for required messages
                    if topic not in self.init_ts.keys():
                        self.init_ts[topic] = msg.header.stamp.to_sec()
                        self.init_seq[topic] = msg.header.seq
                        self.prev_ts[topic] = [0.0]
                        self.msg_intervals[topic] = []
                    # Iterate to obtain the final sequence number and timestamp for tf message
                    self.last_ts[topic] = msg.header.stamp.to_sec()
                    self.last_seq[topic] = msg.header.seq
                    self.msg_intervals[topic].append(
                        msg.header.stamp.to_sec() - self.prev_ts[topic][-1])
                    self.prev_ts[topic].append(msg.header.stamp.to_sec())
                    
                    if 'odom' in topic:
                        self.odoms.append(msg)
                        
                    if 'imu_body' in topic:
                        self.imu_bodys.append(msg)
                        
        self.ang_vel_verify()

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
        plt.legend()
        plt.show()

    def ang_vel_verify(self):
        for i in range(1, len(self.prev_ts["/my_robot_0/imu_body"])):
            for j in range(1, len(self.prev_ts["/my_robot_0/odom"])):
                # find the similar timestamp
                if self.prev_ts["/my_robot_0/imu_body"][i] == self.prev_ts["/my_robot_0/odom"][j]:
                    # Obtain the rotation matrix from odom data
                    orientation = self.odoms[j-1].pose.pose.orientation
                    q = quaternion(orientation.x, orientation.y, orientation.z, orientation.w)
                    rot = q.from_quaternion_to_rotation_matrix()
                    
                    # Obtain local angular velocity from imu data
                    ang_vel = self.imu_bodys[i-1].angular_velocity
                    omega = np.array([ang_vel.x, ang_vel.y, ang_vel.z])
                    
                    # Calculate the angular veloicy w.r.t the world frame
                    omega_world = np.matmul(rot, omega)
                    
                    omega_error = np.abs(np.array(
                        [omega_world[0] - self.odoms[j-1].twist.twist.angular.x,
                         omega_world[1] - self.odoms[j-1].twist.twist.angular.y,
                         omega_world[2] - self.odoms[j-1].twist.twist.angular.z])) > 10**(-6)
                    
                    if omega_error.any():
                        print("Angular Velocity of IMU Data [x: %.5f, y: %.5f, z: %.5f]" 
                              %(omega_world[0], omega_world[1], omega_world[2]) )
                        print("Angular Velocity of ODOM Data:\n", self.odoms[j-1].twist.twist.angular, "\n")
        

if __name__ == '__main__':
    # Define parser arguments
    parser = argparse.ArgumentParser(description="Isaac bag player")
    parser.add_argument("--bag_folder", type=str)
    parser.add_argument("--config_file", type=str)
    args, _ = parser.parse_known_args()

    # load configuration file
    config = confuse.Configuration("IsaacBag", __name__)
    config.set_file(args.config_file)
    config.set_args(args)

    sequences = MsgSequence(config)
    sequences.play_bags()
