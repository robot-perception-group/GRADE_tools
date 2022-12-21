import os
import rosbag
import numpy as np
from rospy.rostime import Time
from cv_bridge import CvBridge


class ReIndex:
    def __init__(self, config):
        print('\n\n ==============  INITIALIZATION  ==============')
        self.config = config
        self.bridge = CvBridge()

        self.final_stamps = {}  # final timestamp for each topic of INPUT BAGS
        self.final_seqs = {}  # final sequence number for each topic of INPUT BAGS
        self.init_seqs = {}  # initial sequence that should be recorded into OUTPUT BAGS
        self.stop_seqs = {}  # final sequence that should be recorded into OUTPUT BAGS
        self.ts_new = {}

        # Load Params
        self.final_stamp = None  # final published timstamp among all topics of INPUT BAGS
        self.stop_stamp = None  # final topic timestamp that should be recorded of OUTPUT BAGS
        self.offset = None  # final timestamp with the time of starting signal

        self.missing_final_msg = False  # Flag to decide whether ignore the last interval

        self.tf_seq = 0 # initial the tf message sequence number
        self.duration = config['time_correction']['duration'].get()
        self.maximum_depth = config['camera']['config']['time_correction_maximum_depth'].get()
        
        # Define requried topic name from CONFIG files
        self.initial_topics()
        
        self.base_frame_id = config['time_correction']['base_frame_id'].get()
        
        # Load Directories        
        self.bag_dir = self.config['path'].get()
        self.reindex_bag_dir = self.bag_dir + "/reindex_bags"
        if not os.path.exists(self.reindex_bag_dir):
            os.makedirs(self.reindex_bag_dir)

        self.bags = os.listdir(self.bag_dir)
        self.bags = [bag for bag in self.bags if (
            '.bag' in bag) and ('.orig' not in bag)]
        self.bags.sort()


    def play_bags(self):
        '''Loop all bags to obtian final sequence number and final timestamp for each topic'''
        print("\n ===============  Loading bags to obtain final messages...  ===============")
        for bag in self.bags:
            print("Loading bag %s" % bag)

            bag_path = os.path.join(self.bag_dir, bag)
            bag = rosbag.Bag(bag_path)

            '''Obtain the Last Stamp'''
            for topic, msg, t in bag.read_messages(topics=self.topics):
                if topic == self.tf_topic:
                    # Filter the redundant transformations
                    if 'fake' in msg.transforms[0].header.frame_id:
                        continue

                    elif 'reference' in msg.transforms[0].child_frame_id:
                        continue

                    elif 'my_robot_0' in msg.transforms[0].child_frame_id:
                        if topic not in self.final_seqs.keys():
                            self.final_seqs[topic] = 0
                        self.final_stamps[topic] = msg.transforms[0].header.stamp.to_sec(
                        )
                        self.final_seqs[topic] += 1

                elif topic == self.signal_topic:
                    continue

                else:
                    self.final_stamps[topic] = msg.header.stamp.to_sec()
                    self.final_seqs[topic] = msg.header.seq

        '''Decide which topic is missing the final messages'''
        self.final_stamp = np.max([self.final_stamps[key]
                                  for key in self.final_stamps.keys()])

        for key in self.final_stamps.keys():
            if self.final_stamps[key] != self.final_stamp:
                # Estimate the number of missing messages
                num = round(
                    (self.final_stamp - self.final_stamps[key])*self.freqs[key])
                print("\n[WARN] Topic %s is missing %d Message!{Final_Msg: %.4f, Final_Stamp: %.4f}" % (
                    key, num, self.final_stamps[key], self.final_stamp))

                # Compensate the messages to get the correct final sequence number
                self.final_seqs[key] += num
                self.missing_final_msg = True

        '''Find the Initial Condition'''
        self.offset = self.final_stamp - self.duration
        for key in self.final_seqs.keys():
            self.init_seqs[key] = self.final_seqs[key] - 60 * self.freqs[key]

            # when missing the final message, we save the messages until the timstamp of [end-1]
            if self.missing_final_msg == True:
                self.stop_seqs[key] = self.final_seqs[key] - 1 * int(
                    self.freqs[key] / self.freqs[self.rgb_topic[0]])
                self.stop_stamp = self.duration - 1 / \
                    self.freqs[self.rgb_topic[0]]
            else:
                self.stop_seqs[key] = self.final_seqs[key]
                self.stop_stamp = self.duration

        '''ReIndex the Bags'''
        for bag in self.bags:
            print("ReIndexing bag", bag)

            bag_path = os.path.join(self.bag_dir, bag)
            bag = rosbag.Bag(bag_path)

            w_bag = rosbag.Bag(os.path.join(self.reindex_bag_dir,
                                            f"{bag.filename.split('/')[-1][:-4]}.bag"), "w")

            for topic, msg, t in bag.read_messages(topics=self.topics):
                if topic != self.signal_topic:
                    # reindex the tf messages
                    if topic == self.tf_topic:
                        # Filter the fake messages
                        if ('fake' in msg.transforms[0].header.frame_id) or ('navigation' in msg.transforms[0].child_frame_id):
                            continue

                        # Process the robot_0 tf messages
                        if ('reference' not in msg.transforms[0].child_frame_id):
                            # filter the message without the range
                            self.tf_seq += 1
                            if self.tf_seq < self.init_seqs[topic] or self.tf_seq > self.stop_seqs[topic]:
                                continue
                            else:
                                msg, t = self.tf_robot_reindex(topic, msg)
                                
                        # Process reference tf transforms
                        else:
                            t_new = msg.transforms[0].header.stamp.to_sec() - self.offset

                            if t_new < -10**(-5) or t_new > self.stop_stamp + 10**(-5):
                                continue
                            elif t_new < 0.:
                                t_new = 10**(-7)

                            t = Time(t_new)
                            msg.transforms[0].header.stamp = t
                    else:
                        # Filter the messages without the range
                        if msg.header.seq < self.init_seqs[topic] or msg.header.seq > self.stop_seqs[topic]:
                            continue
                        else:
                            if topic not in self.ts_new.keys():
                                t_init = msg.header.stamp.to_sec() - self.offset
                                if t_init < 0:
                                    #print(topic, " : ", t_orig)
                                    self.ts_new[topic] = 10**(-7)
                                else:
                                    self.ts_new[topic] = t_init
                            else:
                                self.ts_new[topic] += 1./self.freqs[topic]

                            t = Time(self.ts_new[topic])
                            msg.header.stamp = t

                            # Limit the Distance in Depth Image Messages
                            if topic in self.depth_topic:
                                msg = self.depth_filter(msg)
                else:
                    if msg.data == "starting":
                        print("Starting...")

                    if t.to_sec() - self.offset < 0:
                        t = Time(10**(-8))
                    else:
                        t = Time(t.to_sec() - self.offset)

                w_bag.write(self.rewrite_topic_name(topic), msg, t)

            w_bag.close()

    def initial_topics(self):
        # Define the signal topic
        if self.config['signal_topic'].get() == self.config['topics']['signal_topic'].keys()[0]:
            self.signal_topic = self.config['signal_topic'].get()
            print('Signal Topic: ', self.signal_topic)
        else:
            raise ValueError('The input Signal Topic does not match the config file,,,')
        
        # Define TF topic
        self.tf_topic = self.config['topics']['tf_topic'].keys()[0]
        print('TF Topic:', self.tf_topic)
        # Define RGB image topic
        self.rgb_topic = self.config['topics']['rgb_topic'].keys()
        print('RGB Image Topic: ', self.rgb_topic)
        # Define depth image topic
        self.depth_topic = self.config['topics']['depth_topic'].keys()
        print('Depth Image Topic: ', self.depth_topic)
        
        # Define other topics need to be stored
        self.topics = []
        self.freqs = {}
        print('\nTopics Input: ')
        for topic_type in self.config['topics'].keys():
            for topic_name in self.config['topics'][topic_type].keys():
                self.topics.append(topic_name)
                self.freqs[topic_name] = self.config['topics'][topic_type][topic_name].get()
                print(topic_name, ' : ', self.freqs[topic_name], ' Hz')

    # filter the depth image with large distance
    def depth_filter(self, msg):
        header = msg.header
        depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        img_depth = depth.copy()

        if self.maximum_depth > 0:
            mask = depth > self.maximum_depth
            img_depth[mask] = np.nan

        msg = self.bridge.cv2_to_imgmsg(img_depth, 'passthrough')
        msg.header = header

        return msg


    def tf_robot_reindex(self, topic, msg):
        transforms_new = []

        # Iterate to define the new transformations
        if topic in self.ts_new.keys():
            self.ts_new[topic] += 1./self.freqs[topic]

        for m in msg.transforms:
            # Modify the transform trees
            if m.child_frame_id == self.base_frame_id:
                m.child_frame_id = m.child_frame_id + "_gt"

            # First Recorded TF message
            if topic not in self.ts_new.keys():
                t_init = m.header.stamp.to_sec() - self.offset

                if t_init < 0:
                    #print("tf: ", t_orig)
                    self.ts_new[topic] = 10**(-7)
                else:
                    self.ts_new[topic] = t_init

            t = Time(self.ts_new[topic])

            m.header.stamp = t
            m.header.seq = self.tf_seq
            transforms_new.append(m)

        msg.transforms = transforms_new
        return msg, t


    def rewrite_topic_name(self, topic):
        if topic in self.config["mapping"].get():
            topic_name = self.config["mapping"].get()[topic]
        else:
            topic_name = topic
        return topic_name
