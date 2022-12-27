import os
import rosbag
import argparse
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


def output_gt_pose(args):
    """
    Output Default Ground Truth Pose TXT File
    
    :input args.path: Path to the rosbag sequence[reindex]
    :input args.topic: Camera Pose Topic Name
    
    :output gt_pose.txt: Ground Truth Pose [ts p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    """
    # Loop rosbags to output ground truth poses
    bag_dir = args.path
    bags = os.listdir(os.path.join(bag_dir))
    bags.sort()

    outdir = args.od

    f1 = open(os.path.join(outdir, "gt_pose.txt"), "w")

    TOPIC_CHECK = False
    
    '''Generating Camera gt-pose'''
    for bag in bags:
        # Ignore the nonbag files/folder
        if not bag.endswith('.bag'):
            # print(f"[WARNING] {bag} not being processed... ")
            continue

        print("Playing bag", bag)

        bag_path = os.path.join(bag_dir, bag)
        bag = rosbag.Bag(bag_path)

        for topic, msg, t in bag.read_messages(topics=args.topic):
            TOPIC_CHECK = True

            ts = msg.header.stamp.to_sec()

            p = msg.pose.position
            q = msg.pose.orientation

            f1.write('%f %f %f %f %f %f %f %f\n' %
                     (ts, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    if TOPIC_CHECK:
        print('Ground Truth Pose Topic is valid...')
    else:
        raise ValueError(
            'Ground Truth Pose Topic does not exist in the input rosbag...')
    f1.close()

def output_dynavins(args):
    """
    Extract gt-camera Pose and Estimated Camera Pose from Recorded Bag
    
    :input args.path: Path to the single recorded result rosbag
    :input args.start_time: Start time to output poses
    :input args.end_time: End time to output poses
    
    :output gt_pose_dynavins.txt: ground truth pose [ts p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    :output estimated_pose_dynavins.txt: estimated camera pose from Dynamic-VINS
    """
    bag = rosbag.Bag(args.path)
    outdir = args.od

    f1 = open(os.path.join(outdir,"estimated_pose_dynavins.txt"), "w")
    f2 = open(os.path.join(outdir,"gt_pose_dynavins.txt"), "w")

    ts_init = None
    ts_start = args.start_time
    ts_stop = args.end_time

    max_difference = args.max_difference

    print('The evaluation will start at Timestamp: %.2f' % ts_start)
    print('                     stop at Timestamp: %.2f' % ts_stop)

    # Define the extrinsic Rotation from camera to odom
    T_odom_to_camera = np.eye(4)
    T_odom_to_camera[:3, :3] = np.array([
                                        [0.,       -0.25983896,  0.96565196],
                                        [-1.0,         0.0,          0.0],
                                        [0.0,         -0.96565196, -0.25983896]])
    T_odom_to_camera[:3, 3] = np.array([0.1, 0.0, 0.0])

    ''' Find the Initialization Timestamp'''
    for topic, msg, t in bag.read_messages():
        if 'init_map_time' in topic:
            ts = msg.header.stamp.to_sec()

            if (ts < ts_start) or (ts > ts_stop):
                continue

            ts_init = ts
            break

    '''Find the Initialization Position of Map Frame'''
    for topic, msg, t in bag.read_messages():
        if 'my_robot_0/odom' in topic:
            ts = msg.header.stamp.to_sec()

            if abs(ts_init - ts) < max_difference:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation

                T_w2m = Quaternion(w=q.w, x=q.x, y=q.y,
                                   z=q.z).transformation_matrix
                T_w2m[:3, 3] = [p.x, p.y, p.z]

                break

    print('Initialization Started at :', ts_init)

    T_m2c_init = T_odom_to_camera
    T_w2c_init = np.matmul(T_w2m, T_m2c_init)

    p_init = T_w2c_init[:3, 3]
    q_init = Quaternion(matrix=T_w2c_init)

    # Write the first estimated pose
    f1.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' % (
        ts_init, p_init[0], p_init[1], p_init[2], q_init.x, q_init.y, q_init.z, q_init.w))

    '''Output recorded poses to the txt files'''
    for topic, msg, t in bag.read_messages():
        if 'vins_estimator/camera_pose' in topic:
            ts = msg.header.stamp.to_sec()

            if (ts < ts_start) or (ts > ts_stop):
                continue

            p = msg.pose.position
            q = msg.pose.orientation

            T_m2c = Quaternion(w=q.w, x=q.x, y=q.y,
                               z=q.z).transformation_matrix
            T_m2c[:3, 3] = [p.x, p.y, p.z]

            T_w2c = np.matmul(T_w2m, T_m2c)

            p = T_w2c[:3, 3]
            q = Quaternion(matrix=T_w2c)
            f1.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' %
                     (ts, p[0], p[1], p[2], q.x, q.y, q.z, q.w))

        if 'camera/pose' in topic:
            ts = msg.header.stamp.to_sec()

            if (ts < ts_start) or (ts > ts_stop):
                continue

            p = msg.pose.position
            q = msg.pose.orientation

            f2.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' %
                     (ts, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    f1.close()
    f2.close()

def output_tartan_gt(args):
    """
    Output ground truth camera pose required by TARTAN VO at Image Publish Frequency [30 Hz]
    Transform Default 'gt_pose.txt' to 'gt_pose_tartan.txt'
    [ts p.x, p.y, p.z, q.x, q.y, q.z, q.w] -> [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    
    :input args.path: Path to the 'gt_pose.txt'
    
    :output gt_pose_tartan.txt: ground truth pose [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    """
    outdir = args.od
    f1 = open(os.path.join(outdir, "gt_pose_tartan.txt"), "w")

    with open(args.path) as f:
        data = f.readlines()

    pose_init = [float(i) for i in data[0].split(' ')]
    T_init = Quaternion(x=pose_init[4], y=pose_init[5],
                        z=pose_init[6], w=pose_init[7]).transformation_matrix
    T_init[:3, 3] = [pose_init[1], pose_init[2], pose_init[3]]

    T_init_inv = np.linalg.inv(T_init)

    pose_id = 0
    for i in range(len(data)):
        pose = [float(j) for j in data[i].split(' ')]

        if abs(pose[0] - pose_id/args.image_freq) < args.max_difference:
            T_w2c = Quaternion(
                x=pose[4], y=pose[5], z=pose[6], w=pose[7]).transformation_matrix
            T_w2c[:3, 3] = [pose[1], pose[2], pose[3]]

            T_m2c = np.matmul(T_init_inv, T_w2c)

            p = T_m2c[:3, 3]
            q = Quaternion(matrix=T_m2c)

            f1.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' %
                     (p[0], p[1], p[2], q.x, q.y, q.z, q.w))
            pose_id += 1
    f1.close()


def output_tartan(args):
    """
    Transform the Tartan Format Result to the TUM format Result for ATE Evaluation
    [p.x, p.y, p.z, q.x, q.y, q.z, q.w] -> [ts p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    
    :input args.path: Path to the Tartan VO Format Estimated Result
    
    :output estimated_pose_tartan.txt: Estimated Pose [ts p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    """
    # Transform the tartan format result to the tum format result for evaluation
    outdir = args.od
    f1 = open(os.path.join(outdir, "estimated_pose_tartan.txt"), "w")

    ts = 0.0

    # Transform the result poses to the world frame
    with open(args.path) as f:
        data = f.readlines()

    for i in range(len(data)):

        T = [float(j) for j in data[i].split(' ')]

        f1.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' %
                 (ts, T[0], T[1], T[2], T[3], T[4], T[5], T[6]))

        ts += 1/args.image_freq
    f1.close()


def output_sf(args):
    """
    Transform the StaticFusion Format Result to the TUM Format Result for ATE Evaluation
    [Frame_id p.x, p.y, p.z, q.x, q.y, q.z, q.w] -> [ts p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    
    :input args.path: Path to the StaticFusion Format Estimated Result
    
    :output estimated_pose_sf.txt: Estimated Pose [ts p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    """
    outdir = args.od

    f1 = open(os.path.join(outdir, "estimated_pose_sf.txt"), "w")

    f1.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' %
             (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    ts = 1/args.image_freq

    # Transform the result poses to the world frame
    with open(args.path) as f:
        data = f.readlines()

    for i in range(len(data)):

        T = [float(j) for j in data[i].split(' ') if j != '']

        f1.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' %
                 (ts, T[1], T[2], T[3], T[4], T[5], T[6], T[7]))

        ts += 1/args.image_freq
    f1.close()


def output_vdo_gt(args):
    """
    Extract ground truth camera pose required by VDO-SLAM at Image Publish Frequency [30 Hz]
    
    :input args.path:  Path to the rosbag sequence[reindex]
    :input args.topic: Camera Pose Topic Name
    
    :output gt_pose_vdo.txt: Ground Truth Pose [FrameID R11 R12 R13 t1 R21 R22 R23 t2 R31 R32 R33 t3 0 0 0 1]
    :output times.txt: Timestamp Sequence [ts]
    """
    bag_dir = args.path
    bags = os.listdir(os.path.join(bag_dir))
    bags.sort()

    outdir = args.od

    # for VDO SLAM
    f1 = open(os.path.join(outdir,"pose_gt.txt"), "w")
    f3 = open(os.path.join(outdir, "times.txt"), "w")


    ts_list = []

    '''Generating Timestamps List'''
    for bag in bags:
        if not bag.endswith(".bag"):
            continue
        bag_path = os.path.join(bag_dir, bag)
        bag = rosbag.Bag(bag_path)

        for topic, msg, t in bag.read_messages(topics=["/my_robot_0/camera_link/0/rgb/image_raw"]):
            ts = msg.header.stamp.to_sec()
            ts_list.append(ts)
            f3.write('%.6e\n' % (ts))

    '''Generating Camera gt-pose'''
    frame_index = 0
    for bag in bags:
        if not bag.endswith(".bag"):
            continue
        print("Playing bag", bag)

        bag_path = os.path.join(bag_dir, bag)
        bag = rosbag.Bag(bag_path)

        for topic, msg, t in bag.read_messages(topics=args.topic):
            ts = msg.header.stamp.to_sec()

            if (np.abs(np.array(ts_list)-ts) < 0.005).any():
                p = msg.pose.position
                q = msg.pose.orientation

                T = Quaternion(w=q.w, x=q.x, y=q.y,
                               z=q.z).transformation_matrix
                T[:3, 3] = [p.x, p.y, p.z]

                if frame_index == 0:
                    T_init = T

                T = np.matmul(np.linalg.inv(T_init), T)
                T = T.reshape((16,))
                f1.write('%d  ' % frame_index)
                for i in range(16):
                    f1.write('%.6f  ' % T[i])
                f1.write('\n')
                frame_index += 1
    f1.close()
    f3.close()


def output_vdo(args):
    """
    Transform the VDO-SLAM Format Result to the TUM Format Result for ATE Evaluation
    [FrameID R11 R12 R13 t1 R21 R22 R23 t2 R31 R32 R33 t3 0 0 0 1] -> [ts p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    
    :input args.path: Path to the VDO-SLAM Format Estimated Result
    
    :output estimated_pose_vdo.txt: Estimated Pose [ts p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    """
    # for VDO SLAM
    outdir = args.od
    f1 = open(os.path.join(outdir,"estimated_pose_vdo.txt"), "w")

    # # Obtain the initial transformation from map to world
    # with open("pose_gt_mat.txt") as f:
    #     data = f.readlines()

    # T_init = np.zeros((16,))
    # for i in range(16):
    #     T_init[i] = data[0].split('  ')[i+1]
    # T_init = T_init.reshape(4,4)

    # Transform the result poses to the world frame
    with open(args.path) as f:
        data = f.readlines()

    for i in range(len(data)):
        t = float(data[i].split(' ')[0])/args.image_freq

        T = np.zeros((16,))
        for j in range(16):
            T[j] = float(data[i].split(' ')[j+1])
        T = T.reshape((4, 4))
        # T = np.matmul(T_init, T)
        r = Rotation.from_matrix(T[:3, :3])
        q = r.as_quat()

        f1.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' %
                 (t, T[0, 3], T[1, 3], T[2, 3], q[0], q[1], q[2], q[3]))
    f1.close()


if __name__ == '__main__':
    # Define parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to the desired rosbags")
    parser.add_argument("--od", type=str, help="path to the desired outdir", default=".")
    parser.add_argument("--type", type=str, help="type to output different results")
    parser.add_argument("--topic", type=str, help="camera pose topic")
    parser.add_argument("--start_time", type=float, default=0.0, help="start time")
    parser.add_argument("--end_time", type=float, default=60.0, help="end time")
    parser.add_argument("--image_freq", type=float, default=30.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching', default=0.004)
    args, _ = parser.parse_known_args()

    if args.type == 'groundtruth':
        output_gt_pose(args)
    elif args.type == 'dynavins':
        output_dynavins(args)
    elif args.type == 'tartan_gt':
        output_tartan_gt(args)
    elif args.type == 'tartan':
        output_tartan(args)
    elif args.type == 'staticfusion':
        output_sf(args)
    elif args.type == 'vdo_gt':
        output_vdo_gt(args)
    elif args.type == 'vdo':
        output_vdo(args)
