#!/bin/bash
usage() {
    echo "Usage: ${0} [-t|--type] [-f|--file] [-b|--bag] [-s|--st] [-e|--et]" 1>&2
    exit 1
}

# Setup the Default Parameters
ST=0
ET=60

while [[ $# -gt 0 ]];do
    key=${1}
    case ${key} in
        -t|--type)
        TYPE=${2}
        echo "TYPE : $TYPE"
        shift 2
        ;;
        -f|--file)
        FILE=${2}
        echo "INPUT DATA : $FILE"
        shift 2
        ;;
        -s|--st)
        ST=${2}
        echo "START TIME : $ST"
        shift 2
        ;;
        -e|--et)
        ET=${2}
        echo "END TIME : $ET"
        shift 2
        ;;
        *)
        usage
        shift
        ;;
    esac
done


if [[ $TYPE == "rtabmap" ]];
then
    rtabmap-report --poses ~/.ros/rtabmap.db
    python3 src/tool/evaluate_ate.py ~/.ros/rtabmap_gt.txt ~/.ros/rtabmap_slam.txt  --verbose --plot result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "dynavins" ]];
then
    python3 src/tool/output_pose.py --type dynavins --path $FILE --st $ST --et $ET
    python3 src/tool/evaluate_ate.py gt_pose_dynavins.txt estimated_pose_dynavins.txt  --verbose --plot result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "tartan" ]];
then
    python3 src/tool/output_pose.py --type tartan --path $FILE
    python3 src/tool/evaluate_ate.py gt_pose.txt estimated_pose_tartan.txt  --verbose --plot result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "staticfusion" ]];
then
    python3 src/tool/output_pose.py --type staticfusion --path $FILE
    python3 src/tool/evaluate_ate.py gt_pose.txt estimated_pose_sf.txt  --verbose --plot result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "dynaslam" ]];
then
    python3 src/tool/evaluate_ate.py gt_pose.txt $FILE  --verbose --plot result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "orbslam2" ]];
then
    python3 src/tool/evaluate_ate.py gt_pose.txt $FILE  --verbose --plot result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "vdo" ]];
then
    python3 src/tool/output_pose.py --type vdo --path $FILE
    python3 src/tool/evaluate_ate.py gt_pose.txt estimated_pose_vdo.txt  --verbose --plot result.png --start_time $ST --end_time $ET # -- umeyama True
fi