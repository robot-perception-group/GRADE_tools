#!/bin/bash
usage() {
    echo "Usage: ${0} [-t|--type] [-f|--file] [-b|--bag] [-s|--st] [-e|--et] [-o|--od]" 1>&2
    exit 1
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Setup the Default Parameters
ST=0
ET=60
OD="./"
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
        -o|--od)
        OD=${2}
        echo "OUT DIR : $OD"
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
    # TODO FIX THIS PRE-FIXED path
    rtabmap-report --poses ~/.ros/rtabmap.db
    python3 ${SCRIPT_DIR}/src/tool/evaluate_ate.py ~/.ros/rtabmap_gt.txt ~/.ros/rtabmap_slam.txt  --verbose --plot result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "dynavins" ]];
then
    python3 ${SCRIPT_DIR}/src/tool/output_pose.py --type dynavins --path $FILE --st $ST --et $ET --od $OD
    python3 ${SCRIPT_DIR}/src/tool/evaluate_ate.py ${OD}/gt_pose_dynavins.txt ${OD}/estimated_pose_dynavins.txt  --verbose --plot ${OD}/result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "tartan" ]];
then
    python3 ${SCRIPT_DIR}/src/tool/output_pose.py --type tartan --path $FILE --od $OD
    python3 ${SCRIPT_DIR}/src/tool/evaluate_ate.py ${OD}/gt_pose.txt ${OD}/estimated_pose_tartan.txt  --verbose --plot ${OD}/result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "staticfusion" ]];
then
    python3 ${SCRIPT_DIR}/src/tool/output_pose.py --type staticfusion --path $FILE --od $OD
    python3 ${SCRIPT_DIR}/src/tool/evaluate_ate.py ${OD}/gt_pose.txt ${OD}/estimated_pose_sf.txt  --verbose --plot ${OD}/result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "dynaslam" ]];
then
    python3 ${SCRIPT_DIR}/src/tool/evaluate_ate.py ${OD}/gt_pose.txt $FILE  --verbose --plot ${OD}/result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "orbslam2" ]];
then
    python3 ${SCRIPT_DIR}/src/tool/evaluate_ate.py ${OD}/gt_pose.txt $FILE  --verbose --plot ${OD}/result.png --start_time $ST --end_time $ET # -- umeyama True
elif [[ $TYPE == "vdo" ]];
then
    python3 ${SCRIPT_DIR}/src/tool/output_pose.py --type vdo --path $FILE --od $OD
    python3 ${SCRIPT_DIR}/src/tool/evaluate_ate.py ${OD}/gt_pose.txt ${OD}/estimated_pose_vdo.txt  --verbose --plot ${OD}/result.png --start_time $ST --end_time $ET # -- umeyama True
fi