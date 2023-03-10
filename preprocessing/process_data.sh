#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
usage() {
     echo "Usage: ${0} [-t|--type] [-p|--path]]" 1>&2
     exit 1
}

while [[ $# -gt 0 ]];do
     key=${1}
     case ${key} in
          -t|--type)
          TYPE=${2}
          echo "TYPE : $TYPE"
          shift 2
          ;;
          -p|--path)
          FOLDER=${2}
          echo "INPUT PATH : $FOLDER"
          shift 2
          ;;
          *)
          usage
          shift
          ;;
     esac
done

for FILE in `ls $FOLDER`
do
     FILE="$FOLDER/$FILE"
     if [[ "$FILE" =~ ".active" ]];
     then
          mv $FILE ${FILE:0:-7} #notice it is .bag.active
          rosbag reindex ${FILE:0:-7}
          sleep 5
          rm ${FILE:0:-11}.orig.bag
     fi
     if [[ "$FILE" =~ ".orig.bag" ]];
     then
          rm $FILE
     fi
done

if [[ $TYPE == "bag" ]];
then
     python3 $SCRIPT_DIR/src/bag_process/play_bags.py --path $FOLDER  --config $SCRIPT_DIR/config/bag_process.yaml
elif [[ $TYPE == "file" ]];
then
     python3 $SCRIPT_DIR/src/file_process/play_files.py --path $FOLDER  --config $SCRIPT_DIR/config/file_process.yaml
elif [[ $TYPE == "extract" ]];
then
     python3 $SCRIPT_DIR/src/extract_data/play_bags.py --path $FOLDER  --config $SCRIPT_DIR/config/extract_data.yaml
fi