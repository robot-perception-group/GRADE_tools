#!/bin/bash
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
     fi
     if [[ "$FILE" =~ ".orig" ]];
     then
          rm $FILE
     fi
done


if [[ $TYPE == "bag" ]];
then
     python3 src/bag_process/play_bags.py --path $FOLDER  --config config/bag_process.yaml
elif [[ $TYPE == "file" ]];
then
     python3 src/file_process/play_files.py --path $FOLDER  --config config/file_process.yaml
elif [[ $TYPE == "extract" ]];
then
     python3 src/extract_data/play_bags.py --path $FOLDER  --config config/extract_data.yaml
fi

for FILE in `ls $FOLDER`
do
     FILE="$FOLDER/$FILE"
     if [[ "$FILE" =~ ".bag" ]];
     then
          rm $FILE
     fi
     if [[ "$FILE" =~ "reindex" ]];
     then
          rm -rf $FILE
     fi
done
