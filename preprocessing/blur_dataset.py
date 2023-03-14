import os
import shutil

if __name__ == '__main__':
    # Extract the noisy images from the bags
    main_paths = ['/ps/project/irotate/DE_horiz_flight_lot_obs_cam0', '/ps/project/irotate/DE_lot_obs_cam0']
    for main_path in main_paths:
        datasets = os.listdir(main_path)
        
        exp_name = main_path.split('/')[-1]
        for dataset in datasets:
            local_path = os.path.join('/home/cxu', exp_name, dataset)
            if not os.path.exists(local_path):
                os.makedirs(local_path)
            
                dataset = os.path.join(main_path, dataset)
                files = os.listdir(dataset)
                for file in files:
                    if '.bag' in file:
                        print(f'Copying {file} from ', dataset)
                        shutil.copyfile(os.path.join(dataset, file), os.path.join(local_path, file))
                os.system(f"./process_data.sh --type bag --path {local_path}")