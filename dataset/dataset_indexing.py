import os
from utils.config import Config
def set_dataset_path(config: Config, dataset_name: str = '', seq: str = ''):
    
    if seq is None:
        seq = ''
    config.name = config.name + '_' + dataset_name + '_' + seq.replace("/", "")

    if config.use_dataloader:
        config.data_loader_name = dataset_name
        config.data_loader_seq = seq
        print('Using data loader for specific dataset or specific input data format')
        from dataset.dataloaders import available_dataloaders 
        print('Available dataloaders:', available_dataloaders())

    else:
        if dataset_name == "kitti":
            base_path = config.pc_path.rsplit('/', 3)[0]
            config.pc_path = os.path.join(base_path, 'sequences', seq, "velodyne")
            pose_file_name = seq + '.txt'
            config.pose_path = os.path.join(base_path, 'poses', pose_file_name)
            config.calib_path = os.path.join(base_path, 'sequences', seq, "calib.txt")
            config.label_path = os.path.join(base_path, 'sequences', seq, "labels")
            config.kitti_correction_on = True
            config.correction_deg = 0.195
        else:
            print('Dataset has not been supported yet')
