import yaml
from dataloader.pc_dataset import *
class SemKITTI_nerflidar_nusc(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="./config/NeRFlidar_city.yaml", nusc=None):
        ### label_mapping via nuscenes.yaml
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32,count=-1).reshape((-1, 5))
        # if self.imageset == 'test':
        #     annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        # else:
        #     annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
        #                                  dtype=np.uint32).reshape((-1, 1))
        #     annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
        #     # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
        lidarseg_labels_filename = self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label'
        print('the current lidarseg_labels_filename is:',lidarseg_labels_filename)
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        print('raw_data shape is:',raw_data.shape)
        print('points_label:',points_label.shape)
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)

        data_tuple = (raw_data[:, :3], points_label.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple
def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


if __name__ == '__main__':
    Dataset  = SemKITTI_nerflidar_nusc(data_path='/SSD_DISK/users/zhangjunge/semantickitti_nerf_city/sequences/',imageset='val')