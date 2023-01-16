# -*- coding:utf-8 -*-
# author: Xinge
# @file: pc_dataset.py 

import os
import numpy as np
from torch.utils import data
import yaml
import pickle

REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]

@register_dataset
class SemKITTI_demo(data.Dataset):
    def __init__(self, data_path, imageset='demo',
                 return_ref=True, label_mapping="semantic-kitti.yaml", demo_label_path=None):
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.return_ref = return_ref

        self.im_idx = []
        self.im_idx += absoluteFilePaths(data_path)
        self.label_idx = []
        if self.imageset == 'val':
            print(demo_label_path)
            self.label_idx += absoluteFilePaths(demo_label_path)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'demo':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif self.imageset == 'val':
            annotated_data = np.fromfile(self.label_idx[index], dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple

@register_dataset
class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="semantic-kitti.yaml", nusc=None):
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
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple


@register_dataset
class SemKITTI_nusc(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="nuscenes.yaml", nusc=None):
        self.return_ref = return_ref

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.nusc = nusc

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                self.nusc.get('lidarseg', lidar_sd_token)['filename'])

        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (points[:, :3], points_label.astype(np.uint8))
        if self.return_ref:
            data_tuple += (points[:, 3],)
        return data_tuple

@register_dataset
class SemKITTI_nerflidar(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="NeRFlidar_label.yaml", nusc=None):
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
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 3))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple

@register_dataset
class SemKITTI_nerflidar_nusc(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="NeRFlidar_label.yaml", nusc=None):
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
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])

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


def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    remove_ind = label == 0
    label -= 1
    label[remove_ind] = 255
    return label

from os.path import join
@register_dataset
class SemKITTI_sk_multiscan(data.Dataset):
    def __init__(self, data_path, imageset='train',return_ref=False, label_mapping="semantic-kitti-multiscan.yaml"):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.data_path = data_path
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        multiscan = 2 # additional two frames are fused with target-frame. Hence, 3 point clouds in total
        self.multiscan = multiscan
        self.im_idx = []

        self.calibrations = []
        self.times = []
        self.poses = []

        self.load_calib_poses()

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in range(0, 22):
            seq_folder = join(self.data_path, str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):

        # pose = poses[0][idx]

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        # new_points = hpoints.dot(pose.T)
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        # new_coords = new_coords.dot(pose0[:3, :3])
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        number_idx = int(self.im_idx[index][-10:-4])
        dir_idx = int(self.im_idx[index][-22:-20])

        pose0 = self.poses[dir_idx][number_idx]

        if number_idx - self.multiscan >= 0:

            for fuse_idx in range(self.multiscan):
                plus_idx = fuse_idx + 1

                pose = self.poses[dir_idx][number_idx - plus_idx]

                newpath2 = self.im_idx[index][:-10] + str(number_idx - plus_idx).zfill(6) + self.im_idx[index][-4:]
                raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))

                if self.imageset == 'test':
                    annotated_data2 = np.expand_dims(np.zeros_like(raw_data2[:, 0], dtype=int), axis=1)
                else:
                    annotated_data2 = np.fromfile(newpath2.replace('velodyne', 'labels')[:-3] + 'label',
                                                  dtype=np.int32).reshape((-1, 1))
                    annotated_data2 = annotated_data2 & 0xFFFF  # delete high 16 digits binary

                raw_data2 = self.fuse_multi_scan(raw_data2, pose0, pose)

                if len(raw_data2) != 0:
                    raw_data = np.concatenate((raw_data, raw_data2), 0)
                    annotated_data = np.concatenate((annotated_data, annotated_data2), 0)

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))

        if self.return_ref:
            data_tuple += (raw_data[:, 3], origin_len) # origin_len is used to indicate the length of target-scan


        return data_tuple


# load Semantic KITTI class info

def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        val_ = semkittiyaml['learning_map'][i]
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels_8'][val_]
    return SemKITTI_label_name
def get_cityscape_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        cityyaml = yaml.safe_load(stream)
    cityscape_label_name = dict()
    for i in sorted(list(cityyaml['learning_map'].keys()))[::-1]:
        val_ = cityyaml['learning_map'][i]
        cityscape_label_name[val_] = cityyaml['new_labels'][val_]
    # cityscape_label_name = cityyaml['city_labels']
    # import pdb;pdb.set_trace()
    return cityscape_label_name

def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name
######################################################start for V7
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)
def get_simu_split(path , scene_lists=None , select = 0) :
    scenes = os.listdir(path)
    get_simu = []
    
    for scene in scenes :
        if (scene_lists != None) and scene not in scene_lists :
            continue
        temps = os.listdir(path + "/" + scene + "/velodyne")
        print(scene,len(temps))
        for i, temp in enumerate(temps) :
            # print('real' not in temp , temp)
            # if i%2 != 0 and 'real' not in scene :
            #     continue
            # if int(temp[:6])%10 < select and 'real' not in scene :
            #     continue
            get_simu.append(path + "/" + scene + "/velodyne/" + temp)
    return get_simu
@register_dataset
class SemKITTI_nus32_lidar(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="NeRFlidar_label.yaml", nusc=None):
        ### label_mapping via nuscenes.yaml
        self.return_ref = return_ref

        ####读取label mapping
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        
        ####设置 距离 mask
        self.near = semkittiyaml['near']
        self.far = semkittiyaml['far']
        self.mask_flag = semkittiyaml['mask_flag']
        
        ####设置 simu mode
        self.simu_path = semkittiyaml['simu_path']
        self.simu_flag = semkittiyaml['simu_flag']
        self.hybrid_scene = semkittiyaml['hybrid_simu']['train']
        self.simu_val = semkittiyaml['hybrid_simu']['val']
        simu_num = 0

        #nerf
        self.nerf_mapping = semkittiyaml['nerf_mapping']

        #####设置数据大小data scale  V6
        np.random.seed(0)
        self.data_scale = semkittiyaml['data_scale']
        self.val_only_simu = semkittiyaml['val_only']

        # loading self contrast
        self.compare = semkittiyaml['compare_simu']

        # self.val_data_scale = semkittiyaml['val_data_scale']
        #####dataset 类型
        self.imageset = imageset
        if imageset == 'train':
            split = os.path.join(data_path + "/" + semkittiyaml['split']['train'])
            with open(split, "r") as f:
                data = f.read().splitlines()
            split = data
        elif imageset == 'val':
            split = os.path.join(data_path + "/" + semkittiyaml['split']['val'])
            with open(split, "r") as f:
                data = f.read().splitlines()
            split = data
        else:
            raise Exception('Split must be train/val/test')
        random_scene = np.random.randint(0, high=len(split), size=1110, dtype='uint16')
        self.im_idx = []
        for i , i_folder in enumerate(split):
            if i >= self.data_scale:
                break
            # if self.imageset == "val" :
            #     if i >= 5 :
            #         break
            temp_fold = split[random_scene[i]]
            # if int(temp_fold) < 120 and int(temp_fold) > 110 :
            #     continue
            # # print(len(temp_fold) , temp_fold, data_path)
            temp_path = data_path  + "/"+os.path.join(temp_fold[-4:] ,"velodyne" )
            # temp_path = "/SSD_DISK/users/kuangshaochen/nus_data/sequences/0133/velodyne"
            temp_files = os.listdir(temp_path)
            for one_temp in temp_files :
                self.im_idx.append(temp_path + "/" + one_temp)
        # if self.imageset == 'val' :
        #     # print("load nus real data and return avoid other data \n" , self.imageset , len(self.im_idx) )
        #     print("val dataset close all nus real val")
        #     self.im_idx = []
            # return
        if self.imageset == 'train':
            if self.val_only_simu == True :
                self.im_idx = []
            if self.simu_flag :
                simu_s = get_simu_split(self.simu_path , self.hybrid_scene)
                for simu_ in simu_s :
                    simu_num += 1
                    self.im_idx.append(simu_)
            print("Extra loading simulation train data :" , simu_num)
            # print(" All train set  :" , len(self.im_idx))
        else :
            # print("no val ?" , self.simu_flag)
            if self.simu_flag :
                simu_s = get_simu_split(self.simu_path , self.simu_val)
                for simu_ in simu_s :
                    simu_num += 1
                    self.im_idx.append(simu_)
            print("Extra loading simu val data :" , simu_num)
            pass
        if self.compare['flag'] :
            self.im_idx = []
            simu_s = get_simu_split(self.simu_path , self.compare[self.imageset] , self.compare['select'])
            for simu_ in simu_s :
                simu_num += 1
                self.im_idx.append(simu_)
            print("close all nuscene real data")
            print("loading: ",self.imageset ,self.compare[self.imageset] , len(self.im_idx))
        print("\nCurrent data set is :" , self.imageset ," and len of it is ", len(self.im_idx))
        self.temp_list = ['../nus_data/sequences/0966/velodyne/000015.bin', '../nus_data/sequences/0966/velodyne/000000.bin',
          '../nus_data/sequences/0966/velodyne/000005.bin', '../nus_data/sequences/0966/velodyne/000012.bin',
             '../nus_data/sequences/0966/velodyne/000002.bin', '../nus_data/sequences/0966/velodyne/000003.bin',
         '../nus_data/sequences/0966/velodyne/000038.bin','../nus_data/sequences/0966/velodyne/000017.bin', 
         '../nus_data/sequences/0966/velodyne/000025.bin', '../nus_data/sequences/0966/velodyne/000014.bin']
        self.temp_num = 0

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        # self.im_idx[index] = self.temp_list[self.temp_num]
        # self.temp_num += 1
        # raw_data = np.fromfile(self.im_idx[index], dtype=np.float32,count=-1).reshape((-1, 5))
        # lidarseg_labels_filename = self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label'
        # points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        # print(self.im_idx[index])
        if "semantickitti_nerf_city" in self.im_idx[index] and "real" not in self.im_idx[index]  :
            # print(self.im_idx[index])
            raw_data = np.fromfile(self.im_idx[index], dtype=np.float32, count=-1).reshape((-1, 3))
            lidarseg_labels_filename = self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label'
            
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint32).reshape([-1, 1])
            # print(points_label.shape , raw_data.shape)
            points_label = np.vectorize(self.nerf_mapping.__getitem__)(points_label)
            # print(points_label.shape , raw_data.shape)
        else:
            raw_data = np.fromfile(self.im_idx[index], dtype=np.float32, count=-1).reshape((-1, 5))
            lidarseg_labels_filename = self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label'
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        if self.mask_flag :
            polar_coor = cart2polar(raw_data[:,:3])
            # print("before mask : near ", polar_coor.min(0)[0])
            # print("before mask : far ", polar_coor.max(0)[0])
            near_mask =  (polar_coor[:,0] < self.near)
            far_mask =  (polar_coor[:,0] > self.far)
            mask = near_mask + far_mask
            raw_data = np.delete(raw_data , mask , axis=0)
            points_label = np.delete(points_label , mask , axis=0)
            polar_coor = cart2polar(raw_data[:,:3])
            # print(raw_data.shape)
            # print("after mask : near ", polar_coor.min(0)[0])
            # print("after mask : far ", polar_coor.max(0)[0])
            pass

        # print("load: " , self.im_idx[index])
        # print(polar_coor.shape , points_label.shape)
        data_tuple = (raw_data[:, :3], points_label.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple