import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import LaserScan, SemLaserScan

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)
class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               max_points=150000,   # max number of points present in dataset
               gt=True):            # send ground truth?
    # save deats
    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      temp_path = self.root  + "/"+os.path.join(seq ,"velodyne" )
      temp_files = os.listdir(temp_path)
      for one_temp in temp_files :
        self.scan_files.append(temp_path + "/" + one_temp)
        self.label_files.append((temp_path + "/" + one_temp).replace("velodyne/","labels/").replace(".bin" , ".label"))
    # print(self.label_files)
    # sort for correspondance
    self.scan_files.sort()
    self.label_files.sort()

    print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                    len(self.sequences)))

  def __getitem__(self, index):
    # get item in tensor shape
    scan_file = self.scan_files[index]
    if self.gt:
      label_file = self.label_files[index]
    # print("training dataset getitem??",scan_file)
    # open a semantic laserscan
    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

    # open and obtain scan
    scan.open_scan(scan_file)
    if self.gt:
      scan.open_label(label_file)
      print("before",np.unique(scan.sem_label),)
      # map unused classes to used classes (also for projection)
      # print(label_file)
      scan.sem_label = self.map(scan.sem_label, self.learning_map , label_file)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map ,label_file)
      # import pdb;pdb.set_trace()
      # print("after",np.unique(scan.sem_label))
      quit()
    # make a tensor of the uncompressed data (with the max num points)
    unproj_n_points = scan.points.shape[0]
    unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
    unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
    unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
    unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
    if self.gt:
      unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
      unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
    else:
      unproj_labels = []

    # get points and labels
    proj_range = torch.from_numpy(scan.proj_range).clone()
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
    proj_remission = torch.from_numpy(scan.proj_remission).clone()
    proj_mask = torch.from_numpy(scan.proj_mask)
    if self.gt:
      proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []
    proj_x = torch.full([self.max_points], -1, dtype=torch.long)
    proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
    proj_y = torch.full([self.max_points], -1, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])
    proj = (proj - self.sensor_img_means[:, None, None]
            ) / self.sensor_img_stds[:, None, None]
    proj = proj * proj_mask.float()

    # get name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")
    # print("path_norm: ", path_norm)
    # print("path_seq", path_seq)
    # print("path_name", path_name)

    # return
    return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points

  def __len__(self):
    return len(self.scan_files)

  @staticmethod
  def map(label, mapdict , path=None , label_file = None):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    # print("dataset mapping?")
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    # print(mapdict)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # import pdb;pdb.set_trace()
    # do the mapping
    # print(label_file)

    # label[label > 20] = 0
    return lut[label]

def random_select_scene(path , select , scenelist = None) :
    if "semantickitti_nerf_city" in path :
      return scenelist
    np.random.seed(0)
    with open('/SSD_DISK/users/zhangjunge/nus_data/sequences/train.txt', "r") as f:
        scenes = f.read().splitlines()
    select_scene = []
    random_scene = np.random.randint(0, high=len(scenes), size=len(scenes), dtype='uint16')
    for i in range(len(scenes)) :
      if i > select-1:
        continue
      scene = scenes[random_scene[i]]
      select_scene.append(scene)
    return select_scene
class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               sensor,            # sensor to use
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True,# shuffle training set?
               datacfg = None):  
    super(Parser, self).__init__()

    # if I am training, get the dataset
    
    self.datacfg = datacfg
    
    self.valroot = self.datacfg['valid_path']
    self.trainroot = self.datacfg['train_path']
    self.train_sequences = random_select_scene(self.trainroot, self.datacfg['train_scale'], self.datacfg['train_scene'])

    # self.test_sequences = test_sequences
    with open(self.valroot+'/sequences/val.txt', "r") as f:
        self.valid_sequences = f.read().splitlines()

    # print(self.train_sequences , self.valid_sequences )
    # for idx, scene in enumerate( self.valid_sequences ):
    #   if scene in self.train_sequences:
    #     print(scene)
    #   print("check" , idx)
    # quit()
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.val_lm = learning_map
    if 'semantickitti_nerf_city' in self.valroot :
      self.val_lm = self.datacfg['nerf_use_mapping_real200']
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train

    # number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)
    print(root)
    # Data loading code
    self.train_dataset = SemanticKitti(root=self.trainroot,
                                       sequences=self.train_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       gt=self.gt)

    self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=self.shuffle_train,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.trainloader) > 0
    self.trainiter = iter(self.trainloader)

    self.valid_dataset = SemanticKitti(root=self.valroot,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       gt=self.gt)

    self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.validloader) > 0
    self.validiter = iter(self.validloader)

    # if self.test_sequences:
    #   self.test_dataset = SemanticKitti(root=self.root,
    #                                     sequences=self.test_sequences,
    #                                     labels=self.labels,
    #                                     color_map=self.color_map,
    #                                     learning_map=self.learning_map,
    #                                     learning_map_inv=self.learning_map_inv,
    #                                     sensor=self.sensor,
    #                                     max_points=max_points,
    #                                     gt=False)

    #   self.testloader = torch.utils.data.DataLoader(self.test_dataset,
    #                                                 batch_size=self.batch_size,
    #                                                 shuffle=False,
    #                                                 num_workers=self.workers,
    #                                                 pin_memory=True,
    #                                                 drop_last=True)
      # assert len(self.testloader) > 0
      # self.testiter = iter(self.testloader)

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # put label in color
    return SemanticKitti.map(label, self.color_map)
