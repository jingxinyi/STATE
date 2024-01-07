from torch.utils import data
from torchvision import datasets, transforms
from base import BaseDataLoader, torch
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict


def generate_id(name, angle_1, angle_2):
    angle_1, angle_2 = str(angle_1), str(angle_2)

    return '_'.join([name, angle_1, angle_2])


def parse_id(id):
    id_split = id.split('_')

    return id_split[0], id_split[1], id_split[2]


class ObjectTestDataset(data.Dataset):
    def __init__(self, data_dir, num_input, object_type, image_size):
        self.shapenet_dir = Path(data_dir) / 'shapenet'
        self.shapenet_object_dir = self.shapenet_dir / object_type
        self.image_id_file = self.shapenet_dir / 'id_{}_random_elevation.txt'.format(object_type)
        self.image_ids = np.genfromtxt(self.image_id_file, dtype=np.str)
        self.image_size = image_size

        self.num_input = num_input
        self.angle_range_1 = [0, 18]
        self.angle_scale_1 = 2
        self.angle_range_2 = [0, 3]
        self.angle_scale_2 = 10

    def __getitem__(self, index):
        target_name, target_angle_1, target_angle_2 = parse_id(self.image_ids[index][0])# [0]
        target_img = self.load_image_by_id(self.image_ids[index][0], False)# [0]
        target_img_raw = self.load_image_by_id(self.image_ids[index][0], False)#[0]
        target_pose = self.encode_pose(target_angle_1, target_angle_2)

        source_imgs = []
        source_poses = []
        for i in range(self.num_input):
            _, source_angle_1, source_angle_2 = parse_id(self.image_ids[index][i + 1])

            source_imgs.append(self.load_image_by_id(self.image_ids[index][i + 1]))
            source_poses.append(self.encode_pose(source_angle_1, source_angle_2))
        # for i in range(self.num_input):
        #     angle_1 = np.random.randint(*self.angle_range_1) * self.angle_scale_1
        #     angle_2 = np.random.randint(*self.angle_range_2) * self.angle_scale_2
        #     source_id = generate_id(target_name, angle_1, angle_2)
        #     source_imgs.append(self.load_image_by_id(source_id))
        #     source_poses.append(self.encode_pose(angle_1, angle_2))
        return source_imgs, source_poses, target_img, target_pose, target_img_raw

    def __len__(self):
        return len(self.image_ids)

    def encode_pose(self, angle_1, angle_2):
        angle_1, angle_2 = int(angle_1), int(angle_2)

        encode = np.zeros((self.angle_scale_1 * self.angle_range_1[1] + self.angle_range_2[1], 1))
        encode[angle_1, 0] = 1
        encode[angle_2 // self.angle_scale_2 + self.angle_scale_1 * self.angle_range_1[1], 0] = 1

        return torch.tensor(encode, dtype=torch.float)

    def load_image_by_id(self, id, is_resize=True):
        image = Image.open(str(self.shapenet_object_dir / id) + '.png')

        if is_resize:
            trsfm = transforms.Compose([
                transforms.Resize(size=self.image_size),
                transforms.ToTensor(),
            ])
        else:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
            ])

        return trsfm(image) * 2 - 1


class ObjectDataset(data.Dataset):
    def __init__(self, data_dir, num_input, object_type, image_size, is_single_mode, single_mode_start_id,
                 single_mode_object_number, is_single_mode_fixed_pose):
        self.shapenet_dir = Path(data_dir) / 'shapenet'
        self.shapenet_object_dir = self.shapenet_dir / object_type
        self.image_id_file = self.shapenet_dir / 'id_{}_train.txt'.format(object_type)
        self.image_ids = np.genfromtxt(self.image_id_file, dtype=np.str)
        self.image_size = image_size

        self.is_single_mode = is_single_mode
        self.is_single_mode_fixed_pose = is_single_mode_fixed_pose
        self.single_mode_start_id = single_mode_start_id
        self.single_mode_object_number = single_mode_object_number
        self.single_mode_fixed_poses = [0, 7, 14, 21]

        self.single_mode_ids = self.image_ids[54 * self.single_mode_start_id:
                                              54 * self.single_mode_start_id + 54 * self.single_mode_object_number]
        if self.is_single_mode:
            self.image_ids = list(self.single_mode_ids)

        self.num_input = num_input
        self.angle_range_1 = [0, 18]
        self.angle_scale_1 = 2
        self.angle_range_2 = [0, 3]
        self.angle_scale_2 = 10

    def __getitem__(self, index):
        target_id = self.image_ids[index]

        item = defaultdict(list)
        target_name, target_angle_1, target_angle_2 = parse_id(target_id)

        if self.is_single_mode and self.is_single_mode_fixed_pose:
            item['target_pose'].append(self.encode_pose(28, 0))
            item['target_img'].append(self.load_image_by_id('{}_{}_{}'.format(target_name, 28, 0), False))
        else:
            item['target_pose'].append(self.encode_pose(target_angle_1, target_angle_2))
            item['target_img'].append(self.load_image_by_id(target_id, False))

        for i in range(self.num_input):
            if self.is_single_mode and self.is_single_mode_fixed_pose:
                angle_1 = self.single_mode_fixed_poses[i]
                angle_2 = 0
            else:
                angle_1 = np.random.randint(*self.angle_range_1) * self.angle_scale_1
                angle_2 = np.random.randint(*self.angle_range_2) * self.angle_scale_2
            source_id = generate_id(target_name, angle_1, angle_2)

            item['source_img'].append(self.load_image_by_id(source_id))
            item['source_pose'].append(self.encode_pose(angle_1, angle_2))

        # Parse image data from dict
        target_img = torch.cat([torch.unsqueeze(v, 0) for v in item['target_img'][0]], 0)
        source_imgs = []
        for i in range(len(item['source_img'])):
            source_imgs.append(torch.cat([torch.unsqueeze(v, 0) for v in item['source_img'][i]], 0))

        # Parse pose data from dict
        target_pose = torch.cat([torch.unsqueeze(v, 0) for v in item['target_pose'][0]], 0)
        source_poses = []
        for i in range(len(item['source_img'])):
            source_poses.append(torch.cat([torch.unsqueeze(v, 0) for v in item['source_pose'][i]], 0))

        return source_imgs, source_poses, target_img, target_pose

    def __len__(self):
        return len(self.image_ids)

    def encode_pose(self, angle_1, angle_2):
        angle_1, angle_2 = int(angle_1), int(angle_2)

        encode = np.zeros((self.angle_scale_1 * self.angle_range_1[1] + self.angle_range_2[1], 1))
        encode[angle_1, 0] = 1
        encode[angle_2 // self.angle_scale_2 + self.angle_scale_1 * self.angle_range_1[1], 0] = 1

        return torch.tensor(encode, dtype=torch.float)

    def load_image_by_id(self, id, is_resize=True):
        image = Image.open(str(self.shapenet_object_dir / id) + '.png')

        if is_resize:
            trsfm = transforms.Compose([
                transforms.Resize(size=self.image_size),
                transforms.ToTensor(),
            ])
        else:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
            ])

        return trsfm(image) * 2 - 1


class ObjectDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, num_input, object_type, image_size, is_single_mode, single_mode_start_id,
                 single_mode_object_number, is_single_mode_fixed_pose, shuffle=True, validation_split=0.0,
                 num_workers=1, ):
        self.data_dir = data_dir
        self.dataset = ObjectDataset(data_dir, num_input, object_type, image_size, is_single_mode, single_mode_start_id,
                                     single_mode_object_number, is_single_mode_fixed_pose)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
