import os.path
import pdb
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_dataset_from_configs
from PIL import Image

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        assert (
            "geopifu" in opt.name and opt.no_instance
        )

        ### input A (label maps/ source image)
        ### input B (real images/ target image)
        self.dir_AB = opt.dataroot  # human_render_config location
        self.AB_paths = sorted(make_dataset_from_configs(self.dir_AB))

        if opt.phase == "train":
            self.AB_paths = self.AB_paths[:86976]
        else:
            self.AB_paths = self.AB_paths[86976:]

        self.dataset_size = len(self.AB_paths) 
      
    def __getitem__(self, index):        

        ### input A (label maps)
        volume_id  = index // 4 * 4
        view_id    = index - volume_id
        front_id   = volume_id + view_id
        right_id   = volume_id + (view_id+1) % 4
        back_id    = volume_id + (view_id+2) % 4
        left_id    = volume_id + (view_id+3) % 4
        index_list = [front_id, right_id, back_id, left_id]
        index_list_names = ["front", "right", "back", "left"]

        if index == front_id:
            b_index = back_id
        elif index == back_id:
            b_index = front_id
        elif index == right_id:
            b_index = left_id
        elif index == left_id:
            b_index = right_id

        # Convert a and mask it
        a_path = self.AB_paths[index]
        A = Image.open(a_path).convert('RGB')
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params) # Convert to tensor here
        a_rgb_img = transform_A(A)
        
        # mask_path = a_path.replace("rgbImage", "maskImage")
        # mask = Image.open(mask_path).convert('L')
        # mask = self.transform(mask)
        # mask = np.array(mask)

        # masked = np.array(im.copy())
        # masked[mask == 0] = 0
        # a_rgb_img = transforms.ToTensor()(masked)

        # Convert b and mask it
        b_path = self.AB_paths[b_index]
        B = Image.open(b_path).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        b_rgb_img = transform_B(B)

        # mask_path = b_path.replace("rgbImage", "maskImage")
        # mask = Image.open(mask_path).convert('L')
        # mask = self.transform(mask)
        # mask = np.array(mask)

        # masked = np.array(im.copy())
        # print(masked.shape)
        # print(mask.shape)
        # masked[mask == 0] = 0
        # b_rgb_img = transforms.ToTensor()(masked)
        
        A = a_rgb_img
        B = b_rgb_img

        input_dict = {'label': A, 'inst': 0, 'image': B, 
                        'feat': 0, 'A_path': a_path, 'B_path': b_path}

        return input_dict

    def __len__(self):
        return len(self.AB_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'UnalignedDataset'