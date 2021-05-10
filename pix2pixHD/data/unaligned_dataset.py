import os.path
import pdb
import numpy as np
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_dataset_from_configs
from PIL import Image
import cv2

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
        """
            Unmasked
        """
        if "mask" not in self.opt.name:
            a_path = self.AB_paths[index]
            A = Image.open(a_path).convert('RGB')
            params = get_params(self.opt, A.size)
            transform_A = get_transform(self.opt, params) # Convert to tensor here
            a_rgb_img = transform_A(A)

            # Convert b and mask it
            b_path = self.AB_paths[b_index]
            B = Image.open(b_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            b_rgb_img = transform_B(B)

        else:
            """
                Masked
            """
            # print("Using masked for training...")
            a_path = self.AB_paths[index]
            # a_rgb_img = Image.open(a_path).convert('RGB')
            # read data BGR -> RGB, np.uint8
            a_rgb_img = cv2.imread(a_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}
            a_rgb_img_padded = np.zeros((max(a_rgb_img.shape), max(a_rgb_img.shape), 3), np.uint8) # (1536, 1536, 3)
            a_rgb_img_padded[:,a_rgb_img_padded.shape[0]//2-min(a_rgb_img.shape[:2])//2:a_rgb_img_padded.shape[0]//2+min(a_rgb_img.shape[:2])//2,:] = a_rgb_img # (1536, 1536, 3)

            # resize to (512, 512, 3), np.uint8
            a_rgb_img_padded = cv2.resize(a_rgb_img_padded, (self.opt.fineSize, self.opt.fineSize))
            a_rgb_img = Image.fromarray(a_rgb_img_padded)
            params = get_params(self.opt, a_rgb_img.size)
            self.transform = get_transform(self.opt, params) # Don't convert to tensor, only resize etc
            a_rgb_img = self.transform(a_rgb_img).float()
            
            self.transform_mask = get_transform(self.opt, params, normalize=False)
            mask_path = a_path.replace("rgbImage", "maskImage")
            mask_data = np.round((cv2.imread(mask_path)[:,:,0]).astype(np.float32)/255.) # (1536, 1024)
            mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32) # (1536, 1536)
            mask_data_padded[:,mask_data_padded.shape[0]//2-min(mask_data.shape)//2:mask_data_padded.shape[0]//2+min(mask_data.shape)//2] = mask_data # (1536, 1536)

            # NN resize to (512, 512)
            # a_mask = Image.open(mask_path).convert('L')
            mask_data_padded = cv2.resize(mask_data_padded, (self.opt.fineSize,self.opt.fineSize), interpolation=cv2.INTER_NEAREST)
            mask_data_padded = Image.fromarray(mask_data_padded)
            a_mask = self.transform_mask(mask_data_padded).float()

            a_rgb_img = a_mask.expand_as(a_rgb_img) * a_rgb_img # rgb value where mask is black is black

            # # Convert b and mask it
            b_path = self.AB_paths[b_index]
            # b_rgb_img = Image.open(b_path).convert('RGB')
            b_rgb_img = cv2.imread(b_path)[:,:,::-1] # (1536, 1024, 3), np.uint8, {0,...,255}
            b_rgb_img_padded = np.zeros((max(b_rgb_img.shape), max(b_rgb_img.shape), 3), np.uint8) # (1536, 1536, 3)
            b_rgb_img_padded[:,b_rgb_img_padded.shape[0]//2-min(b_rgb_img.shape[:2])//2:b_rgb_img_padded.shape[0]//2+min(b_rgb_img.shape[:2])//2,:] = b_rgb_img # (1536, 1536, 3)

            # resize to (512, 512, 3), np.uint8
            b_rgb_img_padded = cv2.resize(b_rgb_img_padded, (self.opt.fineSize, self.opt.fineSize))
            b_rgb_img = Image.fromarray(b_rgb_img_padded)
            b_rgb_img = self.transform(b_rgb_img).float()

            mask_path = b_path.replace("rgbImage", "maskImage")
            mask_data = np.round((cv2.imread(mask_path)[:,:,0]).astype(np.float32)/255.) # (1536, 1024)
            mask_data_padded = np.zeros((max(mask_data.shape), max(mask_data.shape)), np.float32) # (1536, 1536)
            mask_data_padded[:,mask_data_padded.shape[0]//2-min(mask_data.shape)//2:mask_data_padded.shape[0]//2+min(mask_data.shape)//2] = mask_data # (1536, 1536)

            # NN resize to (512, 512)
            # b_mask = Image.open(mask_path).convert('L')
            mask_data_padded = cv2.resize(mask_data_padded, (self.opt.fineSize,self.opt.fineSize), interpolation=cv2.INTER_NEAREST)
            mask_data_padded = Image.fromarray(mask_data_padded)
            b_mask = self.transform_mask(mask_data_padded).float()

            b_rgb_img = b_mask.expand_as(b_rgb_img) * b_rgb_img
        
        A = a_rgb_img
        B = b_rgb_img

        input_dict = {'label': A, 'inst': 0, 'image': B, 
                        'feat': 0, 'A_path': a_path, 'B_path': b_path}

        return input_dict

    def __len__(self):
        return len(self.AB_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'UnalignedDataset'