from torch.utils.data import Dataset
from random import random
import torch
import random
from data_loader import label2vectormerge, clips2vectormerge
from torchvision.io.video import read_video
import numpy as np


class MultiViewDataset(Dataset):
    def __init__(self, path, start, end, fps, split, num_views, transform=None, transform_model=None):

        if split != 'Chall':
            # To load the annotations
            self.labels_offence_severity, self.labels_action, self.distribution_offence_severity,self.distribution_action, not_taking, self.number_of_actions = label2vectormerge(path, split, num_views)
            self.clips = clips2vectormerge(path, split, num_views, not_taking)[:20]
            self.distribution_offence_severity = torch.div(self.distribution_offence_severity, len(self.labels_offence_severity))
            self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

            self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
            self.weights_action = torch.div(1, self.distribution_action)
        else:
            self.clips = clips2vectormerge(path, split, num_views, [])

        # INFORMATION ABOUT SELF.LABELS_OFFENCE_SEVERITY
        # self.labels_offence_severity => Tensor of size of the dataset. 
        # each element of self.labels_offence_severity is another tensor of size 4 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if it is not an offence, then the tensor is [1, 0, 0, 0]. 

        # INFORMATION ABOUT SELF.LABELS_ACTION
        # self.labels_action => Tensor of size of the dataset. 
        # each element of self.labels_action is another tensor of size 8 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if the action is a tackling, then the tensor is [1, 0, 0, 0, 0, 0, 0, 0]. 

        # INFORMATION ABOUT SLEF.CLIPS
        # self.clips => list of the size of the dataset
        # each element of the list is another list of size of the number of views. The list contains the paths to all the views of that particular action.

        # The offence_severity groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_offence_severity tensor
        # The type of action groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_action tensor
        
        self.split = split
        self.start = start
        self.end = end
        self.transform = transform
        self.transform_model = transform_model
        self.num_views = num_views

        self.factor = (end - start) / (((end - start) / 25) * fps)
        self.length = len(self.clips)

    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action, 
    def getWeights(self):
        return self.weights_offence_severity, self.weights_action, 


    # RETURNS
    #
    # self.labels_offence_severity[index][0] => tensor of size 4. Example [1, 0, 0, 0] if the action is not an offence
    # self.labels_action[index][0] => tensor of size 8.           Example [1, 0, 0, 0, 0, 0, 0, 0] if the type of action is a tackling
    # videos => tensor of shape V, C, N, H, W with V = number of views, C = number of channels, N = the number of frames, H & W = height & width
    # self.number_of_actions[index] => the id of the action
    #
    def __getitem__(self, index):

        prev_views = []

        for num_view in range(len(self.clips[index])):

            index_view = num_view

            if len(prev_views) == 2:
                continue

            # As we use a batch size > 1 during training, we always randomly select two views even if we have more than two views.
            # As the batch size during validation and testing is 1, we can have 2, 3 or 4 views per action.
            cont = True
            if self.split == 'Train':
                while cont:
                    aux = random.randint(0,len(self.clips[index])-1)
                    if aux not in prev_views:
                        cont = False
                index_view = aux
                prev_views.append(index_view)


            video, _, _ = read_video(self.clips[index][index_view], output_format="THWC")
            frames = video[self.start:self.end,:,:,:]

            final_frames = None

            for j in range(len(frames)):
                if j%self.factor<1:
                    if final_frames == None:
                        final_frames = frames[j,:,:,:].unsqueeze(0)
                    else:
                        final_frames = torch.cat((final_frames, frames[j,:,:,:].unsqueeze(0)), 0)

            final_frames = final_frames.permute(0, 3, 1, 2)

            if self.transform != None:
                final_frames = self.transform(final_frames)

            final_frames = self.transform_model(final_frames)
            final_frames = final_frames.permute(1, 0, 2, 3)
            
            if num_view == 0:
                videos = final_frames.unsqueeze(0)
            else:
                final_frames = final_frames.unsqueeze(0)
                videos = torch.cat((videos, final_frames), 0)

        if self.num_views != 1 and self.num_views != 5:
            videos = videos.squeeze()   

        videos = videos.permute(0, 2, 1, 3, 4)

        if self.split != 'Chall':
            return self.labels_offence_severity[index][0], self.labels_action[index][0], videos, self.number_of_actions[index]
        else:
            return -1, -1, videos, str(index)

    def __len__(self):
        return self.length

class SelfSupervisedDataset(MultiViewDataset):
    def __init__(self, path, start, end, fps, split, num_views, transform=None, transform_model=None, semi_type='views'):
        self.semi_type = semi_type
        super().__init__(path, start, end, fps, split, num_views, transform, transform_model)
        if semi_type == 'views':
            self.length = super().__len__()*2 # double the length of the dataset
        else:
            self.length = super().__len__()

    def __getitem__(self, index):
        if self.semi_type == 'views':
            return self.get_item_views(index)
        else:
            return self.get_item_sync(index)

    def get_item_views(self, index):
        if index >= self.length/2:
            index = int(index - self.length/2)
            x = super().__getitem__(index)[2]
            #pick a random index not equal to the current index
            rand_idx = random.randint(0, self.length/2-1)
            while rand_idx == index:
                rand_idx = random.randint(0, self.length/2-1)
            
            y = super().__getitem__(rand_idx)[2]
            i = random.randint(0, 1)
            j = 1-i
            out_tensor = torch.cat((x[i].unsqueeze(0), y[j].unsqueeze(0)), 0)
            return out_tensor, torch.tensor(0, dtype=torch.float).unsqueeze(0)
        else:
            x = super().__getitem__(index)[2]
            return x, torch.tensor(1, dtype=torch.float).unsqueeze(0)

    def get_item_sync(self, index):
        x = super().__getitem__(index)[2]
        num_frames = x.shape[2]
        window_size = num_frames//2
        end_point = random.randint(window_size, num_frames-window_size) 
        start_point = end_point - window_size
        y = torch.zeros_like(x[:,:,start_point:end_point, :, :])
        y[0] = x[0,:,start_point:end_point, :, :]
        #pick probability of 0.1
        if random.random() < 0.1:
            y[1] = x[1,:,start_point:end_point, :, :]
            return y, torch.tensor(1, dtype=torch.float)
        else:
            direction = np.sign((num_frames-end_point)-start_point)
            if direction == -1:
                translation = random.randint(start_point//2, start_point)
            else:
                direction = 1
                translation = random.randint((num_frames-end_point)//2, num_frames-end_point)
            translation *= direction
            y[1] = x[1,:,start_point+translation:end_point+translation, :, :]
            return y, torch.tensor(0, dtype=torch.float)

    def __len__(self):
        return self.length