import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2 as cv


#    █▀▄ ▄▀█ ▀█▀ ▄▀█ █▀ █▀▀ ▀█▀   █▀▀ █░░ ▄▀█ █▀ █▀
#    █▄▀ █▀█ ░█░ █▀█ ▄█ ██▄ ░█░   █▄▄ █▄▄ █▀█ ▄█ ▄█

class WLASLDataset(Dataset):
    """Class to load WLASL videos and labels."""
    
    def __init__(self, csv_file, transforms=None, frames_limit=0):
        """
        Args:
            csv_file (string): The path to the csv_file
            transforms: The transforms to apply to the videos
            frames_limit (int): The number of frames that every video will have
                                0 when I don't need to limit it
        """
        self.dataframe = pd.read_csv(csv_file)
        self.classes = self.dataframe.label.unique().tolist()
        self.transforms = transforms
        self.frames_limit = frames_limit

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        video = video_to_tensor(path = self.dataframe.iloc[index].path,
                                frames_limit = self.frames_limit)
        if self.transforms:
            video = self.transforms(video)
        
        label = self.dataframe.iloc[index].label
        label = self.classes.index(label)
        
        return video, label


#    █░█ █ █▀▄ █▀▀ █▀█   ▀█▀ █▀█   ▀█▀ █▀▀ █▄░█ █▀ █▀█ █▀█
#    ▀▄▀ █ █▄▀ ██▄ █▄█   ░█░ █▄█   ░█░ ██▄ █░▀█ ▄█ █▄█ █▀▄

def video_to_tensor(path, frames_limit=0):
    cap = cv.VideoCapture(path)
    
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    frames = torch.FloatTensor(3, num_frames, height, width)
    
    for i in range(num_frames):# while(cap.isOpened()):
        #cap.set(cv.CAP_PROP_POS_FRAMES, i) # to set the video position
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame)
        frame = frame.permute(2, 0, 1) # (H x W x C) to (C x H x W)
        frames[:, i, :, :] = frame.float()
    
    cap.release()
    
    if frames_limit == 0: return frames / 255
    
    ##############################
    # Limit the number of frames #
    ##############################
    
    multiplier = num_frames / frames_limit
    selected_frames = [0] * frames_limit
    for i in range(frames_limit):
        selected_frames[i] = (int)(i * multiplier)
    
    limited_frames = torch.FloatTensor(3, frames_limit, height, width)
    
    for i, elem in enumerate(selected_frames):
        limited_frames[:, i, :, :] = frames[:, elem, :, :]
    
    return limited_frames / 255