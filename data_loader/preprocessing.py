import numpy as np
import torch
import os
import pickle

from torch.utils.data.sampler import SubsetRandomSampler


# %%
# def save_subset_paths(path="/home/hackathon/dem/hackII"):
def save_subset_paths(path="C:/Users/Darwin/Desktop/hackII/"):
    input_path = os.path.join(path, "all_frames_5m6b")
    files_list = []
    for file in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path, file)):
            if np.sum(np.load(os.path.join(input_path, file))) > 0:
                files_list.append(file)
    files_list.sort()

    with open("files_list.pkl", "wb") as file:
        pickle.dump(files_list, file, -1)


# %%
# def read_data(path="/home/hackathon/dem/hackII"):
def read_data(path="C:/Users/Darwin/Desktop/hackII", subset_load=True):
    image_path = os.path.join(path, "all_frames_5m6b")
    label_path = os.path.join(path, "all_masks_5m6b")
    if subset_load:
        with open("files_list.pkl", "rb") as file:
            files_list = pickle.load(file)
    else:
        files_list = [file for file in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, file))]
        files_list.sort()

    input_data = [os.path.join(image_path, file) for file in files_list]
    label_data = [os.path.join(label_path, file) for file in files_list]
    return input_data, label_data


# %%
class DEMData(torch.utils.Dataset):
    def __init__(self):
        images, labels = read_data()

        self.images = []
        self.labels = []

        for image, label in (images, labels):
            if np.sum(label) > 0:
                continue

            dem = image[:,:,5]
            dem[dem == 0] = np.nan
            dem = (dem - 800) / 50
            dem[np.isnan(dem)] = 0
            dem[:, :, 5] = dem

            for r in range(4):
                self.images.append(np.rot90(image, r))
                self.labels.append(np.rot90(label, r))

                self.images.append(np.rot90(np.fliplr(image), r))
                self.labels.append(np.rot90(np.fliplr(label), r))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = torch.as_tensor(self.images[item], dtype=torch.float)
        label = torch.as_tensor(self.labels[item], dtype=torch.float)
        return image, label


# %%
def get_data_loaders(dataset, train_batch_size=32, test_batch_size=32):
    indices = np.asarray(range(int(len(dataset) / 8)))
    np.random.seed(42)
    np.random.shuffle(indices)
    indices = np.stack([indices * 8 + i for i in range(8)], axis=-1).reshape(-1)
    split = int(np.floor(0.8 * len(dataset) / 8) * 8)
    train_indices, test_indices = indices[:split], indices[split:]

    sampler_train = SubsetRandomSampler(train_indices)
    sampler_test = SubsetRandomSampler(test_indices)

    dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, sampler=sampler_train)
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, sampler=sampler_test)

    return dataloader_train, dataloader_test