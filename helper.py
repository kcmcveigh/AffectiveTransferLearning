import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    """
    Custom dataset class - for summary images from affective videos dataset
    """

    def __init__(self, labels, paths, transform=None, target_transform=None):
        """
        init custom dataset
        :param labels: a list of integers
        :param paths:a list of strings - each string is
        :param transform (callable, optional): Optional transform to be applied
                on a sample.
        :param target_transform (callable, optional): optional transform on labels
        """
        self.img_labels = labels
        self.img_paths = paths
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        image = image / 255
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def CleanUpDataFrame(par_df,
                     img_file_names,
                     video_name_col='video_lower',
                     image_path_name_col='image_paths',
                     target_name_col='valence_rating'
                     ):
    """
    reformated and clean BBOE dataframe to include paths to summary images, and drop any rows with missing affective ratings
    :param par_df: dataframe from BBOE affective ratings dataset
    :param img_file_names:list of strings for video summary imaages
    :param video_name_col: optional name of video column, default value 'video_lower'
    :param image_path_name_col:optional name of image path, default value 'image_paths'
    :param target_name_col: optional name of affective feature to predict, default value 'valence_rating'
    :return: returns pandas dataframe with target image paths, and affective feature column
    """
    par_df[video_name_col] = [str.lower(video_str.split('_')[0].split('.')[0]) for video_str in par_df.video]#not the prettiest
    img_names_list = [str.lower(img_str.split('/')[-1].split('_')[0]) for img_str in img_file_names]#not the prettiest
    img_path = []
    for vid in par_df[video_name_col]:
        try:
            img_path.append(img_file_names[img_names_list.index(vid)])
        except:
            img_path.append('vid missing')
    par_df[image_path_name_col] = img_path
    val_image_df = par_df[[target_name_col, image_path_name_col]][par_df.image_paths != 'vid missing']
    val_image_df = val_image_df.dropna()
    return val_image_df


def three_class_problem(label):
    """
    convert 5 point likert scale (0 indexed) to 3 values, where any thing below 2 is recoved as zero (negative), equal to 2 is
    recoded to 1 (neutral) and above 2 is recoded 2 (positive)
    :param label: numerical rating of affective feature endorsed for video
    :return: corresponding 3 point value for initial 5 point value
    """
    if label < 2:
        return torch.tensor(0)
    if label == 2:
        return torch.tensor(1)
    if label > 2:
        return torch.tensor(2)


def TrainTestLoop(model_ft,
                  optimizer,
                  criterion,
                  train_dataloader,
                  test_dataloader,
                  num_epochs,
                  device
                  ):
    """
    :param model_ft: pytorch model
    :param optimizer: pytorch optimizer
    :param criterion: loss function
    :param train_dataloader: pytorch dataloader for training set
    :param test_dataloader: pytorch dataloader for testing set
    :param num_epochs: number of epochs to train for
    :param device: device to train should be 'gpu' or 'cpu'
    :return: average accuracies across all epochs list of floats on cpu, testing accuracy for each epoch list of floats
    on cpu
    """
    test_acc_list = []
    epoch_avg_acc_list = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        model_ft.train()
        # Iterate over data.
        epoch_train_acc_list = []
        for inputs, labels in train_dataloader:
            running_loss = 0.0
            running_corrects = 0
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs.softmax(dim=1), labels.long())

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(labels)
            epoch_acc = running_corrects.double() / len(labels)
            epoch_train_acc_list.append(epoch_acc.cpu())
            print(f' Loss: {epoch_loss:.4f} train Acc: {epoch_acc:.4f}')

        model_ft.eval()
        with torch.no_grad():
            test_inputs, test_labels = next(iter(test_dataloader))
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model_ft(test_inputs)
            _, test_preds = torch.max(test_outputs, 1)
            test_acc = torch.sum(test_preds == test_labels.data) / len(test_labels)
        print(f'Test Acc: {test_acc:.4f}')
        epoch_avg_acc_list.append(np.mean(epoch_train_acc_list))
        test_acc_list.append(test_acc.cpu().numpy())
    return epoch_avg_acc_list, test_acc_list
