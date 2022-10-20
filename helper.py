import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, labels,paths, transform=None, target_transform=None):
        self.img_labels = labels
        self.img_paths = paths
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        image = image/255
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def CleanUpDataFrame(par_df,
                     img_file_names,
                     video_name_col = 'video_lower',
                     image_path_name_col = 'image_paths',
                     target_name_col = 'valence_rating'
                    ):
    par_df[video_name_col]=[str.lower(video_str.split('_')[0].split('.')[0]) for video_str in par_df.video]
    img_names_list = [str.lower(img_str.split('/')[-1].split('_')[0]) for img_str in img_file_names]
    img_path = []
    for vid in par_df[video_name_col]:
        try:
            img_path.append(img_file_names[img_names_list.index(vid)])
        except:
            img_path.append('vid missing')
    par_df[image_path_name_col]=img_path
    val_image_df = par_df[[target_name_col,image_path_name_col]][par_df.image_paths!='vid missing']
    val_image_df = val_image_df.dropna()
    return val_image_df

def three_class_problem(label):
    if label<2:
        return torch.tensor(0)
    if label==2:
        return torch.tensor(1)
    if label>2:
        return torch.tensor(2)

def TrainTestLoop(model_ft,
                  optimizer,
                  criterion,
                  train_dataloader,
                  test_dataloader,
                  num_epochs,
                  device
                 ):
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
            test_inputs,test_labels = next(iter(test_dataloader))
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model_ft(test_inputs)
            _, test_preds = torch.max(test_outputs,1)
            test_acc = torch.sum(test_preds == test_labels.data)/len(test_labels)
        print(f'Test Acc: {test_acc:.4f}')
        epoch_avg_acc_list.append(np.mean(epoch_train_acc_list))
        test_acc_list.append(test_acc.cpu().numpy())
    return epoch_avg_acc_list,test_acc_list