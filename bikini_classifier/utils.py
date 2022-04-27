import os
import shutil
import torch
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import _LRScheduler


class corruptedDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]

        try:
            sample = self.loader(path)
        except:
            return None

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device

        torch.save(model.state_dict(), 'init_params.pt')

    def range_test(self, model, iterator, end_lr = 10, num_iter = 100, 
                   smooth_f = 0.05, diverge_th = 5):
        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        
        iterator = IteratorWrapper(iterator)
        
        for iteration in range(num_iter):
            loss = self._train_batch(iterator)

            #update lr
            lr_scheduler.step()
            lrs.append(lr_scheduler.get_lr()[0])

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
                
            if loss < best_loss:
                best_loss = loss

            losses.append(loss)
            
            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break
                       
        #reset model to initial parameters
        model.load_state_dict(torch.load('init_params.pt'))
                    
        return lrs, losses

    def _train_batch(self, iterator):
        self.model.train()
        self.optimizer.zero_grad()
        
        x, y = iterator.get_batch()
        x = x.to(self.device)
        y = y.to(self.device)
        
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y.float())
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)


def plot_images(images, labels, classes, normalize = True):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (15, 15))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        label = classes[labels[i]]
        ax.set_title(label)
        ax.axis('off')


def plot_lr_finder(lrs, losses, skip_start = 5, skip_end = 5):
    
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]
    
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    fig.savefig('lr_finder.png')
    plt.close(fig)


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image    


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_dataset_dir(args, train_dir, test_dir, validation_dir):
    img_classes = os.listdir(args.dataset_dir)

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        return
        
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    for img_class in img_classes:
        class_dir = os.path.join(args.dataset_dir, img_class)
        images = os.listdir(class_dir)
        
        n_train = int(len(images) * args.train_ratio)
        
        train_images = images[:n_train]
        test_images = images[n_train:]
        
        os.makedirs(os.path.join(train_dir, img_class), exist_ok = True)
        os.makedirs(os.path.join(test_dir, img_class), exist_ok = True)
        
        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, img_class, image)
            shutil.copyfile(image_src, image_dst)
            
        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, img_class, image) 
            shutil.copyfile(image_src, image_dst)


def get_data(args, train_dir, test_dir, val_dir):
    train_data = corruptedDataset(root = train_dir, 
                                  transform = transforms.ToTensor())

    img_shape = (320, 320)
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize(img_shape),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = pretrained_means, 
            std = pretrained_stds)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(img_shape),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = pretrained_means, 
            std = pretrained_stds)
    ])

    train_data = corruptedDataset(
        root=train_dir, 
        transform=train_transforms)

    val_data = corruptedDataset(
        root = val_dir, 
        transform = train_transforms)

    test_data = corruptedDataset(
        root = test_dir, 
        transform = test_transforms)

    # n_train_examples = int(len(train_data) * (1 - args.valid_ratio))
    # n_valid_examples = len(train_data) - n_train_examples

    # train_data, valid_data = data.random_split(
    #     train_data,
    #     [n_train_examples, n_valid_examples])

    # valid_data = copy.deepcopy(valid_data)
    # valid_data.dataset.transform = test_transforms

    return train_data, val_data, test_data