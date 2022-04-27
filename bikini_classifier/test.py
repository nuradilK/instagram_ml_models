import torch
import argparse
from tqdm import tqdm

import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import set_seed
from model import MobileNetClf


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--images-dir', required=True, help='Dataset directory path')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    model = MobileNetClf()
    model.load_state_dict(torch.load('best_model.pt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_size = 320
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = pretrained_means, 
            std = pretrained_stds)
    ])

    test_data = datasets.ImageFolder(
        root = args.images_dir, 
        transform = test_transforms)

    test_iterator = data.DataLoader(
        test_data, 
        batch_size = args.batch_size)

    model.to(device)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for (x, y) in tqdm(test_iterator, desc='Eval'):
            x = x.to(device)
            y = y.to(device)

            preds += model(x).tolist()
            targets += y.tolist()

    print(preds, targets)

if __name__ == '__main__':
    main()