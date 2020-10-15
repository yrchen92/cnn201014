import os, time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from PIL import Image

def myParser():
    parser = argparse.ArgumentParser(description='AI Algorithm')
    parser.add_argument('--train_data', default='data/train')
    parser.add_argument('--test_data', default='data/test')
    parser.add_argument('--model_path', default='data/model.pt')
    parser.add_argument('--base_lr', default=0.001)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--max_epoch', default=10)
    args = parser.parse_args()
    return args

class CaptchaData(Dataset):
    def __init__(self, data_path, num_class=36, num_char=4, 
                 transform=None, target_transform=None):
        super(Dataset, self).__init__()
        source = [str(i) for i in range(0, 10)]
        source += [chr(i) for i in range(97, 97+26)]
        self.alphabet = ''.join(source)
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self.make_dataset(self.data_path, self.alphabet, 
                                    self.num_class, self.num_char)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = self.img_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.Tensor(target)

    def make_dataset(self, data_path, alphabet, num_class, num_char):
        img_names = os.listdir(data_path)
        samples = []
        for img_name in img_names:
            img_path = os.path.join(data_path, img_name)
            target_str = img_name.split('.')[0]
            assert len(target_str) == num_char
            target = []
            for char in target_str:
                vec = [0] * num_class
                vec[alphabet.find(char)] = 1
                target += vec
            samples.append((img_path, target))
        return samples

    def img_loader(self, img_path):
        img = Image.open(img_path)
        return img.convert('RGB')

class CNN(nn.Module):
    def __init__(self, num_class=36, num_char=4):
        super(CNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
                #batch*3*180*100
                nn.Conv2d(3, 16, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                #batch*16*90*50
                nn.Conv2d(16, 64, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                #batch*64*45*25
                nn.Conv2d(64, 512, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                #batch*512*22*12
                nn.Conv2d(512, 512, 3, padding=(1, 1)),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                #batch*512*11*6
                )
        self.fc = nn.Linear(512*11*6, self.num_class*self.num_char)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512*11*6)
        x = self.fc(x)
        return x

def calculat_acc(output, target):
    output, target = output.view(-1, 36), target.view(-1, 36)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc

def main(args):
     # data preparation
    transforms = Compose([ToTensor()])
     # training data
    train_dataset = CaptchaData(args.train_data, transform=transforms)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, 
                             shuffle=True, drop_last=True)
    # testing data
    test_data = CaptchaData(args.test_data, transform=transforms)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, 
                                  num_workers=0, shuffle=False, drop_last=True)
    # network
    cnn = CNN()
    if torch.cuda.is_available():
        cnn.cuda()
    if os.path.isfile(args.model_path):
        cnn.load_state_dict(torch.load(args.model_path))
    # optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.base_lr)
    # loss
    criterion = nn.MultiLabelSoftMarginLoss() # CrossEntropyLoss
    # train
    for epoch in range(args.max_epoch):
        start_t = time.time()
        loss_history = []
        acc_history = []
        cnn.train()
        # training
        for img, target in train_data_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = calculat_acc(output, target)
            acc_history.append(float(acc))
            loss_history.append(float(loss))
        print('train_loss: {:.4}|train_acc: {:.4}'.format(
                torch.mean(torch.Tensor(loss_history)),
                torch.mean(torch.Tensor(acc_history)),
                ))
        # testing
        loss_history = []
        acc_history = []
        cnn.eval()
        for img, target in test_data_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)
            acc = calculat_acc(output, target)
            acc_history.append(float(acc))
            loss_history.append(float(loss))
        print('test_loss: {:.4}|test_acc: {:.4}'.format(
                torch.mean(torch.Tensor(loss_history)),
                torch.mean(torch.Tensor(acc_history)),
                ))
        print('epoch: {}|time: {:.4f}'.format(epoch, time.time()-start_t))
        torch.save(cnn.state_dict(), args.model_path)

if __name__ == "__main__":
    args = myParser()
    main(args)
