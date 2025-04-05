import numpy as np
from PIL import Image
import pickle
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

from torchvision import transforms as T
from torchvision.transforms import functional as F
import torchvision.models as models

from PIL import Image

import torch.optim as optim
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

import random
from torch.utils.data import random_split
from itertools import cycle

import datetime

class AttDataset(Dataset):
    """
*json_fp: path to JSON file, contenting a list where each element is a dict of image file path and annotation file path 
```
[
    {
    'img': '/path/to/image.jpg' 
    'json': '/path/to/ann.jpg'
    },
...
]
```
    """
    ATT_CLASSES = {
        "пол": ["мужчина", "женщина", "не определен"],
        "возраст": ["0-9 лет", "10-16 лет", "17-35 лет", "36-50 лет", "50+ лет", "не определен"],
        "голов_убор": ["шапка", "шляпа", "кепка", "капюшон", "платок", "отсутствует", "не определен"],
        "цвет_верх": ["желтый", "зеленый", "синий", "голубой", "красный", "розовый", "белый", "черный", "серый", "коричневый", "оранжевый", "фиолетовый", "не определен"],
        "цвет_низ": ["желтый", "зеленый", "синий", "голубой", "красный", "розовый", "белый", "черный", "серый", "коричневый", "оранжевый", "фиолетовый", "не определен"],
        "тип_верх": ["куртка", "пальто", "жилет", "толстовка", "футболка", "рубашка", "платье", "не определен"],
        "тип_низ": ["брюки", "юбка", "шорты", "не определен"],
        "сумка_спина": ["да", "отсутствует", "не определен"],
        "сумка_рука": ["да", "отсутствует", "не определен"]
    }
    ATT_ORDER = {
        "пол": 0,
        "возраст": 1,
        "голов_убор": 2,
        "цвет_верх": 3,
        "цвет_низ": 4,
        "тип_верх": 5,
        "тип_низ": 6,
        "сумка_спина": 7,
        "сумка_рука": 8
    }
    ATT_LABEL = {v:k for k,v in ATT_ORDER.items()}
    ATT_CLASSES_LABEL = {
        "пол": {i:v for i,v in enumerate(ATT_CLASSES["пол"])},
        "возраст": {i:v for i,v in enumerate(ATT_CLASSES["возраст"])},
        "голов_убор": {i:v for i,v in enumerate(ATT_CLASSES["голов_убор"])},
        "цвет_верх": {i:v for i,v in enumerate(ATT_CLASSES["цвет_верх"])},
        "цвет_низ": {i:v for i,v in enumerate(ATT_CLASSES["цвет_низ"])},
        "тип_верх": {i:v for i,v in enumerate(ATT_CLASSES["тип_верх"])},
        "тип_низ": {i:v for i,v in enumerate(ATT_CLASSES["тип_низ"])},
        "сумка_спина": {i:v for i,v in enumerate(ATT_CLASSES["сумка_спина"])},
        "сумка_рука": {i:v for i,v in enumerate(ATT_CLASSES["сумка_рука"])},
    }
    ATT_CLASSES_ORDER = {
        "пол": {v:i for i,v in enumerate(ATT_CLASSES["пол"])},
        "возраст": {v:i for i,v in enumerate(ATT_CLASSES["возраст"])},
        "голов_убор": {v:i for i,v in enumerate(ATT_CLASSES["голов_убор"])},
        "цвет_верх": {v:i for i,v in enumerate(ATT_CLASSES["цвет_верх"])},
        "цвет_низ": {v:i for i,v in enumerate(ATT_CLASSES["цвет_низ"])},
        "тип_верх": {v:i for i,v in enumerate(ATT_CLASSES["тип_верх"])},
        "тип_низ": {v:i for i,v in enumerate(ATT_CLASSES["тип_низ"])},
        "сумка_спина": {v:i for i,v in enumerate(ATT_CLASSES["сумка_спина"])},
        "сумка_рука": {v:i for i,v in enumerate(ATT_CLASSES["сумка_рука"])},
    }

    ATT_KEYS = ['gender', 'age', 'headwear', 'top_color', 'bottom_color', 'top_type', 'bottom_type', 'backpack', 'handbag']


    def __init__(self, json_fp, size=224, transform=None):
        with open(json_fp) as f:
            self.data = json.load(f)
        self.transform = transform
        self.size = size
        self.center_crop = T.CenterCrop(self.size)

    def __len__(self):
        return len(self.data)

    def letter_box(self, sample):
        img, ann = sample['img'], sample['ann']
        img = self.center_crop(F.resize(img, int(min(img.size)*(self.size/(max(img.size)))), 3))
        return {'img': img, 'ann': ann}

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.data[idx]['img'])
        with open(self.data[idx]['json']) as f:
            ann = json.load(f)
        sample = self.letter_box({'img': image, 'ann': ann})

        if self.transform:
            sample = self.transform(sample)

        return sample

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.subset[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.subset)
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, sample):
        image, ann = sample['img'], sample['ann']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        ann_arr = np.zeros(9)
        for k in ann:
            idx = AttDataset.ATT_ORDER[k]
            ival = AttDataset.ATT_CLASSES_ORDER[k][ann[k]]
            ann_arr[idx] = ival
        image = self.to_tensor(image)#image.transpose((2, 0, 1))
        return {'img': image.float(),
                'ann': torch.from_numpy(ann_arr).type(torch.LongTensor)}

class Normalize(object):
    def __init__(self, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        self.normilize = T.Normalize(mean, std)

    def __call__(self, sample):
        img, ann = sample['img'], sample['ann']
        img = self.normilize(img)

        return {'img': img, 'ann': ann}

class ColorJitter(object):
    def __init__(self):
        self.jitter = T.ColorJitter((0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (-0.05, 0.05))
    
    def __call__(self, sample):
        img, ann = sample['img'], sample['ann']
        img = self.jitter(img)

        return {'img': img, 'ann': ann}

class Sharpness(object):
    def __init__(self):
        self.sharp2 = T.RandomAdjustSharpness(sharpness_factor=2)
        self.sharp0 = T.RandomAdjustSharpness(sharpness_factor=0)
    
    def __call__(self, sample):
        img, ann = sample['img'], sample['ann']
        if random.random() >= 0.5:
            img = self.sharp2(img)
        else:
            img = self.sharp0(img)
        return {'img': img, 'ann': ann}

class HorizontalFlip(object):
    def __init__(self):
        self.flip = T.RandomHorizontalFlip(p=0.5)
    
    def __call__(self, sample):
        img, ann = sample['img'], sample['ann']
        img = self.flip(img)

        return {'img': img, 'ann': ann}

class Affine(object):
    def __init__(self):
        self.affine = T.RandomAffine(degrees=(-10, 10), translate=(0, 0.05), scale=(0.9, 1.1))

    def __call__(self, sample):
        img, ann = sample['img'], sample['ann']
        img = self.affine(img)

        return {'img': img, 'ann': ann}

class Classifier(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        # resnet50 = models.resnet50(weights=None)
        ensemble = {}
        for att in AttDataset.ATT_CLASSES:
            resnet = models.resnet50(weights=None)
            feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
            if backbone:
                feature_extractor.load_state_dict(torch.load(backbone, weights_only=True))
            ensemble[att] = feature_extractor
        # self.dropout_fe = nn.Dropout(1./2.)
        fc_attributes = {k: nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(1./3.),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(1./3.),
            nn.Linear(512, len(AttDataset.ATT_CLASSES[k]))
        ) for k in AttDataset.ATT_CLASSES}
        self.resnet_0 = ensemble["пол"]
        self.resnet_1 = ensemble["возраст"]
        self.resnet_2 = ensemble["голов_убор"]
        self.resnet_3 = ensemble["цвет_верх"]
        self.resnet_4 = ensemble["цвет_низ"]
        self.resnet_5 = ensemble["тип_верх"]
        self.resnet_6 = ensemble["тип_низ"]
        self.resnet_7 = ensemble["сумка_спина"]
        self.resnet_8 = ensemble["сумка_рука"]
        
        self.fc_0 = fc_attributes["пол"]
        self.fc_1 = fc_attributes["возраст"]
        self.fc_2 = fc_attributes["голов_убор"]
        self.fc_3 = fc_attributes["цвет_верх"]
        self.fc_4 = fc_attributes["цвет_низ"]
        self.fc_5 = fc_attributes["тип_верх"]
        self.fc_6 = fc_attributes["тип_низ"]
        self.fc_7 = fc_attributes["сумка_спина"]
        self.fc_8 = fc_attributes["сумка_рука"]
        # for k, v in fc_attributes.items():
        #     setattr(self, f"fc_{AttDataset.ATT_ORDER[k]}", v)

    def forward(self, x):
        return [self.fc_0(torch.squeeze(self.resnet_0(x), (2, 3))),
                self.fc_1(torch.squeeze(self.resnet_1(x), (2, 3))),
                self.fc_2(torch.squeeze(self.resnet_2(x), (2, 3))),
                self.fc_3(torch.squeeze(self.resnet_3(x), (2, 3))),
                self.fc_4(torch.squeeze(self.resnet_4(x), (2, 3))),
                self.fc_5(torch.squeeze(self.resnet_5(x), (2, 3))),
                self.fc_6(torch.squeeze(self.resnet_6(x), (2, 3))),
                self.fc_7(torch.squeeze(self.resnet_7(x), (2, 3))),
                self.fc_8(torch.squeeze(self.resnet_8(x), (2, 3)))]

    def compute_metrics(self, out, target, device='cuda'):
        with torch.no_grad():
            preds = torch.stack([torch.max(o_i, 1)[1] for o_i in out]).to(device)
            target = target.to(device)
            scores = [(preds[jj] == target[:, jj]).sum().div(len(target)).item() for jj in range(len(preds))]
            scores_dict = {AttDataset.ATT_LABEL[i]: scores[i] for i in AttDataset.ATT_LABEL}
            score = np.array(scores).mean()
        return scores_dict, score
            

    def to_text(self, preds):
        res = ""
        for i in range(8):
            res += f"{i+1}. {preds[i]}.\n"
        res+=f"9. {preds[8]}."
        return res

    def to_json(self, preds):
        return {k:v for k,v in zip(AttDataset.ATT_KEYS, preds)}

    def pred(self, image, device='cuda'):
        _to_tensor = T.ToTensor()
        _central_crop = T.CenterCrop(224)
        _norm = T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        image = F.resize(image, int(min(image.size)*(224/(max(image.size)))), 3)
        image = _norm(_to_tensor(_central_crop(image)).float()).unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            preds = self(image)
        # preds = torch.stack([torch.max(o_i, 1)[1] for o_i in preds])
        if device=="cpu":
            preds = [np.argmax(o_i[0]) for o_i in [preds[self.output(jj)] for jj in range(9)]]
        else:
            preds = [torch.max(o_i[0], 0)[1].item() for o_i in preds]
        preds = [AttDataset.ATT_CLASSES_LABEL[AttDataset.ATT_LABEL[i]][preds[i]] for i in range(9)]
        return preds

class AttributesLoss():
    def __init__(self, device):
        self.CE = nn.CrossEntropyLoss().to(device)
        self.device = device
    def __call__(self, out, target):
        # for i in range(9):
        #     losses[i] = self.CE(out, target[:, i])
        # loss = torch.mean(losses
        losses = [self.CE(out[i], target[:, i]) for i in range(9)]
        loss = torch.mean(torch.stack(losses).to(self.device))                 
        return loss


def save(folder, train_losses, val_losses, scores, att_scores, E):
    with open(os.path.join(folder, f"e{E}_results.json"), "w") as f:
        res_obj = {
            "score": scores[-1],
            "score_per_attribute": {k:att_scores[k][-1] for k in att_scores},
            "train_loss": train_losses[-1],
            "val_loss": val_losses[-1]
        }
        json.dump(res_obj, f)
    with open(os.path.join(folder, "results.pickle"), "wb") as f:
        res_obj = [{
            "score": scores[i],
            "score_per_attribute": {k:att_scores[k][i] for k in att_scores},
            "train_loss": train_losses[i],
            "val_loss": val_losses[i]
        } for i in range(E)]
        pickle.dump(res_obj, f)
    with open(os.path.join(folder, "model.txt"), 'w') as f:
        f.write(f"{net}")
    with open(os.path.join(folder, "optimizer.txt"), 'w') as f:
        f.write(f"{optimizer}")
    torch.save(net.state_dict(), os.path.join(folder, f"e{E}_s{scores[-1]:.3f}_model.pt"))
    
    
    fig, axs = plt.subplots(2)
    axs[0].plot(train_losses, "r")
    axs[0].plot(val_losses, "b")
    axs[0].set_xlabel("epoch", fontsize = 15)
    axs[0].set_ylabel("loss",fontsize = 15)
    __set_ty = lambda x,y: min(x + y) + (max(x + y) - min(x + y))*0.8
    axs[0].text(E*0.1,
             __set_ty(val_losses, train_losses),
             f'Train final loss: {train_losses[-1]:.3f}\nVal final loss: {val_losses[-1]:.3f}',
             fontsize = 11)
    axs[1].set_xlabel("epoch", fontsize = 15)
    axs[1].set_ylabel("score",fontsize = 15)
    __set_ty = lambda x: min(x) + (max(x) - min(x))*0.4
    axs[1].text(E*0.5,
             __set_ty(scores),
             f'Final score: {scores[-1]:.3f}',
             fontsize = 13)
    axs[1].plot(scores)
    plt.savefig(os.path.join(folder, "plots.pdf"))
    plt.show()

class Trainer():
    def __init__(self, train_loader, val_loaders, model):
        self.train_loader = train_loader
        self.val_loaders_iter = cycle(val_loaders)
        self.val_loader = next(self.val_loaders_iter)
        self.model = model
    def train(self, optimizer, scheduler, criterion, device, folder, e = 20, log_num=500):
        train_losses = []
        val_losses = []
        scores = []
        att_scores = {k:[] for k in AttDataset.ATT_CLASSES}
        epoch = 0
        try:
            for epoch in range(e):  # loop over the dataset multiple times
                running_loss = 0.0
                for i, data in enumerate(tqdm(self.train_loader), 0):
                    inputs, labels = data['img'].to(device), data['ann'].to(device)
            
                    # zero the parameter gradients
                    optimizer.zero_grad()
            
                    # forward + backward + optimize
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
                    # print statistics
                    running_loss += loss.item()
                    if i % log_num == log_num-1:    # print every log_num mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f}')
        
                train_losses.append(running_loss / len(self.train_loader))  
    
                #evaluation
                print("Eval")
                self.model.eval()
                running_loss = 0.0
                running_score = 0.0
                running_att_score = {k:0.0 for k in AttDataset.ATT_CLASSES}
                for i, data in enumerate(tqdm(self.val_loader, 0)):
                    inputs, labels = data['img'].to(device), data['ann'].to(device)
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)        
                        running_loss += loss.item()
    
                        att_sc, sc = self.model.compute_metrics(outputs, labels)
                        running_score += sc
                        running_att_score = {k:running_att_score[k] + att_sc[k] for k in att_sc}
                print(f'loss: {running_loss / len(self.val_loader):.3f}')
                print(f'SCORE: {running_score / len(self.val_loader):.3f}')
    
                for k in running_att_score:
                    print(f'{k} score: {running_att_score[k] / len(self.val_loader):.3f}')
                val_losses.append(running_loss / len(self.val_loader))
                
                scheduler.step(running_loss / len(self.val_loader))
                
                scores.append(running_score /  len(self.val_loader))
                for k in att_scores:
                    att_scores[k].append(running_att_score[k] / len(self.val_loader))
                self.val_loader = next(self.val_loaders_iter)
                self.model.train()

                if (epoch+1) % 5 == 0:
                    save(folder, train_losses, val_losses, scores, att_scores, epoch+1)
                        
            print('Finished Training')
            epoch+=1
        except KeyboardInterrupt:
            save(folder, train_losses, val_losses, scores, att_scores, epoch)
        return train_losses, val_losses, scores, att_scores, epoch

if __name__ == "__main__":
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    composed = T.Compose([ToTensor(), ColorJitter(), Sharpness(), HorizontalFlip(), Affine(), Normalize()])
    compose_val = T.Compose([ToTensor(), Normalize()])
    # dataset = AttDataset("/home/Tracking/classifier_train_Sep10_2024_05.19.48.json", 224, composed)
    # data_val = AttDataset("/home/Tracking/classifier_val_Sep10_2024_05.19.48.json", 224, compose_val)
    dataset_splits = random_split(AttDataset('/home/Tracking/classifier_dataset_Nov19_2024_05.32.09.json', 224), [0.9, 0.025, 0.025, 0.025, 0.025])
    dataset = DatasetFromSubset(dataset_splits[0])
    dataset.transform = T.Compose([ToTensor(), ColorJitter(), Sharpness(), HorizontalFlip(), Affine(), Normalize()])
    data_val_splits = dataset_splits[1:]
    for i, _ in enumerate(data_val_splits):
        data_val_splits[i] = DatasetFromSubset(data_val_splits[i])
        data_val_splits[i].transform = T.Compose([ToTensor(), Normalize()])

    BS = 32#128
    dloader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=16)
    # dloader_val = DataLoader(data_val, batch_size=32, shuffle=True, num_workers=16)
    dloader_vals = [DataLoader(data_val, batch_size=32, shuffle=True, num_workers=16) for data_val in data_val_splits]

    net = Classifier(backbone = "/home/Tracking/Classifier/resnet50_backbone.pt")
    net = net.to('cuda')
    criterion = AttributesLoss('cuda')
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    E = 200
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(len(dataset) / BS * 0.4), eta_min=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    date_str = str(datetime.datetime.now().strftime("_%b%d_%Y_%H.%M.%S"))
    SAVE_FOLDER = f"resnet_ensemble_bs{BS}_{date_str}"
    os.makedirs(SAVE_FOLDER)
    trainer = Trainer(dloader, dloader_vals, net)
    train_losses, val_losses, scores, att_scores, E = trainer.train(optimizer, scheduler, criterion, 'cuda', SAVE_FOLDER,  e=E, log_num=100)