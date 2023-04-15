import os, glob
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from torch.utils import data
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import MLFlowLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, balanced_accuracy_score, adjusted_mutual_info_score
from sklearn.manifold import TSNE
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from models.combiner import Combiner
# from torchviz import make_dot

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



class LitClassification(pl.LightningModule):
    def __init__(self, cfg, cwd, time_str):
        super().__init__()

        #freeze text encoder and finetune only image encoder
        # substitute _transform() for preprocess()
        self.model, _ = clip.load("RN50", device="cuda")
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.best_score = 0
        self.cfg = cfg
        self.cwd = cwd
        self.time_str = time_str
        self.learning_rate = cfg.lr
        self.batch_size = self.cfg.batch_size
        self.automatic_optimization = False

        self.table = {}

        # text encoder
        for param in self.model.transformer.parameters():
            param.requires_grad = False

        self.acc_stack = np.zeros(10)

    def forward(self, img, txt):
        logits_per_image, logits_per_text = self.model(img, txt)
        
        return logits_per_image, logits_per_text

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.lr)

        # milestones = self.cfg.milestones
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=0.1)

        return [self.optimizer, ], [lr_scheduler, ]
        # return [self.optimizer, ]

    def training_step(self, train_batch, batch_idx):
        img, txt = train_batch
        
        logits_per_image, logits_per_text = self(img, txt)

        label = torch.arange(len(img)).cuda()

        loss = F.cross_entropy(logits_per_image , label) + F.cross_entropy(logits_per_text, label)
        
        self.optimizer.zero_grad()
        self.manual_backward(loss)
        self.model.float()
        self.optimizer.step()
        convert_models_to_mix(self.model)
        
        print(loss.item())

        return loss

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        self.log('train loss', loss)
        self.log('learning rate', self.optimizers().param_groups[0]['lr'])

    def validation_step(self, val_batch, batch_idx):
        img, txt = val_batch
        logits_per_image, logits_per_text = self(img, txt)

        label = torch.arange(len(img)).cuda()
        loss = F.cross_entropy(logits_per_image, label) + F.cross_entropy(logits_per_text, label)

        # T2I retrieval
        rank_5_count_t2i = torch.tensor([0])
        for i in range(len(logits_per_text)):
            sorted_logits, sorted_indices = torch.sort(logits_per_text[i, :], descending=True)
            index = np.argwhere(sorted_indices.cpu().numpy() == i)
            if index < 5: rank_5_count_t2i += 1 
        
        return {"val_loss":loss, "rank_5_count_t2i":rank_5_count_t2i, "data_num": torch.tensor([len(logits_per_image)])}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean().item()
        rank_5_count_t2i = torch.stack([x["rank_5_count_t2i"] for x in outputs]).detach().cpu().numpy().sum()
        data_num = torch.stack([x["data_num"] for x in outputs]).detach().cpu().numpy().sum()

        accr5 = rank_5_count_t2i/data_num
        
        self.log('validation loss', loss)
        self.log('rank_5_count_t2i', accr5)

        state_dict = self.state_dict()
        state_dict_dir = os.path.join(self.cwd, "checkpoint", self.time_str)
        os.makedirs(state_dict_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(state_dict_dir, "model-current.ckpt"))
        if self.update_best(accr5):
            torch.save(state_dict, os.path.join(state_dict_dir, "model-best.ckpt"))

    def update_best(self, new_score):
        if new_score > self.best_score:
            self.best_score = new_score
            return True
        else:
            return False

    def test_step(self, test_batch, batch_idx):
        img, txt, img_pth_g, c_idx = test_batch
        
        logits_per_image, logits_per_text = self(img, txt)

        # I2T retrieval
        rank_1_count_i2t, rank_5_count_i2t, rank_10_count_i2t = torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
        for i in range(len(logits_per_image)):
            sorted_logits, sorted_indices = torch.sort(logits_per_image[i, :], descending=True)

            index = np.argwhere(sorted_indices.cpu().numpy() == i)
            imgpth = img_pth_g[i]
            self.table[imgpth] = np.array(img_pth_g)[sorted_indices.cpu().numpy()[:5]]
            if index < 1: 
                rank_1_count_i2t += 1 
            if index < 5: 
                rank_5_count_i2t += 1 
            if index < 10: 
                rank_10_count_i2t += 1

        # T2I retrieval
        rank_1_count_t2i, rank_5_count_t2i, rank_10_count_t2i = torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
        for i in range(len(logits_per_text)):
            sorted_logits, sorted_indices = torch.sort(logits_per_text[i, :], descending=True)

            index = np.argwhere(sorted_indices.cpu().numpy() == i)

            if index < 1: rank_1_count_t2i += 1 
            if index < 5: rank_5_count_t2i += 1 
            if index < 10: rank_10_count_t2i += 1
        
        return {"rank_1_count_i2t":rank_1_count_i2t, "rank_5_count_i2t":rank_5_count_i2t, "rank_10_count_i2t":rank_10_count_i2t, 
                "rank_1_count_t2i":rank_1_count_t2i, "rank_5_count_t2i":rank_5_count_t2i, "rank_10_count_t2i":rank_10_count_t2i,
                "data_num": torch.tensor([len(logits_per_image)])}

    def test_epoch_end(self, outputs):

        rank_1_count_i2t = torch.stack([x["rank_1_count_i2t"] for x in outputs]).detach().cpu().numpy().sum()
        rank_5_count_i2t = torch.stack([x["rank_5_count_i2t"] for x in outputs]).detach().cpu().numpy().sum()
        rank_10_count_i2t = torch.stack([x["rank_10_count_i2t"] for x in outputs]).detach().cpu().numpy().sum()

        rank_1_count_t2i = torch.stack([x["rank_1_count_t2i"] for x in outputs]).detach().cpu().numpy().sum()
        rank_5_count_t2i = torch.stack([x["rank_5_count_t2i"] for x in outputs]).detach().cpu().numpy().sum()
        rank_10_count_t2i = torch.stack([x["rank_10_count_t2i"] for x in outputs]).detach().cpu().numpy().sum()

        data_num = torch.stack([x["data_num"] for x in outputs]).detach().cpu().numpy().sum()

        print('>>> retrieval {}: acc@1: {}, acc@5: {}, acc@10: {}'.format("i2t", rank_1_count_i2t/data_num, rank_5_count_i2t/data_num, rank_10_count_i2t/data_num))
        print('>>> retrieval {}: acc@1: {}, acc@5: {}, acc@10: {}'.format("t2i", rank_1_count_t2i/data_num, rank_5_count_t2i/data_num, rank_10_count_t2i/data_num))

        table_df = pd.DataFrame.from_dict(self.table).T
        table_df.to_csv(os.path.join(self.cwd, "outputs", "predict_onlyglobal_100_drop07.csv"))

def convert_models_to_mix(model):
    clip.model.convert_weights(model)



class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir, cfg, workers):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.workers = workers
        self.cfg = cfg

        self.train_dataset = VTDataset(data_dir, cfg, "train")
        self.val_dataset = VTDataset(data_dir, cfg, "val")
        self.test_dataset = VTDataset(data_dir, cfg, "test")

        self.num_data = len(self.train_dataset)
        self.cloth_imagelist_dict = self.train_dataset.cloth_imagelist_dict

        self.setup_folds_kai(self.cfg.num_folds)
        self.setup_fold_index(0)



    def setup_folds(self, num_folds: int) -> None:
        # self.splits = [split for split in KFold(num_folds).split(range(self.num_data))]
        self.splits = []
        idxes = list(range(self.num_data))
        # idxes = idxes[12:] + idxes[:12]
        random.shuffle(idxes)
        n = self.num_data // num_folds
        for i in range(num_folds):
            self.splits.append((np.array(idxes[:n*i]+idxes[n*(i+1):]), np.array(idxes[n*i:n*(i+1)])))
        
        self.splits = self.splits[self.cfg.val_fold:] + self.splits[:self.cfg.val_fold]

    def setup_folds_kai(self, num_folds: int) -> None:
        self.splits = []
        idxes = list(self.cloth_imagelist_dict.keys())
        random.shuffle(idxes)
        n = len(idxes) // num_folds
        for i in range(num_folds):
            train_data = []
            val_data = []
            for j in range(len(idxes)):
                if j < n*i or n*(i+1) <= j:
                    train_data += self.cloth_imagelist_dict[idxes[j]]
                else:
                    val_data += self.cloth_imagelist_dict[idxes[j]]
            self.splits.append((np.array(train_data), np.array(val_data)))
            # self.splits.append((np.array(idxes[:n*i]+idxes[n*(i+1):]), np.array(idxes[n*i:n*(i+1)])))

        self.splits = self.splits[self.cfg.val_fold:] + self.splits[:self.cfg.val_fold]


    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.val_dataset, val_indices)
        self.test_fold = Subset(self.test_dataset, val_indices)

        print("len(self.train_fold) = ", len(self.train_fold))
        print("len(self.val_fold) = ", len(self.val_fold))


   
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
                self.train_fold,
                batch_size=self.batch_size,
                num_workers=self.workers,
                shuffle=True,
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
                self.val_fold,
                batch_size=self.batch_size,
                num_workers=self.workers,
                shuffle=False,
            )

    def test_dataloader(self) -> DataLoader:
        torch.manual_seed(42)
        return DataLoader(
                self.test_fold,
                batch_size=self.batch_size,
                num_workers=self.workers,
                shuffle=True,
            )


class VTDataset(Dataset):
    def __init__(self, data_dir, cfg, phase):
        self.data_dir = data_dir
        self.cfg = cfg
        self.phase = phase
        self.filenames = pd.read_csv(os.path.join(data_dir, cfg.labelfile_name), index_col=0).index

        self.cltidx2desc_global = pd.read_csv(os.path.join(data_dir, self.cfg.cltidx2desc_global), index_col=0)

        self.attr2desc_local = {}
        with open(os.path.join(data_dir, self.cfg.attr2desc_local)) as f:
            lines = f.readlines()
        for l in lines:
            l = l.strip("\n")
            a = l.split(",")[0]
            self.attr2desc_local[a] = l[len(a)+1:]
        imgpth2labels = pd.read_csv(os.path.join(data_dir, self.cfg.labelfile_name), index_col=0)
        self.attrs = imgpth2labels.columns

        self.imgpths = imgpth2labels.index.values
        self.labels = imgpth2labels.values

        self.cloth_imagelist_dict = {}
        for i in range(len(self.filenames)):
            cloth = "/".join(self.filenames[i].split("/")[:-1])
            if cloth not in self.cloth_imagelist_dict.keys():
                self.cloth_imagelist_dict[cloth] = [i]
            else:
                self.cloth_imagelist_dict[cloth].append(i)


    def __len__(self):
        return len(self.imgpths)

    def __getitem__(self, idx):
        
        # # local
        # img_path = os.path.join(self.data_dir, str(self.imgpths[idx])).replace("images", self.cfg.local_img_dir)
        # img_pil = Image.open(img_path)
        # img_tensor = _transform(img_pil, 1, self.phase)

        # global
        img_path = os.path.join(self.data_dir, str(self.imgpths[idx])).replace("images", self.cfg.global_img_dir)
        img_pil = Image.open(img_path)
        img_tensor = _transform(img_pil, 1, self.phase)

        # # multi-labelなので, shuffleしてdescriptionを一つ選ぶ.
        # # local
        # attr_idx = np.where(self.labels[idx] == 1)[0]
        # np.random.shuffle(attr_idx)
        # attr = self.attrs[attr_idx[0]]
        # desc = self.attr2desc_local[attr]
        # desc_tokenized = clip.tokenize(desc)[0]

        cltidx = int(img_path.split("/")[7])

        # global
        desc = self.cltidx2desc_global.loc[cltidx, "3"]
        desc_tokenized = clip.tokenize(desc)[0]

        if self.phase == "train" or self.phase == "val":
            return img_tensor, desc_tokenized

        elif self.phase == "test":
            return img_tensor, desc_tokenized, img_path, cltidx



def _transform(img_pil, r, phase):
    h, w = img_pil.size
    crop_size = (224, 224)

    # resize & crop
    if phase == "train":
        trans = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.RandomRotation(10),
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomCrop(crop_size),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    elif phase == "val" or phase == "test":
        trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(crop_size),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        NotImplementedError

    img_tensor = trans(img_pil)

    return img_tensor


def _convert_image_to_rgb(image):
    return image.convert("RGB")



@hydra.main(config_path="./config", config_name="finetune.yaml")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    cwd = hydra.utils.get_original_cwd()
    seed_everything(cfg.seed)
    
    if cfg.train_mode == "train":
        time_str = time.strftime('%m%d%H%M')
        save_dir = time_str + "_{}_{}".format(cfg.dataset, cfg.num_epochs)
    elif cfg.train_mode == "test":
        save_dir = "01160122_FACAD_30" # "onlyglobal_onlyimgenc_10ep_100b_1e5"


    # data_dirs = []
    # for dataset in cfg.datasets:
    #     data_dirs.append(os.path.join(cwd, dataset, "data"))
    data_dir = os.path.join(cwd, cfg.dataset, "data")
    datamodule = DataModule(data_dir, cfg, 4)
    model =LitClassification(cfg, cwd, save_dir)

    # # view model backward
    # data_img = torch.randn(2, 3, 224, 224).cuda()
    # data_txt = torch.randint(0, 100, (2, 77)).cuda()
    # y = model(data_img, data_txt)
    # image = make_dot(y, params=dict(model.named_parameters()))
    # image.format = "png"
    # image.render(os.path.join(cwd, "NeuralNet"))


    mlf_logger = MLFlowLogger(experiment_name="default", tracking_uri="file:{}".format(os.path.join(cwd, "mlruns")))
    mlf_logger.log_hyperparams(cfg)
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.num_epochs, log_every_n_steps=1, accelerator="auto", 
                            logger=mlf_logger, auto_lr_find=False, num_sanity_val_steps=0) # gradient_clip_val=0.5, detect_anomaly=True


    if cfg.train_mode == "train":
        trainer.fit(model, datamodule)
        state_dict_dir = os.path.join(cwd, "checkpoint", save_dir)
        trainer.save_checkpoint(os.path.join(state_dict_dir, "model-last.ckpt"))
        model.load_state_dict(torch.load(os.path.join(state_dict_dir, "model-best.ckpt")), strict=True)
        trainer.test(model, datamodule=datamodule)
    elif cfg.train_mode == "test":
        state_dict_dir = os.path.join(cwd, "checkpoint", save_dir)
        model.load_state_dict(torch.load(os.path.join(state_dict_dir, "model-best.ckpt")), strict=True)
        # torch.save(model.model.state_dict(), os.path.join(state_dict_dir, "model.ckpt"))
        trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    main()
