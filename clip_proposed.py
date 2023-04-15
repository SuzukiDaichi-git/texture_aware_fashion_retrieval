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
from common.metrics import RecallAtK_kaleido
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
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
        self.model_g, _ = clip.load("RN50", device="cuda")
        self.model_l, _ = clip.load("RN50", device="cuda")
        
        feature_dim = self.model_g.visual.output_dim
        projection_dim = 2560
        hidden_dim = 5120
        self.img_combiner = Combiner(feature_dim, projection_dim, hidden_dim)
        self.txt_combiner = Combiner(feature_dim, projection_dim, hidden_dim)
        self.maxpool = torch.nn.MaxPool1d(2)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.feat_dict = {}

        self.metrics = RecallAtK_kaleido()

        # self.model_g.load_state_dict(torch.load(os.path.join(cwd, "checkpoint/12221031_FACAD_10/model_g.ckpt")))
        # self.model_l.load_state_dict(torch.load(os.path.join(cwd, "checkpoint/12221031_FACAD_10/model_l.ckpt")))
        # self.img_combiner.load_state_dict(torch.load(os.path.join(cwd, "checkpoint/12221031_FACAD_10/img_combiner.ckpt")))
        # self.txt_combiner.load_state_dict(torch.load(os.path.join(cwd, "checkpoint/12221031_FACAD_10/txt_combiner.ckpt")))

        self.best_score = 0
        self.cfg = cfg
        self.cwd = cwd
        self.time_str = time_str
        self.learning_rate = cfg.lr
        self.batch_size = self.cfg.batch_size
        self.automatic_optimization = False

        self.table = {}

        # # image encoder for global image
        # for param in self.model_g.visual.parameters():
        #     param.requires_grad = False

        # text encoder
        for param in self.model_g.transformer.parameters():
            param.requires_grad = False
        for param in self.model_l.transformer.parameters():
            param.requires_grad = False
        
        self.acc_stack = np.zeros(10)

    def forward(self, img_g, txt_g, img_l, txt_l_list, img_pth):
        img_feat_g = self.model_g.encode_image(img_g)
        txt_feat_g = self.model_g.encode_text(txt_g)
        img_feat_l = self.model_l.encode_image(img_l)

        # for i in range(6):
        #     if i == 0:
        #         txt_feat_l = self.model_l.encode_text(txt_l_list[i])
        #     else:
        #         txt_feat_l += self.model_l.encode_text(txt_l_list[i])
        # txt_feat_l = txt_feat_l / 6

        txt_feat_l_list = []
        for i in range(len(txt_l_list)):
            txt_feat_l_list.append(self.model_l.encode_text(txt_l_list[i]))

        txt_feat_l, _ = torch.max(torch.stack(txt_feat_l_list), axis=0)


        img_feat_g = img_feat_g.to(torch.float32)
        txt_feat_g = txt_feat_g.to(torch.float32)
        img_feat = self.img_combiner.combine_features(img_feat_g, img_feat_l)
        txt_feat = self.txt_combiner.combine_features(txt_feat_g, txt_feat_l)

        for i in range(len(img_pth)):
            self.feat_dict["/".join(img_pth[i].split("/")[-3:])] = img_feat[i].cpu().detach()

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()

        logits_per_image = logit_scale * img_feat @ txt_feat.t()
        
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam([{'params':self.model_g.parameters()}, {'params':self.model_l.parameters()}, 
                                            {'params':self.img_combiner.parameters(), "lr": 1e-4}, {'params':self.txt_combiner.parameters(), "lr": 1e-4}],
                                            lr=self.learning_rate)
        # self.optimizer = torch.optim.AdamW([{'params':self.model_g.parameters()}, {'params':self.model_l.parameters()}, 
        #                                     {'params':self.img_combiner.parameters(), "lr": 1e-4}],
        #                                     lr=self.learning_rate)

        milestones = self.cfg.milestones
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=0.1)

        return [self.optimizer, ], [lr_scheduler, ]
        # return [self.optimizer, ]


    def training_step(self, train_batch, batch_idx):
        img_g, txt_g, img_l, txt_l_list = train_batch
        
        logits_per_image, logits_per_text = self(img_g, txt_g, img_l, txt_l_list)

        label = torch.arange(len(img_l)).cuda()

        loss = F.cross_entropy(logits_per_image , label) + F.cross_entropy(logits_per_text, label)
        
        self.optimizer.zero_grad()
        if self.current_epoch == 0:
            lr_scale = min(1., float(batch_idx + 1) / 260.)
            for i, pg in enumerate(self.optimizer.param_groups):
                if i < 2:
                    pg["lr"] = 1e-5 * lr_scale
                else:
                    pg["lr"] = 1e-4 * lr_scale
        self.manual_backward(loss)
        self.model_g.float()
        self.model_l.float()
        self.optimizer.step()
        convert_models_to_mix(self.model_g)
        convert_models_to_mix(self.model_l)
        
        print(loss.item())

        return loss

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        self.log('train loss', loss)
        self.log('learning rate', self.optimizers().param_groups[0]['lr'])

    def validation_step(self, val_batch, batch_idx):
        img_g, txt_g, img_l, txt_l_list = val_batch
        logits_per_image, logits_per_text = self(img_g, txt_g, img_l, txt_l_list)

        label = torch.arange(len(img_g)).cuda()
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
        img_g, txt_g, img_l, txt_l_list, img_pth_g, c_idx = test_batch
        
        # img_feat_g = self.model_g.encode_image(img_g)
        # txt_feat_g = self.model_g.encode_text(txt_g)
        # img_feat_l = self.model_l.encode_image(img_l)
        # txt_feat_l = self.model_l.encode_text(txt_l)

        # img_feat_g = img_feat_g.to(torch.float32)
        # txt_feat_g = txt_feat_g.to(torch.float32)
        # img_feat = self.img_combiner.combine_features(img_feat_g, img_feat_l).detach().cpu()
        # txt_feat = self.txt_combiner.combine_features(txt_feat_g, txt_feat_l).detach().cpu()

        # return {"img_feat": img_feat, "txt_feat": txt_feat, "data_num": torch.tensor([len(txt_feat)])}


        logits_per_image, logits_per_text = self(img_g, txt_g, img_l, txt_l_list, img_pth_g)

        # I2T retrieval
        rank_1_count_i2t, rank_5_count_i2t, rank_10_count_i2t = torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
        for i in range(len(logits_per_image)):
            sorted_logits, sorted_indices = torch.sort(logits_per_image[i, :], descending=True)

            index = np.argwhere(sorted_indices.cpu().numpy() == i)

            if index < 1: rank_1_count_i2t += 1 
            if index < 5: rank_5_count_i2t += 1 
            if index < 10: rank_10_count_i2t += 1

        # T2I retrieval
        rank_1_count_t2i, rank_5_count_t2i, rank_10_count_t2i = torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
        for i in range(len(logits_per_text)):
            sorted_logits, sorted_indices = torch.sort(logits_per_text[i, :], descending=True)

            index = np.argwhere(sorted_indices.cpu().numpy() == i)
            imgpth = img_pth_g[i]
            
            self.table[imgpth] = np.array(img_pth_g)[sorted_indices.cpu().numpy()[:5]]
            if index < 1: 
                rank_1_count_t2i += 1
            if index < 5: 
                rank_5_count_t2i += 1 
            if index < 10: 
                rank_10_count_t2i += 1
        
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
        
        torch.save(self.feat_dict, os.path.join(self.cwd, "FACAD/data/test_img_feats.pt"))
        # table_df = pd.DataFrame.from_dict(self.table).T
        # table_df.to_csv(os.path.join(self.cwd, "outputs", "predict_globallocal_100_drop07.csv"))


        # sample_list = {}
        # model_output = {}
        # model_output["scores"] = torch.cat([x["img_feat"] for x in outputs])
        # model_output["targets"] = torch.cat([x["txt_feat"] for x in outputs])
        # data_num = torch.stack([x["data_num"] for x in outputs]).detach().cpu().numpy().sum()

        # sample_list["image_id"] = torch.tensor(range(data_num))
        # sample_list["text_id"] = torch.tensor(range(data_num))
        # sample_list["image_subcat_id"] = torch.zeros(data_num)
        # sample_list["text_subcat_id"] = torch.zeros(data_num)

        # res = self.metrics.calculate(sample_list, model_output)
        # print(res)

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
        # if self.phase == "train":
        #     for i, data_dir in enumerate(data_dirs):
        #         f_l = pd.read_csv(os.path.join(data_dir, "labels_train_processed.csv"), index_col=0)
        #         f_l.loc[:, "data_dir"] = data_dir
        #         if i == 0:
        #             filenames_labels = f_l
        #         else:
        #             filenames_labels = pd.concat([filenames_labels, f_l], axis=0)
                
        # elif self.phase == "val" or "test":
        #     for i, data_dir in enumerate(data_dirs):
        #         f_l = pd.read_csv(os.path.join(data_dir, "labels_val_processed.csv"), index_col=0)
        #         f_l.loc[:, "data_dir"] = data_dir
        #         if i == 0:
        #             filenames_labels = f_l
        #         else:
        #             filenames_labels = pd.concat([filenames_labels, f_l], axis=0)
        #     self.cloth_idxes = filenames_labels["1"].values
        # else:
        #     AttributeError
        # self.data_dirs = filenames_labels.loc[:, "data_dir"].values

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

        # drop_idx = filenames_labels.index[np.where(filenames_labels.sum(axis=1) == 0)[0]]
        # filenames_labels = filenames_labels.drop(index = drop_idx)

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
        
        img_path_l = os.path.join(self.data_dir, str(self.imgpths[idx])).replace("images", self.cfg.local_img_dir)
        img_pil_l = Image.open(img_path_l)
        img_tensor_l = _transform(img_pil_l, 1, self.phase)
        img_path_g = os.path.join(self.data_dir, str(self.imgpths[idx])).replace("images", self.cfg.global_img_dir)
        img_pil_g = Image.open(img_path_g)
        img_tensor_g = _transform(img_pil_g, 1, self.phase)

        attr_idx = np.where(self.labels[idx] == 1)[0]
        desc_tokenized_l_list = []
        for i in range(6):
            attr = self.attrs[attr_idx[i % len(attr_idx)]]
            desc_l = self.attr2desc_local[attr]
            desc_tokenized_l_list.append(clip.tokenize(desc_l)[0])


        # # attributeを連結してテキストにする.
        # attr_idx = np.where(self.labels[idx] == 1)[0]
        # desc_tokenized_l_list = []
        # for i in range(6):
        #     attr = self.attrs[attr_idx[i % len(attr_idx)]]
        #     desc_l = attr
        #     desc_tokenized_l_list.append(clip.tokenize(desc_l)[0])

        # desc_l = "a photo of a"
        # for i, idx in enumerate(attr_idx):
        #     if i != 0:
        #         desc_l += " and"
        #     attr = self.attrs[idx]
        #     desc_l += " " + attr
        # desc_l += " fabric"
        # desc_tokenized_l = clip.tokenize(desc_l)[0]

        cltidx = int(img_path_g.split("/")[7])

        desc_g = self.cltidx2desc_global.loc[cltidx, "3"]
        desc_tokenized_g = clip.tokenize(desc_g)[0]

        if self.phase == "train" or self.phase == "val":
            return img_tensor_g, desc_tokenized_g, img_tensor_l, desc_tokenized_l_list

        elif self.phase == "test":
            return img_tensor_g, desc_tokenized_g, img_tensor_l, desc_tokenized_l_list, img_path_g, cltidx



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
            transforms.Normalize((0.46777044, 0.44531429, 0.40661017), (0.12221994, 0.12145835, 0.14380469)),
            # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    elif phase == "val" or phase == "test":
        trans = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Resize(n_px, interpolation=BICUBIC),
            transforms.CenterCrop(crop_size),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.46777044, 0.44531429, 0.40661017), (0.12221994, 0.12145835, 0.14380469)),
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
        save_dir = "globallocal_adamw_warmup_drop07_max" # "globallocal_adamw_warmup_drop07" # "01180048_FACAD_10" # attrs:"01180048_FACAD_10" # desc:"01180026_FACAD_10" # "globallocal_onlyimgenc_10ep_100b_1e5_1e4"


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
        # torch.save(model.model_g.state_dict(), os.path.join(state_dict_dir, "model_g.ckpt"))
        # torch.save(model.model_l.state_dict(), os.path.join(state_dict_dir, "model_l.ckpt"))
        # torch.save(model.img_combiner.state_dict(), os.path.join(state_dict_dir, "img_combiner.ckpt"))
        # torch.save(model.txt_combiner.state_dict(), os.path.join(state_dict_dir, "txt_combiner.ckpt"))
        trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    main()
