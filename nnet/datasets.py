# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random

# PyTorch
import torch
import torch.nn as nn
import torchvision
import torchaudio
from torchvision.datasets.utils import extract_archive

# Other
import os
import glob
from tqdm import tqdm
import sentencepiece as spm
import numpy as np
import requests
import pickle
import gdown
import multiprocessing

# NeuralNets
from nnet import layers
from nnet import transforms
from nnet import collate_fn
from nnet import transforms
from nnet import collate_fn

###############################################################################
# Datasets
###############################################################################

class Dataset(torch.utils.data.Dataset):

    def __init__(self, batch_size=8, collate_fn=collate_fn.Collate(), root="datasets", shuffle=True):
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.root = root
        self.shuffle = shuffle

class MultiDataset(Dataset):

    def __init__(self, batch_size, collate_fn, datasets, shuffle=True):
        super(MultiDataset, self).__init__(batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, root=None)

        self.datasets = datasets

    def __len__(self):

        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, n):

        ctr = 0
        for dataset in self.datasets:
            ctr_prev = ctr
            ctr += len(dataset)
            if n < ctr:
                return dataset.__getitem__(n - ctr_prev)

class LRS(Dataset):

    """ LRS2 and LRS3 datasets
    
    Lip Reading Sentences 2 (LRS2) Dataset : https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html

    The dataset consists of thousands of spoken sentences from BBC television. Each sentences is up to 100 characters in length. 
    The training, validation and test sets are divided according to broadcast date. The dataset statistics are given in the table below.
    The utterances in the pre-training set correspond to part-sentences as well as multiple sentences, whereas the training set only consists of single full sentences or phrases. 
    There is some overlap between the pre-training and the training sets.
    Although there might be some label noise in the pre-training and the training sets, the test set has undergone additional verification; so, to the best of our knowledge, there are no errors in the test set.

    Infos:
        37 characters: 26 (a-z) letters + apostrophe (') + 10 (0-9) numbers
        total = 144482 samples
        pretrain + train = 142,157 training samples, 224 hours
        160 x 160, 25 fps videos

        - 96,318 pretrain samples, pretrain folder, 195 hours
        - 45,839 train samples, main folder, 28 hours
        - 1,082 val samples, main folder, 0.6 hours
        - 1,243 test samples, main folder, 0.5 hours

    Reference: "Deep Audio-Visual Speech Recognition", Afouras et al.
    https://arxiv.org/abs/1809.02108


    --------------------------------------------------------------------------------------------------------------------
    

    Lip Reading Sentences 3 (LRS3) Dataset : https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html

    The dataset consists of thousands of spoken sentences from TED and TEDx videos. 
    There is no overlap between the videos used to create the test set and the ones used for the pre-train and trainval sets.

    Reference: "LRS3-TED: a large-scale dataset for visual speech recognition", Afouras et al.
    https://arxiv.org/abs/1809.00496

    151,819 total samples
    150,498 pretrain + trainval samples, 438 hours
    118,516 pretrain samples from 5,090 videos, 407 hours
    31,982 trainval samples from 4,004 videos, 30 hours
    1,321 test samples from 412 videos, 1 hour

    37 characters: 26 (a-z) letters + apostrophe (') + 10 (0-9) numbers
    16kHz audio
    25fps 224x224pixel video
    
    """

    def __init__(self, batch_size, collate_fn, version="LRS2", img_mean=(0.5,), img_std=(0.5,), crop_mouth=True,
                 root="", shuffle=True, ascending=False,
                 mode="pretrain+train+val", load_audio=True, load_video=True, video_transform=None,
                 audio_transform=None, download=False, prepare=False, workers_prepare=-1, video_max_length=None,
                 audio_max_length=None, label_max_length=None, tokenizer_path="datasets/LRS3/tokenizerbpe256.model",
                 mean_face_path="media/20words_mean_face.npy", align=False, seg_num=4, my_settings=None):
        super(LRS, self).__init__(batch_size=batch_size, collate_fn=collate_fn, root=root, shuffle=shuffle and not ascending)

        assert version in ["LRS2", "LRS3"]

        # Params
        self.my_settings = my_settings
        self.seg_num = seg_num
        self.version = version
        self.mode = mode
        self.ascending = ascending
        self.load_audio = load_audio
        self.load_video = load_video
        self.video_max_length = video_max_length
        self.audio_max_length = audio_max_length
        self.label_max_length = label_max_length
        self.workers_prepare = multiprocessing.cpu_count() if workers_prepare==-1 else workers_prepare
        self.tokenizer_path = tokenizer_path
        self.crop_mouth = crop_mouth
        self.mean_face_path = mean_face_path
        self.align = align
        self.data_dir = '/home/zcw/Downloads/spilt_cs_acmmm'
        self.paths = glob.glob('{}/{}/*'.format(self.data_dir, self.mode))
        self.paths = sorted(self.paths)
        # print(self.paths)


        # Video Transforms
        self.video_preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
            layers.Permute(dims=(1, 0, 2, 3)),
            torchvision.transforms.Grayscale(),
            layers.Permute(dims=(1, 0, 2, 3)),
            transforms.NormalizeVideo(mean=img_mean, std=img_std),
            video_transform if video_transform != None else nn.Identity()
        ])
        file = open('./data_process/word_list.txt', 'r')
        word_list = file.read().split('\n')
        file.close()
        self.word_list = word_list

        self.new_label = {}
        print('used language:', self.my_settings['language'])
        if self.my_settings['language'] in ['en_cn']:
            f = open('./data_process/new/en_cn_label.txt', 'r')
            data = f.read().split('\n')
            f.close()
            for index, _ in enumerate(data):
                real_label, onehot_list = _.split(':')
                onehot_list = onehot_list.split(',')
                onehot_list = [int(__) for __ in onehot_list]
                self.new_label[index] = onehot_list
        elif self.my_settings['language'] == 'en':
            f = open('./data_process/new/en_label.txt', 'r')
            data = f.read().split('\n')
            f.close()
            for index, _ in enumerate(data):
                real_label, onehot_list = _.split(':')
                onehot_list = onehot_list.split(',')
                onehot_list = [int(__) for __ in onehot_list]
                self.new_label[index + 100] = onehot_list
        elif self.my_settings['language'] == 'cn':
            f = open('./data_process/new/cn_label.txt', 'r')
            data = f.read().split('\n')
            f.close()
            for index, _ in enumerate(data):
                real_label, onehot_list = _.split(':')
                onehot_list = onehot_list.split(',')
                onehot_list = [int(__) for __ in onehot_list]
                self.new_label[index + 200] = onehot_list



        # first run to index
        self.word_list_split = {'en': [], 'cn': [], 'en_cn': []}
        for i in tqdm(range(len(self.paths))):
            npz = np.load(self.paths[i])
            real_label = npz['real_label']
            label_part = self.word_list.index(real_label)
            en_cn = label_part // 100
            if en_cn == 0:
                self.word_list_split['en_cn'].append(i)
            elif en_cn == 1:
                self.word_list_split['en'].append(i)
            elif en_cn == 2:
                self.word_list_split['cn'].append(i)

        np.savez('./data_process/word_list_split_{}_v2.npz'.format(self.mode), **self.word_list_split)

        word_list_split = np.load('./data_process/word_list_split_{}_v2.npz'.format(self.mode))
        self.word_list_split = {'en': word_list_split['en'], 'cn': word_list_split['cn'], 'en_cn': word_list_split['en_cn']}


        if self.my_settings['language'] == 'en':
            self.paths = [self.paths[_] for _ in self.word_list_split['en']]
        elif self.my_settings['language'] == 'cn':
            self.paths = [self.paths[_] for _ in self.word_list_split['cn']]








    def __len__(self):

        return len(self.paths)

    def __getitem__(self, n):

        npz = np.load(self.paths[n])


        video = torch.tensor(npz['video'])
        video_len = torch.tensor(npz['video_length'])
        label_len = torch.tensor(npz['label_length'])
        real_label = npz['real_label']

        video = video.unsqueeze(-1)
        video = torch.repeat_interleave(video, 3, -1)

        # video = l w h c
        video = video.permute(3, 0, 1, 2)
        video = self.video_preprocessing(video)
        video = video.permute(1, 2, 3, 0)

        # Infos Preprocessing
        video_len = torch.tensor(video_len, dtype=torch.long)
        label_len = torch.tensor(label_len, dtype=torch.long)

        t = torch.tensor([n], dtype=torch.long)

        label_part = self.word_list.index(real_label)




        out_dict = {'video': video, 'label': torch.tensor(self.new_label[label_part], dtype=torch.long),
                    'video_len': video_len, 'label_len': label_len,
                    't': t}

        video = torch.concat([out_dict['video'].unsqueeze(0)], 0)

        label_max_len = 20
        label = torch.concat(
            [(torch.concat([out_dict['label'], torch.zeros(label_max_len - out_dict['label'].shape[0]
                                                              , dtype=out_dict['label'].dtype)], 0)).unsqueeze(0)], 0)


        video_len = torch.concat([out_dict['video_len'].unsqueeze(0)], 0)
        label_len = torch.concat([out_dict['label_len'].unsqueeze(0)], 0)
        t = torch.concat([out_dict['t'].unsqueeze(0)], 0)



        return video, None, label, video_len, None, label_len, t, t, t

class CorpusLM(Dataset):

    def __init__(self, batch_size, collate_fn, root="datasets", shuffle=True, download=False, tokenizer_path="datasets/LRS3/tokenizerbpe1024.model", max_length=None, corpus_path="datasets/LibriSpeechCorpus/librispeech-lm-norm.txt"):
        super(CorpusLM, self).__init__(batch_size=batch_size, collate_fn=collate_fn, root=root, shuffle=shuffle)

        # Params
        self.root = root
        self.max_len = max_length

        if isinstance(tokenizer_path, str):
            self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)
        else:
            self.tokenizer = tokenizer_path
        self.corpus = open(corpus_path, 'r').readlines()

    def __getitem__(self, i):

        if self.max_len:
            while len(self.tokenizer.encode(self.corpus[i].replace("\n", "").lower())) > self.max_len:
                i = torch.randint(0, self.__len__(), [])

        label = torch.LongTensor(self.tokenizer.encode(self.corpus[i].replace("\n", "").lower()))

        return label,

    def __len__(self):
        return len(self.corpus)

class LRW(Dataset):

    """ Lip Reading in the Wild (LRW) Dataset : https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html

    The dataset consists of up to 1000 utterances of 500 different words, spoken by hundreds of different speakers. 
    All videos are 29 frames (1.16 seconds) in length, and the word occurs in the middle of the video.

    Infos:
        488,766 train samples
        25,000 val samples
        test samples
        (29, 256, 256, 3) videos
        (1, 19456) audios
    
    """

    def __init__(self, batch_size, collate_fn, root="datasets", shuffle=True, mode="train", img_mean=(0.5,), img_std=(0.5,), crop_mouth=True, load_audio=True, load_video=True, video_transform=None, download=False, prepare=False, mean_face_path="media/20words_mean_face.npy", workers_prepare=-1):
        super(LRW, self).__init__(batch_size=batch_size, collate_fn=collate_fn, root=root, shuffle=shuffle)

        # Params
        self.workers_prepare = multiprocessing.cpu_count() if workers_prepare==-1 else workers_prepare
        self.crop_mouth = crop_mouth
        self.mean_face_path = mean_face_path
        self.load_audio = load_audio
        self.load_video = load_video

        # Download Dataset
        if download:
            self.download()

        # Prepare Dataset
        if prepare:
            self.prepare()

        # Mode
        assert mode in ["train", "val", "test"]

        # Class Dict
        self.class_dict = {}
        for i, path in enumerate(sorted(glob.glob(os.path.join(self.root, "LRW", "lipread_mp4", "*")))):
            c = path.split("/")[-1]
            self.class_dict[i] = c
            self.class_dict[c] = i

        # Paths
        self.paths = glob.glob(os.path.join(self.root, "LRW", "lipread_mp4", "*", mode, "*[0-9].mp4"))
        for i, path in enumerate(self.paths):
                self.paths[i] = path[:-4]

        # Video Transforms
        self.video_preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
            layers.Permute(dims=(1, 0, 2, 3)),
            torchvision.transforms.Grayscale(),
            layers.Permute(dims=(1, 0, 2, 3)),
            transforms.NormalizeVideo(mean=img_mean, std=img_std),
            video_transform if video_transform != None else nn.Identity()
        ])

    def __len__(self):

        return len(self.paths)

    def __getitem__(self, n):

        # Load Video
        if self.load_video:
            if self.crop_mouth:
                video, audio, infos = torchvision.io.read_video(self.paths[n] + "_mouth.mp4")
            else:
                video, audio, infos = torchvision.io.read_video(self.paths[n] + ".mp4")
        else:
            video = None

        # Load Audio
        if self.load_audio:
            audio = torchaudio.load(self.paths[n] + ".flac")[0]
        else:
            audio = None

        # Label
        c = self.paths[n].split("/")[-1].split("_")[0]
        label = self.class_dict[c]

        # Preprocessing
        video = self.video_preprocessing(video.permute(3, 0, 1, 2))
        audio = audio.squeeze(dim=0)
        label = torch.tensor(label)

        return video, audio, label

    class PrepareDataset:

        def __init__(self, paths, mean_face_path):
            self.paths = paths
            self.lip_crop = transforms.LipDetectCrop(mean_face_landmarks_path=mean_face_path)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):

            file_path = self.paths[idx]

            # Load Video
            video, audio , info = torchvision.io.read_video(file_path.replace(".txt", ".mp4"))

            # Save Audio
            torchaudio.save(file_path.replace(".txt", ".flac"), audio, sample_rate=16000)

            # Extract Landmarks
            landmarks_pathname = file_path.replace(".txt", ".npz").replace("lipread_mp4", "LRW_landmarks")
            person_id = 0
            multi_sub_landmarks = np.load(landmarks_pathname, allow_pickle=True)['data']
            landmarks = [None] * len(multi_sub_landmarks)
            for frame_idx in range(len(landmarks)):
                try:
                    landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)]['facial_landmarks']
                except IndexError:
                    continue

            # Interpolate Landmarks
            preprocessed_landmarks = self.lip_crop.landmarks_interpolate(landmarks)

            # Crop
            if not preprocessed_landmarks:
                video = torchvision.transforms.functional.resize(video.permute(3, 0, 1, 2), size=(self.lip_crop.crop_height, self.lip_crop.crop_width)).permute(1, 2, 3, 0)
            else:
                video = self.lip_crop.crop_patch(video.numpy(), preprocessed_landmarks)
                assert video is not None
                video = torch.tensor(video)

            # Save Video
            torchvision.io.write_video(filename=file_path.replace(".txt", "_mouth.mp4"), video_array=video, fps=info["video_fps"], video_codec="libx264")

            return file_path

    def prepare(self):

        # Prepare
        print("Prepare Dataset")
        dataloader = torch.utils.data.DataLoader(
            self.PrepareDataset(
                paths=glob.glob(os.path.join(self.root, "LRW", "lipread_mp4", "*", "*", "*.txt")),
                mean_face_path=self.mean_face_path
            ),
            batch_size=1,
            num_workers=self.workers_prepare,
            collate_fn=collate_fn.Collate(),
        )
        for batch in tqdm(dataloader):
            pass

    def download(self):

        # Print
        print("Download dataset")
        os.makedirs(os.path.join(self.root, "LRW"), exist_ok=True)

        # Download Pretrain
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaa",
            path=os.path.join(self.root, "LRW", "lrw-v1-partaa")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partab",
            path=os.path.join(self.root, "LRW", "lrw-v1-partab")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partac",
            path=os.path.join(self.root, "LRW", "lrw-v1-partac")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partad",
            path=os.path.join(self.root, "LRW", "lrw-v1-partad")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partae",
            path=os.path.join(self.root, "LRW", "lrw-v1-partae")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaf",
            path=os.path.join(self.root, "LRW", "lrw-v1-partaf")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partag",
            path=os.path.join(self.root, "LRW", "lrw-v1-partag")
        )
        os.system("cat " + os.path.join(self.root, "LRW", "lrw-v1*") + " > " +  os.path.join(self.root, "LRW", "lrw-v1.tar"))
        extract_archive(
            from_path=os.path.join(self.root, "LRW", "lrw-v1.tar"),
            to_path=os.path.join(self.root, "LRW")
        )

        # Download Landmarks from https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
        gdown.download("https://drive.google.com/uc?id=12mHlNQKCE2AXkFHzvRyqSbsmOMEs259i", os.path.join(self.root, "LRW", "LRW_landmarks.zip"), quiet=False)
        extract_archive(
            from_path=os.path.join(self.root, "LRW", "LRW_landmarks.zip"),
            to_path=os.path.join(self.root, "LRW")
        )

    def download_file(self, url, path):

        # Download, Open and Write
        with requests.get(url, auth=(os.getenv("LRW_USERNAME"), os.getenv("LRW_PASSWORD")), stream=True) as r:
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)
