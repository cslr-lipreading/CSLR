import datetime
import os
import sys

sys.path.append("../../")

# Imports
import nnet
import torch
import torch.nn as nn
import torchvision

# Architecture

# initial
# interctc_blocks = [3, 6, 9]
# loss_weights = [0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5]
# modified
interctc_blocks = []
loss_weights = [1]



# Beam Search
beamsearch = False
tokenizer_path = None

ngram_path = "datasets/LRS3/6gram_lrs23.arpa"
ngram_offset = 100
beam_size = 16
ngram_alpha = 0.6
ngram_beta = 1.0
ngram_tmp = 1.0
neural_config_path = "configs/LRS23/LM/GPT-Small.py"
neural_checkpoint = "checkpoints_epoch_10_step_2860.ckpt"
neural_alpha = 0.6
neural_beta = 1.0
test_augments = torchvision.transforms.RandomHorizontalFlip(p=1.0)
test_time_aug = False

# Training
batch_size = 24
eval_batch_size = 8
accumulated_steps = 2
eval_training = False
precision = torch.float16
recompute_metrics = True
callback_path = "callbacks/LRS23/VO/EffConfInterCTC"

my_settings = {
    'exp_name': 'GRU_ours',
    'method': ['ours'][0],
    'front_end': ['shufflenet'][0],
    'back_end': ['gru'][0]
}


my_settings['dim_model'] = [256]
my_settings['num_blocks'] = [2]
my_settings['language'] = 'en_cn'

save_path = '{}'.format('./ckpt')
save_path += '/' + datetime.datetime.now().isoformat().split('.')[0]
save_path = save_path.replace(':', '_')
save_path += '_' + my_settings['exp_name']
if not os.path.isdir(save_path):
    os.makedirs(save_path)
os.makedirs('{}/pt'.format(save_path), exist_ok=True)
os.makedirs('{}/res'.format(save_path), exist_ok=True)

my_settings['save_path'] = save_path

if 'en_cn' in my_settings['language']:
    vocab_size = 414
elif my_settings['language'] == 'en':
    vocab_size = 130
elif my_settings['language'] == 'cn':
    vocab_size = 288

model = nnet.VisualEfficientCTC(vocab_size=vocab_size, interctc_blocks=interctc_blocks,
                                test_augments=test_augments if test_time_aug else None, my_settings=my_settings)
model.compile(
    losses=None if test_time_aug else nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
    decoders={
        "outputs": nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path, mode=my_settings['language'])
        if not beamsearch
        else nnet.CTCBeamSearchDecoder(tokenizer_path=tokenizer_path,
                                       beam_size=beam_size,
                                       ngram_path=ngram_path,
                                       ngram_tmp=ngram_tmp,
                                       ngram_alpha=ngram_alpha,
                                       ngram_beta=ngram_beta,
                                       ngram_offset=ngram_offset,
                                       neural_config_path=neural_config_path,
                                       neural_checkpoint=neural_checkpoint,
                                       neural_alpha=neural_alpha,
                                       neural_beta=neural_beta,
                                       test_time_aug=test_time_aug)},
    metrics={"outputs": nnet.WordErrorRate()},
    loss_weights=loss_weights
)

if 'single_en_cn' in my_settings['method'] and (my_settings['load_pretrain_en'] or my_settings['load_pretrain_cn']):
    if my_settings['load_pretrain_en']:
        model_en = torch.load(my_settings['en_pretrain'])
        state_dict_front_end_en = {_.replace('front_end.', ''): model_en[_] for _ in model_en if 'front_end.' in _}
        model.encoder.front_end_en.load_state_dict(state_dict_front_end_en)
        state_dict_back_end_en = {_.replace('back_end.', ''): model_en[_]
                                  for _ in model_en if 'back_end.' in _ and int(_.split('.')[2]) <= 3}
        # for i in state_dict_back_end_en:
        #     print(i, state_dict_back_end_en[i].shape)
        model.encoder.back_end_en.load_state_dict(state_dict_back_end_en)
        # del model.encoder.back_end_en.conformer_blocks[1]

    if my_settings['load_pretrain_cn']:
        model_cn = torch.load(my_settings['cn_pretrain'])
        state_dict_front_end_cn = {_.replace('front_end.', ''): model_cn[_] for _ in model_cn if 'front_end.' in _}
        model.encoder.front_end_cn.load_state_dict(state_dict_front_end_cn)
        state_dict_back_end_cn = {_.replace('back_end.', ''): model_cn[_]
                                 for _ in model_cn if 'back_end.' in _ and int(_.split('.')[2]) <= 3}
        model.encoder.back_end_cn.load_state_dict(state_dict_back_end_cn)
        # del model.encoder.back_end_cn.conformer_blocks[1]


# Load Pretrained
# if lrw_pretrained:
#     lrw_checkpoint = torch.load(lrw_checkpoint, map_location=model.device)
#     for key, value in lrw_checkpoint["model_state_dict"].copy().items():
#         if not "front_end" in key:
#             lrw_checkpoint["model_state_dict"].pop(key)
#     model.encoder.front_end.load_state_dict(
#         {key.replace(".module.", ".").replace("encoder.front_end.", ""): value for key, value in
#          lrw_checkpoint["model_state_dict"].items()})
#
#
# if conformer_pretrained:
#     conformer_checkpoint = torch.load(conformer_checkpoint, map_location=model.device)
#     # print(conformer_checkpoint['model_state_dict'].keys())
#     current_model_proj_shape = {
#         # '0.proj_1.weight': [100, 100],
#         '0.proj_1.weight': [450, 256],
#         '0.proj_1.bias': [450],
#         '0.proj_2.weight': [256, 450],
#         '0.proj_2.bias': [256],
#         '1.proj_1.weight': [450, 360],
#         '1.proj_1.bias': [450],
#         '1.proj_2.weight': [360, 450],
#         '1.proj_2.bias': [360],
#         '2.proj_1.weight': [450, 360],
#         '2.proj_1.bias': [450],
#         '2.proj_2.weight': [360, 450],
#         '2.proj_2.bias': [360]
#     }
#     for key, value in conformer_checkpoint["model_state_dict"].copy().items():
#         if not "back_end" in key:
#             conformer_checkpoint["model_state_dict"].pop(key)
#         if 'proj_1' in key or 'proj_2' in key:
#             name = '.'.join(key.split('.')[-3:])
#
#             conformer_checkpoint["model_state_dict"][key] \
#                 = nn.Parameter(torch.zeros(current_model_proj_shape[name], dtype=torch.float, device=model.device))
#
#     model.encoder.back_end.load_state_dict(
#         {key.replace(".module.", ".").replace("encoder.back_end.", ""): value for key, value in
#          conformer_checkpoint["model_state_dict"].items()})


# Datasets
video_max_length = 400
crop_size = (88, 88)
collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0, "padding": True}, {"axis": 3}, {"axis": 6}],
                            targets_params=({"axis": 2, "padding": True, "padding_value": 0}, {"axis": 5}, {"axis": 7}, {"axis": 8}))
training_video_transform = nn.Sequential(
    torchvision.transforms.RandomCrop(crop_size),
    torchvision.transforms.RandomHorizontalFlip(),
    nnet.Permute(dims=(2, 3, 0, 1)),
    nnet.TimeMaskSecond(T_second=0.4, num_mask_second=1.0, fps=25.0, mean_frame=True),
    nnet.Permute(dims=(2, 3, 0, 1))
)
evaluation_video_transform = torchvision.transforms.CenterCrop(crop_size)
training_dataset = nnet.datasets.MultiDataset(
    batch_size=batch_size,
    collate_fn=collate_fn,
    datasets=[
        nnet.datasets.LRS(
            batch_size=None,
            collate_fn=None,
            version="LRS2",
            mode="train",
            video_max_length=video_max_length,
            video_transform=training_video_transform,
            my_settings=my_settings
        ),
        # nnet.datasets.LRS(
        #     batch_size=None,
        #     collate_fn=None,
        #     version="LRS3",
        #     mode="pretrain+trainval",
        #     video_max_length=video_max_length,
        #     video_transform=training_video_transform
        # )
    ])
evaluation_dataset = [
    nnet.datasets.LRS(
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        version="LRS2",
        mode="test",
        video_transform=evaluation_video_transform,
        shuffle=False,
        my_settings=my_settings
    ),
    # nnet.datasets.LRS(
    #     batch_size=batch_size,
    #     collate_fn=collate_fn,
    #     version="LRS3",
    #     mode="test",
    #     video_transform=evaluation_video_transform
    # )
]
