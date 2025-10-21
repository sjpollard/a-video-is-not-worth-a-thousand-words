import os
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
import torch
import pandas as pd
from itertools import accumulate

# Modified by Sam Pollard
# InternVideo imports
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.modules import CLIP
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datasets.video.video_base_dataset import read_frames_decord, VideoTransform, video_aug
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.datamodules.image.datamodule_base import get_pretrained_tokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask
)
from internvideo.InternVideo1.Downstream.multi_modalities_downstream.CoTrain.modules.InternVideo import tokenize as internvideo_tokenize

# Shapley value imports
import shap
from utils import custom_masker, convert_to_indices, accuracy, threshold_indices, split_string
import egoschema_utils
import hd_epic_utils
import mvbench_utils
import lvbench_utils

torch.manual_seed(0)

device = torch.device("cuda")

def load_egoschema_frames(frames, q_uid, q_dict, video_folder):
    video, _, _ = read_frames_decord(video_path=f'{video_folder}/videos/{q_uid}.mp4', num_frames=frames, mode='test', width=INTERNVIDEO_CONFIG['image_size'], height=INTERNVIDEO_CONFIG['image_size'])
    transform = VideoTransform(mode='test', crop_size=INTERNVIDEO_CONFIG['image_size'], backend=INTERNVIDEO_CONFIG['backend'])
    video = video_aug(video, transform)
    video = video[0].cuda()

    return [video], -1


def load_hd_epic_frames(frames, q_uid, q_dict, video_folder):
    video_frames, input_keys, stride, frame_idx_dict = hd_epic_utils.load_videos(q_dict=q_dict, frames=frames, input_fps=1, video_path=video_folder, height=INTERNVIDEO_CONFIG['image_size'], width=INTERNVIDEO_CONFIG['image_size'])
    video = torch.from_numpy(np.concatenate(video_frames)).permute((0, 3, 1, 2))
    transform = VideoTransform(mode='test', crop_size=INTERNVIDEO_CONFIG['image_size'], backend=INTERNVIDEO_CONFIG['backend'])
    video = video_aug(video, transform)
    video = video[0].cuda()

    return [video], stride


def load_mvbench_frames(frames, q_uid, q_dict, video_folder):
    video, timestamps = mvbench_utils.load_video(q_uid, q_dict, frames, video_path=video_folder, height=INTERNVIDEO_CONFIG['image_size'], width=INTERNVIDEO_CONFIG['image_size'])
    video = torch.from_numpy(video).permute((0, 3, 1, 2))
    transform = VideoTransform(mode='test', crop_size=INTERNVIDEO_CONFIG['image_size'], backend=INTERNVIDEO_CONFIG['backend'])
    video = video_aug(video, transform)
    video = video[0].cuda()

    return [video], -1


def load_lvbench_frames(frames, q_uid, q_dict, video_folder):
    video, timestamps = lvbench_utils.load_video(q_dict, frames, video_path=video_folder, height=INTERNVIDEO_CONFIG['image_size'], width=INTERNVIDEO_CONFIG['image_size'])
    video = torch.from_numpy(video).permute((0, 3, 1, 2))
    transform = VideoTransform(mode='test', crop_size=INTERNVIDEO_CONFIG['image_size'], backend=INTERNVIDEO_CONFIG['backend'])
    video = video_aug(video, transform)
    video = video[0].cuda()

    return [video], -1


EGOSCHEMA_CONFIG = {'qa_folder': '<EGOSCHEMA-PATH>',
                    'video_folder': '<EGOSCHEMA-PATH>',
                    'dataset_fn': egoschema_utils.get_egoschema,
                    'parse_vqa': egoschema_utils.parse_vqa,
                    'load_frames': load_egoschema_frames}


HD_EPIC_CONFIG = {'qa_folder': '<HD-EPIC-ANNOTATIONS-PATH>',
                  'video_folder': '<HD-EPIC-VIDEO-PATH>',
                  'dataset_fn': hd_epic_utils.get_hd_epic,
                  'parse_vqa': hd_epic_utils.parse_vqa,
                  'load_frames': load_hd_epic_frames}


MVBENCH_CONFIG = {'qa_folder': '<MVBENCH_PATH>',
                  'video_folder': '<MVBENCH_PATH>/video',
                  'dataset_fn': mvbench_utils.get_mvbench,
                  'parse_vqa': mvbench_utils.parse_vqa,
                  'load_frames': load_mvbench_frames}


LVBENCH_CONFIG = {'qa_folder': '<LVBENCH_PATH>',
                  'video_folder': '<LVBENCH_PATH>/videos/00000',
                  'dataset_fn': lvbench_utils.get_lvbench,
                  'parse_vqa': lvbench_utils.parse_vqa,
                  'load_frames': load_lvbench_frames}


INTERNVIDEO_VIT_WEIGHTS_PATH = '<INTERNVIDEO-VIT-PATH>'
INTERNVIDEO_MSRVTT_WEIGHTS_PATH = '<INTERNVIDEO-PATH>'
INTERNVIDEO_CONFIG = {'exp_name': 'clip_kc_nc_finetune_msrvttchoice', 
          'seed': 0, 
          'video_datasets': ['msrvtt_choice'], 
          'image_datasets': [], 
          'val_datasets': [], 
          'loss_names': {'vtm': 0, 
                         'mlm': 0, 
                         'mpp': 0, 
                         'vtc': 0, 
                         'vcop': 0, 
                         'dino': 0, 
                         'vqa': 0, 
                         'openend_vqa': 0, 
                         'mc_vqa': 0, 
                         'nlvr2': 0, 
                         'irtr': 0, 
                         'multiple_choice': 1, 
                         'vcr_q2a': 0, 
                         'zs_classify': 0, 
                         'contrastive': 0, 
                         'cap': 0, 
                         'mim': 0}, 
          'val_loss_names': {'vtm': 0, 
                             'mlm': 0, 
                             'mpp': 0, 
                             'vtc': 0, 
                             'vcop': 0, 
                             'dino': 0, 
                             'vqa': 0, 
                             'openend_vqa': 0, 
                             'mc_vqa': 1, 
                             'nlvr2': 0, 
                             'irtr': 0, 
                             'multiple_choice': 0, 
                             'vcr_q2a': 0, 
                             'zs_classify': 0, 
                             'contrastive': 0, 
                             'cap': 0, 
                             'mim': 0}, 
          'batch_size': 1, 
          'linear_evaluation': False, 
          'draw_false_image': 1, 
          'train_transform_keys': ['pixelbert'], 
          'val_transform_keys': ['pixelbert'], 
          'image_size': 224, 
          'patch_size': 16, 
          'max_image_len': -1, 
          'draw_false_video': 1, 
          'video_only': False, 
          'num_frames': None, 
          'vqav2_label_size': 3129, 
          'msrvttqa_label_size': 1501, 
          'max_text_len': 77, 
          'tokenizer': 'bert-base-uncased', 
          'vocab_size': 30522, 
          'whole_word_masking': False, 
          'mlm_prob': 0.15, 
          'draw_false_text': 5, 
          'draw_options_text': 0, 
          'vit': 'vit_base_patch16_224', 
          'hidden_size': 768, 
          'num_heads': 12, 
          'num_layers': 12, 
          'mlp_ratio': 4, 
          'drop_rate': 0.1, 
          'shared_embedding_dim': 512, 
          'save_checkpoints_interval': 1, 
          'optim_type': 'adamw', 
          'learning_rate': 0.0001, 
          'weight_decay': 0.01, 
          'decay_power': 1, 
          'max_epoch': 10, 
          'max_steps': 25000, 
          'warmup_steps': 0.1, 
          'end_lr': 0, 
          'lr_mult': 1, 
          'backend': 'a100', 
          'get_recall_metric': False, 
          'get_ind_recall_metric': False, 
          'retrieval_views': 3, 
          'resume_from': None, 
          'fast_dev_run': False, 
          'val_check_interval': 0.5, 
          'test_only': True, 
          'data_root': '',
          'log_dir': '/result/', 
          'per_gpu_batchsize': 1, 
          'num_gpus': 1, 
          'num_nodes': '', 
          'load_path': INTERNVIDEO_MSRVTT_WEIGHTS_PATH, 
          'num_workers': 1, 
          'precision': 16, 
          'model_dir': '//models/', 
          'clip': INTERNVIDEO_VIT_WEIGHTS_PATH, 
          'clip_type': 'kc_new', 
          'clip_freeze': False, 
          'clip_freeze_text': False, 
          'clip_dpr': 0.0, 
          'prompt_type': 'all', 
          'clip_lr_mult': 1, 
          'clip_no_pretrain': False, 
          'clip_grad_unfreeze_int': 0, 
          'clip_evl_dropout': 0.5, 
          'mim_prob': 0.9, 
          'clip_mlm_decoder_n_layers': 4, 
          'clip_mim_decoder_n_layers': 4, 
          'clip_mim_decoder_width': 512, 
          'clip_cap_decoder_n_layers': 4, 
          'clip_init_zero': True, 
          'clip_qa_type': 'vtc', 
          'clip_mc_type': 'vtc', 
          'clip_wiseft_coef': 0.5,
          'clip_mmt': False, 
          'clip_alt_data': False, 
          'image_data_mult': 1,
          'clip_cls_dropout': 0.5,
          'save_last': True, 
          'save_top_k': 1, 
          'clip_use_checkpoint': False,
          'clip_checkpoint_num': [0, 0, 0], 
          'clip_momentum_ckpt': 1, 
          'clip_momentum_interval': 1}


def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Running experiments"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        dest="dataset",
        choices=['egoschema', 'hd-epic', 'mvbench', 'lvbench'],
        default='egoschema',
        type=str
    )
    parser.add_argument(
        "--frames",
        "-f",
        dest="frames",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--question",
        "-q",
        dest="q_uid",
        default="00b9a0de-c59e-49cb-a127-6081e2fb8c8e",
        type=str
    )
    parser.add_argument(
        "--subset-path",
        "-s",
        dest="subset_path",
        default=None,
        type=str
    )
    parser.add_argument(
        "--answer-replacement",
        dest="answer_replacement",
        default=None,
        type=str
    )
    parser.add_argument(
        "--iterations",
        "-n",
        dest="iterations",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--mode",
        "-m",
        dest="mode",
        choices=['text_only', 'video_only', 'joint'],
        default='joint',
        type=str
    )
    parser.add_argument(
        "--chunks",
        dest="chunks",
        default=1,
        type=int
    )
    parser.add_argument(
        "--chunk-index",
        dest="chunk_index",
        default=0,
        type=int
    )
    parser.add_argument(
        "--force-redo",
        dest="force_redo",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--command",
        "-c",
        dest="command",
        choices=['calculate', 'compare', 'evaluate', 'benchmark'],
        default='calculate',
        type=str
    )
    parser.add_argument(
        "--index",
        "-i",
        dest="index",
        default=0,
        type=int
    )
    parser.add_argument(
        "--masking-direction",
        dest="masking_direction",
        choices=['nothing', 'everything', 'negative', 'positive'],
        default='nothing',
        type=str
    )
    parser.add_argument(
        "--masking-logit",
        dest="masking_logit",
        choices=['ground_truth', 'false', 'all'],
        default='ground_truth',
        type=str
    )
    parser.add_argument(
        "--masking-mode",
        dest="masking_mode",
        choices=['joint', 'video', 'question', 'answer'],
        default='joint',
        type=str
    )
    parser.add_argument(
        "--threshold",
        "-t",
        dest="threshold",
        default=0.0,
        type=float
    )
    parser.add_argument(
        "--force-logit",
        dest="force_logit",
        choices=['first', 'last'],
        default=None,
        type=str
    )
    parser.add_argument(
        "--answer-masking",
        dest="answer_masking",
        choices=['gt', 'all'],
        default='all',
        type=str
    )
    return parser.parse_args()


def reload_model(frames):
    INTERNVIDEO_CONFIG['num_frames'] = frames

    model = CLIP(INTERNVIDEO_CONFIG)
    model.current_tasks = ['multiple_choice']

    model.cuda()
    model.eval()
    return model

# Copied from internvideo/InternVideo1/Downstream/multi_modalities_downstream/CoTrain/datasets/video/video_base_dataset.py
def collate(batch, mlm_collator):
    batch_size = len(batch)
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    video_keys = [k for k in list(dict_batch.keys()) if "video" in k]
    video_sizes = list()

    # global & local video
    for video_key in video_keys:
        video_sizes += [ii.shape for i in dict_batch[video_key] if i is not None for ii in i]
        # print(global_video_sizes, local_video_sizes)

    for size in video_sizes:
        # print(size)
        assert (
            len(size) == 4
        ), f"Collate error, an video should be in shape of (T, N, H, W), instead of given {size}"

    if len(video_keys) != 0:
        global_max_height = max([i[2] for i in video_sizes])
        global_max_width = max([i[3] for i in video_sizes])
        global_min_height = min([i[2] for i in video_sizes])
        global_min_width = min([i[3] for i in video_sizes])
    for video_key in video_keys:
        video = dict_batch[video_key]
        view_size = len(video[0])
        if (view_size == 1 and 
            global_max_height == global_min_height and 
            global_max_width == global_min_width):
            dict_batch[video_key] = [torch.stack([x[0] for x in video])]
            continue
    
        assert False, (view_size, global_max_height, global_min_height, 
            global_max_width, global_min_width)
        new_videos = [
            torch.zeros(batch_size, video_sizes[0][0], 3, global_max_height, global_max_width)
            for _ in range(view_size)
        ]
        for bi in range(batch_size):
            orig_batch = video[bi]
            for vi in range(view_size):
                if orig_batch is None:
                    # new_videos[vi][bi] = None
                    # modify by alex
                    continue
                else:
                    orig = video[bi][vi]
                    # print(orig.size())
                    new_videos[vi][bi, :, :, : orig.shape[-2], : orig.shape[-1]] = orig

        dict_batch[video_key] = new_videos

    txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
    # print(txt_keys)
    if len(txt_keys) != 0:
        texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        draw_text_len = len(encodings)
        flatten_encodings = [e for encoding in encodings for e in encoding]
        flatten_mlms = mlm_collator(flatten_encodings)

        for i, txt_key in enumerate(txt_keys):
            texts, encodings = (
                [d[0] for d in dict_batch[txt_key]],
                [d[1] for d in dict_batch[txt_key]],
            )

            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
            )

            input_ids = torch.zeros_like(mlm_ids)
            attention_mask = torch.zeros_like(mlm_ids)
            for _i, encoding in enumerate(encodings):
                _input_ids, _attention_mask = (
                    torch.tensor(encoding["input_ids"]),
                    torch.tensor(encoding["attention_mask"]),
                )
                input_ids[_i, : len(_input_ids)] = _input_ids
                attention_mask[_i, : len(_attention_mask)] = _attention_mask

            dict_batch[txt_key] = texts
            dict_batch[f"{txt_key}_ids"] = input_ids
            dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
            dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
            dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
            dict_batch[f"{txt_key}_masks"] = attention_mask

        clip_text_ids, clip_special_tokens_mask = internvideo_tokenize(
            dict_batch["text"], truncate=True, return_special_tokens_mask=True)
        dict_batch["clip_text_ids"] = clip_text_ids
        dict_batch["clip_special_tokens_mask"] = clip_special_tokens_mask
    
    return dict_batch


def apply_multi_modal_mask(videos, question, options, tokenize, indices, lengths, mode='joint', parse_tags=None, force_logit=None):
    inputs = []

    for item in indices:

        input = {
            'vid_index': 0,
            'cap_index': 0,
            'raw_index': 0,
            'answer': 0,
            'q_uid': '',
        }

        multi_modal_indices = {}
        intervals = list(zip([0] + list(accumulate(list(lengths.values())[:-1])), accumulate(list(lengths.values()))))
        for i, key in enumerate(lengths.keys()):
            start, end = intervals[i]
            multi_modal_indices.update({key: item[start:end]})            
        if mode == 'text_only':
            input['video'] = [videos[0]]
            masked_question = ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(multi_modal_indices['question'], split_string(question)))))
            masked_options = list(map(lambda y: ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(y[1], split_string(y[0]))))), zip(options, list(map(lambda x: multi_modal_indices[x], [f'option {i}' for i in range(1, len(options) + 1)])))))
            text = list(map(lambda y: (y, tokenize(parse_tags(y)) if parse_tags is not None else tokenize(y)), list(map(lambda x: f"Question: {masked_question} Is it '{x}'?.", masked_options))))
        elif mode == 'video_only':
            for i in range(len(videos)):
                input['video'] = [videos[i] * multi_modal_indices[f'video {i + 1}'].reshape(-1, 1, 1, 1).cuda()]
            text = list(map(lambda y: (y, tokenize(parse_tags(y)) if parse_tags is not None else tokenize(y)), list(map(lambda x: f"Question: {question} Is it '{x}'?.", options))))
        elif mode == 'joint':
            for i in range(len(videos)):
                input['video'] = [videos[i] * multi_modal_indices[f'video {i + 1}'].reshape(-1, 1, 1, 1).cuda()]
            masked_question = ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(multi_modal_indices['question'], split_string(question)))))
            masked_options = list(map(lambda y: ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(y[1], split_string(y[0]))))), zip(options, list(map(lambda x: multi_modal_indices[x], [f'option {i}' for i in range(1, len(options) + 1)])))))
            text = list(map(lambda y: (y, tokenize(parse_tags(y)) if parse_tags is not None else tokenize(y)), list(map(lambda x: f"Question: {masked_question} Is it '{x}'?.", masked_options))))
        
        if force_logit != None:
            if force_logit['logit'] == 'first':
                forced_index = 0
            elif force_logit['logit'] == 'last':
                forced_index = len(options) - 1
            temp = text.pop(force_logit['gt'])
            text.insert(forced_index, temp)

        option_keys = ['text', 'false_text_0', 'false_text_1', 'false_text_2', 'false_text_3']

        for i in range(5):
            if i < len(options):
                input[option_keys[i]] = text[i]
            else:
                input[option_keys[i]] = text[-1]

        inputs.append(input)
    
    return inputs


def internvideo_forward(model, tokenizer, mlm_collator, videos, q_uid, question, options, indices, lengths, mode, parse_tags, force_logit=None):
    indices = torch.from_numpy(indices)

    tokenize = lambda x: tokenizer(
        x,
        padding="longest",
        return_special_tokens_mask=True,
    )

    inputs = apply_multi_modal_mask(videos, question, options, tokenize, indices, lengths, mode, parse_tags, force_logit)

    inputs = collate(inputs, mlm_collator)

    for k in inputs.keys():
        if type(inputs[k]) == torch.Tensor:
            inputs[k] = inputs[k].cuda()

    output = model(inputs)

    logits = output["score"].cpu()

    return logits


def internvideo_shap(args, model, tokenizer, mlm_collator, q_uid, question, options, q_dict, qa_answers):
    with torch.no_grad():
        videos, stride = args.config['load_frames'](args.frames, q_uid, q_dict, args.config['video_folder'])

        if videos[0].size(0) != args.frames:
            model = reload_model(videos[0].size(0))

        if args.dataset == 'hd-epic':
            parse_tags = lambda x: hd_epic_utils.parse_tags(x, stride, q_dict)
        else: 
            parse_tags = None

        x, lengths, names = convert_to_indices([x.size(0) for x in videos], question, options, mode=args.mode)
        x = x[None][None]

        fc = lambda x: internvideo_forward(model, tokenizer, mlm_collator, videos, q_uid, question, options, x, lengths, args.mode, parse_tags)
        
        explainer = shap.Explainer(model=fc,  masker=custom_masker, algorithm='permutation', feature_names=names, seed=0)

        explanation = explainer(x, batch_size=args.batch_size, max_evals=args.iterations)

        del videos
        torch.cuda.empty_cache()
        
        shap_values = np.squeeze(explanation.values).T

        columns = [chr(ord('A') + i) for i in range(len(options))]
        columns[qa_answers[q_uid]] += ': Ground Truth'

        shap_dict = dict(zip(columns, shap_values))

        return pd.DataFrame({'element': names, **shap_dict})


def prepare_internvideo(args):
    INTERNVIDEO_CONFIG['num_frames'] = args.frames

    model = CLIP(INTERNVIDEO_CONFIG)
    model.current_tasks = ['multiple_choice']

    model.cuda()
    model.eval()

    tokenizer = get_pretrained_tokenizer(INTERNVIDEO_CONFIG['tokenizer'])

    collator = (
        DataCollatorForWholeWordMask
        if INTERNVIDEO_CONFIG["whole_word_masking"]
        else DataCollatorForLanguageModeling
    )

    mlm_collator = collator(
        tokenizer=tokenizer, mlm=True, mlm_probability=INTERNVIDEO_CONFIG["mlm_prob"]
    )

    return model, tokenizer, mlm_collator


def calculate_shap_values(args):
    add_to_path = ''
    if args.answer_replacement is not None:
        add_to_path += f'_{args.answer_replacement}'

    results_path = f'shap_results/internvideo/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/internvideo{add_to_path}'
    
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
    else:
        q_uids = [args.q_uid]

    model, tokenizer, mlm_collator = prepare_internvideo(args)

    q_uids = np.array_split(q_uids, args.chunks)[args.chunk_index]

    for q_uid in tqdm(q_uids):

        if args.force_redo or not os.path.exists(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.csv'):

            q, options, q_dict = args.config['parse_vqa'](qa_data, q_uid)

            shap_out = internvideo_shap(args, model, tokenizer, mlm_collator, q_uid, q, options, q_dict, qa_answers)

            print(f'Question: {q}')
            for idx, option in enumerate(options):
                print(f'{chr(ord("A") + idx)}: {option}')
            print(f'Correct answer: {options[qa_answers[q_uid]]}')
            print(shap_out)
            shap_out.to_csv(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.csv', index=False)


def internvideo_logit_difference(args, model, tokenizer, mlm_collator, q_uid, question, options, q_dict, qa_answers):
    with torch.no_grad():
        videos, stride = args.config['load_frames'](args.frames, q_uid, q_dict, args.config['video_folder'])

        if videos[0].size(0) != args.frames:
            model = reload_model(videos[0].size(0))

        if args.dataset == 'hd-epic':
            parse_tags = lambda x: hd_epic_utils.parse_tags(x, stride, q_dict)
        else: 
            parse_tags = None

        x, lengths, names = convert_to_indices([x.size(0) for x in videos], question, options, mode=args.mode)
        x = x[None][None]

        fc = lambda x: internvideo_forward(model, tokenizer, mlm_collator, videos, q_uid, question, options, x, lengths, args.mode, parse_tags)

        logit = fc(x.numpy()[0])[0][qa_answers[q_uid]]

        x[0, 0, args.index] = 0

        logit_hat = fc(x.numpy()[0])[0][qa_answers[q_uid]]

        print(f'Original input: {logit}')
        print(f'Masked input: {logit_hat}')
        print(f'Difference in true logit when masking \"{names[args.index]}\": {logit_hat - logit}')

        del videos
        torch.cuda.empty_cache()

        return


def compare_logit_values(args):
    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    q, options, q_dict = args.config['parse_vqa'](qa_data, args.q_uid)
        
    model, tokenizer, mlm_collator = prepare_internvideo(args)
    internvideo_logit_difference(args, model, tokenizer, mlm_collator, args.q_uid, q, options, q_dict, qa_answers)

    print(f'Question: {q}')
    for idx, option in enumerate(options):
        print(f'{chr(ord("A") + idx)}: {option}')
    print(f'Correct answer: {options[qa_answers[args.q_uid]]}')    


def internvideo_evaluate(args, model, tokenizer, mlm_collator, q_uid, question, options, q_dict, masking_direction, masking_logit, masking_mode):
    with torch.no_grad():
        videos, stride = args.config['load_frames'](args.frames, q_uid, q_dict, args.config['video_folder'])
        
        if videos[0].size(0) != args.frames:
            model = reload_model(videos[0].size(0))

        if args.dataset == 'hd-epic':
            parse_tags = lambda x: hd_epic_utils.parse_tags(x, stride, q_dict)
        else: 
            parse_tags = None

        x, lengths, names = convert_to_indices([x.size(0) for x in videos], question, options, mode=args.mode)
        x = x[None][None]

        if masking_direction != 'nothing':
            add_to_path = ''
            if args.answer_replacement is not None:
                add_to_path += f'_{args.answer_replacement}'

            results_path = f'shap_results/internvideo/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/internvideo{add_to_path}'            
            x = threshold_indices(x, lengths, results_path, q_uid, args.iterations, args.mode, masking_direction, masking_logit, masking_mode, args.threshold, args.answer_masking)

        fc = lambda x: internvideo_forward(model, tokenizer, mlm_collator, videos, q_uid, question, options, x, lengths, args.mode, parse_tags, args.force_logit)

        logits = fc(x.numpy()[0])[0][:len(options)]
        
        pred = logits.argmax().item()

        del videos
        torch.cuda.empty_cache()

        return pred


def evaluate_model(args):
    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
    else:
        q_uids = [args.q_uid]
    
    model, tokenizer, mlm_collator = prepare_internvideo(args)

    preds = []
    labels = []

    if args.force_logit != None:
        args.force_logit = {'logit': args.force_logit}

    for q_uid in tqdm(q_uids):
        q, options, q_dict = args.config['parse_vqa'](qa_data, q_uid)

        if args.force_logit != None:
            if args.force_logit['logit'] == 'first':
                label = 0
            elif args.force_logit['logit'] == 'last':
                label = len(options) - 1
            args.force_logit.update({'gt': qa_answers[q_uid]})
        else:
            label = qa_answers[q_uid]

        pred = internvideo_evaluate(args, model, tokenizer, mlm_collator, q_uid, q, options, q_dict, args.masking_direction, args.masking_logit, args.masking_mode)

        preds.append(pred)
        labels.append(label)

        if args.force_logit != None:
            if args.force_logit['logit'] == 'first':
                forced_index = 0
            elif args.force_logit['logit'] == 'last':
                forced_index = len(options) - 1
            temp = options.pop(args.force_logit['gt'])
            options.insert(forced_index, temp)

        print(f'Question: {q}')
        for idx, option in enumerate(options):
            print(f'{chr(ord("A") + idx)}: {option}')
        print(f'Correct answer: {options[label]}')
        print(f'Predicted answer: {options[pred]}')

    print(f'Accuracy: {accuracy(preds, labels) * 100.0}%')


def benchmark_model(args):
    add_to_path = ''
    if args.answer_replacement is not None:
        add_to_path += f'_{args.answer_replacement}'

    results_path = f'shap_results/internvideo/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/internvideo{add_to_path}'    
    
    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
        name = args.subset_path.split(".")[0].split("/")[-1]
    else:
        q_uids = [args.q_uid]
        name = q_uids[0]
    
    model, tokenizer, mlm_collator = prepare_internvideo(args)

    masking_directions = ['nothing', 'everything', 'positive', 'negative']
    masking_logits = ['ground_truth', 'false', 'all']
    masking_modes = ['joint', 'video', 'question', 'answer']

    results = {}

    for direction in masking_directions:
        if direction == 'nothing':
            results[direction] = {'preds': [], 'accuracy': None}
        elif direction == 'everything':
            results[direction] = {
                mode: {'preds': [], 'accuracy': None} for mode in masking_modes
            }
        else:
            results[direction] = {
                masking_logit: {
                    mode: {'preds': [], 'accuracy': None} for mode in masking_modes
                } for masking_logit in masking_logits
            }

    labels = []

    if args.force_logit != None:
        args.force_logit = {'logit': args.force_logit}

    for q_uid in tqdm(q_uids):
    
        q, options, q_dict = args.config['parse_vqa'](qa_data, q_uid)

        if args.force_logit != None:
            if args.force_logit['logit'] == 'first':
                label = 0
            elif args.force_logit['logit'] == 'last':
                label = len(options) - 1
            args.force_logit.update({'gt': qa_answers[q_uid]})
        else:
            label = qa_answers[q_uid]

        labels.append(label)

        pred = internvideo_evaluate(args, model, tokenizer, mlm_collator, q_uid, q, options, q_dict, masking_directions[0], None, None)
        results[masking_directions[0]]['preds'].append(pred)

        for masking_direction in masking_directions[1:]:
            if masking_direction in ['positive', 'negative']:
                for masking_logit in masking_logits:
                    for masking_mode in masking_modes:
                        pred = internvideo_evaluate(args, model, tokenizer, mlm_collator, q_uid, q, options, q_dict, masking_direction, masking_logit, masking_mode)
                        results[masking_direction][masking_logit][masking_mode]['preds'].append(pred)
            else:
                for masking_mode in masking_modes:
                    pred = internvideo_evaluate(args, model, tokenizer, mlm_collator, q_uid, q, options, q_dict, masking_direction, None, masking_mode)
                    results[masking_direction][masking_mode]['preds'].append(pred)

    results[masking_directions[0]]['accuracy'] = accuracy(results[masking_directions[0]]['preds'], labels)

    for masking_direction in masking_directions[1:]:
        if masking_direction in ['positive', 'negative']:
            for masking_logit in masking_logits:
                for masking_mode in masking_modes:
                    results[masking_direction][masking_logit][masking_mode]['accuracy'] = accuracy(results[masking_direction][masking_logit][masking_mode]['preds'], labels)
        else:
            for masking_mode in masking_modes:
                results[masking_direction][masking_mode]['accuracy'] = accuracy(results[masking_direction][masking_mode]['preds'], labels)

    add_to_path = ''
    if args.answer_masking == 'gt':
        add_to_path += '_gt'
    elif args.force_logit != None:
        add_to_path += f'_{args.force_logit["logit"]}'

    out_path = f'{results_path}/{name}_{args.iterations}_{args.mode}{add_to_path}_benchmark.json'

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def main(args):    
    if args.dataset == 'egoschema':
        config = EGOSCHEMA_CONFIG
    elif args.dataset == 'hd-epic':
        config = HD_EPIC_CONFIG
    elif args.dataset == 'mvbench':
        config = MVBENCH_CONFIG
    elif args.dataset == 'lvbench':
        config = LVBENCH_CONFIG

    args.config = config
    
    if args.command == 'calculate':
        calculate_shap_values(args)
    elif args.command == 'compare':
        compare_logit_values(args)
    elif args.command == 'evaluate' and args.mode == 'joint':
        evaluate_model(args)
    elif args.command == 'benchmark' and args.mode == 'joint':
        benchmark_model(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)