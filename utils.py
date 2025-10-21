import torch
import numpy as np
import pandas as pd
import re
import json
import copy
from collections import Counter


def get_influential_frame_indices(results_path, q_uid, frames=16, iterations=5000, mode='joint', ordering='chronological'):
    shap_out = pd.read_csv(f'{results_path}/{q_uid}_{iterations}_{mode}.csv')

    index = shap_out['element']
    numeric_columns = [x for x in shap_out.columns if x != 'element']

    ground_truth_in_list = ['Ground Truth' in x for x in numeric_columns]
    ground_truth_index = ground_truth_in_list.index(True)
    ground_truth_shap = shap_out[numeric_columns[ground_truth_index]]
    
    frame_shap = ground_truth_shap[index.str.contains('frame_\d+', regex=True)]

    if ordering == 'chronological':
        sorted_indices = frame_shap.abs().argsort()
        frame_indices = sorted_indices[sorted_indices < frames].index.to_list()
    elif ordering == 'ranking':
        frame_indices = frame_shap.abs().sort_values(ascending=False).index.to_list()[:frames]

    return frame_indices


def get_gemini_frame_indices(ranking_path, q_uid, frames):
    gemini_indices_f = open(f'{ranking_path}/{q_uid}.json')
    gemini_indices = json.load(gemini_indices_f)

    string_indices = gemini_indices[q_uid]

    if string_indices == None:
        return None
    elif 'frame_' in string_indices:
        parsed_indices = list(map(lambda x: int(x.split('frame_')[-1]), string_indices.split(',')))
    else:
        parsed_indices = list(map(lambda x: int(x), string_indices.split(',')))

    parsed_indices_copy = copy.deepcopy(parsed_indices)

    expected_indices = list(range(frames))

    missing_indices = Counter(expected_indices) - Counter(parsed_indices)
    duplicate_indices = Counter(parsed_indices) - Counter(expected_indices)

    if sum(duplicate_indices.values()) > 0:
        for x in list(set(duplicate_indices.elements())):
            indices_in_parsed = [i for i, y in enumerate(parsed_indices_copy) if y == x][1:]
            indices_in_parsed.reverse()
            for z in indices_in_parsed:
                parsed_indices_copy.pop(z)

    if sum(missing_indices.values()) > 0:
        parsed_indices_copy += sorted(missing_indices.elements())

    parsed_indices = parsed_indices_copy

    return parsed_indices


def split_string(string):
    return re.findall(r"[\w']+|<[^<>]*>|[.,!?;:\"\/()_-]", string)


def custom_masker(mask, x):
    return torch.from_numpy(mask)[None] * x


def convert_to_indices(num_frames_per_video, question, choices, mode='joint'):
    if mode == 'text_only':
        split_question = split_string(question)
        split_choices = list(map(lambda x: split_string(x), choices))
        indices = torch.ones(len(split_question + sum(split_choices, [])), dtype=torch.int)
        lengths = {}
        lengths.update({'question': len(split_question)})
        list(map(lambda x: lengths.update({f'option {x[0]}': len(x[1])}), enumerate(split_choices, start=1)))
        names = split_question + sum(split_choices, [])
    elif mode == 'video_only':
        total_frames = 0
        lengths = {}
        for i, num_frames in enumerate(num_frames_per_video, start=1):
            total_frames += num_frames
            lengths.update({f'video {i}': num_frames})
        indices = torch.ones(total_frames, dtype=torch.int)
        names = [f'frame_{x}' for x in range(total_frames)]
    elif mode == 'joint':
        split_question = split_string(question)
        split_choices = list(map(lambda x: split_string(x), choices))
        total_frames = 0
        lengths = {}
        for i, num_frames in enumerate(num_frames_per_video, start=1):
            total_frames += num_frames
            lengths.update({f'video {i}': num_frames})
        indices = torch.ones(total_frames + len(split_question + sum(split_choices, [])), dtype=torch.int)
        lengths.update({'question': len(split_question)})
        list(map(lambda x: lengths.update({f'option {x[0]}': len(x[1])}), enumerate(split_choices, start=1)))
        names = [f'frame_{x}' for x in range(total_frames)] + split_question + sum(split_choices, [])

    return indices, lengths, names


def accuracy(preds, labels):
    return torch.sum(torch.tensor(preds) == torch.tensor(labels)).item()/len(preds)


def threshold_indices(x, lengths, results_path, q_uid, iterations, mode, masking_direction, masking_logit, masking_mode, threshold, answer_masking):
    shap_out = pd.read_csv(f'{results_path}/{q_uid}_{iterations}_{mode}.csv')

    numeric_columns = [x for x in shap_out.columns if x != 'element']

    shap_out_norm = shap_out.copy()
    shap_out_norm[numeric_columns] = shap_out_norm[numeric_columns] / shap_out_norm[numeric_columns].abs().max()

    if masking_logit == 'ground_truth':
        ground_truth_in_list = ['Ground Truth' in x for x in numeric_columns]
        ground_truth_index = ground_truth_in_list.index(True)
        metric = shap_out_norm[numeric_columns[ground_truth_index]]
    elif masking_logit == 'false':
        false_in_list = ['Ground Truth' not in x for x in numeric_columns]
        false_indices = [i for i, x in enumerate(false_in_list) if x == True]
        metric = shap_out_norm[[numeric_columns[x] for x in false_indices]].mean(axis='columns')
    elif masking_logit == 'all':
        metric = shap_out_norm[numeric_columns].mean(axis='columns')

    total_length = sum(lengths.values())
    video_length = sum(lengths[y] for y in [x for x in list(lengths.keys()) if 'video' in x])

    if masking_direction == 'everything':
        masked_indices = np.zeros(total_length, dtype=bool)
    if masking_direction == 'negative':
        masked_indices = (metric > threshold).to_numpy()
    elif masking_direction == 'positive':
        masked_indices = (metric < threshold).to_numpy()

    if masking_mode == 'joint':
        if answer_masking == 'gt':
            ground_truth_in_list = ['Ground Truth' in x for x in numeric_columns]
            ground_truth_index = ground_truth_in_list.index(True)
            options_mask = np.concatenate(list(map(lambda x: np.ones(lengths[f'option {x + 1}'], dtype=bool) if x + 1 == ground_truth_index else np.zeros(lengths[f'option {x + 1}'], dtype=bool), range(len(numeric_columns)))))
        elif answer_masking == 'all':
            options_mask =  np.ones(total_length - video_length - lengths['question'], dtype=bool)
        mode_mask = np.concatenate((np.zeros(video_length + lengths['question'], dtype=bool), options_mask))
        np.ones(sum(lengths.values()), dtype=bool)
    elif masking_mode == 'video':
        mode_mask = np.concatenate((np.ones(video_length, dtype=bool), np.zeros(total_length - video_length, dtype=bool)))
    elif masking_mode == 'question':
        mode_mask = np.concatenate((np.zeros(video_length, dtype=bool), np.ones(lengths['question'], dtype=bool), np.zeros(total_length - video_length - lengths['question'], dtype=bool)))
    elif masking_mode == 'answer':
        if answer_masking == 'gt':
            ground_truth_in_list = ['Ground Truth' in x for x in numeric_columns]
            ground_truth_index = ground_truth_in_list.index(True)
            options_mask = np.concatenate(list(map(lambda x: np.ones(lengths[f'option {x + 1}'], dtype=bool) if x + 1 == ground_truth_index else np.zeros(lengths[f'option {x + 1}'], dtype=bool), range(len(numeric_columns)))))
        elif answer_masking == 'all':
            options_mask =  np.ones(total_length - video_length - lengths['question'], dtype=bool)
        mode_mask = np.concatenate((np.zeros(video_length + lengths['question'], dtype=bool), options_mask))

    masked_indices = masked_indices | np.logical_not(mode_mask)

    return x * masked_indices