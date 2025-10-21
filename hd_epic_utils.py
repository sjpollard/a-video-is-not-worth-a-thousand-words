import json
import os
import numpy as np
import pandas as pd
import glob
import datetime
import subprocess
import re
import time
import decord
import copy
from tqdm import tqdm

decord.bridge.set_bridge('native')

QUESTION_TYPES = []

def get_hd_epic(hd_epic_path, answer_replacement=None):
    if answer_replacement is not None:
        qa_data_f = open(f'{hd_epic_path}/questions_{answer_replacement}.json')
        qa_data = json.load(qa_data_f)

        qa_answers_f = open(f'{hd_epic_path}/subset_answers_{answer_replacement}.json')
        qa_answers = json.load(qa_answers_f)
    else:
        qa_data = {}

        q_files = glob.glob(f'{hd_epic_path}/*.json')
        q_files = [x for x in q_files if ('question' not in x and 'subset_answers' not in x)]

        for q_file in q_files:
            with open(q_file) as f:
                tmp_questions = json.load(f)
            qa_data.update(tmp_questions)

        qa_answers = {}
        for q_uid in qa_data.keys():
            qa_answers.update({q_uid: qa_data[q_uid]['correct_idx']})

    return qa_data, qa_answers


def parse_vqa(qa_data, q_uid):
    q_dict = qa_data[q_uid]

    q = q_dict['question']
    if q[-1] != "?":
        q = q + "?"
    choices = q_dict['choices']
    choices = list(map(lambda x: ', '.join(x) if isinstance(x, list) else x, choices))
    return q, choices, q_dict


def get_random_hd_epic_subset(hd_epic_path, num_questions, subset_name='hd-epic', seed=0):
    if not os.path.exists(f'subsets'):
        os.mkdir(f'subsets')

    rng = np.random.default_rng(seed=seed)
    qa_data, qa_answers = get_hd_epic(hd_epic_path)

    q_uids = list(qa_data.keys())

    question_types = list(dict.fromkeys(list(map(lambda x: '_'.join(x.split('_')[:-1]), q_uids))))
    if num_questions % len(question_types) == 0:
        random_q_uids = []
        for question_type in question_types:
            random_q_uids.extend(rng.choice([x for x in list(qa_data.keys()) if question_type in x], size=num_questions // len(question_types), replace=False))
    else:
        raise Exception('Number of questions must be divisible by number of question types')

    pd.DataFrame({'q_uid': random_q_uids}).to_csv(f'subsets/{subset_name}.csv', index=False)


def replace_answers(hd_epic_path, answer_replacement_type='easy', num_new_negatives=5, seed=0):
    qa_data, qa_answers = get_hd_epic(hd_epic_path)

    q_uids = list(qa_answers.keys())
        
    rng = np.random.default_rng(seed=seed)

    temp_qa_data = copy.deepcopy(qa_data)
    temp_qa_answers = copy.deepcopy(qa_answers)

    if answer_replacement_type == 'easy':
        q_uids_copy = copy.copy(q_uids)

        while not all([q_uids[i] != x for i, x in enumerate(q_uids_copy)]):
            rng.shuffle(q_uids_copy)

        for (q_uid, negative_q_uid) in zip(q_uids, q_uids_copy):
            q, options, q_dict = parse_vqa(qa_data, q_uid)
            original_negative_indices = list(range(len(options)))
            original_negative_indices.remove(qa_answers[q_uid])
            new_negative_indices = list(range(len(options)))
            new_negative_indices.remove(qa_answers[negative_q_uid])
            for (original_negative_index, new_negative_index) in zip(original_negative_indices, new_negative_indices):
                temp_qa_data[q_uid]['choices'][original_negative_index] = qa_data[negative_q_uid]['choices'][new_negative_index]
    elif answer_replacement_type == 'hard':
        for q_uid in tqdm(q_uids):
            q_uids_copy = copy.copy(q_uids)
            q_uids_copy.remove(q_uid)

            question_type = '_'.join(q_uid.split('_')[:-1])

            q_uids_copy = [x for x in q_uids_copy if question_type in x]

            _, options, _ = parse_vqa(qa_data, q_uid)

            possible_new_negatives = []

            for negative_q_uid in q_uids_copy:
                _, temp_options, _ = parse_vqa(qa_data, negative_q_uid)
                possible_new_negatives.extend(temp_options)

            possible_new_negatives = list(set(possible_new_negatives))

            if len(possible_new_negatives) < num_new_negatives:
                if question_type in ['object_motion_object_movement_counting', '3d_perception_fixture_interaction_counting']:
                    possible_new_negatives = list(map(lambda x: str(x), range(0, num_new_negatives + 5)))
                    possible_new_negatives = [x for x in possible_new_negatives if x not in options]
                elif question_type in ['3d_perception_fixture_location']:
                    possible_new_negatives = [x for x in possible_new_negatives if x != options[qa_answers[q_uid]]]
                    possible_new_negatives.extend(list(map(lambda x: f'{(int(x.split(" ")[0]) % 12) * 30} degrees', possible_new_negatives)))
        
            new_negatives = rng.choice(possible_new_negatives, size=num_new_negatives, replace=False)
            options.extend(new_negatives)

            new_options = list(enumerate(options))

            rng.shuffle(new_options)

            options_list = []

            for j, (i, option) in enumerate(new_options):
                if i == qa_answers[q_uid]:
                    temp_qa_answers[q_uid] = j
                    temp_qa_data[q_uid]['correct_idx'] = j

                options_list.append(option)

            temp_qa_data[q_uid]['choices'] = options_list

    answer_replacement = answer_replacement_type if answer_replacement_type == 'easy' else f'{answer_replacement_type}_{num_new_negatives}' 

    with open(f'{hd_epic_path}/questions_{answer_replacement}.json', 'w') as f:
        json.dump(temp_qa_data, f)

    with open(f'{hd_epic_path}/subset_answers_{answer_replacement}.json', 'w') as f:
        json.dump(temp_qa_answers, f)


def print_hd_epic_subset(subset_path, hd_epic_path, answer_replacement=None):
    qa_data, qa_answers = get_hd_epic(hd_epic_path, answer_replacement)

    subset_df = pd.read_csv(subset_path)
    
    for q_uid in subset_df['q_uid']:
        print(qa_data[q_uid])


def dump_hd_epic_subset(subset_path, hd_epic_path, answer_replacement=None):
    qa_data, qa_answers = get_hd_epic(hd_epic_path, answer_replacement)

    subset_df = pd.read_csv(subset_path)
    
    subset_jsons = []

    for q_uid in subset_df['q_uid']:
        subset_jsons.append({q_uid: qa_data[q_uid]})
    
    add_to_path = ''
    if answer_replacement is not None:
        add_to_path += f'_{answer_replacement}'

    with open(f'{subset_path.split(".")[0]}{add_to_path}.json', 'w', encoding='utf-8') as f:
        json.dump(subset_jsons, f, ensure_ascii=False, indent=4)


def secs_from_time_str(s):
    t = datetime.datetime.strptime(s, '%H:%M:%S.%f').time()
    total_seconds = t.hour*3600 + t.minute*60 + t.second + t.microsecond/1e6
    return int(total_seconds)


def time_from_tag(x, stride=1.0, question=None):
    seconds = secs_from_time_str(x.groups()[0])
    input_id = x.groups()[1]

    if input_id not in list(question["inputs"].keys()):
        input_start_seconds = 0
    else:
        if question == None or 'start_time' not in question["inputs"][input_id]:
            input_start_seconds = 0
        else:
            input_start_str = question["inputs"][input_id]["start_time"]
            input_start_seconds = secs_from_time_str(input_start_str)

    seconds = seconds - input_start_seconds

    seconds /= stride

    if seconds >= 3600:
        parsed_time = time.strftime('%H:%M:%S', time.gmtime(seconds))
    else:
        parsed_time = time.strftime('%M:%S', time.gmtime(seconds))
    
    return parsed_time


def bbox_from_tag(x):
    coords = [float(x) for x in x.groups()]
    coords = [int(x / 1408 * 1000) for x in coords]
    return f'({", ".join([str(x) for x in coords])})'


def parse_tags(text, stride, question=None):
    time_pattern = r"<TIME\s+([\d:.]+)\s+(.+?)>"

    text = re.sub(time_pattern, lambda x: time_from_tag(x, stride=stride, question=question), text)

    bbox_pattern = r"<BBOX\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*>"
    text = re.sub(bbox_pattern, lambda x: bbox_from_tag(x), text)
    return text


def get_video_info_from_question(q_dict, input_fps):
    video_ids = []
    start_secs = []
    end_secs = []
    input_keys = []

    for k, v in q_dict['inputs'].items():
        video_ids.append(v['id'])
        input_keys.append(k)
        if 'image' in k:
            t = secs_from_time_str(v['time'])
            start_secs.append(t)
            end_secs.append(t + (1.0 / input_fps))
        else:
            start = secs_from_time_str(v['start_time']) if 'start_time' in v else -1
            end = secs_from_time_str(v['end_time']) if 'end_time' in v else -1
            if end - start < 1.0 / input_fps:
                end = start + 1.0 / input_fps
            start_secs.append(start)
            end_secs.append(end)

    return video_ids, start_secs, end_secs, input_keys


def calculate_frame_idxs(video_ids, input_keys, frames, start_secs, end_secs, input_fps, video_path):
    video_lens = []
    local_frame_idxs = []
    image_frame_idxs = []

    for i in range(len(video_ids)):
        if 'video' in input_keys[i]:
            if start_secs[i] == -1 or end_secs[i] == -1:
                cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {get_full_video_path(video_path, video_ids[i])}'
                result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
                secs = float(result.stdout)
                video_lens.append(secs)
                local_frame_idxs.extend([(input_keys[i], x) for x in np.arange(0, secs * input_fps, dtype=int)])
            else:
                video_lens.append(end_secs[i] - start_secs[i])
                local_frame_idxs.extend([(input_keys[i], x) for x in np.arange(start_secs[i] * input_fps, end_secs[i] * input_fps, dtype=int)])
        elif 'image' in input_keys[i]:
            image_frame_idxs.extend([(input_keys[i], x) for x in np.arange(start_secs[i] * input_fps, end_secs[i] * input_fps, dtype=int)])

    video_len = float(sum(video_lens))

    number_of_videos = sum(list(map(lambda x: 1 if 'video' in x else 0, input_keys)))
    number_of_images = sum(list(map(lambda x: 1 if 'image' in x else 0, input_keys)))

    video_frames_to_sample = frames - number_of_images

    if video_frames_to_sample > len(local_frame_idxs):
        video_frames_to_sample = len(local_frame_idxs)

    global_frame_idxs = []
    stride = -1

    if number_of_videos > 0:
        stride = video_len / video_frames_to_sample
        global_frame_idxs = np.linspace(0, (video_len - 1) * input_fps, video_frames_to_sample, dtype=int)

    return stride, global_frame_idxs, local_frame_idxs, image_frame_idxs


def get_full_video_path(video_path, video_id):
    participant_id = video_id.split('-')[0]
    full_video_path = os.path.join(video_path, participant_id, video_id + '.mp4')
    return full_video_path


def load_frame_idxs(video_id, local_frame_idxs, video_path,height, width):
    full_video_path = get_full_video_path(video_path, video_id)
        
    vr = decord.VideoReader(full_video_path, width=width, height=height, ctx=decord.cpu(0))

    frames = vr.get_batch(local_frame_idxs).asnumpy()
    return frames


def load_videos(q_dict, frames, input_fps, video_path, height=-1, width=-1):
    video_ids, start_secs, end_secs, input_keys = get_video_info_from_question(q_dict, input_fps)

    stride, global_frame_idxs, local_frame_idxs, image_frame_idxs = calculate_frame_idxs(video_ids, input_keys, frames, start_secs, end_secs, input_fps, video_path)

    frame_idx_dict = {x: [] for x in input_keys}

    for global_frame_idx in global_frame_idxs:
        input_key, local_frame_idx = local_frame_idxs[global_frame_idx]
        frame_idx_dict[input_key].append(local_frame_idx)

    for image_frame_idx in image_frame_idxs:
        input_key, frame_idx = image_frame_idx
        frame_idx_dict[input_key].append(frame_idx)

    video_frames = []
    filtered_input_keys = []
    for video_id, input_key in zip(video_ids, input_keys):
        frames = load_frame_idxs(video_id, frame_idx_dict[input_key], video_path, height, width)
        if frames.shape[0] > 0:
            video_frames.append(frames)
            filtered_input_keys.append(input_key)
        else:
            frame_idx_dict.pop(input_key, None)

    return video_frames, filtered_input_keys, stride, frame_idx_dict


def get_video_paths(q_uid, q_dict, video_folder):
    video_ids, start_secs, end_secs, input_keys = get_video_info_from_question(q_dict, 1)

    video_paths = []

    for video_id in video_ids:
        video_paths.append(get_full_video_path(video_folder, video_id))

    return video_paths