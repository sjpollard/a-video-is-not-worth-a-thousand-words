import json
import os
import numpy as np
import pandas as pd
import glob
import decord
import copy

decord.bridge.set_bridge('native')


def rename_videos(video_path):
    for video_json_path in glob.glob(f'{video_path}/*.json'):
        print(video_json_path)
        current_video_path = video_json_path.split('.')[0] + '.mp4'
        if os.path.exists(current_video_path):
            video_json_f = open(video_json_path)
            video_json = json.load(video_json_f)
            new_video_path = current_video_path.replace(video_json['key'], video_json['url'].split('=')[-1])
            os.rename(current_video_path, new_video_path)


def get_lvbench(lvbench_path, answer_replacement=None):
    if answer_replacement is not None:
        qa_data_f = open(f'{lvbench_path}/questions_{answer_replacement}.json')
        qa_data = json.load(qa_data_f)

        qa_answers_f = open(f'{lvbench_path}/subset_answers_{answer_replacement}.json')
        qa_answers = json.load(qa_answers_f)
    else:
        qa_data = {}
        qa_answers = {}

        original_qa_data_f = open(lvbench_path + '/data/video_info.meta.jsonl')
        original_qa_data = [json.loads(line) for line in original_qa_data_f]

        for video in original_qa_data:
            questions = video.pop('qa', None)
            for question in questions:
                q_uid = '_'.join([x.replace(' ', '_') for x in question['question_type']]) + '_' + str(question['uid'])

                split_instruction = question['question'].split('\n')

                question.update({'question': split_instruction[0]})
                question.update({'options': [x[4:]for x in split_instruction[1:]]})
                question.update({'answer': ord(question['answer']) - ord('A')})

                tmp = dict(video)
                tmp.update(question)
                qa_data.update({q_uid: tmp})
                qa_answers.update({q_uid: question['answer']})

    return qa_data, qa_answers


def parse_vqa(qa_data, q_uid):
    q_dict = qa_data[q_uid]

    q = q_dict['question'].capitalize()
    if q[-1] != "?":
        q = q + "?"
    options = q_dict['options']
    options = list(map(lambda x: x.lower().strip().capitalize(), options))

    return q, options, q_dict


def get_random_lvbench_subset(lvbench_path, lvbench_video_path, num_questions, subset_name='lvbench', seed=0):
    if not os.path.exists(f'subsets'):
        os.mkdir(f'subsets')

    rng = np.random.default_rng(seed=seed)
    qa_data, qa_answers = get_lvbench(lvbench_path)

    q_uids = [x for x in qa_data.keys() if os.path.exists(lvbench_video_path + f'/{qa_data[x]["key"]}.mp4')]

    question_types = []
    for q_uid in q_uids:
        question_types.extend(qa_data[q_uid]['question_type'])
    question_types = list(dict.fromkeys(question_types))

    if num_questions % len(question_types) == 0:
        random_q_uids = []
        for question_type in question_types:
                random_q_uids.extend(rng.choice([x for x in list(q_uids) if qa_data[x]['question_type'][0] == question_type], size=num_questions // len(question_types), replace=False))
    else:
        raise Exception('Number of questions must be divisible by number of question types')

    pd.DataFrame({'q_uid': random_q_uids}).to_csv(f'subsets/{subset_name}.csv', index=False)


def replace_answers(lvbench_path, answer_replacement_type='easy', num_new_negatives=5, seed=0):
    qa_data, qa_answers = get_lvbench(lvbench_path)

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
                temp_qa_data[q_uid]['options'][original_negative_index] = qa_data[negative_q_uid]['options'][new_negative_index]
    elif answer_replacement_type == 'hard':
        for q_uid in q_uids:
            q_uids_copy = copy.copy(q_uids)
            q_uids_copy.remove(q_uid)

            _, options, _ = parse_vqa(qa_data, q_uid)

            possible_new_negatives = []
            for negative_q_uid in q_uids_copy:
                if bool(set(qa_data[negative_q_uid]['question_type']) & set(qa_data[q_uid]['question_type'])):
                    _, temp_options, _ = parse_vqa(qa_data, negative_q_uid)
                    possible_new_negatives.extend(temp_options)

            new_negatives = rng.choice(possible_new_negatives, size=num_new_negatives, replace=False)
            options.extend(new_negatives)

            new_options = list(enumerate(options))

            rng.shuffle(new_options)

            options_list = []

            for j, (i, option) in enumerate(new_options):
                if i == qa_answers[q_uid]:
                    temp_qa_answers[q_uid] = j
                    temp_qa_data[q_uid]['answer'] = j

                options_list.append(option)

            temp_qa_data[q_uid]['options'] = options_list
    
    answer_replacement = answer_replacement_type if answer_replacement_type == 'easy' else f'{answer_replacement_type}_{num_new_negatives}' 

    with open(f'{lvbench_path}/questions_{answer_replacement}.json', 'w') as f:
        json.dump(temp_qa_data, f)

    with open(f'{lvbench_path}/subset_answers_{answer_replacement}.json', 'w') as f:
        json.dump(temp_qa_answers, f)


def print_lvbench_subset(subset_path, lvbench_path, answer_replacement=None):
    qa_data, qa_answers = get_lvbench(lvbench_path, answer_replacement)

    subset_df = pd.read_csv(subset_path)
    
    for q_uid in subset_df['q_uid']:
        print(qa_data[q_uid])


def dump_lvbench_subset(subset_path, lvbench_path, answer_replacement=None):
    qa_data, qa_answers = get_lvbench(lvbench_path, answer_replacement)

    subset_df = pd.read_csv(subset_path)
    
    subset_jsons = []

    for q_uid in subset_df['q_uid']:
        subset_jsons.append({q_uid: qa_data[q_uid]})
    
    add_to_path = ''
    if answer_replacement is not None:
        add_to_path += f'_{answer_replacement}'

    with open(f'{subset_path.split(".")[0]}{add_to_path}.json', 'w', encoding='utf-8') as f:
        json.dump(subset_jsons, f, ensure_ascii=False, indent=4)


def load_video(q_dict, frames, video_path, height=-1, width=-1):
    video_id = q_dict['key']

    vr = decord.VideoReader(video_path + f'/{video_id}.mp4', ctx=decord.cpu(0), width=width, height=height, num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    frame_indices = np.linspace(0, max_frame - 1, frames, dtype=int)

    return vr.get_batch(frame_indices).asnumpy(), frame_indices / fps


def get_video_paths(q_uid, q_dict, video_folder):
    video_id = q_dict['key']
    return [video_folder + f'/{video_id}.mp4']