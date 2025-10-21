import json
import os
import numpy as np
import pandas as pd
import copy


def get_egoschema(egoschema_path, answer_replacement=None):
    if answer_replacement is not None:
        qa_data_f = open(f'{egoschema_path}/questions_{answer_replacement}.json')
    else:
        qa_data_f = open(f'{egoschema_path}/questions.json')
    temp_qa_data = json.load(qa_data_f)

    qa_data = {}

    for dictionary in temp_qa_data:
        qa_data.update({dictionary['q_uid']: dictionary})

    if answer_replacement is not None:
        qa_answers_f = open(f'{egoschema_path}/subset_answers_{answer_replacement}.json')
    else:
        qa_answers_f = open(f'{egoschema_path}/subset_answers.json')
    qa_answers = json.load(qa_answers_f)

    return qa_data, qa_answers


def parse_vqa(qa_data, q_uid):
    q_dict = qa_data[q_uid]

    q = q_dict['question'].capitalize()
    if q[-1] != "?":
        q = q + "?"
    num_options = len([x for x in list(q_dict.keys()) if 'option' in x])
    options = list(map(lambda x: q_dict[f'option {x}'], range(num_options)))
    options = list(map(lambda x: x.lower().strip().capitalize(), options))

    return q, options, q_dict


def find_extreme_q_uids(egoschema_path, answer_replacement=None):
    qa_data, qa_answers = get_egoschema(egoschema_path, answer_replacement)

    lengths = []

    for q_uid in qa_answers.keys():
        q, options, q_dict = parse_vqa(qa_data, q_uid)

        lengths.append(len((' '.join([q] + options)).split(' ')))

    shortest_index, longest_index = lengths.index(min(lengths)), lengths.index(max(lengths))

    shortest_q_uid = list(qa_answers.keys())[shortest_index]
    q, options, q_dict = parse_vqa(qa_data, shortest_q_uid)
    print(shortest_q_uid + '\n' + ' '.join([q] + options))

    longest_q_uid = list(qa_answers.keys())[longest_index]
    q, options, q_dict = parse_vqa(qa_data, longest_q_uid)

    print(longest_q_uid + '\n' + ' '.join([q] + options))


def get_random_egoschema_subset(egoschema_path, num_questions, subset_name='egoschema', seed=0):
    if not os.path.exists(f'subsets'):
        os.mkdir(f'subsets')

    rng = np.random.default_rng(seed=seed)
    qa_data, qa_answers = get_egoschema(egoschema_path)

    q_uids = list(qa_answers.keys())

    random_q_uids = rng.choice(q_uids, size=num_questions, replace=False)
    pd.DataFrame({'q_uid': random_q_uids}).to_csv(f'subsets/{subset_name}.csv', index=False)


def replace_answers(egoschema_path, answer_replacement_type='easy', num_new_negatives=5, seed=0):
    qa_data, qa_answers = get_egoschema(egoschema_path)

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
                temp_qa_data[q_uid][f'option {original_negative_index}'] = qa_data[negative_q_uid][f'option {new_negative_index}']
    elif answer_replacement_type == 'hard':
        for q_uid in q_uids:
            q_uids_copy = copy.copy(q_uids)
            q_uids_copy.remove(q_uid)

            _, options, _ = parse_vqa(qa_data, q_uid)

            possible_new_negatives = []
            for negative_q_uid in q_uids_copy:
                _, temp_options, _ = parse_vqa(qa_data, negative_q_uid)
                possible_new_negatives.extend(temp_options)

            new_negatives = rng.choice(possible_new_negatives, size=num_new_negatives, replace=False)
            options.extend(new_negatives)

            new_options = list(enumerate(options))

            rng.shuffle(new_options)

            for j, (i, option) in enumerate(new_options):
                if i == qa_answers[q_uid]:
                    temp_qa_answers[q_uid] = j

                if f'option {j}' in list(temp_qa_data[q_uid].keys()):
                    temp_qa_data[q_uid][f'option {j}'] = option
                elif f'option {j}' not in list(temp_qa_data[q_uid].keys()):
                    temp_qa_data[q_uid].update({f'option {j}': option})

    temp_qa_data = [temp_qa_data[q_uid] for q_uid in list(temp_qa_data.keys())]

    answer_replacement = answer_replacement_type if answer_replacement_type == 'easy' else f'{answer_replacement_type}_{num_new_negatives}' 

    with open(f'{egoschema_path}/questions_{answer_replacement}.json', 'w') as f:
        json.dump(temp_qa_data, f)

    with open(f'{egoschema_path}/subset_answers_{answer_replacement}.json', 'w') as f:
        json.dump(temp_qa_answers, f)


def print_egoschema_subset(subset_path, egoschema_path, answer_replacement=None):
    qa_data, qa_answers = get_egoschema(egoschema_path, answer_replacement)

    subset_df = pd.read_csv(subset_path)
    
    for q_uid in subset_df['q_uid']:
        print(qa_data[q_uid])


def dump_egoschema_subset(subset_path, egoschema_path, answer_replacement=None):
    qa_data, qa_answers = get_egoschema(egoschema_path, answer_replacement)

    subset_df = pd.read_csv(subset_path)
    
    subset_jsons = []

    for q_uid in subset_df['q_uid']:
        subset_jsons.append(qa_data[q_uid])
    
    add_to_path = ''
    if answer_replacement is not None:
        add_to_path += f'_{answer_replacement}'

    with open(f'{subset_path.split(".")[0]}{add_to_path}.json', 'w', encoding='utf-8') as f:
        json.dump(subset_jsons, f, ensure_ascii=False, indent=4)


def get_video_paths(q_uid, q_dict, video_folder):
    return [f'{video_folder}/videos/{q_uid}.mp4']