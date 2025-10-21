import json
import os
import numpy as np
import pandas as pd
import glob
import decord
from PIL import Image

decord.bridge.set_bridge('native')

data_list = {
    "action_sequence": ("action_sequence.json", "star/Charades_v1_480", "video", True), # has start & end
    "action_prediction": ("action_prediction.json", "star/Charades_v1_480", "video", True), # has start & end
    "action_antonym": ("action_antonym.json", "ssv2_video", "video", False),
    "fine_grained_action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos", "video", False),
    "unexpected_action": ("unexpected_action.json", "FunQA_test/test", "video", False),
    "object_existence": ("object_existence.json", "clevrer/video_validation", "video", False),
    "object_interaction": ("object_interaction.json", "star/Charades_v1_480", "video", True), # has start & end
    "object_shuffle": ("object_shuffle.json", "perception/videos", "video", False),
    "moving_direction": ("moving_direction.json", "clevrer/video_validation", "video", False),
    "action_localization": ("action_localization.json", "sta/sta_video", "video", True),  # has start & end
    "scene_transition": ("scene_transition.json", "scene_qa/video", "video", False),
    "action_count": ("action_count.json", "perception/videos", "video", False),
    "moving_count": ("moving_count.json", "clevrer/video_validation", "video", False),
    "moving_attribute": ("moving_attribute.json", "clevrer/video_validation", "video", False),
    "state_change": ("state_change.json", "perception/videos", "video", False),
    "fine_grained_pose": ("fine_grained_pose.json", "nturgbd", "video", False),
    "character_order": ("character_order.json", "perception/videos", "video", False),
    "egocentric_navigation": ("egocentric_navigation.json", "vlnqa", "video", False),
    "episodic_reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq", "frame", True),  # has start & end, read frame
    "counterfactual_inference": ("counterfactual_inference.json", "clevrer/video_validation", "video", False),
}


def get_mvbench(mvbench_path, answer_replacement=None):
    qa_data = {}
    qa_answers = {}

    for q_file in glob.glob(mvbench_path + '/json/*.json'):
        tmp_questions_f = open(q_file)
        tmp_questions = json.load(tmp_questions_f)
        task_name = q_file.split('/')[-1].split('.')[0]
        for question_data in tmp_questions:
            index = tmp_questions.index(question_data)
            qa_data.update({f'{task_name}_{index}': question_data})
            qa_answers.update({f'{task_name}_{index}': question_data['candidates'].index(question_data['answer'])})

    return qa_data, qa_answers


def parse_vqa(qa_data, q_uid):
    q_dict = qa_data[q_uid]

    q = q_dict['question'].capitalize()
    if q[-1] != "?":
        q = q + "?"
    options = q_dict['candidates']
    options = list(map(lambda x: x.lower().strip().capitalize(), options))

    return q, options, q_dict


def get_random_mvbench_subset(mvbench_path, num_questions, subset_name='mvbench', seed=0):
    if not os.path.exists(f'subsets'):
        os.mkdir(f'subsets')

    rng = np.random.default_rng(seed=seed)
    qa_data, qa_answers = get_mvbench(mvbench_path)

    q_uids = list(qa_data.keys())

    question_types = list(dict.fromkeys(list(map(lambda x: '_'.join(x.split('_')[:-1]), q_uids))))
    if num_questions % len(question_types) == 0:
        random_q_uids = []
        for question_type in question_types:
            if question_type != 'fine_grained_pose':
                size = 2 * (num_questions // len(question_types)) if question_type == 'counterfactual_inference' else num_questions // len(question_types)
                random_q_uids.extend(rng.choice([x for x in list(qa_data.keys()) if question_type in x], size=size, replace=False))
    else:
        raise Exception('Number of questions must be divisible by number of question types')

    pd.DataFrame({'q_uid': random_q_uids}).to_csv(f'subsets/{subset_name}.csv', index=False)


def print_mvbench_subset(subset_path, mvbench_path):
    qa_data, qa_answers = get_mvbench(mvbench_path)

    subset_df = pd.read_csv(subset_path)
    
    for q_uid in subset_df['q_uid']:
        print(qa_data[q_uid])


def dump_mvbench_subset(subset_path, mvbench_path):
    qa_data, qa_answers = get_mvbench(mvbench_path)

    subset_df = pd.read_csv(subset_path)
    
    subset_jsons = []

    for q_uid in subset_df['q_uid']:
        subset_jsons.append({q_uid: qa_data[q_uid]})
    
    with open(f'{subset_path.split(".")[0]}.json', 'w', encoding='utf-8') as f:
        json.dump(subset_jsons, f, ensure_ascii=False, indent=4)


def random_performance(subset_path, mvbench_path):
    qa_data, qa_answers = get_mvbench(mvbench_path)

    subset_df = pd.read_csv(subset_path)

    random_chances = []

    for q_uid in subset_df['q_uid']:
        _, options, _ = parse_vqa(qa_data, q_uid)
        random_chance = 1.0 / len(options)
        random_chances.append(random_chance)

    print(sum(random_chances) / len(subset_df))


def get_indices(frames, fps, max_frame, bound=None, first_idx=0):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / frames
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(frames)
    ])
    return frame_indices


def read_video(frames, video_path, bound, height=-1, width=-1):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), width=width, height=height, num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    frame_indices = get_indices(frames, fps, max_frame, bound, first_idx=0)

    return vr.get_batch(frame_indices).asnumpy(), frame_indices / fps
   
    
def read_frame(frames, video_path, bound, fps=3, height=-1, width=-1):
    max_frame = len(os.listdir(video_path))
    images = []
    frame_indices = get_indices(frames, fps, max_frame, bound, first_idx=1) # frame_idx starts from 1

    for frame_index in frame_indices:
        image = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
        if height > 0 and width > 0:
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        images.append(np.asarray(image))

    return np.stack(images), frame_indices / fps


def load_video(q_uid, q_dict, frames, video_path, height=-1, width=-1):
    task_name = '_'.join(q_uid.split('_')[:-1])
    task_specifics = data_list[task_name]

    bound = None
    if task_specifics[3]:
        bound = (q_dict['start'], q_dict['end'])

    if task_specifics[2] == 'frame':
        video, timestamps = read_frame(frames, f'{video_path}/{task_specifics[1]}/{q_dict["video"]}', bound, height=height, width=width)
    elif task_specifics[2] == 'video':
        video, timestamps = read_video(frames, f'{video_path}/{task_specifics[1]}/{q_dict["video"]}', bound, height=height, width=width)

    return video, timestamps


def get_video_paths(q_uid, q_dict, video_folder):
    task_name = '_'.join(q_uid.split('_')[:-1])
    task_specifics = data_list[task_name]

    if task_specifics[2] == 'frame':
        return [f'{video_folder}/{task_specifics[1]}/{q_dict["video"]}']
    elif task_specifics[2] == 'video':
        return [f'{video_folder}/{task_specifics[1]}/{q_dict["video"]}']