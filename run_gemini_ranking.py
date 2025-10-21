from google import genai
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd
import os
import json

import egoschema_utils
import hd_epic_utils
import mvbench_utils
import lvbench_utils

EGOSCHEMA_CONFIG = {'qa_folder': '<EGOSCHEMA-PATH>',
                    'video_folder': '<EGOSCHEMA-PATH>',
                    'system_prompt': "You are an expert video analyser, and your job is to answer the multiple choice question by giving only the letter identifying the answer. Do not give any other information. For example, acceptable answers are 'A' or 'B' or 'C' etc.. You must give an answer, even if you are not sure.",
                    'dataset_fn': egoschema_utils.get_egoschema,
                    'parse_vqa': egoschema_utils.parse_vqa}


HD_EPIC_CONFIG = {'qa_folder': '<HD-EPIC-ANNOTATIONS-PATH>',
                  'video_folder': '<HD-EPIC-VIDEO-PATH>',
                  'system_prompt': "You are an expert video analyser, and your job is to answer the multiple choice question by giving only the letter identifying the answer. Do not give any other information. For example, acceptable answers are 'A' or 'B' or 'C' etc.. You must give an answer, even if you are not sure. Videos are at 1 fps, and timestamps are MM:SS. Bounding boxes are in the format (ymin, xmin, ymax, xmax) relative to an image size of 1000x1000.",
                  'dataset_fn': hd_epic_utils.get_hd_epic,
                  'parse_vqa': hd_epic_utils.parse_vqa}


MVBENCH_CONFIG = {'qa_folder': '<MVBENCH_PATH>',
                  'video_folder': '<MVBENCH_PATH>/video',
                  'system_prompt': 'Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.',
                  'dataset_fn': mvbench_utils.get_mvbench,
                  'parse_vqa': mvbench_utils.parse_vqa}


LVBENCH_CONFIG = {'qa_folder': '<LVBENCH_PATH>',
                  'video_folder': '<LVBENCH_PATH>/videos/00000',
                  'system_prompt': "You are an expert video analyser, and your job is to answer the multiple choice question by giving only the letter identifying the answer. Do not give any other information. For example, acceptable answers are 'A' or 'B' or 'C' etc.. You must give an answer, even if you are not sure.",
                  'dataset_fn': lvbench_utils.get_lvbench,
                  'parse_vqa': lvbench_utils.parse_vqa}


def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Running experiments"
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        type=str
    )
    parser.add_argument(
        "--gemini-version",
        dest="gemini_version",
        choices=['gemini-2.5-pro', 'gemini-2.5-flash'],
        default='gemini-2.5-pro',
        type=str
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
        "--model",
        dest="model",
        choices=['frozenbilm', 'internvideo', 'videollama2', 'llava_video', 'longva', 'videollama3'],
        default='frozenbilm',
        type=str
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
        "--force-redo",
        dest="force_redo",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--command",
        "-c",
        dest="command",
        choices=['rank'],
        default='rank',
        type=str
    )
    return parser.parse_args()


def load_frames(dataset, video_path):
    num_frames = len(os.listdir(video_path))

    if dataset == 'hd-epic':
        num_frames = num_frames - 1

    frames = []

    for frame_index in range(num_frames):
        frame = Image.open(os.path.join(video_path, f'frame_{frame_index}.png'))
        frames.append(frame)

    return frames


def gemini_rank(args, frames, question, options, q_dict, stride):    
    if args.dataset == 'hd-epic':
        parse_tags = lambda x: hd_epic_utils.parse_tags(x, stride, q_dict)
    else: 
        parse_tags = None

    system_prompt ='You will be given frames from a video and a question with multiple-choice answer options.' \
                   '\nframe_0: {frame 0} ,... ,frame_n-1:{frame n-1}' \
                   '\nQuestion: {question}' \
                   '\nOptions: {answer choices}' \
                   '\nYou do not need to answer the question; return a comma separated list of frame ids in the order of their importance for answering the above question. ' \
                   'Order the frame ids by importance, do not leave them in chronological order. ' \
                   f'Respond with exactly {len(frames)} frame ids, returning only this comma separated list and excluding all other textual output.'

    if args.gemini_version == 'gemini-2.5-pro':
        config = genai.types.GenerateContentConfig(system_instruction=system_prompt)
    else:
        config = genai.types.GenerateContentConfig(thinking_config=genai.types.ThinkingConfig(thinking_budget=0), system_instruction=system_prompt)

    options_instruction = ''.join([f'({chr(ord("A") + i)}) {options[i]}\n' for i in range(len(options))])
    qa_contents = f"Question:\n{question}\nOptions:\n{options_instruction}"
    qa_contents = parse_tags(qa_contents) if parse_tags is not None else qa_contents

    contents = []
    for i, frame in enumerate(frames):
        contents.append(f'frame_{i}: ')
        contents.append(frame)

    contents.append(qa_contents)
    contents.append('The ordered list of frames for answering this multiple choice question is: ')

    client = genai.Client(api_key=args.api_key)

    response = client.models.generate_content(
        model=args.gemini_version, 
        config=config,
        contents=contents
    )

    return response.text


def rank_frames(args):
    if not os.path.exists(f'ranks/{args.gemini_version}/{args.model}/{args.dataset}'):
        os.makedirs(f'ranks/{args.gemini_version}/{args.model}/{args.dataset}')

    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'])

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
    else:
        q_uids = [args.q_uid]
    
    for q_uid in tqdm(q_uids):

        redo_null = False

        if os.path.exists(f'ranks/{args.gemini_version}/{args.model}/{args.dataset}/{q_uid}.json'):
            ranks_f = open(f'ranks/{args.gemini_version}/{args.model}/{args.dataset}/{q_uid}.json')
            ranks = json.load(ranks_f)
            if ranks[q_uid] is None:
                redo_null = True

        if args.force_redo or not os.path.exists(f'ranks/{args.gemini_version}/{args.model}/{args.dataset}/{q_uid}.json') or redo_null:

            outputs = {}

            frames = load_frames(args.dataset, f'all_frames/{args.model}/{args.dataset}/{q_uid}')

            stride = -1

            if args.dataset == 'hd-epic':
                with open(f'all_frames/{args.model}/{args.dataset}/{q_uid}/stride.json') as f:
                    stride = json.load(f)['stride']

            q, options, q_dict = args.config['parse_vqa'](qa_data, q_uid)

            label = qa_answers[q_uid]

            output = gemini_rank(args, frames, q, options, q_dict, stride)

            print(output)

            outputs.update({q_uid : output})
        
            with open(f'ranks/{args.gemini_version}/{args.model}/{args.dataset}/{q_uid}.json', 'w', encoding='utf-8') as f:
                json.dump(outputs, f, ensure_ascii=False, indent=4)
    

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

    if args.command == 'rank':
        rank_frames(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)