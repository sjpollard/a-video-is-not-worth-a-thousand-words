import os
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
import torch
import pandas as pd
from itertools import accumulate
import copy
from PIL import Image

# VideoLLaMA3 imports
from videollama3.videollama3 import model_init, mm_infer_logits

# Shapley value imports
import shap
from utils import custom_masker, convert_to_indices, accuracy, threshold_indices, split_string, get_influential_frame_indices
import egoschema_utils
import hd_epic_utils
import mvbench_utils
import lvbench_utils

torch.manual_seed(0)

device = torch.device("cuda")

def load_egoschema_frames(frames, q_uid, q_dict, processor, video_folder):
    video, timestamps = processor.load_video(
                                   f'{video_folder}/videos/{q_uid}.mp4',
                                   start_time=None,
                                   end_time=None,
                                   precise_time=True,
                                   fps=1,
                                   max_frames=frames,
                                   )

    return [video], ['video 1'], -1, [timestamps]


def load_hd_epic_frames(frames, q_uid, q_dict, processor, video_folder):
    videos, input_keys, stride, frame_idx_dict = hd_epic_utils.load_videos(q_dict=q_dict, frames=frames, input_fps=1, video_path=video_folder, height=336, width=336)
    timestamps = []
    for key in frame_idx_dict.keys():
        current_input_timestamps = []
        for video_timestamps in frame_idx_dict[key]:
            current_input_timestamps.append(video_timestamps / stride)
        timestamps.append(current_input_timestamps)

    return videos, input_keys, stride, timestamps


def load_mvbench_frames(frames, q_uid, q_dict, processor, video_folder):
    video, timestamps = mvbench_utils.load_video(q_uid, q_dict, frames, video_path=video_folder, height=336, width=336)

    return [video], ['video 1'], -1, [timestamps]


def load_lvbench_frames(frames, q_uid, q_dict, processor, video_folder):
    video, timestamps = lvbench_utils.load_video(q_dict, frames, video_path=video_folder, height=336, width=336)

    return [video], ['video 1'], -1, [timestamps]


EGOSCHEMA_CONFIG = {'qa_folder': '<EGOSCHEMA-PATH>',
                    'video_folder': '<EGOSCHEMA-PATH>',
                    'system_prompt': "You are an expert video analyser, and your job is to answer the multiple choice question by giving only the letter identifying the answer. Do not give any other information. For example, acceptable answers are 'A' or 'B' or 'C' etc.. You must give an answer, even if you are not sure.",
                    'dataset_fn': egoschema_utils.get_egoschema,
                    'parse_vqa': egoschema_utils.parse_vqa,
                    'load_frames': load_egoschema_frames}


HD_EPIC_CONFIG = {'qa_folder': '<HD-EPIC-ANNOTATIONS-PATH>',
                  'video_folder': '<HD-EPIC-VIDEO-PATH>',
                  'system_prompt': "You are an expert video analyser, and your job is to answer the multiple choice question by giving only the letter identifying the answer. Do not give any other information. For example, acceptable answers are 'A' or 'B' or 'C' etc.. You must give an answer, even if you are not sure. Videos are at 1 fps, and timestamps are MM:SS. Bounding boxes are in the format (ymin, xmin, ymax, xmax) relative to an image size of 1000x1000.",
                  'dataset_fn': hd_epic_utils.get_hd_epic,
                  'parse_vqa': hd_epic_utils.parse_vqa,
                  'load_frames': load_hd_epic_frames}


MVBENCH_CONFIG = {'qa_folder': '<MVBENCH_PATH>',
                  'video_folder': '<MVBENCH_PATH>/video',
                  'system_prompt': 'Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.',
                  'dataset_fn': mvbench_utils.get_mvbench,
                  'parse_vqa': mvbench_utils.parse_vqa,
                  'load_frames': load_mvbench_frames}


LVBENCH_CONFIG = {'qa_folder': '<LVBENCH_PATH>',
                  'video_folder': '<LVBENCH_PATH>/videos/00000',
                  'system_prompt': "You are an expert video analyser, and your job is to answer the multiple choice question by giving only the letter identifying the answer. Do not give any other information. For example, acceptable answers are 'A' or 'B' or 'C' etc.. You must give an answer, even if you are not sure.",
                  'dataset_fn': lvbench_utils.get_lvbench,
                  'parse_vqa': lvbench_utils.parse_vqa,
                  'load_frames': load_lvbench_frames}


VIDEOLLAMA3_WEIGHTS_PATH = "<VIDEOLLAMA3-PATH>"


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
        default=180,
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
        choices=['calculate', 'compare', 'evaluate', 'benchmark', 'save_frames'],
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
    parser.add_argument(
        "--saved-frames",
        dest="saved_frames",
        default=16,
        type=int
    )
    return parser.parse_args()


def apply_multi_modal_mask(videos, question, options, indices, lengths, mode='joint', force_logit=None):
    multi_modal_indices = {}
    intervals = list(zip([0] + list(accumulate(list(lengths.values())[:-1])), accumulate(list(lengths.values()))))
    masked_videos = copy.deepcopy(videos)
    for i, key in enumerate(lengths.keys()):
        start, end = intervals[i]
        multi_modal_indices.update({key: indices[start:end]})
    if mode == 'text_only':
        question = ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(multi_modal_indices['question'], split_string(question)))))
        options = list(map(lambda y: ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(y[1], split_string(y[0]))))), zip(options, list(map(lambda x: multi_modal_indices[x], [f'option {i}' for i in range(1, len(options) + 1)])))))
    elif mode == 'video_only':
        for i in range(len(videos)):
            masked_videos[i] = list(map(lambda x: x[0] * x[1].item(), zip(videos[i], multi_modal_indices[f'video {i + 1}'])))
    elif mode == 'joint':
        for i in range(len(videos)):
            masked_videos[i] = list(map(lambda x: x[0] * x[1].item(), zip(videos[i], multi_modal_indices[f'video {i + 1}'])))
        question = ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(multi_modal_indices['question'], split_string(question)))))
        options = list(map(lambda y: ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(y[1], split_string(y[0]))))), zip(options, list(map(lambda x: multi_modal_indices[x], [f'option {i}' for i in range(1, len(options) + 1)])))))
    
    if force_logit != None:
        if force_logit['logit'] == 'first':
            forced_index = 0
        elif force_logit['logit'] == 'last':
            forced_index = len(options) - 1
        temp = options.pop(force_logit['gt'])
        options.insert(forced_index, temp)

    return masked_videos, question, options


def videollama3_forward(model, processor, system_prompt, videos, timestamps, input_keys, question, options, indices, lengths, answer_ids, mode, parse_tags, force_logit=None):
    indices = torch.from_numpy(indices)
    indices = indices[0]

    masked_videos, question, options = apply_multi_modal_mask(videos, question, options, indices, lengths, mode, force_logit)
    masked_videos = processor.process_images(masked_videos, merge_size=2, return_tensors='pt')

    options_instruction = ''.join([f'({chr(ord("A") + i)}) {options[i]}\n' for i in range(len(options))])
    instruction = f"Select the best answer to the following multiple-choice question based on the video.\n{question}\nOptions:\n{options_instruction}Answer with the option\'s letter from the given choices directly and only give the best option. The best answer is: "
    instruction = parse_tags(instruction) if parse_tags is not None else instruction

    system_message = [
        {'role': 'system', 'content': system_prompt}
    ]

    if len(input_keys) > 1:
        contents = []
        for video_timestamps, input_key in zip(timestamps, input_keys):
            contents.extend([{"type": "text", "text": input_key},
                             {"type": "video", "num_frames": len(video_timestamps), "timestamps": video_timestamps}])
        contents.append({"type": "text", "text": instruction})
    else:
        contents = [{"type": "video", "num_frames": len(timestamps[0]), "timestamps": timestamps[0]},
                    {"type": "text", "text": instruction}]

    conversation = [
        {
            "role": "user",
            "content": contents
        }
    ]

    prompt = processor.apply_chat_template(system_message + conversation, tokenize=False, add_generation_prompt=True)

    text = processor.process_text(
        prompt,
        masked_videos,
        padding=False,
        padding_side=None,
        return_tensors="pt"
    )

    outputs = mm_infer_logits(
        {**masked_videos, **text},
        model=model,
        tokenizer=processor.tokenizer,
        modal=['video'] * len(input_keys),
        do_sample=False,
    )

    output_tokens = processor.tokenizer.convert_ids_to_tokens(outputs.sequences[0])

    option_in_tokens = [x in output_tokens for x in [chr(ord('A') + i) for i in range(len(options))]]
    
    if True in option_in_tokens:
        token_index = output_tokens.index([chr(ord('A') + i) for i in range(len(options))][option_in_tokens.index(True)])
    else:
        token_index = 0

    logits = outputs.logits[token_index][0].cpu()

    return logits[answer_ids][None]


def videollama3_shap(args, model, processor, q_uid, question, options, q_dict, qa_answers):
    with torch.no_grad():
        
        videos, input_keys, stride, timestamps = args.config['load_frames'](args.frames, q_uid, q_dict, processor, args.config['video_folder'])
                
        if args.dataset == 'hd-epic':
            parse_tags = lambda x: hd_epic_utils.parse_tags(x, stride, q_dict)
        else: 
            parse_tags = None

        x, lengths, names = convert_to_indices([len(x) for x in videos], question, options, mode=args.mode)
        x = x[None][None]

        answer_ids = [processor.tokenizer(x)['input_ids'][0] for x in [chr(ord('A') + i) for i in range(len(options))]]

        fc = lambda x: videollama3_forward(model, processor, args.config['system_prompt'], videos, timestamps, input_keys, question, options, x, lengths, answer_ids, args.mode, parse_tags)
        
        explainer = shap.Explainer(model=fc,  masker=custom_masker, algorithm='permutation', feature_names=names, seed=0)

        explanation = explainer(x, batch_size=1, max_evals=args.iterations)

        del videos
        torch.cuda.empty_cache()

        shap_values = np.squeeze(explanation.values).T

        columns = [chr(ord('A') + i) for i in range(len(options))]
        columns[qa_answers[q_uid]] += ': Ground Truth'

        shap_dict = dict(zip(columns, shap_values))

        return pd.DataFrame({'element': names, **shap_dict})


def prepare_videollama3(args):
    model, processor = model_init(model_path=VIDEOLLAMA3_WEIGHTS_PATH, max_visual_tokens=16384, device_map="cuda:0")

    return model, processor


def calculate_shap_values(args): 
    add_to_path = ''
    if args.answer_replacement is not None:
        add_to_path += f'_{args.answer_replacement}'

    results_path = f'shap_results/videollama3/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/videollama3{add_to_path}'

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
    else:
        q_uids = [args.q_uid]

    model, processor = prepare_videollama3(args)

    q_uids = np.array_split(q_uids, args.chunks)[args.chunk_index]

    for q_uid in tqdm(q_uids):

        if args.force_redo or not os.path.exists(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.csv'):

            q, options, q_dict = args.config['parse_vqa'](qa_data, q_uid)
        
            shap_out = videollama3_shap(args, model, processor, q_uid, q, options, q_dict, qa_answers)

            print(f'Question: {q}')
            for idx, option in enumerate(options):
                print(f'{chr(ord("A") + idx)}: {option}')
            print(f'Correct answer: {options[qa_answers[q_uid]]}')
            print(shap_out)
            shap_out.to_csv(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.csv', index=False)


def videollama3_logit_difference(args, model, processor, q_uid, question, options, q_dict, qa_answers):
    with torch.no_grad():
        
        videos, input_keys, stride, timestamps = args.config['load_frames'](args.frames, q_uid, q_dict, processor, args.config['video_folder'])
                
        if args.dataset == 'hd-epic':
            parse_tags = lambda x: hd_epic_utils.parse_tags(x, stride, q_dict)
        else: 
            parse_tags = None

        x, lengths, names = convert_to_indices([len(x) for x in videos], question, options, mode=args.mode)
        x = x[None][None]

        answer_ids = [processor.tokenizer(x)['input_ids'][0] for x in [chr(ord('A') + i) for i in range(len(options))]]

        fc = lambda x: videollama3_forward(model, processor, args.config['system_prompt'], videos, timestamps, input_keys, question, options, x, lengths, answer_ids, args.mode, parse_tags)

        logits = fc(x.numpy()[0])[0]

        x[0, 0, args.index] = 0

        logits_hat = fc(x.numpy()[0])[0]

        print(f'Original input: {logits}')
        print(f'Masked input: {logits_hat}')
        print(f'Difference in true logit when masking \"{names[args.index]}\": {logits_hat[qa_answers[q_uid]] - logits[qa_answers[q_uid]]}')

        del videos
        torch.cuda.empty_cache()

        return


def compare_logit_values(args):
    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    q, options, q_dict = args.config['parse_vqa'](qa_data, args.q_uid)

    model, processor = prepare_videollama3(args)
    videollama3_logit_difference(args, model, processor, args.q_uid, q, options, q_dict, qa_answers)

    print(f'Question: {q}')
    for idx, option in enumerate(options):
        print(f'{chr(ord("A") + idx)}: {option}')
    print(f'Correct answer: {options[qa_answers[args.q_uid]]}')    


def videollama3_evaluate(args, model, processor, q_uid, question, options, q_dict, masking_direction, masking_logit, masking_mode):
    with torch.no_grad():
        
        videos, input_keys, stride, timestamps = args.config['load_frames'](args.frames, q_uid, q_dict, processor, args.config['video_folder'])
        
        if args.dataset == 'hd-epic':
            parse_tags = lambda x: hd_epic_utils.parse_tags(x, stride, q_dict)
        else: 
            parse_tags = None

        x, lengths, names = convert_to_indices([len(x) for x in videos], question, options, mode=args.mode)
        x = x[None][None]

        answer_ids = [processor.tokenizer(x)['input_ids'][0] for x in [chr(ord('A') + i) for i in range(len(options))]]

        if masking_direction != 'nothing':
            add_to_path = ''
            if args.answer_replacement is not None:
                add_to_path += f'_{args.answer_replacement}'

            results_path = f'shap_results/videollama3/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/videollama3{add_to_path}'
            x = threshold_indices(x, lengths, results_path, q_uid, args.iterations, args.mode, masking_direction, masking_logit, masking_mode, args.threshold, args.answer_masking)

        fc = lambda x: videollama3_forward(model, processor, args.config['system_prompt'], videos, timestamps, input_keys, question, options, x, lengths, answer_ids, args.mode, parse_tags, args.force_logit)

        logits = fc(x.numpy()[0])[0]

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
    
    model, processor = prepare_videollama3(args)

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

        pred = videollama3_evaluate(args, model, processor, q_uid, q, options, q_dict, args.masking_direction, args.masking_logit, args.masking_mode)

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

    results_path = f'shap_results/videollama3/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/videollama3{add_to_path}'

    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
        name = args.subset_path.split(".")[0].split("/")[-1]
    else:
        q_uids = [args.q_uid]
        name = q_uids[0]
    
    model, processor = prepare_videollama3(args)

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

        pred = videollama3_evaluate(args, model, processor, q_uid, q, options, q_dict, masking_directions[0], None, None)
        results[masking_directions[0]]['preds'].append(pred)

        for masking_direction in masking_directions[1:]:
            if masking_direction in ['positive', 'negative']:
                for masking_logit in masking_logits:
                    for masking_mode in masking_modes:
                        pred = videollama3_evaluate(args, model, processor, q_uid, q, options, q_dict, masking_direction, masking_logit, masking_mode)
                        results[masking_direction][masking_logit][masking_mode]['preds'].append(pred)
            else:
                for masking_mode in masking_modes:
                    pred = videollama3_evaluate(args, model, processor, q_uid, q, options, q_dict, masking_direction, None, masking_mode)
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


def save_frames(args):
    add_to_path = ''
    if args.answer_replacement is not None:
        add_to_path += f'_{args.answer_replacement}'

    results_path = f'shap_results/videollama3/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/videollama3{add_to_path}'

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
    else:
        q_uids = [args.q_uid]  

    model, processor = prepare_videollama3(args)
    del model

    frames_path = ('frames' if args.saved_frames != -1 else 'all_frames') + f'/videollama3/{args.dataset}'

    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)
    
    for q_uid in q_uids:        
        if not os.path.exists(f'{frames_path}/{q_uid}'):
            os.makedirs(f'{frames_path}/{q_uid}')

        q, options, q_dict = args.config['parse_vqa'](qa_data, q_uid)

        videos, input_keys, stride, timestamps = args.config['load_frames'](args.frames, q_uid, q_dict, processor, args.config['video_folder'])

        if args.saved_frames != -1:
            frame_indices = get_influential_frame_indices(results_path=results_path, q_uid=q_uid, frames=args.saved_frames, iterations=args.iterations, mode=args.mode)
            video = np.concatenate(videos)
            video = video[frame_indices]
        else:
            video = np.concatenate(videos)
            frame_indices = list(range(video.shape[0]))

        if args.dataset == 'egoschema':
            frames = [video[i].transpose(1, 2, 0) for i in range(video.shape[0])]
        else:
            frames = [video[i] for i in range(video.shape[0])]

        frames = list(map(lambda x: Image.fromarray(x), frames))

        for i, frame in enumerate(frames):
            frame.save(fp=f'{frames_path}/{q_uid}/frame_{frame_indices[i]}.png')

        if args.dataset == 'hd-epic':
            with open(f'{frames_path}/{q_uid}/stride.json', 'w') as f:
                json.dump({'stride': stride}, f)


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
    elif args.command == 'save_frames':
        save_frames(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)