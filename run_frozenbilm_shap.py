import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
from itertools import accumulate
import json

# FrozenBiLM imports
from frozenbilm.util.misc import get_mask
from frozenbilm.model.deberta import DebertaV2ForMaskedLM
from transformers import DebertaV2Tokenizer

# Shapley value imports
import shap
from utils import custom_masker, convert_to_indices, accuracy, threshold_indices, split_string
import egoschema_utils
import hd_epic_utils
import mvbench_utils
import lvbench_utils

torch.manual_seed(0)

device = torch.device("cuda:0")

def load_egoschema_features(q_uid, video_folder):
    video = torch.from_numpy(np.load(f'{video_folder}/{q_uid}.npy').astype('float32'))
    video = video.unsqueeze(0).to(device)

    return [video], -1


def load_hd_epic_features(q_uid, video_folder):
    dict = np.load(f'{video_folder}/{q_uid}.npy', allow_pickle=True).item()
    video = torch.from_numpy(dict['features'].astype('float32'))
    video = video.unsqueeze(0).to(device)
    stride = dict['stride']

    return [video], stride


def load_mvbench_features(q_uid, video_folder):
    video = torch.from_numpy(np.load(f'{video_folder}/{q_uid}.npy').astype('float32'))
    video = video.unsqueeze(0).to(device)

    return [video], -1


def load_lvbench_features(q_uid, video_folder):
    video = torch.from_numpy(np.load(f'{video_folder}/{q_uid}.npy').astype('float32'))
    video = video.unsqueeze(0).to(device)

    return [video], -1


EGOSCHEMA_CONFIG = {'qa_folder': '<EGOSCHEMA-PATH>',
                    'video_folder': 'frozenbilm/egoschema_features_10',
                    'dataset_fn': egoschema_utils.get_egoschema,
                    'parse_vqa': egoschema_utils.parse_vqa,
                    'load_features': load_egoschema_features}


HD_EPIC_CONFIG = {'qa_folder': '<HD-EPIC-ANNOTATIONS-PATH>',
                  'video_folder': 'frozenbilm/hd_epic_features_10',
                  'dataset_fn': hd_epic_utils.get_hd_epic,
                  'parse_vqa': hd_epic_utils.parse_vqa,
                  'load_features': load_hd_epic_features}


MVBENCH_CONFIG = {'qa_folder': '<MVBENCH_PATH>',
                  'video_folder': 'frozenbilm/mvbench_features_10',
                  'dataset_fn': mvbench_utils.get_mvbench,
                  'parse_vqa': mvbench_utils.parse_vqa,
                  'load_features': load_mvbench_features}


LVBENCH_CONFIG = {'qa_folder': '<LVBENCH_PATH>',
                  'video_folder': 'frozenbilm/lvbench_features_10',
                  'dataset_fn': lvbench_utils.get_lvbench,
                  'parse_vqa': lvbench_utils.parse_vqa,
                  'load_features': load_lvbench_features}


FROZENBILM_WEIGHTS_PATH = "<FROZENBILM-PATH>"


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
        default=64,
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


def apply_multi_modal_mask(videos, video_mask, question, options, tokenizer, indices, lengths, mode='joint', parse_tags=None, force_logit=None):
    batch_dict = {'video': [], 'video_mask': [], 'text': []}

    for item in indices:
        multi_modal_indices = {}
        intervals = list(zip([0] + list(accumulate(list(lengths.values())[:-1])), accumulate(list(lengths.values()))))
        for i, key in enumerate(lengths.keys()):
            start, end = intervals[i]
            multi_modal_indices.update({key: item[start:end]})            
        if mode == 'text_only':
            batch_dict['video'].append(videos)
            masked_question = ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(multi_modal_indices['question'], split_string(question)))))
            masked_options = list(map(lambda y: ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(y[1], split_string(y[0]))))), zip(options, list(map(lambda x: multi_modal_indices[x], [f'option {i}' for i in range(1, len(options) + 1)])))))
            text = list(map(lambda x: f"Question: {masked_question} Is it '{x}'? {tokenizer.mask_token}.", masked_options))
        elif mode == 'video_only':
            for i in range(len(videos)):
                batch_dict['video'].append(videos[i] * multi_modal_indices[f'video {i + 1}'].reshape(1, -1, 1).to(device))
            text = list(map(lambda x: f"Question: {question} Is it '{x}'? {tokenizer.mask_token}.", options))
        elif mode == 'joint':
            for i in range(len(videos)):
                batch_dict['video'].append(videos[i] * multi_modal_indices[f'video {i + 1}'].reshape(1, -1, 1).to(device))
            masked_question = ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(multi_modal_indices['question'], split_string(question)))))
            masked_options = list(map(lambda y: ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(y[1], split_string(y[0]))))), zip(options, list(map(lambda x: multi_modal_indices[x], [f'option {i}' for i in range(1, len(options) + 1)])))))
            text = list(map(lambda x: f"Question: {masked_question} Is it '{x}'? {tokenizer.mask_token}.", masked_options))
        
        if force_logit != None:
            if force_logit['logit'] == 'first':
                forced_index = 0
            elif force_logit['logit'] == 'last':
                forced_index = len(options) - 1
            temp = text.pop(force_logit['gt'])
            text.insert(forced_index, temp)

        batch_dict['text'].append(list(map(lambda x: parse_tags(x), text)) if parse_tags is not None else text)
        batch_dict['video_mask'].append(video_mask)

    batch_dict['video'] = torch.cat(batch_dict['video'])
    batch_dict['video_mask'] = torch.cat(batch_dict['video_mask'])
    batch_dict['text'] = [list(x) for x in zip(*batch_dict['text'])]

    return batch_dict


def frozenbilm_forward(model, tokenizer, videos, video_mask, question, options, indices, lengths, mode, parse_tags, force_logit=None):
    indices = torch.from_numpy(indices)

    batch_dict = apply_multi_modal_mask(videos, video_mask, question, options, tokenizer, indices, lengths, mode, parse_tags, force_logit)

    logits_list = []
    for choice in batch_dict['text']:

        encoded = tokenizer(
            choice,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt")
        
        output = model(video=batch_dict['video'],
                    video_mask=batch_dict['video_mask'],
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                    )

        logits = output["logits"]
        logits = logits[:, videos[0].size(1):][encoded["input_ids"] == tokenizer.mask_token_id]
        logits_list.append(logits.softmax(-1)[:, 0].cpu())

        yes_scores = torch.stack(logits_list, 1)
        
    return yes_scores


def frozenbilm_shap(args, model, tokenizer, q_uid, question, options, q_dict, qa_answers):
    with torch.no_grad():
        
        videos, stride = args.config['load_features'](q_uid, args.config['video_folder'])

        video_mask = get_mask(
            torch.tensor(args.frames, dtype=torch.long).unsqueeze(0), videos[0].size(1)
        ).to(device)

        if args.dataset == 'hd-epic':
            parse_tags = lambda x: hd_epic_utils.parse_tags(x, stride, q_dict)
        else: 
            parse_tags = None

        x, lengths, names = convert_to_indices([x.size(1) for x in videos], question, options, mode=args.mode)
        x = x[None][None]

        fc = lambda x: frozenbilm_forward(model, tokenizer, videos, video_mask, question, options, x, lengths, args.mode, parse_tags)

        explainer = shap.Explainer(model=fc,  masker=custom_masker, algorithm='permutation', feature_names=names, seed=0)

        explanation = explainer(x, batch_size=args.batch_size, max_evals=args.iterations)

        del videos
        del video_mask
        torch.cuda.empty_cache()

        shap_values = np.squeeze(explanation.values).T
        columns = [chr(ord('A') + i) for i in range(len(options))]
        columns[qa_answers[q_uid]] += ': Ground Truth'

        shap_dict = dict(zip(columns, shap_values))

        return pd.DataFrame({'element': names, **shap_dict})


def prepare_frozenbilm(args):
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
    model = DebertaV2ForMaskedLM.from_pretrained(
        features_dim=768,
        max_feats=args.frames,
        freeze_lm=False,
        freeze_mlm=False,
        ft_ln=False,
        ds_factor_attn=8,
        ds_factor_ff=8,
        dropout=0.1,
        n_ans=2,
        freeze_last=False,
        pretrained_model_name_or_path="microsoft/deberta-v2-xlarge",
        )

    checkpoint = torch.load(FROZENBILM_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    
    model.to(device)
    model.eval()

    tok_yes = torch.tensor(tokenizer("Yes",
                                     add_special_tokens=False,
                                     max_length=1,
                                     truncation=True,
                                     padding="max_length",)["input_ids"],
                           dtype=torch.long,)

    tok_no = torch.tensor(tokenizer("No",
                                    add_special_tokens=False,
                                    max_length=1,
                                    truncation=True,
                                    padding="max_length",)["input_ids"],
                          dtype=torch.long,)


    a2tok = torch.stack([tok_yes, tok_no])
    model.set_answer_embeddings(
        a2tok.to(model.device), freeze_last=False
    )
    return model, tokenizer


def calculate_shap_values(args):
    add_to_path = ''
    if args.answer_replacement is not None:
        add_to_path += f'_{args.answer_replacement}'

    results_path = f'shap_results/frozenbilm/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/frozenbilm{add_to_path}'

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
    else:
        q_uids = [args.q_uid]
    
    model, tokenizer = prepare_frozenbilm(args)

    q_uids = np.array_split(q_uids, args.chunks)[args.chunk_index]

    for q_uid in tqdm(q_uids):
    
        if args.force_redo or not os.path.exists(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.csv'):

            q, options, q_dict = args.config['parse_vqa'](qa_data, q_uid)

            shap_out = frozenbilm_shap(args, model, tokenizer, q_uid, q, options, q_dict, qa_answers)

            print(f'Question: {q}')
            for idx, option in enumerate(options):
                print(f'{chr(ord("A") + idx)}: {option}')
            print(f'Correct answer: {options[qa_answers[q_uid]]}')
            print(shap_out)
            shap_out.to_csv(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.csv', index=False)


def frozenbilm_logit_difference(args, model, tokenizer, q_uid, question, options, q_dict, qa_answers):
    with torch.no_grad():
        videos, stride = args.config['load_features'](q_uid, args.config['video_folder'])
        
        video_mask = get_mask(
            torch.tensor(args.frames, dtype=torch.long).unsqueeze(0), videos[0].size(1)
        ).to(device)
        
        if args.dataset == 'hd-epic':
            parse_tags = lambda x: hd_epic_utils.parse_tags(x, stride, q_dict)
        else: 
            parse_tags = None

        x, lengths, names = convert_to_indices([x.size(1) for x in videos], question, options, mode=args.mode)
        x = x[None][None]
        
        fc = lambda x: frozenbilm_forward(model, tokenizer, videos, video_mask, question, options, x, lengths, args.mode, parse_tags)

        logit = fc(x.numpy()[0])[0][qa_answers[q_uid]]

        x[0, 0, args.index] = 0

        logit_hat = fc(x.numpy()[0])[0][qa_answers[q_uid]]

        print(f'Original input: {logit}')
        print(f'Masked input: {logit_hat}')
        print(f'Difference in true logit when masking \"{names[args.index]}\": {logit_hat - logit}')

        del videos
        del video_mask
        torch.cuda.empty_cache()

        return


def compare_logit_values(args):
    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    q, options, q_dict = args.config['parse_vqa'](qa_data, args.q_uid)

    model, tokenizer = prepare_frozenbilm(args)
    frozenbilm_logit_difference(args, model, tokenizer, args.q_uid, q, options, q_dict, qa_answers)

    print(f'Question: {q}')
    for idx, option in enumerate(options):
        print(f'{chr(ord("A") + idx)}: {option}')
    print(f'Correct answer: {options[qa_answers[args.q_uid]]}')    


def frozenbilm_evaluate(args, model, tokenizer, q_uid, question, options, q_dict, masking_direction, masking_logit, masking_mode):
    with torch.no_grad():
        videos, stride = args.config['load_features'](q_uid, args.config['video_folder'])
        
        video_mask = get_mask(
            torch.tensor(args.frames, dtype=torch.long).unsqueeze(0), videos[0].size(1)
        ).to(device)

        if args.dataset == 'hd-epic':
            parse_tags = lambda x: hd_epic_utils.parse_tags(x, stride, q_dict)
        else: 
            parse_tags = None

        x, lengths, names = convert_to_indices([x.size(1) for x in videos], question, options, mode=args.mode)
        x = x[None][None]
        
        if masking_direction != 'nothing':
            add_to_path = ''
            if args.answer_replacement is not None:
                add_to_path += f'_{args.answer_replacement}'

            results_path = f'shap_results/frozenbilm/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/frozenbilm{add_to_path}'            
            x = threshold_indices(x, lengths, results_path, q_uid, args.iterations, args.mode, masking_direction, masking_logit, masking_mode, args.threshold, args.answer_masking)

        fc = lambda x: frozenbilm_forward(model, tokenizer, videos, video_mask, question, options, x, lengths, args.mode, parse_tags, args.force_logit)

        logits = fc(x.numpy()[0])[0]

        pred = logits.argmax().item()

        del videos
        del video_mask
        torch.cuda.empty_cache()

        return pred


def evaluate_model(args):

    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
    else:
        q_uids = [args.q_uid]
    
    model, tokenizer = prepare_frozenbilm(args)

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

        pred = frozenbilm_evaluate(args, model, tokenizer, q_uid, q, options, q_dict, args.masking_direction, args.masking_logit, args.masking_mode)

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

    results_path = f'shap_results/frozenbilm/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/frozenbilm{add_to_path}'    
    
    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
        name = args.subset_path.split(".")[0].split("/")[-1]
    else:
        q_uids = [args.q_uid]
        name = q_uids[0]
    
    model, tokenizer = prepare_frozenbilm(args)

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

        pred = frozenbilm_evaluate(args, model, tokenizer, q_uid, q, options, q_dict, masking_directions[0], None, None)
        results[masking_directions[0]]['preds'].append(pred)

        for masking_direction in masking_directions[1:]:
            if masking_direction in ['positive', 'negative']:
                for masking_logit in masking_logits:
                    for masking_mode in masking_modes:
                        pred = frozenbilm_evaluate(args, model, tokenizer, q_uid, q, options, q_dict, masking_direction, masking_logit, masking_mode)
                        results[masking_direction][masking_logit][masking_mode]['preds'].append(pred)
            else:
                for masking_mode in masking_modes:
                    pred = frozenbilm_evaluate(args, model, tokenizer, q_uid, q, options, q_dict, masking_direction, None, masking_mode)
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