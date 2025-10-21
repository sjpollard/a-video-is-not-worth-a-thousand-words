import numpy as np
import argparse
import pandas as pd
import json
from itertools import accumulate

from matplotlib import pyplot as plt

from utils import convert_to_indices, threshold_indices, split_string
import egoschema_utils
import hd_epic_utils
import mvbench_utils
import lvbench_utils


EGOSCHEMA_CONFIG = {'qa_folder': '<EGOSCHEMA-PATH>',
                    'dataset_fn': egoschema_utils.get_egoschema,
                    'parse_vqa': egoschema_utils.parse_vqa}


HD_EPIC_CONFIG = {'qa_folder': '<HD-EPIC-ANNOTATIONS-PATH>',
                  'dataset_fn': hd_epic_utils.get_hd_epic,
                  'parse_vqa': hd_epic_utils.parse_vqa}


MVBENCH_CONFIG = {'qa_folder': '<MVBENCH_PATH>',
                  'dataset_fn': mvbench_utils.get_mvbench,
                  'parse_vqa': mvbench_utils.parse_vqa}


LVBENCH_CONFIG = {'qa_folder': '<LVBENCH_PATH>',
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
        "--gemini-version",
        dest="gemini_version",
        choices=['gemini-2.5-pro', 'gemini-2.5-flash'],
        default='gemini-2.5-pro',
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
        "--command",
        "-c",
        dest="command",
        choices=['plot', 'metrics', 'distribution', 'print_masking', 'split_metrics'],
        default='plot',
        type=str
    )
    parser.add_argument(
        "--logits",
        "-l",
        dest="logits",
        choices=['all', 'gt'],
        default='all',
        type=str
    )
    parser.add_argument(
        "--occurrences",
        "-o",
        dest="occurrences",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--frames",
        "-f",
        dest="frames",
        default=180,
        type=int,
    )
    return parser.parse_args()


def plot_all_logits(args, results_path, q_uid):
    shap_out = pd.read_csv(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.csv')

    width = 1.0 / len(shap_out.columns[1:])
    fig, ax = plt.subplots(figsize=[(width * len(shap_out)), 4.8])

    x = np.arange(len(shap_out))
    multiplier = 0

    for column in shap_out.columns[1:]:
        shap = shap_out[column]
        offset = width * multiplier
        ax.bar(x + offset, shap, width, label=column)
        multiplier += 1

    ax.set_xticks(x + (((len(shap_out.columns[1:]) - 1) / 2) * width), shap_out['element'])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlim([(-0.5 * width) - width, len(shap_out) - (0.5 * width) + width])
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.pdf')
    plt.close()


def plot_gt_logit(args, results_path, q_uid):
    shap_out = pd.read_csv(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.csv')

    fig, ax = plt.subplots(figsize=[(0.2 * len(shap_out)), 4.8])

    x = np.arange(len(shap_out))
    width = 0.5

    numeric_columns = [x for x in shap_out.columns if x != 'element']

    ground_truth_in_list = ['Ground Truth' in x for x in numeric_columns]
    ground_truth_index = ground_truth_in_list.index(True)

    shap = shap_out[numeric_columns[ground_truth_index]]
    ax.bar(x, shap, label=numeric_columns[ground_truth_index])

    ax.set_xticks(x, shap_out['element'])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlim([(-0.5 * width) - 0.5, len(shap_out) - (0.5 * width)])
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}_gt.pdf')
    plt.close()


def plot_shap_values(args):
    add_to_path = ''
    if args.answer_replacement is not None:
        add_to_path += f'_{args.answer_replacement}'

    results_path = f'shap_results/{args.model}/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/{args.model}'
    
    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
    else:
        q_uids = [args.q_uid]

    for q_uid in q_uids:
        if args.logits == 'all':
            plot_all_logits(args, results_path, q_uid)
        elif args.logits == 'gt':
            plot_gt_logit(args, results_path, q_uid)


def calculate_shap_modality_contrib(shap, lengths):
    contrib = shap.abs()

    v_modality_contrib = np.sum(contrib[:lengths['video 1']])
    q_modality_contrib = np.sum(contrib[lengths['video 1']:lengths['video 1'] + lengths['question']])
    a_modality_contrib = np.sum(contrib[lengths['video 1'] + lengths['question']:sum(lengths.values())])
    modality_contrib = v_modality_contrib + q_modality_contrib + a_modality_contrib

    v_modality_contrib = np.array([v_modality_contrib / modality_contrib]).item()
    q_modality_contrib = np.array([q_modality_contrib / modality_contrib]).item()
    a_modality_contrib = np.array([a_modality_contrib / modality_contrib]).item()   
    
    shap_contrib = pd.DataFrame([np.array([v_modality_contrib, q_modality_contrib, a_modality_contrib])], columns=[np.array(['Modality Contribution', 'Modality Contribution', 'Modality Contribution']),
                                                                            np.array(['V', 'Q', 'A'])])
    return shap_contrib


def calculate_shap_per_feature_contrib(shap, lengths):
    contrib = shap.abs()

    v_per_feature_contrib = np.mean(contrib[:lengths['video 1']])
    q_per_feature_contrib = np.mean(contrib[lengths['video 1']:lengths['video 1'] + lengths['question']])
    a_per_feature_contrib = np.mean(contrib[lengths['video 1'] + lengths['question']:sum(lengths.values())])
    modality_contrib = v_per_feature_contrib + q_per_feature_contrib + a_per_feature_contrib

    v_per_feature_contrib = np.array([v_per_feature_contrib / modality_contrib]).item()
    q_per_feature_contrib = np.array([q_per_feature_contrib / modality_contrib]).item()
    a_per_feature_contrib = np.array([a_per_feature_contrib / modality_contrib]).item()   
    
    shap_contrib = pd.DataFrame([np.array([v_per_feature_contrib, q_per_feature_contrib, a_per_feature_contrib])], columns=[np.array(['Per-Feature Contribution', 'Per-Feature Contribution', 'Per-Feature Contribution']),
                                                                            np.array(['V', 'Q', 'A'])])
    return shap_contrib


def calculate_shap_ratio(shap, lengths):
    v_ratio = sum(shap[:lengths['video 1']] >= 0) / lengths['video 1']
    q_ratio = sum(shap[lengths['video 1']:lengths['video 1'] + lengths['question']] >= 0) / lengths['question']
    a_ratio = sum(shap[lengths['video 1'] + lengths['question']:sum(lengths.values())] >= 0) / sum(lengths.values())

    shap_ratio = pd.DataFrame([np.array([v_ratio, q_ratio, a_ratio])], columns=[np.array(['Ratio', 'Ratio', 'Ratio']),
                                                                                np.array(['V', 'Q', 'A'])])
    return shap_ratio


def calculate_shap_mean(shap, lengths):
    v_mean = np.mean(shap[:lengths['video 1']])
    q_mean = np.mean(shap[lengths['video 1']:lengths['video 1'] + lengths['question']])
    a_mean = np.mean(shap[lengths['video 1'] + lengths['question']:sum(lengths.values())])

    shap_mean = pd.DataFrame([np.array([v_mean, q_mean, a_mean])], columns=[np.array(['Mean', 'Mean', 'Mean']),
                                                                            np.array(['V', 'Q', 'A'])])
    return shap_mean


def calculate_shap_var(shap, lengths):
    v_var = np.var(shap[:lengths['video 1']])
    q_var = np.var(shap[lengths['video 1']:lengths['video 1'] + lengths['question']])
    a_var = np.var(shap[lengths['video 1'] + lengths['question']:sum(lengths.values())])

    shap_var = pd.DataFrame([np.array([v_var, q_var, a_var])], columns=[np.array(['Var', 'Var', 'Var']),
                                                                        np.array(['V', 'Q', 'A'])])
    return shap_var


def calculate_shap_metrics(args):
    add_to_path = ''
    if args.answer_replacement is not None:
        add_to_path += f'_{args.answer_replacement}'

    results_path = f'shap_results/{args.model}/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/{args.model}'

    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
        name = args.subset_path.split(".")[0].split("/")[-1]
    else:
        q_uids = [args.q_uid]

    shap_metrics_list = []

    modality_dominance = np.zeros((3, 3))
    per_feature_modality_dominance = np.zeros((3, 3))

    modalities = ['V', 'Q', 'A']

    for q_uid in q_uids:
        shap_out = pd.read_csv(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.csv', keep_default_na=False)

        num_frames = len(shap_out[shap_out['element'].str.contains('frame_\d+', regex=True)])

        numeric_columns = [x for x in shap_out.columns if x != 'element']
        shap_out[numeric_columns] = shap_out[numeric_columns] / shap_out[numeric_columns].abs().max()

        q, options, q_dict = args.config['parse_vqa'](qa_data, q_uid)

        _, lengths, _ = convert_to_indices([num_frames], q, options, args.mode)

        ground_truth_in_list = ['Ground Truth' in x for x in numeric_columns]
        ground_truth_index = ground_truth_in_list.index(True)
        ground_truth_shap = shap_out[numeric_columns[ground_truth_index]]
        ground_truth_shap_modality_contrib = calculate_shap_modality_contrib(ground_truth_shap, lengths)
        ground_truth_shap_per_feature_contrib = calculate_shap_per_feature_contrib(ground_truth_shap, lengths)
        ground_truth_shap_ratio = calculate_shap_ratio(ground_truth_shap, lengths)
        ground_truth_shap_mean = calculate_shap_mean(ground_truth_shap, lengths)
        ground_truth_shap_var = calculate_shap_var(ground_truth_shap, lengths)
        modality_dominance[0][modalities.index(ground_truth_shap_modality_contrib['Modality Contribution'].idxmax(axis=1).item())] += 1
        per_feature_modality_dominance[0][modalities.index(ground_truth_shap_per_feature_contrib['Per-Feature Contribution'].idxmax(axis=1).item())] += 1

        false_in_list = ['Ground Truth' not in x for x in numeric_columns]
        false_indices = [i for i, x in enumerate(false_in_list) if x == True]
        false_shap = shap_out[[numeric_columns[x] for x in false_indices]].mean(axis='columns')
        false_shap_modality_contrib = calculate_shap_modality_contrib(false_shap, lengths)
        false_shap_per_feature_contrib = calculate_shap_per_feature_contrib(false_shap, lengths)
        false_shap_ratio = calculate_shap_ratio(false_shap, lengths)
        false_shap_mean = calculate_shap_mean(false_shap, lengths)
        false_shap_var = calculate_shap_var(false_shap, lengths)
        modality_dominance[1][modalities.index(false_shap_modality_contrib['Modality Contribution'].idxmax(axis=1).item())] += 1
        per_feature_modality_dominance[1][modalities.index(false_shap_per_feature_contrib['Per-Feature Contribution'].idxmax(axis=1).item())] += 1

        all_shap = shap_out[numeric_columns].mean(axis='columns')
        all_shap_modality_contrib = calculate_shap_modality_contrib(all_shap, lengths)
        all_shap_per_feature_contrib = calculate_shap_per_feature_contrib(all_shap, lengths)
        all_shap_ratio = calculate_shap_ratio(all_shap, lengths)
        all_shap_mean = calculate_shap_mean(all_shap, lengths)
        all_shap_var = calculate_shap_var(all_shap, lengths)
        modality_dominance[2][modalities.index(all_shap_modality_contrib['Modality Contribution'].idxmax(axis=1).item())] += 1
        per_feature_modality_dominance[2][modalities.index(all_shap_per_feature_contrib['Per-Feature Contribution'].idxmax(axis=1).item())] += 1

        shap_modality_contrib = pd.concat([ground_truth_shap_modality_contrib, false_shap_modality_contrib, all_shap_modality_contrib], axis=0)
        shap_per_feature_contrib = pd.concat([ground_truth_shap_per_feature_contrib, false_shap_per_feature_contrib, all_shap_per_feature_contrib], axis=0)
        shap_ratio = pd.concat([ground_truth_shap_ratio, false_shap_ratio, all_shap_ratio], axis=0)
        shap_mean = pd.concat([ground_truth_shap_mean, false_shap_mean, all_shap_mean], axis=0)
        shap_var = pd.concat([ground_truth_shap_var, false_shap_var, all_shap_var], axis=0)
        shap_metrics = pd.concat([shap_modality_contrib, shap_per_feature_contrib, shap_ratio, shap_mean, shap_var], axis=1)

        index = [[f'{q_uid}'] * 3,
                 ['Ground Truth', 'False', 'All']]

        shap_metrics.index = index

        shap_metrics_list.append(shap_metrics)

        if args.subset_path is None:
            shap_metrics.to_csv(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}_metrics.csv', index=True)

    modality_dominance = modality_dominance / len(q_uids)
    per_feature_modality_dominance = per_feature_modality_dominance / len(q_uids)

    shap_modality_dominance = pd.DataFrame(modality_dominance, index=['Ground Truth', 'False', 'All'], columns=[['Modality Dominance', 'Modality Dominance', 'Modality Dominance'],
                                                                                                                ['V', 'Q', 'A']])

    shap_per_feature_modality_dominance = pd.DataFrame(per_feature_modality_dominance, index=['Ground Truth', 'False', 'All'], columns=[['Per-Feature Dominance', 'Per-Feature Dominance', 'Per-Feature Dominance'],
                                                                                                                ['V', 'Q', 'A']])

    aggregate_shap_metrics = pd.concat(shap_metrics_list, axis=0)

    subset_shap_metrics = aggregate_shap_metrics.groupby(level=1, sort=False).mean()
    subset_shap_metrics = pd.concat([subset_shap_metrics, shap_modality_dominance, shap_per_feature_modality_dominance], axis=1) 

    if args.subset_path is not None:
        aggregate_shap_metrics.to_csv(f'{results_path}/{name}_{args.iterations}_{args.mode}_q_uid_metrics.csv', index=True)
        subset_shap_metrics.to_csv(f'{results_path}/{name}_{args.iterations}_{args.mode}_metrics.csv', index=True)


def calculate_shap_distribution(args):
    add_to_path = ''
    if args.answer_replacement is not None:
        add_to_path += f'_{args.answer_replacement}'

    results_path = f'shap_results/{args.model}/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/{args.model}'

    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
        name = args.subset_path.split(".")[0].split("/")[-1]
    else:
        q_uids = [args.q_uid]

    distribution = {}

    for q_uid in q_uids:
        shap_out = pd.read_csv(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.csv')

        num_frames = len(shap_out[shap_out['element'].str.contains('frame_\d+', regex=True)])

        numeric_columns = [x for x in shap_out.columns if x != 'element']
        shap_out[numeric_columns] = shap_out[numeric_columns] / shap_out[numeric_columns].abs().max()

        q, options, q_dict = args.config['parse_vqa'](qa_data, q_uid)

        _, lengths, names = convert_to_indices([num_frames], q, options, args.mode)

        ground_truth_in_list = ['Ground Truth' in x for x in numeric_columns]
        ground_truth_index = ground_truth_in_list.index(True)
        ground_truth_shap = shap_out.drop(columns=[x for x in numeric_columns if x != numeric_columns[ground_truth_index]])

        unique_words = set(names)

        for word in unique_words:
            word_shap = ground_truth_shap[ground_truth_shap['element'] == word].drop(columns='element')
            if word not in distribution.keys():
                distribution.update({word: {'sum': word_shap.sum().item(), 'num': len(word_shap)}})
            else:
                distribution[word]['sum'] += word_shap.sum().item()
                distribution[word]['num'] += len(word_shap)

    for word in list(distribution.keys()):
        distribution[word].update({'mean': distribution[word]['sum'] / distribution[word]['num']})
        if distribution[word]['num'] < args.occurrences:
            distribution.pop(word, None)

    sorted_words = sorted(distribution, key=lambda x: distribution[x]['mean'], reverse=True)

    word_shap = pd.DataFrame({'element': sorted_words, 'mean': [distribution[x]['mean'] for x in sorted_words], 'num': [distribution[x]['num'] for x in sorted_words]})
    
    if args.subset_path is not None:
        word_shap.to_csv(f'{results_path}/{name}_{args.iterations}_{args.mode}_word_shap.csv', index=False)


def print_masking(args):
    add_to_path = ''
    if args.answer_replacement is not None:
        add_to_path += f'_{args.answer_replacement}'

    results_path = f'shap_results/{args.model}/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/{args.model}'

    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
        name = args.subset_path.split(".")[0].split("/")[-1]
    else:
        q_uids = [args.q_uid]

    masking_direction = 'negative'
    masking_logit = 'ground_truth'
    masking_mode = 'joint'

    for q_uid in q_uids:
        shap_out = pd.read_csv(f'{results_path}/{q_uid}_{args.iterations}_{args.mode}.csv')

        num_frames = len(shap_out[shap_out['element'].str.contains('frame_\d+', regex=True)])

        numeric_columns = [x for x in shap_out.columns if x != 'element']
        shap_out[numeric_columns] = shap_out[numeric_columns] / shap_out[numeric_columns].abs().max()

        q, options, q_dict = args.config['parse_vqa'](qa_data, q_uid)

        x, lengths, names = convert_to_indices([num_frames], q, options, args.mode)

        results_path = f'shap_results/{args.model}/{args.subset_path.split(".")[0].split("/")[-1]}' if args.subset_path is not None else f'shap_results/{args.model}'
        indices = threshold_indices(x, lengths, results_path, q_uid, args.iterations, args.mode, masking_direction, masking_logit, masking_mode, 0)

        multi_modal_indices = {}
        intervals = list(zip([0] + list(accumulate(list(lengths.values())[:-1])), accumulate(list(lengths.values()))))
        for i, key in enumerate(lengths.keys()):
            start, end = intervals[i]
            multi_modal_indices.update({key: indices[start:end]})

        question = ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(multi_modal_indices['question'], split_string(q)))))
        options = list(map(lambda y: ' '.join(list(map(lambda x: x[1] if x[0] == 1 else ' ', zip(y[1], split_string(y[0]))))), zip(options, list(map(lambda x: multi_modal_indices[x], [f'option {i}' for i in range(1, len(options) + 1)])))))

        print(f'Question: {question}')
        for idx, option in enumerate(options):
            print(f'{chr(ord("A") + idx)}: {option}')
        print(f'Correct answer: {options[qa_answers[q_uid]]}\n')
        

def split_metrics(args):
    add_to_path = ''
    if args.answer_replacement is not None:
        add_to_path += f'_{args.answer_replacement}'

    results_path = f'shap_results/{args.model}/{args.subset_path.split(".")[0].split("/")[-1]}{add_to_path}' if args.subset_path is not None else f'shap_results/{args.model}'

    qa_data, qa_answers = args.config['dataset_fn'](args.config['qa_folder'], args.answer_replacement)

    if args.subset_path is not None:
        q_uids = pd.read_csv(args.subset_path)['q_uid'].to_list()
        name = args.subset_path.split(".")[0].split("/")[-1]
    else:
        q_uids = [args.q_uid]

    aggregate_shap_metrics = pd.read_csv(f'{results_path}/{name}_{args.iterations}_{args.mode}_q_uid_metrics.csv', index_col=[0, 1], header=[0, 1])

    results_f = open(f'{results_path}/{name}_{args.iterations}_{args.mode}_benchmark.json')
    results = json.load(results_f)

    preds = results['nothing']['preds']

    correct_mask = []
    incorrect_mask = []

    for i, q_uid in enumerate(q_uids):
        correct = preds[i] == qa_answers[q_uid]
        correct_mask.extend([correct] * 3)
        incorrect_mask.extend([not correct] * 3)

    print(aggregate_shap_metrics[correct_mask].groupby(level=1, sort=False).mean())
    print(aggregate_shap_metrics[incorrect_mask].groupby(level=1, sort=False).mean())


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

    if args.command == 'plot':
        plot_shap_values(args)
    elif args.command == 'metrics' and args.mode == 'joint':
        calculate_shap_metrics(args)
    elif args.command == 'distribution':
        calculate_shap_distribution(args)
    elif args.command == 'print_masking':
        print_masking(args)
    elif args.command == 'split_metrics':
        split_metrics(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)