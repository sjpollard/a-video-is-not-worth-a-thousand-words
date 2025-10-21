import glob
from math import sqrt
import matplotlib
import pandas as pd
import itertools
import json
import numpy as np
import colorsys
import subprocess
import re
from matplotlib import pyplot as plt
from PIL import Image, ImageColor
from wordcloud import WordCloud
from sklearn.metrics import mean_squared_error

from utils import convert_to_indices, split_string, get_gemini_frame_indices, get_influential_frame_indices
from run_shap_metrics import calculate_shap_modality_contrib, calculate_shap_per_feature_contrib, calculate_shap_ratio

import egoschema_utils
import hd_epic_utils
import mvbench_utils
import lvbench_utils


EGOSCHEMA_PATH = '<EGOSCHEMA-PATH>',

HD_EPIC_PATH = '<HD-EPIC-ANNOTATIONS-PATH>',

MVBENCH_PATH = '<MVBENCH_PATH>',

LVBENCH_PATH = '<LVBENCH_PATH>'


def get_dataset_specifics(dataset):
    if dataset == 'egoschema':
        return egoschema_utils.get_egoschema, egoschema_utils.parse_vqa, '<EGOSCHEMA-PATH>', egoschema_utils.get_video_paths
    elif dataset == 'hd-epic':
        return hd_epic_utils.get_hd_epic, hd_epic_utils.parse_vqa, '<HD-EPIC-ANNOTATIONS-PATH>', hd_epic_utils.get_video_paths
    elif dataset == 'mvbench':
        return mvbench_utils.get_mvbench, mvbench_utils.parse_vqa, '<MVBENCH_PATH>', mvbench_utils.get_video_paths
    elif dataset == 'lvbench':
        return lvbench_utils.get_lvbench, lvbench_utils.parse_vqa, '<LVBENCH_PATH>', lvbench_utils.get_video_paths


def get_dataset_video_averages(video_folder, dataset, subset_path):
    get_dataset, parse_vqa, dataset_path, get_video_paths = get_dataset_specifics(dataset)
    q_uids = pd.read_csv(subset_path)['q_uid'].to_list()

    video_paths = []

    for q_uid in q_uids:
        qa_data, qa_answers = get_dataset(dataset_path)

        q, options, q_dict = parse_vqa(qa_data, q_uid)

        video_paths.extend(get_video_paths(q_uid, q_dict, video_folder))

    video_paths = list(set(video_paths))
    
    lengths = 0

    for video_path in video_paths:
        if re.search(r'\.mp4|\.webm', video_path):
            length = float(subprocess.run(f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video_path}', shell=True, check=True, stdout=subprocess.PIPE).stdout)
            lengths += length
        else:
            lengths += 3.0 / len(glob.glob(f'{video_path}/*'))

    return lengths / len(video_paths)


def get_dataset_text_averages(dataset, subset_path):
    get_dataset, parse_vqa, dataset_path, _ = get_dataset_specifics(dataset)
    qa_data, qa_answers = get_dataset(dataset_path)

    q_uids = pd.read_csv(subset_path)['q_uid'].to_list()

    question_lengths = 0
    answer_lengths = 0

    for q_uid in q_uids:
        q, options, _ = parse_vqa(qa_data, q_uid)
        question_lengths += len(q.split(' '))
        answer_lengths += sum(list(map(lambda x: len(x.split(' ')), options)))
        print(question_lengths, answer_lengths)
    average_question_length = question_lengths / len(q_uids)
    average_answer_length = answer_lengths / len(q_uids)

    return average_question_length, average_answer_length, len(q_uids)


def is_word_whitelisted(word):
    if 'frame_' in word:
        return False
    if word.lower() == 'c':
        return False
    if word.lower() == "c's":
        return False
    return True


def get_single_color_func(color):
    old_r, old_g, old_b = ImageColor.getrgb(color)
    rgb_max = 255.
    h, s, v = colorsys.rgb_to_hsv(old_r / rgb_max, old_g / rgb_max,
                                  old_b / rgb_max)
    
    def single_color_func(word=None, font_size=None, position=None,
                          orientation=None, font_path=None, random_state=None):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return 'rgb({:.0f}, {:.0f}, {:.0f})'.format(r * rgb_max, g * rgb_max,
                                                    b * rgb_max)
    return single_color_func


#Modified from https://amueller.github.io/word_cloud/auto_examples/colored_by_group.html
class GroupedColorFunc(object):
    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


def get_vocab_freq(word_shap):
    clean_df = word_shap[word_shap.element.apply(is_word_whitelisted)]
    vocab_freq = ' '.join(clean_df.apply(lambda x: ' '.join([x.element for i in range(x.num)]), axis=1).to_list())
    return vocab_freq.lower()


def plot_wordcloud(model, dataset, iterations=5000, mode='joint'):
    results_path = f'shap_results/{model}/{dataset}'

    word_shap = pd.read_csv(f'{results_path}/{dataset}_{iterations}_{mode}_word_shap.csv')

    vocab_freq = get_vocab_freq(word_shap)

    cmap = matplotlib.colormaps['RdBu']
    normalize = matplotlib.colors.SymLogNorm(vmin=-1, vmax=1, linthresh=0.001, linscale=0.001, base=10)

    col_dict = {}
    for _, row in word_shap.iterrows():
        if is_word_whitelisted(row.element):
            hex_col = matplotlib.colors.rgb2hex(cmap(normalize(row['mean'])))
            if hex_col not in col_dict:
                col_dict[hex_col] = []
            col_dict[hex_col].append(row.element.lower())

    grouped_colour_func = GroupedColorFunc(col_dict, 'black')

    wc = WordCloud(random_state=None, width=800, height=500, mode='RGBA', color_func=grouped_colour_func, collocations=False, background_color='white').generate(vocab_freq)

    wc.to_file(f'shap_results/{model}/{dataset}/word_cloud.png')


def calculate_spearmans_correlation(q_uid, frames, iterations, mode, results_path, ranking_path):
    if frames > 1:
        shapley_ranking = pd.Series(get_influential_frame_indices(results_path=results_path, q_uid=q_uid, frames=frames, iterations=iterations, mode=mode, ordering='ranking'))
        gemini_ranking = get_gemini_frame_indices(ranking_path=ranking_path, q_uid=q_uid, frames=frames)

        if gemini_ranking == None:
            spearmans_correlation = None
        else:
            gemini_ranking = pd.Series(gemini_ranking)
            spearmans_correlation = gemini_ranking.corr(shapley_ranking, method='spearman')
    else:
        spearmans_correlation = None

    return spearmans_correlation


def plot_rank_violins(iterations=5000, mode='joint', gemini_version='gemini-2.5-pro', answer_replacement=None):
    add_to_path = ''
    if answer_replacement is not None:
        add_to_path += f'_{answer_replacement}'

    models=['llava_video', 'longva', 'videollama3']
    datasets=['egoschema', 'hd-epic', 'mvbench', 'lvbench']

    model_dict = {'llava_video': f'LLaVA-Video', 
                  'longva': f'LongVA', 
                  'videollama3': f'VideoLLaMA3'}
    
    dataset_dict = {'egoschema': f'EgoSchema', 
                    'hd-epic': f'HD-EPIC', 
                    'mvbench': f'MVBench', 
                    'lvbench': f'LVBench'}

    fig, axs = plt.subplots(ncols=len(models), figsize=(8, 4.5), sharex=True, sharey=True)

    colours = matplotlib.color_sequences['tab10']

    for j, model in enumerate(models):

        all_correlations_list = []

        for i, dataset in enumerate(datasets):
            q_uids = pd.read_csv(f'subsets/{dataset}.csv')['q_uid'].to_list()

            correlations_list = []

            for q_uid in q_uids:
                results_path = f'shap_results/{model}/{dataset}{add_to_path}'
                ranking_path = f'ranks/{gemini_version}/{model}/{dataset}{add_to_path}'

                shap_out = pd.read_csv(f'{results_path}/{q_uid}_{iterations}_{mode}.csv', keep_default_na=False)

                num_frames = len(shap_out[shap_out['element'].str.contains('frame_\d+', regex=True)])

                spearmans_correlation = calculate_spearmans_correlation(q_uid=q_uid, frames=num_frames, iterations=iterations, mode=mode, results_path=results_path, ranking_path=ranking_path)

                if spearmans_correlation is not None:
                    correlations_list.append(spearmans_correlation)

            all_correlations_list.append(correlations_list)

        violin = axs[j].violinplot(dataset=all_correlations_list, showextrema=False, vert=False)
        
        axs[j].set_yticks(np.arange(1, len(datasets) + 1), labels=[dataset_dict[x] for x in datasets])
        axs[j].set_xlim(-1, 1)

        axs[j].set_title(model_dict[models[j]])

    fig.savefig('ranks/rank_violin_plots.png', dpi=400, bbox_inches='tight')


def plot_violins(iterations=5000, mode='joint'):
    models=['frozenbilm', 'internvideo', 'videollama2', 'llava_video', 'longva', 'videollama3']
    datasets=['egoschema', 'hd-epic', 'mvbench', 'lvbench']

    px = 1/plt.rcParams['figure.dpi']

    model_dict = {'frozenbilm': f'FBLM', 
                  'internvideo': f'IV', 
                  'videollama2': f'VL2', 
                  'llava_video': f'L-V', 
                  'longva': f'LVA', 
                  'videollama3': f'VL3'}
    
    dataset_dict = {'egoschema': f'EgoSchema', 
                    'hd-epic': f'HD-EPIC', 
                    'mvbench': f'MVBench', 
                    'lvbench': f'LVBench'}

    fig, axs = plt.subplots(nrows=len(models), ncols=len(datasets), figsize=(5.5, 8.5), sharex=True, sharey=True)

    colours = matplotlib.color_sequences['Dark2']
    labels = ['V', 'Q', 'A']

    for j, model in enumerate(models):
        for i, dataset in enumerate(datasets):
            results_path = f'shap_results/{model}/{dataset}'

            q_uids = pd.read_csv(f'subsets/{dataset}.csv')['q_uid'].to_list()

            get_dataset, parse_vqa, dataset_path, _ = get_dataset_specifics(dataset)

            qa_data, qa_answers = get_dataset(dataset_path)

            v_shap_list = []
            q_shap_list = []
            a_shap_list = []

            for q_uid in q_uids:
                shap_out = pd.read_csv(f'{results_path}/{q_uid}_{iterations}_{mode}.csv')

                num_frames = len(shap_out[shap_out['element'].str.contains('frame_\d+', regex=True)])

                q, options, q_dict = parse_vqa(qa_data, q_uid)

                _, lengths, _ = convert_to_indices([num_frames], q, options, mode)

                numeric_columns = [x for x in shap_out.columns if x != 'element']
                shap_out[numeric_columns] = shap_out[numeric_columns] / shap_out[numeric_columns].abs().max()

                ground_truth_in_list = ['Ground Truth' in x for x in numeric_columns]
                ground_truth_index = ground_truth_in_list.index(True)
                ground_truth_shap = shap_out[numeric_columns[ground_truth_index]]

                v_shap = ground_truth_shap[:lengths['video 1']]
                q_shap = ground_truth_shap[lengths['video 1']:lengths['video 1'] + lengths['question']]
                a_shap = ground_truth_shap[lengths['video 1'] + lengths['question']:sum(lengths.values())]

                v_shap_list.extend(v_shap.to_list())
                q_shap_list.extend(q_shap.to_list())
                a_shap_list.extend(a_shap.to_list())

            violin = axs[j][i].violinplot(dataset=[v_shap_list, q_shap_list, a_shap_list], showextrema=False)

            for body, colour in zip(violin['bodies'], colours[:len(violin['bodies'])]):
                body.set_facecolor(colour)
                body.set_edgecolor(colour)
                body.set_alpha(1)
            
            axs[j][i].set_xticks(np.arange(1, len(labels) + 1), labels=labels)
            axs[j][i].set_xlim(0.25, len(labels) + 0.75)

            axs[0][i].set_title(dataset_dict[datasets[i]])

        axs[j][0].set_ylabel(model_dict[models[j]])

    plt.subplots_adjust(bottom=0.05, top=0.95)

    fig.savefig('shap_results/violin_plots.png', dpi=400)


def plot_attributions(model, subset_path, iterations=5000, mode='joint'):
    results_path = f'shap_results/{model}/{subset_path.split(".")[0].split("/")[-1]}'

    q_uids = pd.read_csv(subset_path)['q_uid'].to_list()

    name = subset_path.split(".")[0].split("/")[-1]

    shap_list = []

    num_frames_list = []

    for q_uid in q_uids:
        shap_out = pd.read_csv(f'{results_path}/{q_uid}_{iterations}_{mode}.csv')

        numeric_columns = [x for x in shap_out.columns if x != 'element']
        shap_out[numeric_columns] = shap_out[numeric_columns] / shap_out[numeric_columns].abs().max()

        num_frames = len(shap_out[shap_out['element'].str.contains('frame_\d+', regex=True)])
        num_frames_list.append(num_frames)

        ground_truth_in_list = ['Ground Truth' in x for x in numeric_columns]
        ground_truth_index = ground_truth_in_list.index(True)
        ground_truth_shap = shap_out[numeric_columns[ground_truth_index]]

        ground_truth_shap.name = q_uid

        shap_list.append(ground_truth_shap)

    shap_grid = pd.concat(shap_list, axis=1).transpose()

    shap_grid = shap_grid.truncate(after=max(num_frames_list) + 19, axis=1)

    px = 1/plt.rcParams['figure.dpi']

    width = len(shap_grid.columns) * px
    height = len(q_uids) * px

    fig, ax = plt.subplots(figsize=(width, height))

    ax = plt.gca()

    ax.set_axis_off()

    cmap = matplotlib.colormaps['RdBu']
    normalize = matplotlib.colors.SymLogNorm(vmin=-1, vmax=1, linthresh=0.001, linscale=0.001, base=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.imshow(shap_grid.to_numpy(), cmap=cmap, norm=normalize)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.savefig(f'{results_path}/{name}_heatmap.png')


def plot_shapley_visualisation(model, dataset, q_uid, tune_font=0, gt_lines=None, iterations=5000, mode='joint'):
    shap_out = pd.read_csv(f'shap_results/{model}/{dataset}/{q_uid}_{iterations}_{mode}.csv')
        
    numeric_columns = [x for x in shap_out.columns if x != 'element']
    shap_out[numeric_columns] = shap_out[numeric_columns] / shap_out[numeric_columns].abs().max()

    ground_truth_in_list = ['Ground Truth' in x for x in numeric_columns]
    ground_truth_index = ground_truth_in_list.index(True)
    ground_truth_shap = shap_out[numeric_columns[ground_truth_index]]

    num_frames = len(shap_out[shap_out['element'].str.contains('frame_\d+', regex=True)])

    frame_paths = glob.glob(f'frames/{model}/{dataset}/{q_uid}/frame_*')
    get_frame_index_from_path = lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0])
    frame_paths = sorted(frame_paths, key=get_frame_index_from_path)

    frames = []
    frame_indices = []

    for frame_path in frame_paths:
        frame_index = get_frame_index_from_path(frame_path)
        frame_indices.append(frame_index)
        frames.append(Image.open(frame_path).resize((336, 336)))

    get_dataset, parse_vqa, dataset_path, _ = get_dataset_specifics(dataset)

    qa_data, qa_answers = get_dataset(dataset_path)

    q, options, q_dict = parse_vqa(qa_data, q_uid)

    _, lengths, _ = convert_to_indices([num_frames], q, options, mode)

    per_feature_contribution = calculate_shap_per_feature_contrib(ground_truth_shap, lengths)
    v_per_feature_contrib = per_feature_contribution['Per-Feature Contribution']['V'].item()
    q_per_feature_contrib = per_feature_contribution['Per-Feature Contribution']['Q'].item()
    a_per_feature_contrib = per_feature_contribution['Per-Feature Contribution']['A'].item()

    text =  f'{q} ' + ' '.join(options)

    mosaic = [['metrics'] + [f'frame_{x}' for x in frame_indices[:len(frame_indices) // 2]] + ['colorbar'],
              ['metrics'] + [f'frame_{x}' for x in frame_indices[len(frame_indices) // 2:]] + ['colorbar'],
              ['metrics'] + ['text'] * (len(frame_indices) // 2) + ['colorbar']]

    metrics_ratio = 0.015
    colorbar_ratio = 0.015
    height_ratios = [0.25, 0.25, 0.5]

    px = 1/plt.rcParams['figure.dpi']

    video_width = 224 * ((len(frame_indices)) // 2)
    width = (video_width + (video_width * metrics_ratio + video_width * colorbar_ratio)) * px
    height = 224 * 4 * px

    width_ratios = [metrics_ratio] + [(1 - (colorbar_ratio + metrics_ratio)) / (len(frames) // 2)] * (len(frames) // 2) + [colorbar_ratio]
    fig, axs = plt.subplot_mosaic(mosaic=mosaic, height_ratios=height_ratios, width_ratios=width_ratios, figsize=(width, height), subplot_kw={'xticks': [], 'yticks': []})

    fig_size = fig.get_size_inches()

    cmap = matplotlib.colormaps['RdBu']
    normalize = matplotlib.colors.SymLogNorm(vmin=-1, vmax=1, linthresh=0.001, linscale=0.001, base=10)

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap), cax=axs['colorbar'])
    cbar.set_ticks([1, 0.1, 0.01, 0, -0.01, -0.1, -1], labels=[1, 0.1, 0.01, 0, -0.01, -0.1, -1], fontsize=16, rotation=270)

    for frame, frame_index in zip(frames, frame_indices):
        shap = ground_truth_shap.iloc[frame_index]
        axs[f'frame_{frame_index}'].imshow(frame)
        for spine in axs[f'frame_{frame_index}'].spines.values():
            spine.set_edgecolor(cmap(normalize(shap)))
            spine.set_linewidth(10)
    
    for spine in axs['text'].spines.values():
        spine.set_visible(False)

    elements = split_string(text)

    newline_indices = list(itertools.accumulate([lengths[f'option {x + 1}'] for x in range(len(options) - 1)], initial=lengths['question']))

    fontsize = (0.5 * 72 * (0.6 * fig_size[1])) / sqrt(len(elements)) + tune_font

    gt_x_start = 0.01
    x_start = 0.01 if gt_lines is None else 0.05
    y_start = 0.9
    line_height = fontsize / (72 * 3)
    max_width = 0.95

    x, y = x_start, y_start

    for i, word in enumerate(elements):
        shap = ground_truth_shap.iloc[num_frames + i].item()
        colour = 'white' if abs(shap) > 0.15 else 'black'
        t = axs['text'].text(x, y, word, color=colour, fontsize=fontsize, va='top', ha='left',
                    bbox=dict(facecolor=cmap(normalize(shap)), edgecolor='none', pad=1.5, alpha=1.0))

        renderer = fig.canvas.get_renderer()
        bbox = t.get_window_extent(renderer=renderer)
        inv = axs['text'].transData.inverted()
        bbox_data = inv.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
        word_width = bbox_data[1][0] - bbox_data[0][0]
        word_end_x = bbox_data[1][0]

        x += word_width

        if word_end_x + word_width > max_width or i + 1 in newline_indices:
            x = x_start
            y -= line_height
            if i + 1 == newline_indices[0]:
                y -= line_height

    if gt_lines is not None:
        axs['text'].text(gt_x_start, y_start - gt_lines * line_height, 'GT:', color='black', fontsize=fontsize, va='top', ha='left')

    video_metric_string = '$PFC_V = $' + f'{v_per_feature_contrib:.2f}'
    question_metric_string = '$PFC_Q = $' + f'{q_per_feature_contrib:.2f}'
    answer_metric_string = '$PFC_A = $' + f'{a_per_feature_contrib:.2f}'

    axs['metrics'].text(0.5, 0.75, s=video_metric_string, fontsize=20, rotation='vertical', horizontalalignment='center', verticalalignment='bottom')
    axs['metrics'].text(0.5, 0.5, s=question_metric_string, fontsize=20, rotation='vertical', horizontalalignment='center', verticalalignment='center')
    axs['metrics'].text(0.5, 0.25, s=answer_metric_string, fontsize=20, rotation='vertical', horizontalalignment='center', verticalalignment='top')

    for spine in axs['metrics'].spines.values():
        spine.set_visible(False)

    aspect_ratio = fig_size[0] / fig_size[1]
    edge_gap = 0.025
    gap = 0.05
    plt.subplots_adjust(left=edge_gap, right=1 - edge_gap, bottom=edge_gap * aspect_ratio, top=1 - (edge_gap * aspect_ratio), wspace=gap, hspace=gap * aspect_ratio)
    
    output_path = f'frames/{model}/{dataset}/{q_uid}/shapley_visualisation.png'

    fig.savefig(output_path, dpi=200)


def plot_shapley_error(model, q_uid):
    filepaths = glob.glob(f'shap_results/{model}/{q_uid}*.csv')
    filepaths.sort(key = lambda x: int(x.split(q_uid)[1].split('_')[1]))
    
    values_list = []
    iterations_list = []

    for filepath in filepaths:
        iterations = int(filepath.split('_')[2])
        iterations_list.append(iterations)

        shap_out = pd.read_csv(filepath)
        numeric_columns = [x for x in shap_out.columns if x != 'element']

        ground_truth_in_list = ['Ground Truth' in x for x in numeric_columns]
        ground_truth_index = ground_truth_in_list.index(True)
        ground_truth_shap = shap_out[numeric_columns[ground_truth_index]]
        values_list.append(ground_truth_shap.to_numpy())

    errors = []

    for values in values_list:
        errors.append(mean_squared_error(values_list[-1], values))

    print(errors)

    fig, ax = plt.subplots()

    x = np.arange(len(iterations_list))

    ax.plot(x, errors, marker='x')

    ax.set_xticks(x, iterations_list)

    ax.set_ylabel('MSE')

    ax.set_xlabel('Iterations')

    fig.savefig(f'shap_results/{model}/{q_uid}_errors.png', dpi=400, bbox_inches='tight')


def plot_answer_replacement(iterations=5000, mode='joint'):
    models = ['videollama2', 'llava_video', 'longva', 'videollama3']
    datasets = ['egoschema', 'hd-epic', 'lvbench']
    answer_replacement_types = ['easy', 'none', 'hard_5', 'hard_10', 'hard_15', 'hard_20']
    metrics = ['$PFC_V$', '$PFC_Q$', '$PFC_A$', 'Accuracy']

    px = 1/plt.rcParams['figure.dpi']

    model_dict = {'videollama2': f'VideoLLaMA2', 
                  'llava_video': f'LLaVA-Video', 
                  'longva': f'LongVA', 
                  'videollama3': f'VideoLLaMA3'}
    
    dataset_dict = {'egoschema': f'EgoSchema', 
                    'hd-epic': f'HD-EPIC', 
                    'lvbench': f'LVBench'}
    
    answer_replacement_dict = {'easy': 'Easy',
                               'none': 'Default',
                               'hard_5': 'New-5',
                               'hard_10': 'New-10',
                               'hard_15': 'New-15',
                               'hard_20': 'New-20'}

    colours = matplotlib.color_sequences['tab10']
    linestyles = ['solid', 'dotted', 'dashed']
    markers = ['x', '.', '^', 's']

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), layout='tight', sharex=True)

    all_lines = []

    for k, model in enumerate(models):

        lines = []

        for j, dataset in enumerate(datasets):

            per_feature_contributions = []
            accuracies = []

            for answer_replacement in answer_replacement_types:
                results_path = f'shap_results/{model}/{dataset}_{answer_replacement}' if answer_replacement != 'none' else f'shap_results/{model}/{dataset}'
                metrics_out = pd.read_csv(f'{results_path}/{dataset}_{iterations}_{mode}_metrics.csv', index_col=0, header=[0, 1])['Per-Feature Contribution'].loc['Ground Truth']
                per_feature_contributions.append(metrics_out.to_numpy())

                results_f = open(f'shap_results/{model}/{dataset}_{answer_replacement}/{dataset}_{iterations}_{mode}_benchmark.json') if answer_replacement != 'none' else open(f'shap_results/{model}/{dataset}/{dataset}_{iterations}_{mode}_benchmark.json')
                results = json.load(results_f)
                accuracy = results['nothing']['accuracy']
                accuracies.append(accuracy)
            
            per_feature_contributions = np.concatenate([per_feature_contributions], axis=1)

            per_feature_contributions = np.swapaxes(per_feature_contributions, 0, 1)

            x = np.arange(len(answer_replacement_types))

            for i, metric in enumerate(metrics):

                if i in [0, 1, 2]:
                    line, = axs[i // 2][i % 2].plot(x, per_feature_contributions[i], label=metric, color=colours[k], linestyle=linestyles[j], marker='.')
                    if i == 0:
                        axs[i // 2][i % 2].set_ylim(0.0, 0.5)
                    elif i == 1:
                        axs[i // 2][i % 2].set_ylim(0.2, 0.7)
                    elif i == 2:
                        axs[i // 2][i % 2].set_ylim(0.2, 0.7)
                else:
                    line, = axs[i // 2][i % 2].plot(x, accuracies, label=metric, color=colours[k], linestyle=linestyles[j], marker='.')
                    axs[i // 2][i % 2].set_ylim(0.0, 1.0)
                    lines.append(line)

                axs[i // 2][i % 2].set_xticks(np.arange(0, len(answer_replacement_types)), labels=[answer_replacement_dict[x] for x in answer_replacement_types], rotation=45)
                axs[i // 2][i % 2].set_xlim(0, len(answer_replacement_types) - 1)

                axs[i // 2][i % 2].set_title(metrics[i])

        all_lines.append(lines)

    axs[0][1].legend(loc='upper left', handles=[line[0] for line in all_lines], labels=[model_dict[x] for x in models])
    axs[1][0].legend(loc='upper right', handles=all_lines[0], labels=[dataset_dict[x] for x in datasets])

    fig.savefig('shap_results/answer_replacement.png', dpi=400)


def print_benchmark_table(models=['frozenbilm', 'internvideo', 'videollama2', 'llava_video', 'longva', 'videollama3'], 
                          datasets=['egoschema', 'hd-epic', 'mvbench', 'lvbench'],
                          masking_directions=['nothing', 'everything'],
                          logits=['ground_truth'],
                          iterations=5000, 
                          mode='joint',
                          force_logit=None,
                          answer_masking='all',
                          answer_replacement=None):
    
    suffix = '{}'

    model_dict = {'frozenbilm': f'\\frozenbilm{suffix}', 
                  'internvideo': f'\\internvideo{suffix}', 
                  'videollama2': f'\\videollamatwo{suffix}', 
                  'llava_video': f'\\llavavideo{suffix}', 
                  'longva': f'\\longva{suffix}', 
                  'videollama3': f'\\videollamathree{suffix}'}
    
    dataset_dict = {'egoschema': f'\\egoschema{suffix}', 
                    'hd-epic': f'\\hdepic{suffix}', 
                    'mvbench': f'\\mvbench{suffix}', 
                    'lvbench': f'\\lvbench{suffix}'}

    masking_directions_dict = {'nothing': ('Nothing', 1, ['None']),
                               'everything': ('Everything', 4, ['All', 'Video', 'Question', 'Answer']),
                               'positive': ('Positive', 4, ['All', 'Video', 'Question', 'Answer']),
                               'negative': ('Negative', 4, ['All', 'Video', 'Question', 'Answer'])}

    add_to_file = ''
    if answer_masking == 'gt':
        add_to_file += '_gt'
    elif force_logit != None:
        add_to_file += f'_{force_logit}'
    
    add_to_path = ''
    if answer_replacement != None:
        add_to_path += f'_{answer_replacement}'

    model_accuracies_list = []

    mask = []

    for model in models:

        dataset_accuracies = []

        dataset_mask = []

        for dataset in datasets:
            results_f = open(f'shap_results/{model}/{dataset}{add_to_path}/{dataset}_{iterations}_{mode}{add_to_file}_benchmark.json')
            results = json.load(results_f)

            accuracies = []

            modality_mask = []

            for masking_direction in masking_directions:
                if masking_direction == 'nothing':
                    accuracies.append(results[masking_direction]['accuracy'])
                    modality_mask.append(-2)
                elif masking_direction == 'everything':
                    for modal in results[masking_direction].keys():
                        diff = results[masking_direction][modal]['accuracy'] - results['nothing']['accuracy'] 
                        accuracies.append(diff)
                        if diff < 0:
                            modality_mask.append(0)
                        elif diff > 0:
                            modality_mask.append(1)
                        elif diff == 0:
                            modality_mask.append(-1)
                else:
                    for logit in logits:
                        for modal in results[masking_direction][logit]:
                            diff = results[masking_direction][logit][modal]['accuracy'] - results['nothing']['accuracy']
                            accuracies.append(diff)
                            if diff < 0:
                                modality_mask.append(0)
                            elif diff > 0:
                                modality_mask.append(1)
                            elif diff == 0:
                                modality_mask.append(-1)
            dataset_mask.append(modality_mask)
            dataset_accuracies.append(accuracies)
        
        mask.append(np.array(dataset_mask).T)

        model_accuracies = pd.DataFrame(dataset_accuracies).transpose()
        model_accuracies_list.append(model_accuracies)

    mask = np.concatenate(mask)

    index_names = ['Model', 'Modality']

    columns = [dataset_dict[x] for x in datasets]

    index = [list(itertools.chain.from_iterable([[model_dict[x]] * sum([masking_directions_dict[y][1] for y in masking_directions]) for x in models])),
             list(itertools.chain.from_iterable([masking_directions_dict[x][2] for x in masking_directions])) * len(models)]

    benchmark = pd.concat(model_accuracies_list, axis=0).set_index(index)
    index = benchmark.index.set_names(index_names)
    benchmark = benchmark.set_index(index)
    benchmark.columns = columns

    s = benchmark.style
    s.clear()
    s.format('{:.2f}')  

    cmap = matplotlib.colors.ListedColormap(colors=['black', 'grey', 'red', 'green'])

    s.text_gradient(cmap=cmap, gmap=mask, axis=None)

    print(s.to_latex(column_format='c' * (len(index_names) + len(columns)), multicol_align='c', convert_css=True))


def print_metric_table(models=['frozenbilm', 'internvideo', 'videollama2', 'llava_video', 'longva', 'videollama3'], 
                       datasets=['egoschema', 'hd-epic', 'mvbench', 'lvbench'],
                       metrics=['Modality Contribution', 'Per-Feature Contribution'],
                       iterations=5000, 
                       mode='joint',
                       first_index='model',
                       short_model_names=False,
                       logits='Ground Truth',
                       answer_replacement=None):

    prefix = '\\' if not short_model_names else '\\s'
    suffix = '{}'

    model_dict = {'frozenbilm':f'{prefix}frozenbilm{suffix}', 
                  'internvideo': f'{prefix}internvideo{suffix}', 
                  'videollama2': f'{prefix}videollamatwo{suffix}', 
                  'llava_video': f'{prefix}llavavideo{suffix}', 
                  'longva': f'{prefix}longva{suffix}', 
                  'videollama3': f'{prefix}videollamathree{suffix}'}
    
    dataset_dict = {'egoschema':f'\\egoschema{suffix}', 
                    'hd-epic': f'\\hdepic{suffix}', 
                    'mvbench': f'\\mvbench{suffix}', 
                    'lvbench': f'\\lvbench{suffix}'}

    rows = []
        
    add_to_path = ''
    if answer_replacement != None:
        add_to_path += f'_{answer_replacement}'
    
    if first_index == 'model':
        for model in models:
            for dataset in datasets:
                results_f = open(f'shap_results/{model}/{dataset}{add_to_path}/{dataset}_{iterations}_{mode}_benchmark.json')
                results = json.load(results_f)
                metrics_out = pd.read_csv(f'shap_results/{model}/{dataset}{add_to_path}/{dataset}_{iterations}_{mode}_metrics.csv', index_col=0, header=[0, 1])[metrics]
                accuracy_column = pd.DataFrame(results['nothing']['accuracy'], index=[logits], columns=[['Acc'], ['']])
                rows.append(pd.concat([metrics_out.loc[[logits]], accuracy_column], axis=1))

        if len(datasets) > 1:
            index = [list(itertools.chain.from_iterable([[model_dict[x]] * len(datasets) for x in models])),
                     list(itertools.chain.from_iterable([[dataset_dict[x]] for x in datasets] * len(models)))]
        else:
            index = [model_dict[x] for x in models]
    elif first_index == 'dataset':
        for dataset in datasets:
            for model in models:
                results_f = open(f'shap_results/{model}/{dataset}{add_to_path}/{dataset}_{iterations}_{mode}_benchmark.json')
                results = json.load(results_f)
                metrics_out = pd.read_csv(f'shap_results/{model}/{dataset}{add_to_path}/{dataset}_{iterations}_{mode}_metrics.csv', index_col=0, header=[0, 1])[metrics]
                accuracy_column = pd.DataFrame(results['nothing']['accuracy'], index=[logits], columns=[['Acc'], ['']])
                rows.append(pd.concat([metrics_out.loc[[logits]], accuracy_column], axis=1))
                
        if len(models) > 1:
            index = [list(itertools.chain.from_iterable([[dataset_dict[x]] * len(models) for x in datasets])),
                 list(itertools.chain.from_iterable([[model_dict[x]] for x in models] * len(datasets)))]
        else:
            index = [dataset_dict[x] for x in datasets]

    shap_metrics = pd.concat(rows, axis=0)
    shap_metrics.index = index

    s = shap_metrics.style
    s.clear()
    s.format('{:.2f}')  

    s.background_gradient(cmap='RdBu', vmin=shap_metrics[metrics[0]].min().min(), vmax=shap_metrics[metrics[0]].max().max(), subset=[[metrics[0], 'V'],
                                                                                                                                               [metrics[0], 'Q'],
                                                                                                                                               [metrics[0], 'A']])
    
    s.background_gradient(cmap='RdBu', vmin=shap_metrics[metrics[1]].min().min(), vmax=shap_metrics[metrics[1]].max().max(), subset=[[metrics[1], 'V'],
                                                                                                                                               [metrics[1], 'Q'],
                                                                                                                                               [metrics[1], 'A']])
    
    if shap_metrics.index is pd.MultiIndex:
        index_columns = len(shap_metrics.index[0])
    else:
        index_columns = 1

    print(s.to_latex(column_format='c' * (index_columns + len(shap_metrics.columns) + 1), multicol_align='c', convert_css=True))
