---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: A Video Is Not Worth a Thousand Words
---

# A Video Is Not Worth a Thousand Words

[Sam Pollard](https://sjpollard.github.io), [Michael Wray](https://mwray.github.io)

University of Bristol

arXiv

<img src="assets/intro.png" width="922" height="448"/>

# Abstract

As we become increasingly dependent on vision language models (VLMs) to answer questions about the world around us, there is a significant amount of research devoted to increasing both the difficulty of video question answering (VQA) datasets, and the context lengths of the models that they evaluate. The reliance on large language models as backbones has lead to concerns about potential text dominance, and the exploration of interactions between modalities is underdeveloped. How do we measure whether we're heading in the right direction, with the complexity that multi-modal models introduce? We propose a joint method of computing both feature attributions and modality scores based on Shapley values, where both the features and modalities are arbitrarily definable. Using these metrics, we compare 6 VLM models of varying context lengths on 4 representative datasets, focusing on multiple-choice VQA. In particular, we consider video frames and whole textual elements as equal features in the hierarchy, and the multiple-choice VQA task as an interaction between three modalities: video, question and answer. Our results demonstrate a dependence on text and show that the multiple-choice VQA task devolves into a model's ability to ignore distractors.

# Method

<img src="assets/method.gif" width="480" height="270"/>

Given an input VQA-tuple, we want to know how much each feature contributes towards the model output. We calculate these contributions using [Shapley values](https://en.wikipedia.org/wiki/Shapley_value), resulting in output much like the above animation. With these contributions we then calculate a modality contribution (from the contribution of the entire modality) and a per-feature contribution (from the average contribution of each feature). Here are input modalities are video, question and answer.

# Results

<img src="assets/table.png" width="480" height="270"/>

In this table we show our metrics for a range of model and dataset combinations. Red cells are closer to minimum contribution and blue cells are closer to maximum contribution. We see that generally video has low modality contribution, and even the stronger models show extremely low per-frame contributions. Furthermore, question features are consistently less important than answer features. Does this really make sense?

# Analysis

<img src="assets/smarter_than_a_vlm.gif" width="480" height="270"/>

Which of these two frames do you think is more important for answering the given question? It's *obviously* the one on the right. However, the Shapley values indicate that the one on the left actually contributes more. Just take a look at the teaser figure at the top of the page. Feature contributions often don't correlate with common sense, even in extremely simple scenarios.


# Links
[Code](https://github.com/sjpollard/a-video-is-not-worth-a-thousand-words) | [arXiv]()

# Bibtex

```

```

