import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler


class HPMBatchSampler(Sampler):
    """Hard Prototype Mining (HPM) batch sampler.

    Implements the HPM procedure:
    - Build speaker indexes over dataset labels.
    - Use confusion matrix to infer similarity between classes.
    - For each iteration, sample S speakers and gather I most similar speakers
      (including the speaker itself).
    - From each selected speaker group sample U utterances.
    - Return a batch of indices of size S * I * U.
    """

    def __init__(
        self,
        ds,
        confusion_matrix=None,
        n_selected_speakers=8,
        n_similar_speakers=4,
        n_utterances_per_speaker=4,
        random_seed=42,
        drop_last=True,
        **kwargs,
    ):
        self.ds = ds
        self.confusion_matrix = confusion_matrix
        self.n_selected_speakers = n_selected_speakers
        self.n_similar_speakers = n_similar_speakers
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.random_seed = random_seed
        self.drop_last = drop_last

        self.build_label_to_ind()

    def build_label_to_ind(self):
        self.speaker_to_indices = defaultdict(list)

        if hasattr(self.ds, "_index"):
            for idx, entry in enumerate(self.ds._index):
                label = entry.get("label", entry.get("labels"))
                self.speaker_to_indices[int(label)].append(idx)
        else:
            for idx in range(len(self.ds)):
                sample = self.ds[idx]
                label = sample.get("label", sample.get("labels"))
                self.speaker_to_indices[int(label)].append(idx)

        self.speaker_ids = sorted(self.speaker_to_indices.keys())
        self.num_speakers = len(self.speaker_ids)
        self.label_to_position = {label: pos for pos, label in enumerate(self.speaker_ids)}
        self.position_to_label = {pos: label for label, pos in self.label_to_position.items()}

    def __iter__(self):
        if self.num_speakers == 0:
            return iter([])
        speaker_candidates = self.speaker_ids.copy()
        random.Random(self.random_seed).shuffle(speaker_candidates)
        batches = []
        for i in range(0, self.num_speakers, self.n_selected_speakers):
            selected_positions = speaker_candidates[i : i + self.n_selected_speakers]
            batch_indices = []
            for pos in selected_positions:
                if self.similarity_matrix is not None:
                    sim_row = self.similarity_matrix[pos]
                    if np.all(np.isnan(sim_row)):
                        similar_positions = [pos]
                    else:
                        order = np.argsort(sim_row)[::-1]
                        similar_positions = list(order[: self.n_similar_speakers])
                        if pos not in similar_positions:
                            similar_positions[0] = pos
                else:
                    similar_positions = [pos]

                if pos not in similar_positions:
                    similar_positions = [pos] + [x for x in similar_positions if x != pos][: self.n_similar_speakers - 1]
                similar_positions = similar_positions[: self.n_similar_speakers]

                similar_labels = [self.position_to_label[p] for p in similar_positions]

                for target_label in similar_labels:
                    utterances = self.speaker_to_indices.get(target_label, [])
                    if len(utterances) == 0:
                        continue

                    if len(utterances) >= self.n_utterances_per_speaker:
                        chosen = random.sample(utterances, self.n_utterances_per_speaker)
                    else:
                        chosen = random.choices(utterances, k=self.n_utterances_per_speaker)
                    batch_indices.extend(chosen)

            if len(batch_indices) == 0:
                continue
            if self.drop_last and len(batch_indices) < self.n_selected_speakers * self.n_similar_speakers * self.n_utterances_per_speaker:
                continue
            batches.append(batch_indices)
        return iter(batches)

    def __len__(self):
        if self.n_selected_speakers <= 0:
            return 0
        return math.ceil(self.num_speakers / self.n_selected_speakers)
