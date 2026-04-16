import math
import random
import torch
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class CriterionSimiliarityMatrix:
    def __init__(self, num_speakers):
        self.sm = torch.zeros((num_speakers, num_speakers))

class HardPrototypeMiningBatchSampler(Sampler):
    def __init__(
        self,
        ds,
        n_selected_speakers=8,
        n_similar_speakers=4,
        n_utterances_per_speaker=4,
        random_seed=42,
        num_iter=3,
        drop_last=True,
        **kwargs,
    ):
        self.ds = ds
        self.num_iter = num_iter
        self.n_selected_speakers = n_selected_speakers
        self.n_similar_speakers = n_similar_speakers
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.random_seed = random_seed
        self.drop_last = drop_last
        self.unique_labels = sorted(self.ds.get_labels())
        self.num_unique_speakers = len(self.unique_labels)
        self.label_to_mtrx_id = {label: i for i, label in enumerate(self.unique_labels)}
        self.mtrx_id_to_label = {i: label for i, label in enumerate(self.unique_labels)}
        self.speaker_to_indices = defaultdict(list)
        for idx, item in enumerate(self.ds.load_index()):
            self.speaker_to_indices[item["label"]].append(idx)
        self.full_mtrx_id_list = list(range(self.num_unique_speakers)) * self.num_iter
        self.criterion = CriterionSimiliarityMatrix(self.num_unique_speakers)

    def __iter__(self):
        if not self.full_mtrx_id_list:
            return iter([])
            
        rng = random.Random(self.random_seed)
        candidates = list(self.full_mtrx_id_list)
        rng.shuffle(candidates)
        
        batches = []
        for i in range(0, len(candidates), self.n_selected_speakers):
            selected_mtrx_ids = candidates[i : i + self.n_selected_speakers]
            
            if self.drop_last and len(selected_mtrx_ids) < self.n_selected_speakers:
                continue
            
            batch_indices = []
            for mtrx_id in selected_mtrx_ids:
                sim_row = self.criterion.sm[mtrx_id]
                top_mtrx_indices = torch.argsort(sim_row, descending=True)[:self.n_similar_speakers]
                mtrx_id_set = top_mtrx_indices.tolist()
                if mtrx_id not in mtrx_id_set:
                    mtrx_id_set[-1] = mtrx_id
                
                for s_id in mtrx_id_set:
                    label = self.mtrx_id_to_label[s_id]
                    all_utts = self.speaker_to_indices[label]
                    
                    if len(all_utts) >= self.n_utterances_per_speaker:
                        chosen = rng.sample(all_utts, self.n_utterances_per_speaker)
                    else:
                        chosen = rng.choices(all_utts, k=self.n_utterances_per_speaker)
                    batch_indices.extend(chosen)
            
            if batch_indices:
                batches.append(batch_indices)

        return iter(batches)

    def __len__(self):
        total_items = len(self.full_mtrx_id_list)
        if self.drop_last:
            return total_items // self.n_selected_speakers
        return math.ceil(total_items / self.n_selected_speakers)
