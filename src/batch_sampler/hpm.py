import math
import random
import numpy as np
from torch.utils.data import Sampler

class HardPrototypeMiningBatchSampler(Sampler):
    def __init__(
        self,
        ds,
        similarity_matrix,
        n_selected_speakers=8,
        n_similar_speakers=4,
        n_utterances_per_speaker=4,
        random_seed=42,
        drop_last=True,
        **kwargs,
    ):
        self.ds = ds
        self.similarity_matrix = similarity_matrix.detach().cpu().numpy()
        self.n_selected_speakers = n_selected_speakers
        self.n_similar_speakers = n_similar_speakers
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.random_seed = random_seed
        self.drop_last = drop_last
        self.num_speakers = self.similarity_matrix.shape[0]
        self.total_samples = len(ds)
        self.utterances_per_speaker = self.total_samples // self.num_speakers
        self.speaker_start_idx = [i * self.utterances_per_speaker for i in range(self.num_speakers)]

    def _get_indices_for_speaker(self, speaker_id):
        """Return list of all dataset indices belonging to speaker `speaker_id`."""
        start = self.speaker_start_idx[speaker_id]
        return list(range(start, start + self.utterances_per_speaker))

    def __iter__(self):
        if self.num_speakers == 0:
            return iter([])

        speaker_candidates = list(range(self.num_speakers))
        random.Random(self.random_seed).shuffle(speaker_candidates)

        batches = []
        for i in range(0, self.num_speakers, self.n_selected_speakers):
            selected_speakers = speaker_candidates[i : i + self.n_selected_speakers]
            batch_indices = []
            for speaker_id in selected_speakers:
                sim_row = self.similarity_matrix[speaker_id]
                if np.all(np.isnan(sim_row)):
                    similar_speakers = [speaker_id]
                else:
                    order = np.argsort(sim_row)[::-1]
                    similar_speakers = list(order[:self.n_similar_speakers])
                    if speaker_id not in similar_speakers:
                        similar_speakers[0] = speaker_id
                if speaker_id not in similar_speakers:
                    similar_speakers = [speaker_id] + [
                        s for s in similar_speakers if s != speaker_id
                    ][:self.n_similar_speakers - 1]
                similar_speakers = similar_speakers[:self.n_similar_speakers]

                for sim_spk in similar_speakers:
                    all_utterances = self._get_indices_for_speaker(sim_spk)
                    if len(all_utterances) == 0:
                        continue

                    if len(all_utterances) >= self.n_utterances_per_speaker:
                        chosen = random.sample(all_utterances, self.n_utterances_per_speaker)
                    else:
                        chosen = random.choices(all_utterances, k=self.n_utterances_per_speaker)
                    batch_indices.extend(chosen)
            if not batch_indices:
                continue
            if self.drop_last and len(batch_indices) < self.n_selected_speakers * self.n_similar_speakers * self.n_utterances_per_speaker:
                continue
            batches.append(batch_indices)

        return iter(batches)

    def __len__(self):
        if self.n_selected_speakers <= 0:
            return 0
        return math.ceil(self.num_speakers / self.n_selected_speakers)