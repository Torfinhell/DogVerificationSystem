import torch


import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.

    Args:
        dataset_items (list[dict]): list of objects from dataset.__getitem__.
    Returns:
        result_batch (dict): dict containing batched tensors.
            - "audio": padded audio tensor of shape (batch, max_len)
            - "audio_lengths": original audio lengths before padding (batch,)
            - "spectral_feat_lengths": original spectral feature lengths before padding (batch,)
            - "label": stacked label tensor of shape (batch,)
            - "breed": stacked breed tensor (if available)
            - (optionally other keys, handled as lists if not padded)
    """
    batch = {}
    for key in dataset_items[0].keys():
        values = [item[key] for item in dataset_items]

        if key == "audio":
            transposed = [v.T for v in values]
            audio_lengths = [t.shape[0] for t in transposed]
            batch["audio_lengths"] = torch.tensor(audio_lengths)
            batch[key] = pad_sequence(transposed, batch_first=True)  
        elif key == "spectral_feat":
            spectral_lengths = [v.shape[1] for v in values]
            batch["spectral_feat_lengths"] = torch.tensor(spectral_lengths)
            transposed = [v.T for v in values]
            padded = pad_sequence(transposed, batch_first=True)
            batch[key] = padded.permute(0, 2, 1)   
        elif key == "label":
            batch[key] = torch.tensor(values)
        elif key == "breed":
            batch[key] = torch.tensor(values, dtype=torch.long)
        else:
            batch[key] = values
    return batch