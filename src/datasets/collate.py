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
            - "labels": stacked labels tensor of shape (batch,)
            - (optionally other keys, handled as lists if not padded)
    """
    batch = {}
    for key in dataset_items[0].keys():
        values = [item[key] for item in dataset_items]

        if key == "audio":
            batch[key] = pad_sequence(values, batch_first=True)
        elif key == "spectral_feat":
            transposed = [v.T for v in values] 
            padded = pad_sequence(transposed, batch_first=True)  
            batch[key] = padded.permute(0, 2, 1)  
        elif key == "labels":
            batch[key] = torch.tensor(values)  
        else:
            batch[key] = values

    return batch