"""
Concatenate several :class:`torch.utils.data.Dataset` instances into one label space.

Each source dataset is expected to expose integer class IDs in ``0 .. num_local_classes-1``
(``"labels"`` or ``"label"``). This wrapper assigns disjoint global indices and records the
layout in ``map.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from src.utils.io_utils import ROOT_PATH


def _get_label_from_sample(sample: dict) -> int:
    if "labels" in sample:
        v = sample["labels"]
    elif "label" in sample:
        v = sample["label"]
    else:
        raise KeyError("Sample must contain 'labels' or 'label'")
    if isinstance(v, torch.Tensor):
        return int(v.item())
    return int(v)


def _set_label_on_sample(sample: dict, global_label: int) -> dict:
    out = dict(sample)
    if "labels" in out:
        out["labels"] = global_label
    if "label" in out:
        out["label"] = global_label
    return out


class DatasetCombine(Dataset):
    """
    Stack multiple datasets with independent local label spaces into one global
    ``0 .. num_classes-1`` space.

    Global index for dataset ``i`` and local label ``k`` is ``offset[i] + k``.

    ``map.json`` stores:

    - ``num_classes``: total number of concatenated classes.
    - One entry per source: ``{DATASET_NAME: offset}`` — ``DATASET_NAME`` is the
      name you pass for that dataset; value is its starting index in the concat space.
    - ``local_to_global``: nested ``{DATASET_NAME: {local_str: global_label}}``.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        num_local_classes: Sequence[int],
        dataset_names: Sequence[str],
        map_json_path: Path | str | None = None,
    ):
        """
        Args:
            datasets: Source datasets; each ``__getitem__`` returns a dict with
                ``"labels"`` or ``"label"`` (local class index).
            num_local_classes: Number of distinct local classes per dataset,
                in the same order as ``datasets``.
            dataset_names: Unique name per dataset; used as JSON keys for offsets
                and for ``local_to_global`` (same order as ``datasets``).
            map_json_path: Where to write the mapping. Default:
                ``ROOT_PATH / "data" / "datasets" / "concat_label_map.json"``.
        """
        if len(datasets) != len(num_local_classes) or len(datasets) != len(dataset_names):
            raise ValueError(
                f"len(datasets)={len(datasets)}, len(num_local_classes)={len(num_local_classes)}, "
                f"len(dataset_names)={len(dataset_names)} — all must match"
            )
        if len(datasets) == 0:
            raise ValueError("datasets must be non-empty")

        _reserved = {"num_classes", "local_to_global"}
        names = [str(n) for n in dataset_names]
        if len(set(names)) != len(names):
            raise ValueError(f"dataset_names must be unique, got {names}")
        bad = [n for n in names if n in _reserved]
        if bad:
            raise ValueError(f"dataset_names cannot be reserved keys {_reserved}: {bad}")

        self._datasets: list[Dataset] = list(datasets)
        self._num_local = [int(n) for n in num_local_classes]
        self._dataset_names = names

        offsets: list[int] = []
        o = 0
        for n in self._num_local:
            if n <= 0:
                raise ValueError(f"num_local_classes must be positive, got {n}")
            offsets.append(o)
            o += n
        self._offsets = offsets
        self._num_classes = o

        self._cumulative_lengths: list[int] = []
        cum_len = 0
        for ds in self._datasets:
            ln = len(ds)
            self._cumulative_lengths.append(cum_len)
            cum_len += ln
        self._total_len = cum_len

        if map_json_path is None:
            map_json_path = ROOT_PATH / "data" / "datasets" / "concat_label_map.json"
        self._map_path = Path(map_json_path)
        self._write_map_json()
        self._index: list[dict] | None = None
        if all(hasattr(ds, "_index") for ds in self._datasets):
            merged: list[dict] = []
            for i, ds in enumerate(self._datasets):
                off = self._offsets[i]
                for entry in ds._index:
                    e = dict(entry)
                    if "label" in e:
                        e["label"] = off + int(e["label"])
                    elif "labels" in e:
                        e["labels"] = off + int(e["labels"])
                    else:
                        raise KeyError(
                            f"Index entry from {self._dataset_names[i]!r} has no 'label' or 'labels'"
                        )
                    merged.append(e)
            self._index = merged

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _write_map_json(self) -> None:
        name_to_offset: dict[str, int] = {}
        local_to_global: dict[str, dict[str, int]] = {}
        for i, off in enumerate(self._offsets):
            name = self._dataset_names[i]
            name_to_offset[name] = off
            local_to_global[name] = {
                str(local): off + local for local in range(self._num_local[i])
            }

        payload = {
            "num_classes": self._num_classes,
            **name_to_offset,
            "local_to_global": local_to_global,
        }
        self._map_path.parent.mkdir(parents=True, exist_ok=True)
        with self._map_path.open("w") as f:
            json.dump(payload, f, indent=2)

    def _locate(self, index: int) -> tuple[int, int]:
        """Map flat ``index`` to ``(dataset_idx, index_within_dataset)``."""
        if index < 0 or index >= self._total_len:
            raise IndexError(f"Index {index} out of range [0, {self._total_len})")
        for i, ds in enumerate(self._datasets):
            start = self._cumulative_lengths[i]
            end = start + len(ds)
            if start <= index < end:
                return i, index - start
        raise RuntimeError("unreachable")

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, index: int) -> dict:
        ds_idx, local_i = self._locate(int(index))
        sample = self._datasets[ds_idx][local_i]
        if not isinstance(sample, dict):
            raise TypeError(
                f"Dataset {self._dataset_names[ds_idx]!r} ({ds_idx}) must return a dict "
                f"from __getitem__, got {type(sample)}"
            )
        local_label = _get_label_from_sample(sample)
        nloc = self._num_local[ds_idx]
        if not (0 <= local_label < nloc):
            raise ValueError(
                f"Dataset {self._dataset_names[ds_idx]!r} ({ds_idx}) returned local label "
                f"{local_label}, expected in [0, {nloc})"
            )
        global_label = self._offsets[ds_idx] + local_label
        return _set_label_on_sample(sample, global_label)


def read_num_classes_from_concat_map(map_json_path: Path | str | None = None) -> int:
    """Read ``num_classes`` from a ``concat_label_map.json`` written by :class:`DatasetCombine`."""
    if map_json_path is None:
        map_json_path = ROOT_PATH / "data" / "datasets" / "concat_label_map.json"
    path = Path(map_json_path)
    with path.open() as f:
        data = json.load(f)
    return int(data["num_classes"])
