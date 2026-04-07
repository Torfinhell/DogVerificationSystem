from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
from torch.utils.data import Dataset
import torch
from torch.utils.data import ConcatDataset

from src.utils.io_utils import ROOT_PATH


class DatasetCombine(ConcatDataset):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        num_local_classes: Sequence[int] | None = None,
        dataset_names: Sequence[str] | None = None,
        map_json_path: Path | str | None = None,
    ):
        """
        Args:
            datasets: Source datasets; each ``__getitem__`` returns a dict with
                ``"label"`` (local class index).
            num_local_classes: Number of distinct local classes per dataset,
                in the same order as ``datasets``. If None, gets from ds.num_classes.
            dataset_names: Unique name per dataset; used as JSON keys for offsets
                and for ``local_to_global`` (same order as ``datasets``).
            map_json_path: Where to write the mapping. Default:
                ``ROOT_PATH / "data" / "datasets" / "concat_label_map.json"``.
        """
        super().__init__(datasets)

        if num_local_classes is None:
            self._num_local = [ds.num_classes for ds in datasets]
        else:
            self._num_local = [int(n) for n in num_local_classes]

        if dataset_names is None:
            self._dataset_names = [f"dataset_{i}" for i in range(len(datasets))]
        else:
            self._dataset_names = list(dataset_names)

        offsets: list[int] = []
        o = 0
        for n in self._num_local:
            if n <= 0:
                raise ValueError(f"num_local_classes must be positive, got {n}")
            offsets.append(o)
            o += n
        self._offsets = offsets
        self._num_classes = o

        if map_json_path is None:
            map_json_path = ROOT_PATH / "data" / "datasets" / "concat_label_map.json"
        self._map_path = Path(map_json_path)
        self._write_map_json()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _write_map_json(self) -> None:
        if self._map_path.exists():
            return
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

    def __getitem__(self, index: int) -> dict:
        dataset_idx = 0
        local_idx = index
        for i, ds in enumerate(self.datasets):
            if local_idx < len(ds):
                dataset_idx = i
                break
            local_idx -= len(ds)
        
        sample = self.datasets[dataset_idx][local_idx]
        if not isinstance(sample, dict):
            raise TypeError(
                f"Dataset {self._dataset_names[dataset_idx]!r} ({dataset_idx}) must return a dict "
                f"from __getitem__, got {type(sample)}"
            )
        
        # Extract local label from sample
        v = sample["label"]
        local_label = int(v.item()) if isinstance(v, torch.Tensor) else int(v)
        
        nloc = self._num_local[dataset_idx]
        if not (0 <= local_label < nloc):
            raise ValueError(
                f"Dataset {self._dataset_names[dataset_idx]!r} ({dataset_idx}) returned local label "
                f"{local_label}, expected in [0, {nloc})"
            )
        
        # Compute and set global label
        global_label = self._offsets[dataset_idx] + local_label
        out = dict(sample)
        out["label"] = global_label
        return out


def read_num_classes_from_concat_map(map_json_path: Path | str | None = None) -> int:
    """Read ``num_classes`` from a ``concat_label_map.json`` written by :class:`DatasetCombine`."""
    if map_json_path is None:
        map_json_path = ROOT_PATH / "data" / "datasets" / "concat_label_map.json"
    path = Path(map_json_path)
    with path.open() as f:
        data = json.load(f)
    return int(data["num_classes"])
