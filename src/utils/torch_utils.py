from __future__ import annotations

import torch


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert a string representation of a data type to a torch.dtype.

    Args:
        dtype_str (str): String representation of the data type.
    Returns:
        dtype (torch.dtype): Corresponding torch data type.
    """
    dtype_mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if dtype_str not in dtype_mapping:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")
    return dtype_mapping[dtype_str]


def dtype_to_str(dtype: torch.dtype) -> str:
    """
    Convert a torch.dtype to its string representation.

    Args:
        dtype (torch.dtype): Torch data type.
    Returns:
        dtype_str (str): String representation of the data type.
    """
    dtype_mapping = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.bfloat16: "bfloat16",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    if dtype not in dtype_mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return dtype_mapping[dtype]


def set_tf32_allowance(allow: bool) -> None:
    """
    Set the allowance for TF32 precision on NVIDIA Ampere GPUs.

    Args:
        allow (bool): If True, enables TF32 for matrix multiplications and convolutions.
                      If False, disables TF32.
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow
