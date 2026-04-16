from pathlib import Path

from src.datasets.base_dataset import BaseDataset

#TODO 
class CustomDirAudioDataset(BaseDataset):
    def __init__(self, transcription_dir=None, text=None, *args, **kwargs):
        assert transcription_dir is not None or text is not None, "Text should be provided"
        data = []
        if(transcription_dir is not None):
            assert Path(transcription_dir).exists(), "transcription dir should exist"
            for path in Path(transcription_dir).iterdir():
                if path.suffix in [".txt"]:
                    with path.open() as f:
                         data.append(
                             {"text":f.read().strip(),
                              "transcription_path":path,
                             }
                         )
        if(text is not None):
            data.append(
                {
                "text":text,
                "transcription_path":Path("from_cmd.txt"),
                }
            )
        super().__init__(data, *args, **kwargs)