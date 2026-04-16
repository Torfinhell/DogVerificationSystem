from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src.logger.utils import confusion_matrix_figure, similarity_matrix_figure, sphere_plot_tensor

from sklearn.metrics import confusion_matrix
import seaborn as sns

class WandBWriter:
    """
    Class for experiment tracking via WandB.

    See https://docs.wandb.ai/.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        entity=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        """
        API key is expected to be provided by the user in the terminal.

        Args:
            logger (Logger): logger that logs output.
            project_config (dict): config for the current experiment.
            project_name (str): name of the project inside experiment tracker.
            entity (str | None): name of the entity inside experiment
                tracker. Used if you work in a team.
            run_id (str | None): the id of the current run.
            run_name (str | None): the name of the run. If None, random name
                is given.
            mode (str): if online, log data to the remote server. If
                offline, log locally.
        """
        self.run_id = run_id
        self.wandb = None
        self._mutable_tables: dict[str, object] = {}

        try:
            import wandb

            wandb.login()

            wandb.init(
                project=project_name,
                entity=entity,
                config=project_config,
                name=run_name,
                resume="allow",  # resume the run if run_id existed
                id=self.run_id,
                mode=mode,
                save_code=kwargs.get("save_code", False),
            )
            self.wandb = wandb

        except ImportError:
            logger.warning("For use wandb install it via \n\t pip install wandb")

        self.step = 0
        # the mode is usually equal to the current partition name
        # used to separate Partition1 and Partition2 metrics
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor
        training/evaluation speed.

        Args:
            step (int): current step.
            mode (str): current mode (partition name).
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        elif self.wandb is not None:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        """
        Update object_name (scalar, image, etc.) with the
        current mode (partition name). Used to separate metrics
        from different partitions.

        Args:
            object_name (str): current object name.
        Returns:
            object_name (str): updated object name.
        """
        return f"{object_name}_{self.mode}" if self.mode else object_name

    def add_checkpoint(self, checkpoint_path, save_dir):
        """
        Log checkpoints to the experiment tracker.

        The checkpoints will be available in the files section
        inside the run_name dir.

        Args:
            checkpoint_path (str): path to the checkpoint file.
            save_dir (str): path to the dir, where checkpoint is saved.
        """
        if self.wandb is None:
            return
        self.wandb.save(checkpoint_path, base_path=save_dir)

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.

        Args:
            scalar_name (str): name of the scalar to use in the tracker.
            scalar (float): value of the scalar.
        """
        if self.wandb is None:
            return
        self.wandb.log(
            {
                self._object_name(scalar_name): scalar,
            },
            step=self.step,
        )

    def add_scalars(self, scalars):
        """
        Log several scalars to the experiment tracker.

        Args:
            scalars (dict): dict, containing scalar name and value.
        """
        if self.wandb is None:
            return
        self.wandb.log(
            {
                self._object_name(scalar_name): scalar
                for scalar_name, scalar in scalars.items()
            },
            step=self.step,
        )

    def add_image(self, image_name, image):
        """
        Log an image to the experiment tracker.

        Args:
            image_name (str): name of the image to use in the tracker.
            image (Path | ndarray | Image): image in the WandB-friendly
                format.
        """
        if self.wandb is None:
            return
        self.wandb.log(
            {self._object_name(image_name): self.wandb.Image(image)}, step=self.step
        )

    def add_audio(self, audio_name, audio, sample_rate=None):
        """
        Log an audio to the experiment tracker.

        Args:
            audio_name (str): name of the audio to use in the tracker.
            audio (Path | ndarray): audio in the WandB-friendly format.
            sample_rate (int): audio sample rate.
        """
        if self.wandb is None:
            return
        audio = audio.detach().cpu().numpy().squeeze()
        self.wandb.log(
            {
                self._object_name(audio_name): self.wandb.Audio(
                    audio, sample_rate=sample_rate
                )
            },
            step=self.step,
        )

    def add_text(self, text_name, text):
        """
        Log text to the experiment tracker.

        Args:
            text_name (str): name of the text to use in the tracker.
            text (str): text content.
        """
        if self.wandb is None:
            return
        self.wandb.log(
            {self._object_name(text_name): self.wandb.Html(text)}, step=self.step
        )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """
        Log histogram to the experiment tracker.

        Args:
            hist_name (str): name of the histogram to use in the tracker.
            values_for_hist (Tensor): array of values to calculate
                histogram of.
            bins (int | str): the definition of bins for the histogram.
        """
        if self.wandb is None:
            return
        values_for_hist = values_for_hist.detach().cpu().numpy()
        np_hist = np.histogram(values_for_hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(values_for_hist, bins=512)

        hist = self.wandb.Histogram(np_histogram=np_hist)

        self.wandb.log({self._object_name(hist_name): hist}, step=self.step)

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Log table to the experiment tracker (append rows across calls for the same keyed name).

        Args:
            table_name (str): name of the table to use in the tracker.
            table (DataFrame): table content.
        """
        if self.wandb is None:
            return
        oname = self._object_name(table_name)
        table = table.copy()
        table["step"] = self.step
        if oname not in self._mutable_tables:
            self._mutable_tables[oname] = self.wandb.Table(
                dataframe=table, log_mode="MUTABLE"
            )
        else:
            for _, row in table.iterrows():
                self._mutable_tables[oname].add_data(*row.tolist())
        self.wandb.log(
            {oname: self._mutable_tables[oname]},
            step=self.step,
        )

    def add_plot_3d(
        self,
        plot_name: str,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        title: str | None = None,
    ):
        """
        Log a 3D sphere plot of embeddings (PCA coords, L2-normalized) as a W&B image.
        """
        if self.wandb is None:
            return
        points_3d = embeddings.detach().float().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        t = sphere_plot_tensor(
            points_3d, labels, title=title or plot_name
        )
        img = t.permute(1, 2, 0).numpy()
        self.wandb.log(
            {self._object_name(plot_name): self.wandb.Image(img)},
            step=self.step,
        )

    def add_confusion_matrix_image(
        self,
        name: str,
        y_true,
        y_pred,
        labels: list, 
        title: str | None = None,
    ):
        """Log a confusion matrix image from raw preds/labels."""
        if self.wandb is None:
            return
        cm_array = confusion_matrix(y_true, y_pred)
        img = confusion_matrix_figure(
            cm_array, 
            title=title or name, 
            class_names=labels
        )
        self.wandb.log(
            {self._object_name(name): self.wandb.Image(img)},
            step=self.step,
        )

    def add_similarity_matrix_image(
        self,
        name: str,
        similarity_matrix,
        title: str | None = None,
        cmap: str = "coolwarm",
    ):
        """
        Log a similarity matrix image.
        
        Args:
            name: Name of the matrix in the tracker.
            similarity_matrix: 2D array or tensor representing similarity scores.
            title: Optional title for the figure.
            cmap: Colormap to use (default "coolwarm").
        """
        if self.wandb is None:
            return
        sm_array = similarity_matrix.detach().cpu().numpy()   
        img = similarity_matrix_figure(
            sm_array,
            title=title or name,
            cmap=cmap,
        )
        self.wandb.log(
            {self._object_name(name): self.wandb.Image(img)},
            step=self.step,
        )
