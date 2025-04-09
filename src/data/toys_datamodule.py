from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import transforms
from data._recdataset import RecDataset, get_seq_dic, get_rating_matrix
import os


class ToysDataModule(LightningDataModule):
    """`LightningDataModule` for the LastFM dataset.

    Description of LastFM dataset: 

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html

    Attributes:
    batch_size : number of batch size in the optimization

    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seq_dic_conf: Optional[Dict[str, Any]]= None, # this should be args to the function get_seq_dic
        others: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a `LastFMDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # no need to download the data
        # I believe this should be a class of torch.utils.data.Dataset
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # config is used to be args here.
            # seq_dic obtain from get_seq_dic
            seq_dic, max_item, num_users = get_seq_dic(self.hparams.seq_dic_conf) # it needs data_file and data_name

            # update the parameters
            self.hparams.others.item_size = max_item + 1
            self.hparams.others.num_users = num_users + 1
            self.hparams.others.same_target_path = os.path.join(self.hparams.seq_dic_conf.data_dir, self.hparams.seq_dic_conf.data_name+'_same_target.npy')
            # get the rating matrix
            self.hparams.others.valid_rating_matrix, self.hparams.others.test_rating_matrix = get_rating_matrix(self.hparams.seq_dic_conf.data_name, seq_dic, max_item)
            # prepare the dataset as train, validation and test
            # each data set should contains tensors of user_id, input_ids, answer, and neg_answer
            self.data_train = RecDataset(self.hparams.others, seq_dic['user_seq'], data_type='train')
            self.data_val = RecDataset(self.hparams.others, seq_dic['user_seq'], data_type='valid')
            self.data_test = RecDataset(self.hparams.others, seq_dic['user_seq'], data_type='test')
            # prepare the samplers based on the dataset
            self.train_sampler = RandomSampler(self.data_train)
            self.eval_sampler = SequentialSampler(self.data_val)
            self.test_sampler = SequentialSampler(self.data_test)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            sampler=self.train_sampler,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            sampler=self.eval_sampler,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            sampler=self.test_sampler,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = ToysDataModule()
