from typing import Any, Dict, Tuple, List, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG
from utils.metrics import Recall, NDCG
import numpy as np
# NDCG https://lightning.ai/docs/torchmetrics/stable/retrieval/normalized_dcg.html
# HitRate https://lightning.ai/docs/torchmetrics/stable/retrieval/hit_rate.html


class SeqRecLitModule(LightningModule):
    """Example of a `LightningModule` for Sequential Recommendation System

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module, # let's say we use SASRec then this should target sasrec which has parent to the abstract_model SequentialRecModel
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool
    ) -> None:
        """Initialize a `SeqRecLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = self.net.calculate_loss

        # the metrics are for retrieval information.
        # the missing values are treated as negative.
        # the negative value will be treated as false recommended items.
        # monitor the validation and test are enough.

        # NDCG, top_k
        self.val_ndcg_5 = NDCG(top_k=5, sync_on_compute=False)#RetrievalNormalizedDCG(top_k = 5)
        self.test_ndcg_5 = NDCG(top_k=5, sync_on_compute=False)# RetrievalNormalizedDCG(top_k = 5)

        self.val_ndcg_10 = NDCG(top_k=10, sync_on_compute=False)#RetrievalNormalizedDCG(top_k = 10)
        self.test_ndcg_10 =NDCG(top_k=10, sync_on_compute=False)#RetrievalNormalizedDCG(top_k = 10)
        
        # HitRate, top_k
        # example on using the hitrate metric

        '''
        from torch import tensor
        preds = tensor([0.2, 0.3, 0.5])
        target = tensor([True, False, True])
        retrieval_hit_rate(preds, target, top_k=2)
        '''

        # important to note regarding the usage of torchmetrics and practical use case in Industry
        # https://github.com/Lightning-AI/torchmetrics/issues/2611
        # still an open issue.
        # it is also potentially slower
        # https://github.com/Lightning-AI/torchmetrics/issues/2287
        self.val_hr_5 = Recall(top_k=5, sync_on_compute=False)#RetrievalHitRate(top_k = 5)
        self.test_hr_5 = Recall(top_k=5, sync_on_compute=False)#RetrievalHitRate(top_k = 5)

        self.val_hr_10 = Recall(top_k=10, sync_on_compute=False)#RetrievalHitRate(top_k = 10)
        self.test_hr_10 = Recall(top_k=10, sync_on_compute=False)#RetrievalHitRate(top_k = 10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_ndcg_5_best = MaxMetric()

        # for tracking best so far validation accuracy
        self.test_ndcg_5_best = MaxMetric()

        # the retrieval is based on list of items, therefore I need to collect the outputs for each step
        # and then calculate the metrics.

        # collect outputs of `*_step`, this is to calculate the metrics at the end of the epoch later.
        self.val_step_outputs = {"preds": [], "answers": []}
        self.test_step_outputs = {"preds": [], "answers": []}

        # collecting the item embeddings
        self.items = []

    def rating_matrix(self, valid_rating_matrix=None, test_rating_matrix=None):
        self.valid_rating_matrix = valid_rating_matrix
        self.test_rating_matrix = test_rating_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_ndcg_5.reset()
        self.val_ndcg_10.reset()
        self.val_hr_5.reset()
        self.val_hr_10.reset()
        self.val_ndcg_5_best.reset()
        self.items.append(self.net.item_embeddings.weight.data.clone().cpu().detach().numpy())

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions. (seq_out will be some highly dimensional tensor for time being)
            - A tensor of target labels.
        """
        user_ids, input_ids, answers, neg_answer, same_target = batch
        seq_out = self.net.forward(input_ids)
        preds = self.net.predict(input_ids, user_ids)
        # print(self.net.item_embeddings(torch.tensor([20]))[0,:4]) #0 is the tensor, 1 is the backward gradient attribute.
        # print('Train' , self.net.item_embeddings.weight.cpu().detach().numpy()[20,:4])
        # self.items.append(self.net.item_embeddings.weight.data.clone().cpu().detach().numpy())

        return user_ids, input_ids, seq_out, preds, answers, neg_answer, same_target

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.
        performing this and log for on_epoch will make the model calculate over the entire dataset. (1 epoch)
        as in batch_idx, batch in enumerate(dataloader): in the training loop.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        user_ids, input_ids, seq_out, _, targets, neg_answer, same_target = self.model_step(batch)
        try:
            loss = self.criterion(seq_out, targets, neg_answer, same_target, user_ids)
        except:
            loss = self.criterion(input_ids, targets, neg_answer, same_target, user_ids)
        # I believe the loss is a scalar
        # preds is a tensor of shape [batch_size, hidden_size]
        # targets is a tensor of shape [batch_size, 1]

        # update and log metrics
        # the original SASRec uses rating matrix to get the top-k items and then calculate the metrics
        self.train_loss(loss) # this one is okay

        # thing to be clarified
        # 1. the usage of torchmetrics ndcg and hitrate (OK)
        # 2. the usage of the rating matrix and the top-k items
        # 3. the latest common point between the original seqrec and if I want to use torchmetrics
        # 4. if necessary, I will use a custom metric.

        # ndcg and the hr using the torchmetrics needs indexes input
        # the indexes is a query where the target and prediction belongs
        # this should be the same for our case so we can just put the same values like numpy.zeros.
        # it seems that we need to convert targets as well to binary by using ... in set(targets).
        # in that case we need to get the corresponding items.

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        # the scenario is that when the epoch ends, we want to calculate the metrics.
        # the metrics I used currently is using the torchmetrics
        # for example if we want to use the ndcg, we will require index, prediction, and target.
        # in which the index should be the same for our case in seqrec module
        # as for the targets, it should be a one-hot value means that the predicted items is in the target.
        # the position itself indicates the relevance of the item.
        # for example,
        # preds = torch.tensor([[0.5, 0.2, 0.3]])
        # targets = torch.tensor([[0, 1, 0]]) which means the second item exists in the target (the top-1).

        # in training, the data loader is shuffled randomly it means there is no way to trace back the user id.

        # probably the code below still needs to be clarified.
        # the original code only monitor the loss in this stage.
        # print('Train' , self.net.item_embeddings.weight.cpu().detach().numpy()[0,:4])
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        user_ids, _,_, preds, targets, _, _ = self.model_step(batch)
        # they will iterate based on the sequence so it will start from the user id 0 and ends in the last user id.
        # this is the difference between the training and validation.
        # here we can trace back the user id.
        # no validation loss, because there is no neg_answer here

        # we want to pre-process the preds to get the rating pred and batch_user_index
        rating_pred, batch_user_index = self._preprocess(preds, user_ids)
        # rating_pred, batch_user are numpy and stored in cpu
        # filter the rating pred based on the train matrix, the rating_matrix comes from the data set.
        rating_pred = self._filter_ratingpred(self.valid_rating_matrix, rating_pred, batch_user_index)
        #targets = targets.unsqueeze(1) # to make it two dimensions
        # obtain the pred_list and answer_list
        # get one-hot encoding for the targets
        #targets_list = self._one_hot(targets, rating_pred.shape[1])
        #rating_pred = torch.tensor(rating_pred)

        pred_list, answer_list = self._evaluate(rating_pred, targets, batch_idx)
        # up to here the pred_list and answer_list are tensor

        # this one below should seems need to use collect_outputs and gather_outputs.
        #local_vars = {"preds": rating_pred, "answers": targets_list}
        local_vars = {"preds": pred_list, "answers": answer_list}

        # we want to append the results to the dictionary that contains the key for the necessary outputs.
        self.val_step_outputs = self._collect_step_outputs(self.val_step_outputs, local_vars) # this is like an append for the dictionary.

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        # we want to gather the outputs that are collected during the epoch training based on the key
        preds = self._gather_step_outputs(self.val_step_outputs, "preds") # it will give the tensor of the values for the key preds
        targets = self._gather_step_outputs(self.val_step_outputs, "answers") # to make it two dimensions
        #indexes = torch.zeros_like(targets) # this is the same as the original code, the indexes are the same for the targets and preds.
        #print('preds',preds)
        #preds = preds.cuda()
        #targets = targets.cuda()
        self.val_ndcg_5(preds, targets)
        self.val_ndcg_10(preds, targets)
        self.val_hr_5(preds, targets)
        self.val_hr_10(preds, targets)

        #self.val_ndcg_5(preds, targets, indexes)
        #self.val_ndcg_10(preds, targets, indexes)
        #self.val_hr_5(preds, targets, indexes)
        #self.val_hr_10(preds, targets, indexes)

        self.log("val/ndcg_5", self.val_ndcg_5, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ndcg_10", self.val_ndcg_10, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/hr_5", self.val_hr_5, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/hr_10", self.val_hr_10, on_step=False, on_epoch=True, prog_bar=True)


        acc = self.val_ndcg_5.compute()  # get current val ndcg_5 will also use for the reference for checkpoint
        self.val_ndcg_5_best(acc)  # update best so far val ndcg_5
        # log `val_ndcg_5_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/ndcg_best", self.val_ndcg_5_best.compute(), sync_dist=True, prog_bar=True)

        self.val_step_outputs = self._clear_epoch_outputs(self.val_step_outputs) # this is to clear the dictionary that contains the key for the necessary outputs.

        # storing the item embeddings into .npz file
        # this should be at the end of the validation epoch
        # the item embeddings are
        # print('Storing the item embeddings')
        # print('Test' , self.net.item_embeddings.weight.data.cpu().detach().numpy()[0,:4])
        self.items.append(self.net.item_embeddings.weight.data.clone().cpu().detach().numpy())


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # the same as validation_step only differs in the data set
        user_ids, _, _, preds, targets, _, _ = self.model_step(batch)
        # update and log metrics
        # no test loss also

        # we want to pre-process the preds to get the rating pred and batch_user_index
        rating_pred, batch_user_index = self._preprocess(preds, user_ids)
        # rating_pred, batch_user are numpy and stored in cpu
        # filter the rating pred based on the train matrix
        rating_pred = self._filter_ratingpred(self.test_rating_matrix, rating_pred, batch_user_index)
        # obtain the pred_list and answer_list
        #targets = targets.unsqueeze(1) # to make it two dimensions
        # obtain the pred_list and answer_list
        # get one-hot encoding for the targets
        #targets_list = self._one_hot(targets, rating_pred.shape[1])
        #rating_pred = torch.tensor(rating_pred)
        pred_list, answer_list = self._evaluate(rating_pred, targets, batch_idx)
        # we want to append the results to the dictionary that contains the key for the necessary outputs.
        #local_vars = {"preds": rating_pred, "answers": targets_list}
        local_vars = {"preds": pred_list, "answers": answer_list}

        self.test_step_outputs = self._collect_step_outputs(self.test_step_outputs, local_vars) # this is like an append for the dictionary.

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # the same as the validation epoch end but only different data sets


        # we want to gather the outputs that are collected during the epoch training based on the key

        preds = self._gather_step_outputs(self.test_step_outputs, "preds") # it will give the tensor of the values for the key preds
        targets = self._gather_step_outputs(self.test_step_outputs, "answers") # making it two dimensions
        #indexes = torch.zeros_like(targets) # this is the same as the original code, the indexes are the same for the targets and preds.

        self.test_ndcg_5(preds, targets)
        self.test_ndcg_10(preds, targets)
        self.test_hr_5(preds, targets)
        self.test_hr_10(preds, targets)        
        
        #self.test_ndcg_5(preds, targets, indexes)
        #self.test_ndcg_10(preds, targets, indexes)
        #self.test_hr_5(preds, targets, indexes)
        #self.test_hr_10(preds, targets, indexes)

        self.log("test/ndcg_5", self.test_ndcg_5, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ndcg_10", self.test_ndcg_10, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/hr_5", self.test_hr_5, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/hr_10", self.test_hr_10, on_step=False, on_epoch=True, prog_bar=True)


        acc = self.test_ndcg_5.compute()  # get current val ndcg_5 will also use for the reference for checkpoint
        self.test_ndcg_5_best(acc)  # update best so far val ndcg_5
        # log `val_ndcg_5_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("test/ndcg_best", self.test_ndcg_5_best.compute(), sync_dist=True, prog_bar=True)

        self.test_step_outputs = self._clear_epoch_outputs(self.test_step_outputs) # this is to clear the dictionary that contains the key for the necessary outputs.


    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/ndcg_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    # below is to enable the accumulation of the results and calculate the metrics at the end of the epoch

    def _collect_step_outputs(
        self, outputs_dict: Dict[str, List[torch.Tensor]], local_vars
    ) -> Dict[str, List[torch.Tensor]]:
        """Collects user-defined attributes of outputs at the end of a `*_step` in dict.
        They use the locals() so we will get the dictionary with the key as the variable name and the value as the value.
        for example
        def a():
            x = 10
            y = 20
            print(locals())
        a()
        will print {'x': 10, 'y': 20}
        """
        for key in outputs_dict.keys():
            val = local_vars.get(key, []) # to get the value of the key in the local_vars, if not exist then return [] the default is None, a method for dictionary.
            outputs_dict[key].append(val) # the type of the value is a list of tensor.
        return outputs_dict

    def _gather_step_outputs(
        self, outputs_dict: Optional[Dict[str, List[torch.Tensor]]], key: str
    ) -> torch.Tensor:
        if key not in outputs_dict.keys():
            raise AttributeError(f"{key} not in {outputs_dict}")

        outputs = torch.cat([output for output in outputs_dict[key]]) # we will get the values as a concatenated tensor for the given key by the user
        return outputs

    def _clear_epoch_outputs(
        self, outputs_dict: Dict[str, List[torch.Tensor]]
    ) -> Dict[str, List[torch.Tensor]]:
        """Clears the outputs collected during each epoch.
        This needs to be called in on_train_end and so on.
        """
        for key in outputs_dict.keys():
            outputs_dict[key].clear()

        return outputs_dict
    
    #### below comes from the original SeqRec code
    
    def _filter_ratingpred(self, train_matrix, rating_pred, batch_user_index):
        # in the original code, the train_matrix is defined when either valid or test is called.
        # here I will just call it as a variable
        # later in the validate or test, we use train_matrix = self.hparams.others.valid_rating_matrix and so on.
        try:
            rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0 # not sure the effect of this.
            # we can get the train_matrix from the data set.
            # the original call the train_matrix in the main.py
            # maybe we also need this in the main function of the train.py
        except: # bert4rec
            rating_pred = rating_pred[:, :-1] # they reverse it, maybe due to the bidirectional. 
            rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0
        return rating_pred 

    def _preprocess(self, preds, user_ids):
        # get the last output of preds as the next item
        recommend_output = preds[:, -1, :]
        # get the rating prediction
        rating_pred = self._predict_full(recommend_output)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        # get the batch user index
        batch_user_index = user_ids.cpu().numpy()
        return rating_pred, batch_user_index

    def _predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.net.item_embeddings.weight # calls the abstract model SeqRecModel
        # [batch hidden_size ]
        # import pdb; pdb.set_trace()
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred  
    
    def _evaluate(self, rating_pred, answers, batch_idx):
        
        '''
        This is the original code for the model evaluation that is called for each batch (step).
        So probably we need to call this in the validation_step and test_step.
        The need numpy
        rating_pred needs to be in the numpy and detach, but not sure if we need that in the lightning.
        '''
        
        ind = np.argpartition(rating_pred, -20)[:, -20:]
        # Take the corresponding values from the corresponding dimension 
        # according to the returned subscript to get the sub-table of each row of topk
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        # Sort the sub-tables in order of magnitude.
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        # retrieve the original subscript from index again
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
        
        #batch_pred_list = rating_pred[:, -20:][np.arange(len(rating_pred)),::-1] # we will get the top-20 items logits, and reverse it.
        # up to here the important variables are batch_pred_list and answers

        # this one below should seems need to use collect_outputs and gather_outputs.
        global pred_lists, answer_lists # something to do with sanity check
        if batch_idx == 0: # i comes from the loop in the original code, in our case we don't have this since we use the trainer of the lightning.
            pred_lists = np.array(batch_pred_list)
            answer_lists = answers.cpu().data.numpy()
        else:
            pred_lists = np.append(pred_lists, batch_pred_list, axis=0)
            answer_lists = np.append(answer_lists, answers.cpu().data.numpy(), axis=0)

        pred_lists = torch.tensor(pred_lists)
        answer_lists = torch.tensor(answer_lists)
        return pred_lists, answer_lists
    # originally they will call the get_full_score which is written by themselves, but in our case we want to use the torchmetrics so we may need to pre-process it first.
    # the pre-process is needed because the torchmetrics need the indexes of the items and the target.
    # in the newsreclib, if we want to use the gather, we need to initialize the dictionary that contains the key for the necessary outputs.
    # for example dic = {'preds': [], 'answers': []}
    # also it seems that implementing the torchmetrics requires tensors for the inputs so probably, pred_list and answer_list should be tensor.

    #### one hot is for the torchmetrics ndcg
    def _one_hot(self, targets, num_items):
        # get the one-hot encoding for the targets
        one_hot = torch.zeros(targets.size(0), num_items)
        one_hot.scatter_(1, targets, int(1))
        return torch.tensor(one_hot,dtype=torch.long)     
    
if __name__ == "__main__":
    _ = SeqRecLitModule(None, None, None, None)
