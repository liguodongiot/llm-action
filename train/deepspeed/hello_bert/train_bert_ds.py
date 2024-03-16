import datetime
import json
import pathlib
import re
import string
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import random
import datasets
import fire
import loguru
import numpy as np
import pytz
import sh
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.models.roberta import RobertaConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaPreTrainedModel,
)

logger = loguru.logger

######################################################################
############### Dataset Creation Related Functions ###################
######################################################################

TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def collate_function(batch: List[Tuple[List[int], List[int]]],
                     pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collect a list of masked token indices, and labels, and
    batch them, padding to max length in the batch.
    """
    max_length = max(len(token_ids) for token_ids, _ in batch)
    padded_token_ids = [
        token_ids +
        [pad_token_id for _ in range(0, max_length - len(token_ids))]
        for token_ids, _ in batch
    ]
    padded_labels = [
        labels + [pad_token_id for _ in range(0, max_length - len(labels))]
        for _, labels in batch
    ]
    src_tokens = torch.LongTensor(padded_token_ids)
    tgt_tokens = torch.LongTensor(padded_labels)
    attention_mask = src_tokens.ne(pad_token_id).type_as(src_tokens)
    return {
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "attention_mask": attention_mask,
    }


def masking_function(
        text: str,
        tokenizer: TokenizerType,
        mask_prob: float,
        random_replace_prob: float,
        unmask_replace_prob: float,
        max_length: int,
) -> Tuple[List[int], List[int]]:
    """Given a text string, randomly mask wordpieces for Bert MLM
    training.

    Args:
        text (str):
            The input text
        tokenizer (TokenizerType):
            The tokenizer for tokenization
        mask_prob (float):
            What fraction of tokens to mask
        random_replace_prob (float):
            Of the masked tokens, how many should be replaced with
            random tokens (improves performance)
        unmask_replace_prob (float):
            Of the masked tokens, how many should be replaced with
            the original token (improves performance)
        max_length (int):
            The maximum sequence length to consider. Note that for
            Bert style models, this is a function of the number of
            positional embeddings you learn

    Returns:
        Tuple[List[int], List[int]]:
            The masked token ids (based on the tokenizer passed),
            and the output labels (padded with `tokenizer.pad_token_id`)
    """
    # Note: By default, encode does add the BOS and EOS token
    # Disabling that behaviour to make this more clear
    tokenized_ids = ([tokenizer.bos_token_id] +
                     tokenizer.encode(text,
                                      add_special_tokens=False,
                                      truncation=True,
                                      max_length=max_length - 2) +
                     [tokenizer.eos_token_id])
    seq_len = len(tokenized_ids)
    tokenized_ids = np.array(tokenized_ids)
    subword_mask = np.full(len(tokenized_ids), False)

    # Masking the BOS and EOS token leads to slightly worse performance
    low = 1
    high = len(subword_mask) - 1
    mask_choices = np.arange(low, high)
    num_subwords_to_mask = max(
        int((mask_prob * (high - low)) + np.random.rand()), 1)
    subword_mask[np.random.choice(mask_choices,
                                  num_subwords_to_mask,
                                  replace=False)] = True

    # Create the labels first
    labels = np.full(seq_len, tokenizer.pad_token_id)
    labels[subword_mask] = tokenized_ids[subword_mask]

    tokenized_ids[subword_mask] = tokenizer.mask_token_id

    # Now of the masked tokens, choose how many to replace with random and how many to unmask
    rand_or_unmask_prob = random_replace_prob + unmask_replace_prob
    if rand_or_unmask_prob > 0:
        rand_or_unmask = subword_mask & (np.random.rand(len(tokenized_ids)) <
                                         rand_or_unmask_prob)
        if random_replace_prob == 0:
            unmask = rand_or_unmask
            rand_mask = None
        elif unmask_replace_prob == 0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = unmask_replace_prob / rand_or_unmask_prob
            decision = np.random.rand(len(tokenized_ids)) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)
        if unmask is not None:
            tokenized_ids[unmask] = labels[unmask]
        if rand_mask is not None:
            weights = np.ones(tokenizer.vocab_size)
            weights[tokenizer.all_special_ids] = 0
            probs = weights / weights.sum()
            num_rand = rand_mask.sum()
            tokenized_ids[rand_mask] = np.random.choice(tokenizer.vocab_size,
                                                        num_rand,
                                                        p=probs)
    return tokenized_ids.tolist(), labels.tolist()


class WikiTextMLMDataset(Dataset):
    """A [Map style dataset](https://pytorch.org/docs/stable/data.html)
    for iterating over the wikitext dataset. Note that this assumes
    the dataset can fit in memory. For larger datasets
    you'd want to shard them and use an iterable dataset (eg: see
    [Infinibatch](https://github.com/microsoft/infinibatch))

    Args:
        Dataset (datasets.arrow_dataset.Dataset):
            The wikitext dataset
        masking_function (Callable[[str], Tuple[List[int], List[int]]])
            The masking function. To generate one training instance,
            the masking function is applied to the `text` of a dataset
            record

    """
    def __init__(
        self,
        dataset: datasets.arrow_dataset.Dataset,
        masking_function: Callable[[str], Tuple[List[int], List[int]]],
    ) -> None:
        self.dataset = dataset
        self.masking_function = masking_function

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        tokens, labels = self.masking_function(self.dataset[idx]["text"])
        return (tokens, labels)


T = TypeVar("T")


class InfiniteIterator(object):
    def __init__(self, iterable: Iterable[T]) -> None:
        self._iterable = iterable
        self._iterator = iter(self._iterable)

    def __iter__(self):
        return self

    def __next__(self) -> T:
        next_item = None
        try:
            next_item = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._iterable)
            next_item = next(self._iterator)
        return next_item


def create_data_iterator(
        mask_prob: float,
        random_replace_prob: float,
        unmask_replace_prob: float,
        batch_size: int,
        max_seq_length: int = 512,
        tokenizer: str = "roberta-base",
) -> InfiniteIterator:
    """Create the dataloader.

    Args:
        mask_prob (float):
            Fraction of tokens to mask
        random_replace_prob (float):
            Fraction of masked tokens to replace with random token
        unmask_replace_prob (float):
            Fraction of masked tokens to replace with the actual token
        batch_size (int):
            The batch size of the generated tensors
        max_seq_length (int, optional):
            The maximum sequence length for the MLM task. Defaults to 512.
        tokenizer (str, optional):
            The tokenizer to use. Defaults to "roberta-base".

    Returns:
        InfiniteIterator:
            The torch DataLoader, wrapped in an InfiniteIterator class, to
            be able to continuously generate samples

    """
    #wikitext_dataset = datasets.load_dataset("wikitext",
    wikitext_dataset = datasets.load_dataset("/home/guodong.li/code/wikitext.py",
                                             "wikitext-2-v1",
                                             split="train")
    wikitext_dataset = wikitext_dataset.filter(
        lambda record: record["text"] != "").map(
            lambda record: {"text": record["text"].rstrip("\n")})
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("/home/guodong.li/model/roberta-base")

    masking_function_partial = partial(
        masking_function,
        tokenizer=tokenizer,
        mask_prob=mask_prob,
        random_replace_prob=random_replace_prob,
        unmask_replace_prob=unmask_replace_prob,
        max_length=max_seq_length,
    )
    dataset = WikiTextMLMDataset(wikitext_dataset, masking_function_partial)
    collate_fn_partial = partial(collate_function,
                                 pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn_partial)

    return InfiniteIterator(dataloader)


######################################################################
############### Model Creation Related Functions #####################
######################################################################


class RobertaLMHeadWithMaskedPredict(RobertaLMHead):
    def __init__(self,
                 config: RobertaConfig,
                 embedding_weight: Optional[torch.Tensor] = None) -> None:
        super(RobertaLMHeadWithMaskedPredict, self).__init__(config)
        if embedding_weight is not None:
            self.decoder.weight = embedding_weight

    def forward(  # pylint: disable=arguments-differ
        self,
        features: torch.Tensor,
        masked_token_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """The current `transformers` library does not provide support
        for masked_token_indices. This function provides the support, by
        running the final forward pass only for the masked indices. This saves
        memory

        Args:
            features (torch.Tensor):
                The features to select from. Shape (batch, seq_len, h_dim)
            masked_token_indices (torch.Tensor, optional):
                The indices of masked tokens for index select. Defaults to None.
                Shape: (num_masked_tokens,)

        Returns:
            torch.Tensor:
                The index selected features. Shape (num_masked_tokens, h_dim)

        """
        if masked_token_indices is not None:
            features = torch.index_select(
                features.view(-1, features.shape[-1]), 0, masked_token_indices)
        return super().forward(features)


class RobertaMLMModel(RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig, encoder: RobertaModel) -> None:
        super().__init__(config)
        self.encoder = encoder
        self.lm_head = RobertaLMHeadWithMaskedPredict(
            config, self.encoder.embeddings.word_embeddings.weight)
        self.lm_head.apply(self._init_weights)

    def forward(
            self,
            src_tokens: torch.Tensor,
            attention_mask: torch.Tensor,
            tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """The forward pass for the MLM task

        Args:
            src_tokens (torch.Tensor):
                The masked token indices. Shape: (batch, seq_len)
            attention_mask (torch.Tensor):
                The attention mask, since the batches are padded
                to the largest sequence. Shape: (batch, seq_len)
            tgt_tokens (torch.Tensor):
                The output tokens (padded with `config.pad_token_id`)

        Returns:
            torch.Tensor:
                The MLM loss
        """
        # shape: (batch, seq_len, h_dim)
        sequence_output, *_ = self.encoder(input_ids=src_tokens,
                                           attention_mask=attention_mask,
                                           return_dict=False)

        pad_token_id = self.config.pad_token_id
        # (labels have also been padded with pad_token_id)
        # filter out all masked labels
        # shape: (num_masked_tokens,)
        masked_token_indexes = torch.nonzero(
            (tgt_tokens != pad_token_id).view(-1)).view(-1)
        # shape: (num_masked_tokens, vocab_size)
        prediction_scores = self.lm_head(sequence_output, masked_token_indexes)
        # shape: (num_masked_tokens,)
        target = torch.index_select(tgt_tokens.view(-1), 0,
                                    masked_token_indexes)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), target)
        return masked_lm_loss


def create_model(num_layers: int, num_heads: int, ff_dim: int, h_dim: int,
                 dropout: float) -> RobertaMLMModel:
    """Create a Bert model with the specified `num_heads`, `ff_dim`,
    `h_dim` and `dropout`

    Args:
        num_layers (int):
            The number of layers
        num_heads (int):
            The number of attention heads
        ff_dim (int):
            The intermediate hidden size of
            the feed forward block of the
            transformer
        h_dim (int):
            The hidden dim of the intermediate
            representations of the transformer
        dropout (float):
            The value of dropout to be used.
            Note that we apply the same dropout
            to both the attention layers and the
            FF layers

    Returns:
        RobertaMLMModel:
            A Roberta model for MLM task

    """
    roberta_config_dict = {
        "attention_probs_dropout_prob": dropout,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": dropout,
        "hidden_size": h_dim,
        "initializer_range": 0.02,
        "intermediate_size": ff_dim,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 514,
        "model_type": "roberta",
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_layers,
        "pad_token_id": 1,
        "type_vocab_size": 1,
        "vocab_size": 50265,
    }
    roberta_config = RobertaConfig.from_dict(roberta_config_dict)
    roberta_encoder = RobertaModel(roberta_config)
    roberta_model = RobertaMLMModel(roberta_config, roberta_encoder)
    return roberta_model


######################################################################
########### Experiment Management Related Functions ##################
######################################################################


def get_unique_identifier(length: int = 8) -> str:
    """Create a unique identifier by choosing `length`
    random characters from list of ascii characters and numbers
    """
    alphabet = string.ascii_lowercase + string.digits
    uuid = "".join(alphabet[ix]
                   for ix in np.random.choice(len(alphabet), length))
    return uuid


def create_experiment_dir(checkpoint_dir: pathlib.Path,
                          all_arguments: Dict[str, Any]) -> pathlib.Path:
    """Create an experiment directory and save all arguments in it.
    Additionally, also store the githash and gitdiff. Finally create
    a directory for `Tensorboard` logs. The structure would look something
    like
        checkpoint_dir
            `-experiment-name
                |- hparams.json
                |- githash.log
                |- gitdiff.log
                `- tb_dir/

    Args:
        checkpoint_dir (pathlib.Path):
            The base checkpoint directory
        all_arguments (Dict[str, Any]):
            The arguments to save

    Returns:
        pathlib.Path: The experiment directory
    """
    # experiment name follows the following convention
    # {exp_type}.{YYYY}.{MM}.{DD}.{HH}.{MM}.{SS}.{uuid}
    current_time = datetime.datetime.now(pytz.timezone("US/Pacific"))
    expname = "bert_pretrain.{0}.{1}.{2}.{3}.{4}.{5}.{6}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
        current_time.second,
        get_unique_identifier(),
    )
    exp_dir = checkpoint_dir / expname
    exp_dir.mkdir(exist_ok=False)
    hparams_file = exp_dir / "hparams.json"
    with hparams_file.open("w") as handle:
        json.dump(obj=all_arguments, fp=handle, indent=2)
    # Save the git hash
    try:
        gitlog = sh.git.log("-1", format="%H", _tty_out=False, _fg=False)
        with (exp_dir / "githash.log").open("w") as handle:
            handle.write(gitlog.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_128:
        logger.info("Seems like the code is not running from"
                    " within a git repo, so hash will"
                    " not be stored. However, it"
                    " is strongly advised to use"
                    " version control.")
    # And the git diff
    try:
        gitdiff = sh.git.diff(_fg=False, _tty_out=False)
        with (exp_dir / "gitdiff.log").open("w") as handle:
            handle.write(gitdiff.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_129:
        logger.info("Seems like the code is not running from"
                    " within a git repo, so diff will"
                    " not be stored. However, it"
                    " is strongly advised to use"
                    " version control.")
    # Finally create the Tensorboard Dir
    tb_dir = exp_dir / "tb_dir"
    tb_dir.mkdir()
    return exp_dir


######################################################################
################ Checkpoint Related Functions ########################
######################################################################


def load_model_checkpoint(
    load_checkpoint_dir: pathlib.Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
    """Loads the optimizer state dict and model state dict from the load_checkpoint_dir
    into the passed model and optimizer. Searches for the most recent checkpoint to
    load from

    Args:
        load_checkpoint_dir (pathlib.Path):
            The base checkpoint directory to load from
        model (torch.nn.Module):
            The model to load the checkpoint weights into
        optimizer (torch.optim.Optimizer):
            The optimizer to load the checkpoint weigths into

    Returns:
        Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
            The checkpoint step, model with state_dict loaded and
            optimizer with state_dict loaded

    """
    logger.info(
        f"Loading model and optimizer checkpoint from {load_checkpoint_dir}")
    checkpoint_files = list(
        filter(
            lambda path: re.search(r"iter_(?P<iter_no>\d+)\.pt", path.name) is
            not None,
            load_checkpoint_dir.glob("*.pt"),
        ))
    assert len(checkpoint_files) > 0, "No checkpoints found in directory"
    checkpoint_files = sorted(
        checkpoint_files,
        key=lambda path: int(
            re.search(r"iter_(?P<iter_no>\d+)\.pt", path.name).group("iter_no")
        ),
    )
    latest_checkpoint_path = checkpoint_files[-1]
    checkpoint_step = int(
        re.search(r"iter_(?P<iter_no>\d+)\.pt",
                  latest_checkpoint_path.name).group("iter_no"))

    state_dict = torch.load(latest_checkpoint_path)
    model.load_state_dict(state_dict["model"], strict=True)
    optimizer.load_state_dict(state_dict["optimizer"])
    logger.info(
        f"Loading model and optimizer checkpoints done. Loaded from {latest_checkpoint_path}"
    )
    return checkpoint_step, model, optimizer


######################################################################
######################## Driver Functions ############################
######################################################################


def train(
        checkpoint_dir: str = None,
        load_checkpoint_dir: str = None,
        # Dataset Parameters
        mask_prob: float = 0.15,
        random_replace_prob: float = 0.1,
        unmask_replace_prob: float = 0.1,
        max_seq_length: int = 512,
        tokenizer: str = "roberta-base",
        # Model Parameters
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 512,
        h_dim: int = 256,
        dropout: float = 0.1,
        # Training Parameters
        batch_size: int = 8,
        num_iterations: int = 10000,
        checkpoint_every: int = 1000,
        log_every: int = 10,
        local_rank: int = -1,
) -> pathlib.Path:
    """Trains a [Bert style](https://arxiv.org/pdf/1810.04805.pdf)
    (transformer encoder only) model for MLM Task

    Args:
        checkpoint_dir (str):
            The base experiment directory to save experiments to
        mask_prob (float, optional):
            The fraction of tokens to mask. Defaults to 0.15.
        random_replace_prob (float, optional):
            The fraction of masked tokens to replace with random token.
            Defaults to 0.1.
        unmask_replace_prob (float, optional):
            The fraction of masked tokens to leave unchanged.
            Defaults to 0.1.
        max_seq_length (int, optional):
            The maximum sequence length of the examples. Defaults to 512.
        tokenizer (str, optional):
            The tokenizer to use. Defaults to "roberta-base".
        num_layers (int, optional):
            The number of layers in the Bert model. Defaults to 6.
        num_heads (int, optional):
            Number of attention heads to use. Defaults to 8.
        ff_dim (int, optional):
            Size of the intermediate dimension in the FF layer.
            Defaults to 512.
        h_dim (int, optional):
            Size of intermediate representations.
            Defaults to 256.
        dropout (float, optional):
            Amout of Dropout to use. Defaults to 0.1.
        batch_size (int, optional):
            The minibatch size. Defaults to 8.
        num_iterations (int, optional):
            Total number of iterations to run the model for.
            Defaults to 10000.
        checkpoint_every (int, optional):
            Save checkpoint after these many steps.

            ..note ::

                You want this to be frequent enough that you can
                resume training in case it crashes, but not so much
                that you fill up your entire storage !

            Defaults to 1000.
        log_every (int, optional):
            Print logs after these many steps. Defaults to 10.
        local_rank (int, optional):
            Which GPU to run on (-1 for CPU). Defaults to -1.

    Returns:
        pathlib.Path: The final experiment directory

    """
    device = (torch.device("cuda", local_rank) if (local_rank > -1)
              and torch.cuda.is_available() else torch.device("cpu"))
    ################################
    ###### Create Exp. Dir #########
    ################################
    if checkpoint_dir is None and load_checkpoint_dir is None:
        logger.error("Need to specify one of checkpoint_dir"
                     " or load_checkpoint_dir")
        return
    if checkpoint_dir is not None and load_checkpoint_dir is not None:
        logger.error("Cannot specify both checkpoint_dir"
                     " and load_checkpoint_dir")
        return
    if checkpoint_dir:
        logger.info("Creating Experiment Directory")
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        all_arguments = {
            # Dataset Params
            "mask_prob": mask_prob,
            "random_replace_prob": random_replace_prob,
            "unmask_replace_prob": unmask_replace_prob,
            "max_seq_length": max_seq_length,
            "tokenizer": tokenizer,
            # Model Params
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "h_dim": h_dim,
            "dropout": dropout,
            # Training Params
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "checkpoint_every": checkpoint_every,
        }
        exp_dir = create_experiment_dir(checkpoint_dir, all_arguments)
        logger.info(f"Experiment Directory created at {exp_dir}")
    else:
        logger.info("Loading from Experiment Directory")
        load_checkpoint_dir = pathlib.Path(load_checkpoint_dir)
        assert load_checkpoint_dir.exists()
        with (load_checkpoint_dir / "hparams.json").open("r") as handle:
            hparams = json.load(handle)
        # Set the hparams
        # Dataset Params
        mask_prob = hparams.get("mask_prob", mask_prob)
        tokenizer = hparams.get("tokenizer", tokenizer)
        random_replace_prob = hparams.get("random_replace_prob",
                                          random_replace_prob)
        unmask_replace_prob = hparams.get("unmask_replace_prob",
                                          unmask_replace_prob)
        max_seq_length = hparams.get("max_seq_length", max_seq_length)
        # Model Params
        ff_dim = hparams.get("ff_dim", ff_dim)
        h_dim = hparams.get("h_dim", h_dim)
        dropout = hparams.get("dropout", dropout)
        num_layers = hparams.get("num_layers", num_layers)
        num_heads = hparams.get("num_heads", num_heads)
        # Training Params
        batch_size = hparams.get("batch_size", batch_size)
        _num_iterations = hparams.get("num_iterations", num_iterations)
        num_iterations = max(num_iterations, _num_iterations)
        checkpoint_every = hparams.get("checkpoint_every", checkpoint_every)
        exp_dir = load_checkpoint_dir
    # Tensorboard writer
    tb_dir = exp_dir / "tb_dir"
    assert tb_dir.exists()
    summary_writer = SummaryWriter(log_dir=tb_dir)
    ################################
    ###### Create Datasets #########
    ################################
    logger.info("Creating Datasets")
    data_iterator = create_data_iterator(
        mask_prob=mask_prob,
        random_replace_prob=random_replace_prob,
        unmask_replace_prob=unmask_replace_prob,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )
    logger.info("Dataset Creation Done")
    ################################
    ###### Create Model ############
    ################################
    logger.info("Creating Model")
    model = create_model(
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        h_dim=h_dim,
        dropout=dropout,
    )
    model = model.to(device)
    logger.info("Model Creation Done")
    ################################
    ###### Create Optimizer #######
    ################################
    logger.info("Creating Optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    logger.info("Optimizer Creation Done")
    ################################
    #### Load Model checkpoint #####
    ################################
    start_step = 1
    if load_checkpoint_dir is not None:
        checkpoint_step, model, optimizer = load_model_checkpoint(
            load_checkpoint_dir, model, optimizer)
        start_step = checkpoint_step + 1

    ################################
    ####### The Training Loop ######
    ################################
    logger.info(
        f"Total number of model parameters: {sum([p.numel() for p in model.parameters()]):,d}"
    )
    model.train()
    losses = []
    for step, batch in enumerate(data_iterator, start=start_step):
        if step >= num_iterations:
            break
        optimizer.zero_grad()
        # Move the tensors to device
        for key, value in batch.items():
            batch[key] = value.to(device)
        # Forward pass
        loss = model(**batch)
        # Backward pass
        loss.backward()
        # Optimizer Step
        optimizer.step()
        losses.append(loss.item())
        if step % log_every == 0:
            logger.info("Loss: {0:.4f}".format(np.mean(losses)))
            summary_writer.add_scalar(f"Train/loss", np.mean(losses), step)
        if step % checkpoint_every == 0:
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(obj=state_dict,
                       f=str(exp_dir / f"checkpoint.iter_{step}.pt"))
            logger.info("Saved model to {0}".format(
                (exp_dir / f"checkpoint.iter_{step}.pt")))
    # Save the last checkpoint if not saved yet
    if step % checkpoint_every != 0:
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(obj=state_dict,
                   f=str(exp_dir / f"checkpoint.iter_{step}.pt"))
        logger.info("Saved model to {0}".format(
            (exp_dir / f"checkpoint.iter_{step}.pt")))

    return exp_dir


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(0)
    fire.Fire(train)
(llama-venv-py310-cu117) [guodong.li@ai-app-2-46 HelloDeepSpeed]$ cat train_bert_ds.py
"""
Modified version of train_bert.py that adds DeepSpeed
"""

import os
import datetime
import json
import pathlib
import re
import string
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import random
import datasets
import fire
import logging
import loguru
import numpy as np
import pytz
import sh
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.models.roberta import RobertaConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaPreTrainedModel,
)


def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


######################################################################
####################### Logging Functions ############################
######################################################################

logger = loguru.logger


def log_dist(message: str,
             ranks: List[int] = [],
             level: int = logging.INFO) -> None:
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks:
        if level == logging.INFO:
            logger.info(f'[Rank {my_rank}] {message}')
        if level == logging.ERROR:
            logger.error(f'[Rank {my_rank}] {message}')
        if level == logging.DEBUG:
            logger.debug(f'[Rank {my_rank}] {message}')


######################################################################
############### Dataset Creation Related Functions ###################
######################################################################

TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def collate_function(batch: List[Tuple[List[int], List[int]]],
                     pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collect a list of masked token indices, and labels, and
    batch them, padding to max length in the batch.
    """
    max_length = max(len(token_ids) for token_ids, _ in batch)
    padded_token_ids = [
        token_ids +
        [pad_token_id for _ in range(0, max_length - len(token_ids))]
        for token_ids, _ in batch
    ]
    padded_labels = [
        labels + [pad_token_id for _ in range(0, max_length - len(labels))]
        for _, labels in batch
    ]
    src_tokens = torch.LongTensor(padded_token_ids)
    tgt_tokens = torch.LongTensor(padded_labels)
    attention_mask = src_tokens.ne(pad_token_id).type_as(src_tokens)
    return {
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "attention_mask": attention_mask,
    }


def masking_function(
        text: str,
        tokenizer: TokenizerType,
        mask_prob: float,
        random_replace_prob: float,
        unmask_replace_prob: float,
        max_length: int,
) -> Tuple[List[int], List[int]]:
    """Given a text string, randomly mask wordpieces for Bert MLM
    training.

    Args:
        text (str):
            The input text
        tokenizer (TokenizerType):
            The tokenizer for tokenization
        mask_prob (float):
            What fraction of tokens to mask
        random_replace_prob (float):
            Of the masked tokens, how many should be replaced with
            random tokens (improves performance)
        unmask_replace_prob (float):
            Of the masked tokens, how many should be replaced with
            the original token (improves performance)
        max_length (int):
            The maximum sequence length to consider. Note that for
            Bert style models, this is a function of the number of
            positional embeddings you learn

    Returns:
        Tuple[List[int], List[int]]:
            The masked token ids (based on the tokenizer passed),
            and the output labels (padded with `tokenizer.pad_token_id`)
    """
    # Note: By default, encode does add the BOS and EOS token
    # Disabling that behaviour to make this more clear
    tokenized_ids = ([tokenizer.bos_token_id] +
                     tokenizer.encode(text,
                                      add_special_tokens=False,
                                      truncation=True,
                                      max_length=max_length - 2) +
                     [tokenizer.eos_token_id])
    seq_len = len(tokenized_ids)
    tokenized_ids = np.array(tokenized_ids)
    subword_mask = np.full(len(tokenized_ids), False)

    # Masking the BOS and EOS token leads to slightly worse performance
    low = 1
    high = len(subword_mask) - 1
    mask_choices = np.arange(low, high)
    num_subwords_to_mask = max(
        int((mask_prob * (high - low)) + np.random.rand()), 1)
    subword_mask[np.random.choice(mask_choices,
                                  num_subwords_to_mask,
                                  replace=False)] = True

    # Create the labels first
    labels = np.full(seq_len, tokenizer.pad_token_id)
    labels[subword_mask] = tokenized_ids[subword_mask]

    tokenized_ids[subword_mask] = tokenizer.mask_token_id

    # Now of the masked tokens, choose how many to replace with random and how many to unmask
    rand_or_unmask_prob = random_replace_prob + unmask_replace_prob
    if rand_or_unmask_prob > 0:
        rand_or_unmask = subword_mask & (np.random.rand(len(tokenized_ids)) <
                                         rand_or_unmask_prob)
        if random_replace_prob == 0:
            unmask = rand_or_unmask
            rand_mask = None
        elif unmask_replace_prob == 0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = unmask_replace_prob / rand_or_unmask_prob
            decision = np.random.rand(len(tokenized_ids)) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)
        if unmask is not None:
            tokenized_ids[unmask] = labels[unmask]
        if rand_mask is not None:
            weights = np.ones(tokenizer.vocab_size)
            weights[tokenizer.all_special_ids] = 0
            probs = weights / weights.sum()
            num_rand = rand_mask.sum()
            tokenized_ids[rand_mask] = np.random.choice(tokenizer.vocab_size,
                                                        num_rand,
                                                        p=probs)
    return tokenized_ids.tolist(), labels.tolist()


class WikiTextMLMDataset(Dataset):
    """A [Map style dataset](https://pytorch.org/docs/stable/data.html)
    for iterating over the wikitext dataset. Note that this assumes
    the dataset can fit in memory. For larger datasets
    you'd want to shard them and use an iterable dataset (eg: see
    [Infinibatch](https://github.com/microsoft/infinibatch))

    Args:
        Dataset (datasets.arrow_dataset.Dataset):
            The wikitext dataset
        masking_function (Callable[[str], Tuple[List[int], List[int]]])
            The masking function. To generate one training instance,
            the masking function is applied to the `text` of a dataset
            record

    """
    def __init__(
        self,
        dataset: datasets.arrow_dataset.Dataset,
        masking_function: Callable[[str], Tuple[List[int], List[int]]],
    ) -> None:
        self.dataset = dataset
        self.masking_function = masking_function

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        tokens, labels = self.masking_function(self.dataset[idx]["text"])
        return (tokens, labels)


T = TypeVar("T")


class InfiniteIterator(object):
    def __init__(self, iterable: Iterable[T]) -> None:
        self._iterable = iterable
        self._iterator = iter(self._iterable)

    def __iter__(self):
        return self

    def __next__(self) -> T:
        next_item = None
        try:
            next_item = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._iterable)
            next_item = next(self._iterator)
        return next_item


def create_data_iterator(
        mask_prob: float,
        random_replace_prob: float,
        unmask_replace_prob: float,
        batch_size: int,
        max_seq_length: int = 512,
        tokenizer: str = "roberta-base",
) -> InfiniteIterator:
    """Create the dataloader.

    Args:
        mask_prob (float):
            Fraction of tokens to mask
        random_replace_prob (float):
            Fraction of masked tokens to replace with random token
        unmask_replace_prob (float):
            Fraction of masked tokens to replace with the actual token
        batch_size (int):
            The batch size of the generated tensors
        max_seq_length (int, optional):
            The maximum sequence length for the MLM task. Defaults to 512.
        tokenizer (str, optional):
            The tokenizer to use. Defaults to "roberta-base".

    Returns:
        InfiniteIterator:
            The torch DataLoader, wrapped in an InfiniteIterator class, to
            be able to continuously generate samples

    """
    #wikitext_dataset = datasets.load_dataset("wikitext",
    wikitext_dataset = datasets.load_dataset("/home/guodong.li/code/wikitext.py",
                                             "wikitext-2-v1",
                                             split="train")
    wikitext_dataset = wikitext_dataset.filter(
        lambda record: record["text"] != "").map(
            lambda record: {"text": record["text"].rstrip("\n")})
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("/home/guodong.li/model/roberta-base")

    masking_function_partial = partial(
        masking_function,
        tokenizer=tokenizer,
        mask_prob=mask_prob,
        random_replace_prob=random_replace_prob,
        unmask_replace_prob=unmask_replace_prob,
        max_length=max_seq_length,
    )
    dataset = WikiTextMLMDataset(wikitext_dataset, masking_function_partial)
    collate_fn_partial = partial(collate_function,
                                 pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn_partial)

    return InfiniteIterator(dataloader)


######################################################################
############### Model Creation Related Functions #####################
######################################################################


class RobertaLMHeadWithMaskedPredict(RobertaLMHead):
    def __init__(self,
                 config: RobertaConfig,
                 embedding_weight: Optional[torch.Tensor] = None) -> None:
        super(RobertaLMHeadWithMaskedPredict, self).__init__(config)
        if embedding_weight is not None:
            self.decoder.weight = embedding_weight

    def forward(  # pylint: disable=arguments-differ
        self,
        features: torch.Tensor,
        masked_token_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """The current `transformers` library does not provide support
        for masked_token_indices. This function provides the support, by
        running the final forward pass only for the masked indices. This saves
        memory

        Args:
            features (torch.Tensor):
                The features to select from. Shape (batch, seq_len, h_dim)
            masked_token_indices (torch.Tensor, optional):
                The indices of masked tokens for index select. Defaults to None.
                Shape: (num_masked_tokens,)

        Returns:
            torch.Tensor:
                The index selected features. Shape (num_masked_tokens, h_dim)

        """
        if masked_token_indices is not None:
            features = torch.index_select(
                features.view(-1, features.shape[-1]), 0, masked_token_indices)
        return super().forward(features)


class RobertaMLMModel(RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig, encoder: RobertaModel) -> None:
        super().__init__(config)
        self.encoder = encoder
        self.lm_head = RobertaLMHeadWithMaskedPredict(
            config, self.encoder.embeddings.word_embeddings.weight)
        self.lm_head.apply(self._init_weights)

    def forward(
            self,
            src_tokens: torch.Tensor,
            attention_mask: torch.Tensor,
            tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """The forward pass for the MLM task

        Args:
            src_tokens (torch.Tensor):
                The masked token indices. Shape: (batch, seq_len)
            attention_mask (torch.Tensor):
                The attention mask, since the batches are padded
                to the largest sequence. Shape: (batch, seq_len)
            tgt_tokens (torch.Tensor):
                The output tokens (padded with `config.pad_token_id`)

        Returns:
            torch.Tensor:
                The MLM loss
        """
        # shape: (batch, seq_len, h_dim)
        sequence_output, *_ = self.encoder(input_ids=src_tokens,
                                           attention_mask=attention_mask,
                                           return_dict=False)

        pad_token_id = self.config.pad_token_id
        # (labels have also been padded with pad_token_id)
        # filter out all masked labels
        # shape: (num_masked_tokens,)
        masked_token_indexes = torch.nonzero(
            (tgt_tokens != pad_token_id).view(-1)).view(-1)
        # shape: (num_masked_tokens, vocab_size)
        prediction_scores = self.lm_head(sequence_output, masked_token_indexes)
        # shape: (num_masked_tokens,)
        target = torch.index_select(tgt_tokens.view(-1), 0,
                                    masked_token_indexes)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), target)
        return masked_lm_loss


def create_model(num_layers: int, num_heads: int, ff_dim: int, h_dim: int,
                 dropout: float) -> RobertaMLMModel:
    """Create a Bert model with the specified `num_heads`, `ff_dim`,
    `h_dim` and `dropout`

    Args:
        num_layers (int):
            The number of layers
        num_heads (int):
            The number of attention heads
        ff_dim (int):
            The intermediate hidden size of
            the feed forward block of the
            transformer
        h_dim (int):
            The hidden dim of the intermediate
            representations of the transformer
        dropout (float):
            The value of dropout to be used.
            Note that we apply the same dropout
            to both the attention layers and the
            FF layers

    Returns:
        RobertaMLMModel:
            A Roberta model for MLM task

    """
    roberta_config_dict = {
        "attention_probs_dropout_prob": dropout,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": dropout,
        "hidden_size": h_dim,
        "initializer_range": 0.02,
        "intermediate_size": ff_dim,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 514,
        "model_type": "roberta",
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_layers,
        "pad_token_id": 1,
        "type_vocab_size": 1,
        "vocab_size": 50265,
    }
    roberta_config = RobertaConfig.from_dict(roberta_config_dict)
    roberta_encoder = RobertaModel(roberta_config)
    roberta_model = RobertaMLMModel(roberta_config, roberta_encoder)
    return roberta_model


######################################################################
########### Experiment Management Related Functions ##################
######################################################################


def get_unique_identifier(length: int = 8) -> str:
    """Create a unique identifier by choosing `length`
    random characters from list of ascii characters and numbers
    """
    alphabet = string.ascii_lowercase + string.digits
    uuid = "".join(alphabet[ix]
                   for ix in np.random.choice(len(alphabet), length))
    return uuid


def create_experiment_dir(checkpoint_dir: pathlib.Path,
                          all_arguments: Dict[str, Any]) -> pathlib.Path:
    """Create an experiment directory and save all arguments in it.
    Additionally, also store the githash and gitdiff. Finally create
    a directory for `Tensorboard` logs. The structure would look something
    like
        checkpoint_dir
            `-experiment-name
                |- hparams.json
                |- githash.log
                |- gitdiff.log
                `- tb_dir/

    Args:
        checkpoint_dir (pathlib.Path):
            The base checkpoint directory
        all_arguments (Dict[str, Any]):
            The arguments to save

    Returns:
        pathlib.Path: The experiment directory
    """
    # experiment name follows the following convention
    # {exp_type}.{YYYY}.{MM}.{DD}.{HH}.{MM}.{SS}.{uuid}
    current_time = datetime.datetime.now(pytz.timezone("US/Pacific"))
    expname = "bert_pretrain.{0}.{1}.{2}.{3}.{4}.{5}.{6}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
        current_time.second,
        get_unique_identifier(),
    )
    exp_dir = checkpoint_dir / expname
    if not is_rank_0():
        return exp_dir
    exp_dir.mkdir(exist_ok=False)
    hparams_file = exp_dir / "hparams.json"
    with hparams_file.open("w") as handle:
        json.dump(obj=all_arguments, fp=handle, indent=2)
    # Save the git hash
    try:
        gitlog = sh.git.log("-1", format="%H", _tty_out=False, _fg=False)
        with (exp_dir / "githash.log").open("w") as handle:
            handle.write(gitlog.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_128:
        log_dist(
            "Seems like the code is not running from"
            " within a git repo, so hash will"
            " not be stored. However, it"
            " is strongly advised to use"
            " version control.",
            ranks=[0],
            level=logging.INFO)
    # And the git diff
    try:
        gitdiff = sh.git.diff(_fg=False, _tty_out=False)
        with (exp_dir / "gitdiff.log").open("w") as handle:
            handle.write(gitdiff.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_129:
        log_dist(
            "Seems like the code is not running from"
            " within a git repo, so diff will"
            " not be stored. However, it"
            " is strongly advised to use"
            " version control.",
            ranks=[0],
            level=logging.INFO)
    # Finally create the Tensorboard Dir
    tb_dir = exp_dir / "tb_dir"
    tb_dir.mkdir(exist_ok=False)
    return exp_dir


######################################################################
################ Checkpoint Related Functions ########################
######################################################################


def load_model_checkpoint(
    load_checkpoint_dir: pathlib.Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
    """Loads the optimizer state dict and model state dict from the load_checkpoint_dir
    into the passed model and optimizer. Searches for the most recent checkpoint to
    load from

    Args:
        load_checkpoint_dir (pathlib.Path):
            The base checkpoint directory to load from
        model (torch.nn.Module):
            The model to load the checkpoint weights into
        optimizer (torch.optim.Optimizer):
            The optimizer to load the checkpoint weigths into

    Returns:
        Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
            The checkpoint step, model with state_dict loaded and
            optimizer with state_dict loaded

    """
    log_dist(
        f"Loading model and optimizer checkpoint from {load_checkpoint_dir}",
        ranks=[0],
        level=logging.INFO)
    checkpoint_files = list(
        filter(
            lambda path: re.search(r"iter_(?P<iter_no>\d+)\.pt", path.name) is
            not None,
            load_checkpoint_dir.glob("*.pt"),
        ))
    assert len(checkpoint_files) > 0, "No checkpoints found in directory"
    checkpoint_files = sorted(
        checkpoint_files,
        key=lambda path: int(
            re.search(r"iter_(?P<iter_no>\d+)\.pt", path.name).group("iter_no")
        ),
    )
    latest_checkpoint_path = checkpoint_files[-1]
    checkpoint_step = int(
        re.search(r"iter_(?P<iter_no>\d+)\.pt",
                  latest_checkpoint_path.name).group("iter_no"))

    state_dict = torch.load(latest_checkpoint_path)
    model.load_state_dict(state_dict["model"], strict=True)
    optimizer.load_state_dict(state_dict["optimizer"])
    log_dist(
        f"Loading model and optimizer checkpoints done. Loaded from {latest_checkpoint_path}",
        ranks=[0],
        level=logging.INFO)
    return checkpoint_step, model, optimizer


######################################################################
######################## Driver Functions ############################
######################################################################


def train(
        checkpoint_dir: str = None,
        load_checkpoint_dir: str = None,
        # Dataset Parameters
        mask_prob: float = 0.15,
        random_replace_prob: float = 0.1,
        unmask_replace_prob: float = 0.1,
        max_seq_length: int = 512,
        tokenizer: str = "roberta-base",
        # Model Parameters
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 512,
        h_dim: int = 256,
        dropout: float = 0.1,
        # Training Parameters
        batch_size: int = 8,
        num_iterations: int = 10000,
        checkpoint_every: int = 1000,
        log_every: int = 10,
        local_rank: int = -1,
) -> pathlib.Path:
    """Trains a [Bert style](https://arxiv.org/pdf/1810.04805.pdf)
    (transformer encoder only) model for MLM Task

    Args:
        checkpoint_dir (str):
            The base experiment directory to save experiments to
        mask_prob (float, optional):
            The fraction of tokens to mask. Defaults to 0.15.
        random_replace_prob (float, optional):
            The fraction of masked tokens to replace with random token.
            Defaults to 0.1.
        unmask_replace_prob (float, optional):
            The fraction of masked tokens to leave unchanged.
            Defaults to 0.1.
        max_seq_length (int, optional):
            The maximum sequence length of the examples. Defaults to 512.
        tokenizer (str, optional):
            The tokenizer to use. Defaults to "roberta-base".
        num_layers (int, optional):
            The number of layers in the Bert model. Defaults to 6.
        num_heads (int, optional):
            Number of attention heads to use. Defaults to 8.
        ff_dim (int, optional):
            Size of the intermediate dimension in the FF layer.
            Defaults to 512.
        h_dim (int, optional):
            Size of intermediate representations.
            Defaults to 256.
        dropout (float, optional):
            Amout of Dropout to use. Defaults to 0.1.
        batch_size (int, optional):
            The minibatch size. Defaults to 8.
        num_iterations (int, optional):
            Total number of iterations to run the model for.
            Defaults to 10000.
        checkpoint_every (int, optional):
            Save checkpoint after these many steps.

            ..note ::

                You want this to be frequent enough that you can
                resume training in case it crashes, but not so much
                that you fill up your entire storage !

            Defaults to 1000.
        log_every (int, optional):
            Print logs after these many steps. Defaults to 10.
        local_rank (int, optional):
            Which GPU to run on (-1 for CPU). Defaults to -1.

    Returns:
        pathlib.Path: The final experiment directory

    """
    device = (torch.device("cuda", local_rank) if (local_rank > -1)
              and torch.cuda.is_available() else torch.device("cpu"))
    ################################
    ###### Create Exp. Dir #########
    ################################
    if checkpoint_dir is None and load_checkpoint_dir is None:
        log_dist(
            "Need to specify one of checkpoint_dir"
            " or load_checkpoint_dir",
            ranks=[0],
            level=logging.ERROR)
        return
    if checkpoint_dir is not None and load_checkpoint_dir is not None:
        log_dist(
            "Cannot specify both checkpoint_dir"
            " and load_checkpoint_dir",
            ranks=[0],
            level=logging.ERROR)
        return
    if checkpoint_dir:
        log_dist("Creating Experiment Directory",
                 ranks=[0],
                 level=logging.INFO)
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        all_arguments = {
            # Dataset Params
            "mask_prob": mask_prob,
            "random_replace_prob": random_replace_prob,
            "unmask_replace_prob": unmask_replace_prob,
            "max_seq_length": max_seq_length,
            "tokenizer": tokenizer,
            # Model Params
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "h_dim": h_dim,
            "dropout": dropout,
            # Training Params
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "checkpoint_every": checkpoint_every,
        }
        exp_dir = create_experiment_dir(checkpoint_dir, all_arguments)
        log_dist(f"Experiment Directory created at {exp_dir}",
                 ranks=[0],
                 level=logging.INFO)
    else:
        log_dist("Loading from Experiment Directory",
                 ranks=[0],
                 level=logging.INFO)
        load_checkpoint_dir = pathlib.Path(load_checkpoint_dir)
        assert load_checkpoint_dir.exists()
        with (load_checkpoint_dir / "hparams.json").open("r") as handle:
            hparams = json.load(handle)
        # Set the hparams
        # Dataset Params
        mask_prob = hparams.get("mask_prob", mask_prob)
        tokenizer = hparams.get("tokenizer", tokenizer)
        random_replace_prob = hparams.get("random_replace_prob",
                                          random_replace_prob)
        unmask_replace_prob = hparams.get("unmask_replace_prob",
                                          unmask_replace_prob)
        max_seq_length = hparams.get("max_seq_length", max_seq_length)
        # Model Params
        ff_dim = hparams.get("ff_dim", ff_dim)
        h_dim = hparams.get("h_dim", h_dim)
        dropout = hparams.get("dropout", dropout)
        num_layers = hparams.get("num_layers", num_layers)
        num_heads = hparams.get("num_heads", num_heads)
        # Training Params
        batch_size = hparams.get("batch_size", batch_size)
        _num_iterations = hparams.get("num_iterations", num_iterations)
        num_iterations = max(num_iterations, _num_iterations)
        checkpoint_every = hparams.get("checkpoint_every", checkpoint_every)
        exp_dir = load_checkpoint_dir
    # Tensorboard writer
    if is_rank_0():
        tb_dir = exp_dir / "tb_dir"
        assert tb_dir.exists()
        summary_writer = SummaryWriter(log_dir=tb_dir)
    ################################
    ###### Create Datasets #########
    ################################
    log_dist("Creating Datasets", ranks=[0], level=logging.INFO)
    data_iterator = create_data_iterator(
        mask_prob=mask_prob,
        random_replace_prob=random_replace_prob,
        unmask_replace_prob=unmask_replace_prob,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )
    log_dist("Dataset Creation Done", ranks=[0], level=logging.INFO)
    ################################
    ###### Create Model ############
    ################################
    log_dist("Creating Model", ranks=[0], level=logging.INFO)
    model = create_model(
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        h_dim=h_dim,
        dropout=dropout,
    )
    log_dist("Model Creation Done", ranks=[0], level=logging.INFO)
    ################################
    ###### DeepSpeed engine ########
    ################################
    log_dist("Creating DeepSpeed engine", ranks=[0], level=logging.INFO)
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu"
            }
        }
    }
    log_dist("-----------------------", ranks=[0], level=logging.INFO)
    log_dist(str(ds_config), ranks=[0], level=logging.INFO)

    model, _, _, _ = deepspeed.initialize(model=model,
                                          model_parameters=model.parameters(),
                                          config=ds_config)
    log_dist("DeepSpeed engine created", ranks=[0], level=logging.INFO)
    ################################
    #### Load Model checkpoint #####
    ################################
    start_step = 1
    if load_checkpoint_dir is not None:
        _, client_state = model.load_checkpoint(load_dir=load_checkpoint_dir)
        checkpoint_step = client_state['checkpoint_step']
        start_step = checkpoint_step + 1

    ################################
    ####### The Training Loop ######
    ################################
    log_dist(
        f"Total number of model parameters: {sum([p.numel() for p in model.parameters()]):,d}",
        ranks=[0],
        level=logging.INFO)
    model.train()
    losses = []
    for step, batch in enumerate(data_iterator, start=start_step):
        if step >= num_iterations:
            break
        # Move the tensors to device
        for key, value in batch.items():
            batch[key] = value.to(device)
        # Forward pass
        loss = model(**batch)
        # Backward pass
        model.backward(loss)
        # Optimizer Step
        model.step()
        losses.append(loss.item())
        if step % log_every == 0:
            log_dist("Loss: {0:.4f}".format(np.mean(losses)),
                     ranks=[0],
                     level=logging.INFO)
            if is_rank_0():
                summary_writer.add_scalar(f"Train/loss", np.mean(losses), step)
        if step % checkpoint_every == 0:
            model.save_checkpoint(save_dir=exp_dir,
                                  client_state={'checkpoint_step': step})
            log_dist("Saved model to {0}".format(exp_dir),
                     ranks=[0],
                     level=logging.INFO)
    # Save the last checkpoint if not saved yet
    if step % checkpoint_every != 0:
        model.save_checkpoint(save_dir=exp_dir,
                              client_state={'checkpoint_step': step})
        log_dist("Saved model to {0}".format(exp_dir),
                 ranks=[0],
                 level=logging.INFO)

    return exp_dir


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(0)
    fire.Fire(train)
