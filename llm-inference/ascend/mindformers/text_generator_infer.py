# mindformers/inference/infers/text_generator_infer.py
"""Text Generator Infer."""

import abc
import time
from typing import Union, List, Optional

import numpy as np
import mindspore_lite as mslite
from mindspore_lite import Model

from mindformers.tools.logger import logger
from mindformers.models import BaseTokenizer
from mindformers.generation import GenerationConfig, LogitsProcessorList
from mindformers.generation.logits_process import RepetitionPenaltyLogitsProcessor, LogitNormalization, \
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from mindformers.generation.streamers import BaseStreamer
from mindformers.generation.utils import softmax_with_threads

from .base_infer import BaseInfer


class BaseInputsOfInfer:
    """
    BaseInputsOfInfer interface.
    """
    @abc.abstractmethod
    def get_inputs(self, model: Model, **kwargs):
        pass

    def get_lite_tensor_list(self, inputs, model):
        input_list = []
        for item in inputs:
            if item is None:
                continue
            input_list.append(item)
        lite_inputs = model.get_inputs()
        for input_np, tensor in zip(input_list, lite_inputs):
            tensor.set_data_from_numpy(input_np)
        return lite_inputs


class CommonInputsOfInfer(BaseInputsOfInfer):
    """
    common infer inputs of llm models.
    """
    # pylint: disable=W0221
    def get_inputs(self, model: Model, input_ids=None, current_index=None, valid_length=None,
                   init_reset=None, is_first_iteration=True, **kwargs):
        if not is_first_iteration:
            inputs_tmp = []
            for i in range(len(current_index)):
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
                # use numpy to slice array to avoid complie ascend slice op
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
            input_ids = np.array(inputs_tmp, dtype=np.int32)
        inputs = [input_ids, current_index, init_reset, valid_length]
        lite_inputs = self.get_lite_tensor_list(inputs, model)
        return lite_inputs


class LlamaInputsOfInfer(BaseInputsOfInfer):
    """
    common infer inputs of llm models.
    """
    # pylint: disable=W0613, W0221
    def get_inputs(self, model: Model, input_ids=None, current_index=None, valid_length=None,
                   init_reset=None, is_first_iteration=True, **kwargs):
        if not is_first_iteration:
            inputs_tmp = []
            for i in range(len(valid_length)):
                current_index_tmp = valid_length[i] - 1  # multibatch
                # use numpy to slice array to avoid complie ascend slice op
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
            input_ids = np.array(inputs_tmp, dtype=np.int32)
        inputs = [input_ids, valid_length, kwargs['batch_index'], kwargs['zactivate_len']]
        lite_inputs = self.get_lite_tensor_list(inputs)
        return lite_inputs

    # pylint: disable=W0221
    def get_lite_tensor_list(self, inputs):
        input_tensors = []
        for item in inputs:
            input_tensors.append(mslite.Tensor(item))
        return input_tensors


class GLMInputsOfInfer(BaseInputsOfInfer):
    """
    glm infer inputs.
    """
    def get_masks_np(self, input_ids, tokenizer: BaseTokenizer):
        batch_size, seq_length = input_ids.shape
        context_lengths = [list(seq).index(tokenizer.bos_token_id) for seq in input_ids]
        attention_mask = np.tril(np.ones((batch_size, seq_length, seq_length)))
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask = np.expand_dims(attention_mask, axis=1)
        attention_mask = np.array(attention_mask < 0.5, np.bool_)
        return attention_mask

    def get_position_ids_np(self, input_ids, mask_positions, tokenizer: BaseTokenizer,
                            use_gmasks=None, position_encoding_2d=True):
        """Get position ids from input_ids and mask_positions with numpy"""
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [list(seq).index(tokenizer.bos_token_id) for seq in input_ids]
        if position_encoding_2d:
            position_ids = np.repeat(np.expand_dims(np.arange(seq_length), 0), batch_size, axis=0)
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [np.concatenate((
                np.zeros(context_length, np.int32),
                np.arange(seq_length - context_length, dtype=np.int32) + 1
            )) for context_length in context_lengths]
            block_position_ids = np.stack(block_position_ids, axis=0)
            position_ids = np.stack((position_ids, block_position_ids), axis=1)
        else:
            position_ids = np.repeat(np.expand_dims(np.arange(seq_length), 0), batch_size, axis=0)
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[context_length:] = mask_positions[i]
        return position_ids

    def create_position_ids_np(self, input_ids, tokenizer, position_encoding_2d=True):
        """Get position ids from input_ids with numpy"""
        mask, gmask = tokenizer.mask_token_id, tokenizer.gmask_token_id
        seqs = list(input_ids)

        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gmask if gmask in seq else mask
            use_gmask = mask_token == gmask
            mask_positions.append(list(seq).index(mask_token))
            use_gmasks.append(use_gmask)
        position_ids = self.get_position_ids_np(input_ids, mask_positions, tokenizer,
                                                use_gmasks=None, position_encoding_2d=position_encoding_2d)
        return position_ids

    # pylint: disable=W0221
    def get_inputs(self, model: Model, input_ids=None, current_index=None, valid_length=None,
                   init_reset=None, tokenizer=None, position_encoding_2d=True, is_first_iteration=True, **kwargs):
        attention_mask_ = self.get_masks_np(input_ids, tokenizer).astype(np.int32)
        position_ids_ = self.create_position_ids_np(input_ids, tokenizer, position_encoding_2d).astype(np.int32)

        if not is_first_iteration:
            attention_mask = []
            position_ids = []
            inputs_tmp = []
            for i in range(len(current_index)):
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1]  # multibatch
                # use numpy to slice array to avoid complie ascend slice op
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
                position_ids.append(position_ids_[i][:, current_index_tmp:current_index_tmp + 1])
                attention_mask.append(attention_mask_[i][:, current_index_tmp:current_index_tmp + 1, :])
            input_ids = np.array(inputs_tmp, np.int32)
            position_ids = np.array(position_ids, np.int32)
            attention_mask = np.array(attention_mask, np.int32)
        else:
            attention_mask = attention_mask_
            position_ids = position_ids_

        inputs = [input_ids, position_ids, attention_mask, current_index, init_reset, valid_length]
        lite_inputs = self.get_lite_tensor_list(inputs, model)
        return lite_inputs


class InputOfInfer:
    """
    Input of llm model.
    """
    MAPPING = {
        "bloom": CommonInputsOfInfer,
        "llama": LlamaInputsOfInfer,
        "codellama": LlamaInputsOfInfer,
        "glm2": CommonInputsOfInfer,
        "glm3": CommonInputsOfInfer,
        "gpt2": CommonInputsOfInfer,
        "codegeex2": CommonInputsOfInfer,
        "glm": GLMInputsOfInfer,
        "internlm": LlamaInputsOfInfer,
        "common": CommonInputsOfInfer
    }

    @classmethod
    def get_inputs(cls, model_name: str, model, **kwargs):
        """
        Get input tensor list of mslite.

        Args:
            model_name: str, model name.
            model: mindspore_lite.Model. mode obj of mslite.

        Returns:
            tensor list of mslite.
        """
        name = ""
        if model_name not in InputOfInfer.MAPPING:
            for k in InputOfInfer.MAPPING:
                if model_name.startswith(k):
                    name = k
                    break
            if not name:
                logger.warning("Model name not in support maps.Common input format will be used to do inference.")
                name = "common"
        else:
            name = model_name
        return InputOfInfer.MAPPING[name]().get_inputs(model, **kwargs)


class TextGeneratorInfer(BaseInfer):
    """
    Text generator infer implement class.
    """
    # pylint: disable=W0221
    def infer(self,
              inputs: Union[str, List[str]],
              do_sample: bool = False,
              top_k: int = 1,
              top_p: float = 1.0,
              temperature: float = 1.0,
              repetition_penalty: float = 1.0,
              eos_token_id: int = 2,
              pad_token_id: int = 0,
              max_length: int = 256,
              is_sample_acceleration: bool = False,
              add_special_tokens: bool = False,
              streamer: Optional[BaseStreamer] = None,
              **kwargs):
        """
        text generator inference api

        Args:
            inputs(List(str), List(List(str))): The token id list or a list of token id list.
            do_sample(bool): Whether to do sampling on the candidate ids.
                If set True it will be enabled, and set it to be False to disable the sampling, equivalent to topk 1.
                If set None, it follows the setting in the configureation in the model. Default None.
            top_k(int): Determine the topK numbers token id as candidate. This should be a positive number.
                If set None, it follows the setting in the configureation in the model. Default 1.
            top_p(float): The accumulation probability of the candidate token ids below the top_p will be select as the
                condaite ids. The valid value of top_p is between (0, 1]. If the value is larger than 1,
                top_K algorithm will be enabled. If set None, it follows the setting in the configureation in the model.
                Default 1.
            temperature (`float`, *optional*, defaults to 1.0): The value used to modulate the next token probabilities.
            eos_token_id(int): The end of sentence token id. If set None, it follows the setting in the configureation
                in the model. Default 2.
            pad_token_id(int): The padding of sentence token id. If set None, it follows the setting in the
                configureation in the model. Default 0.
            repetition_penalty(float): The penalty factor of the frequency that generated words. The If set 1,
                the repetition_penalty will not be enabled. If set None, it follows the setting in the configureation in
                the model. Default 1.
            max_length: The maximum length of the generated words. If set None, it follows the setting in the
                configureation in the model. Default 256.
            is_sample_acceleration: The postprocess are processing in model. Default False.
            add_special_tokens: Add special tokens for preprocess.
            streamer: The streamer that generator uses.

        Returns:
            outputs of model infer
        """
        input_ids = self.preprocess(inputs, add_special_tokens)
        print("input_ids: ", input_ids)
        start_time = time.perf_counter()
        output_ids = self.generate(input_ids, do_sample, top_k, top_p, temperature,
                                   repetition_penalty, eos_token_id, pad_token_id,
                                   max_length, is_sample_acceleration, streamer, **kwargs)
        end_time = time.perf_counter()
        gen_time = end_time - start_time
        print("output_ids: ", output_ids)
        input_token_lens = len(input_ids[0])
        output_token_lens = len(output_ids[0])
        new_token_lens = output_token_lens - input_token_lens
        print("gen_time: ", gen_time, "new_token_lens: ", new_token_lens)
        output_ids = output_ids[0][len(input_ids[0]):]

        outputs = self.postprocess(output_ids)
        return outputs, gen_time, new_token_lens

    # pylint: disable=W0613
    def preprocess(self, input_data, add_special_tokens=False, **kwargs):
        """preprocess."""
        if self.model_name.startswith('glm3'):
           return self.tokenizer.build_batch_input(input_data)["input_ids"]

        tokens = self.tokenizer(input_data, add_special_tokens=add_special_tokens)
        input_ids = tokens["input_ids"]
        input_list = []
        if isinstance(input_data, str):
            input_list.append(input_ids)
        else:
            input_list = input_ids
        return input_list

    # pylint: disable=W0613
    def postprocess(self, predict_data, **kwargs):
        """postprocess."""
        outputs = self.tokenizer.decode(predict_data, skip_special_tokens=True)
        return outputs

    def _get_logits_processor(self,
                              generation_config: GenerationConfig,
                              logits_processor: Optional[LogitsProcessorList]):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # instantiate processors list
        processors = LogitsProcessorList()

        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty=generation_config.repetition_penalty))
        processors = self._merge_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors

    def _merge_processor_list(self,
                              default_list: LogitsProcessorList,
                              custom_list: LogitsProcessorList):
        """merge custom processor list with default list."""
        if not custom_list:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `.generate()`, but it has already been created with the values {default}."
                        f" {default} has been created by passing the corresponding arguments to generate or"
                        f" by the model's config default values. If you just want to change the default values"
                        f" of {object_type} consider passing them as arguments to `.generate()`"
                        f" instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def _get_logits_warper(self, generation_config: GenerationConfig):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        min_tokens_to_keep = 1
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers

    def generate(self, input_ids, do_sample, top_k, top_p, temperature, repetition_penalty, eos_token_id,
                 pad_token_id, max_length, is_sample_acceleration, streamer, **kwargs):
        """token generator."""
        total_time = time.time()
        sampler_dict = {"do_sample": do_sample, "top_k": top_k, "top_p": top_p, "temperature": temperature,
                        "repetition_penalty": repetition_penalty, "max_length": max_length, **kwargs}
        generation_config = GenerationConfig(**sampler_dict)
        if not generation_config.do_sample:
            generation_config.top_p = 1.0
            generation_config.top_k = 0

        logits_processor = LogitsProcessorList()
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            logits_processor=logits_processor,
        )
        logits_warper = self._get_logits_warper(generation_config)

        if streamer:
            streamer.put(np.array(input_ids[0]))

        batch_size = len(input_ids)
        valid_length = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length.append(np.max(np.argwhere(np.array(input_ids[i]) != pad_token_id)) + 1)
        valid_length = np.array(valid_length, np.int32)

        print("self.dynamic", self.dynamic)
        if self.dynamic:
            pad_length = max_length - valid_length
            real_pad_length = max(valid_length) - valid_length
            target_length = kwargs["max_new_tokens"] + max(valid_length) if kwargs.get(
                "max_new_tokens") else max_length
        else:
            target_length = self.seq_length if max_length > self.seq_length else max_length
            # pad original input ids to seq_length
            pad_length = self.seq_length - valid_length
        pad_input_ids = np.array([
            np.pad(input_ids[i], (0, pad_length[i]),
                   'constant', constant_values=pad_token_id) for i in range(len(input_ids))
        ], np.int32)

        # setup is_first_iteration flag for incremental infer
        is_first_iteration = True
        is_finished = [False] * batch_size
        use_past = self.full_model and self.cache_model

        if self.dynamic:
            batch_size_gear, act_len_gear = self.dynshape_gears.match(batch_size, max_length)
            bs_pad = batch_size_gear - batch_size
            pad_input_ids = np.pad(pad_input_ids, ((0, bs_pad), (0, 0)), 'constant', constant_values=pad_token_id)
            valid_length = np.pad(valid_length, (0, bs_pad), 'constant', constant_values=1)
            is_finished += [True] * bs_pad
            real_input_ids = np.array([np.pad(input_ids[i], (0, real_pad_length[i]), 'constant',
                                              constant_values=pad_token_id) for i in range(len(input_ids))], np.int32)
            real_input_ids = np.pad(real_input_ids, ((0, bs_pad), (0, 0)), 'constant', constant_values=pad_token_id)
        input_ids = real_input_ids if self.dynamic else pad_input_ids
        activate_len = np.zeros((act_len_gear if self.dynamic else self.seq_length), dtype=np.int64)
        batch_index = np.arange(len(input_ids), dtype=np.int64)

        origin_len = np.sum(valid_length)
        while np.sum(is_finished) != batch_size:
            start_time = time.time()
            seq_length = input_ids.shape[1]
            current_index = [valid_length[i] - 1 + i * seq_length for i in range(batch_size)]
            current_index = np.array(current_index, np.int32)
            logger.debug("validate length: %s", valid_length)

            if use_past:
                outputs = self._inc_infer(input_ids, current_index, valid_length, is_first_iteration,
                                          batch_index=batch_index, zactivate_len=activate_len, **kwargs)
            else:
                outputs = self._full_infer(input_ids, current_index, is_sample_acceleration,
                                           batch_index=batch_index, zactivate_len=activate_len, **kwargs)

            if self.dynamic:
                input_ids = pad_input_ids

            if not is_sample_acceleration:
                logits = outputs[0].get_data_to_numpy()
                vocab_size = logits.shape[-1]
                if len(logits.shape) < 3:
                    logits = logits.reshape(batch_size, -1, vocab_size)

                # gather logits
                if is_first_iteration and logits.shape[1] > 1:
                    logits = np.array([logits[i][current_index[i] - i * seq_length, :] for i in range(batch_size)])
                else:
                    logits = np.array([logits[i][0, :] for i in range(batch_size)])

                logits = logits.reshape(-1, vocab_size)
                log_probs = logits_processor(input_ids, logits, is_finished)
                p = logits_warper(input_ids, log_probs, is_finished)
                p_args = np.tile(np.arange(logits.shape[-1]), (batch_size, 1))
            else:
                p = outputs[0].get_data_to_numpy().astype(np.int32)
                p_args = outputs[1].get_data_to_numpy()

            if generation_config.do_sample:
                p_norms = softmax_with_threads(p, is_finished)

            # Random select a token as final output for this round
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                # target_index = np.random.choice(len(p[i]), p=p[i])
                if generation_config.do_sample:
                    # multinomial sample
                    p_norm = p_norms[i]
                    target_index = np.random.choice(len(p[i]), p=p_norm)
                else:
                    # greedy
                    target_index = np.argmax(p[i])

                # Stop judgment when length exceeds target_length.
                if valid_length[i] >= target_length:
                    is_finished[i] = True
                    continue

                # add next token to input_ids.
                target = p_args[i][target_index]
                input_ids[i, valid_length[i]] = target
                if streamer:
                    streamer.put(np.asarray([target]))
                valid_length[i] += int(1)

                # Stop judgment when output is EOS token, with the output
                # is appended to input_ids and streamer.
                if target == eos_token_id:
                    is_finished[i] = True
                    continue

            is_first_iteration = not use_past
            logger.debug(f"one token takes {time.time() - start_time} s")

        # Return valid outputs out of padded outputs
        output_ids = []
        for i in range(batch_size):
            output_ids.append(input_ids[i, : int(valid_length[i])].astype(np.int32))
        logger.debug("The output is: %s", output_ids)

        generate_len = np.sum(valid_length) - origin_len
        total_time = time.time() - total_time
        logger.info("total time: %s s; generated tokens: %s tokens; generate speed: %s tokens/s",
                    total_time, generate_len, generate_len / total_time)

        if streamer:
            streamer.end()

        return output_ids

    def _inc_infer(self, input_ids, current_index, valid_length, is_first_iteration, **kwargs):
        """kvcache infer"""
        if is_first_iteration:
            init_reset = np.array([False])
            lite_inputs = self.get_predict_inputs(self.full_model, input_ids, current_index,
                                                  valid_length, init_reset, is_first_iteration, **kwargs)
            outputs = self.full_model.predict(lite_inputs)
        else:
            init_reset = np.array([True])
            lite_inputs = self.get_predict_inputs(self.cache_model, input_ids, current_index,
                                                  valid_length, init_reset, is_first_iteration, **kwargs)
            outputs = self.cache_model.predict(lite_inputs)
        return outputs

    def _full_infer(self, input_ids, current_index, is_npu_acceleration, **kwargs):
        """infer"""
        # get inputs
        if is_npu_acceleration:
            lite_inputs = self.get_predict_inputs(self.full_model, input_ids, current_index, **kwargs)
        else:
            lite_inputs = self.get_predict_inputs(self.full_model, input_ids, **kwargs)
        # do infer
        outputs = self.full_model.predict(lite_inputs)
        return outputs

    def get_predict_inputs(self, mode: Model, input_ids, current_index=None,
                           valid_length=None, init_reset=None, is_first_iteration=True, **kwargs):
        """Get inputs of llm model for mslite."""
        return InputOfInfer.get_inputs(self.model_name, mode, input_ids=input_ids, current_index=current_index,
                                       valid_length=valid_length, init_reset=init_reset, tokenizer=self.tokenizer,
                                       is_first_iteration=is_first_iteration, **kwargs)
