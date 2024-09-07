import gc

import numpy
import psutil
import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from llm_attacks_bishopfox import get_decoded_token
from llm_attacks_bishopfox import get_decoded_tokens
from llm_attacks_bishopfox import get_embedding_matrix
from llm_attacks_bishopfox import get_embeddings 
from llm_attacks_bishopfox import get_encoded_token 
from llm_attacks_bishopfox import get_encoded_tokens 

from llm_attacks_bishopfox.attack.attack_classes import AdversarialContent
from llm_attacks_bishopfox.attack.attack_classes import AdversarialContentList

class PaddingException(Exception):
    pass

class NullPaddingTokenException(PaddingException):
    pass

def print_stats(function_name):
    print(f"---")
    print(f"[{function_name}] Resource statistics")
    mem_info = psutil.Process().memory_info()
    print(f"System: {mem_info}")
    for i in range(torch.cuda.device_count()):
        cuda_device_name = torch.cuda.get_device_properties(i).name
        cuda_device_total_memory = torch.cuda.get_device_properties(i).total_memory
        cuda_device_reserved_memory = torch.cuda.memory_reserved(i)
        cuda_device_reserved_allocated_memory = torch.cuda.memory_allocated(i)
        cuda_device_reserved_unallocated_memory = cuda_device_reserved_memory - cuda_device_reserved_allocated_memory
        print(f"CUDA device {i} ({cuda_device_name}) - Total memory: {cuda_device_total_memory}, reserved memory: {cuda_device_reserved_memory}, reserved allocated memory: {cuda_device_reserved_allocated_memory}, reserved unallocated memory: {cuda_device_reserved_unallocated_memory}")
    print(f"---")

def create_new_quantized_tensor(default_value, size, device, quantized_dtype, scale, zero_point):
    temp_dtype = torch.uint8
    if quantized_dtype == torch.qint8:
        temp_dtype = torch.int8
    if quantized_dtype == torch.qint32:
        temp_dtype = torch.int32
    result_int = torch.randint(default_value, (default_value + 1), size=size, device=device, dtype=temp_dtype)

    result = torch._make_per_tensor_quantized_tensor(result_int, scale, zero_point) 
    return result

def get_first_value_from_tensor(t):
    if isinstance(t, torch.Tensor):
        tensor_item = t.tolist()
        if len(tensor_item) > 1:
            tensor_item = tensor_item[0]
        return get_first_value_from_tensor(tensor_item)
    return t
    
def get_padded_target_token_ids(tokenizer, loss_slice, target_ids):
    #print(f"[get_padded_target_token_ids] Debug: target_ids = {target_ids}, loss_slice = {loss_slice}")
    result = target_ids
    return_tensor = None
    input_is_list_of_lists = False
    
    original_target_ids_length = len(target_ids)
    
    target_ids_as_list = None
    if isinstance(target_ids, list):
        target_ids_as_list = copy.deepcopy(target_ids)
        return_tensor = False
    if isinstance(target_ids, torch.Tensor):
        target_ids_as_list = target_ids.tolist()
        return_tensor = True
    
    if return_tensor is None:
        raise PaddingException(f"Couldn't pad the object '{target_ids}' because it was not a list or a tensor.")
    
    if len(target_ids_as_list) == 0:
        return result
    
    len_loss_slice = loss_slice.stop - loss_slice.start
    len_comparison = len(target_ids_as_list)

    # Calls to this function with output from e.g. get_logits are passing a multidimensional array of values that need to be padded
    if isinstance(target_ids_as_list[0], list):
        #print(f"[get_padded_target_token_ids] Debug: target_ids_as_list is a multidimensional array")
        input_is_list_of_lists = True
        len_comparison = len(target_ids_as_list[0])

    #print(f"[get_padded_target_token_ids] Debug: target_ids_as_list = {target_ids_as_list}, len_loss_slice = {len_loss_slice}")
    
    if len_loss_slice > len_comparison:
        if tokenizer.pad_token_id is None:
            # This should never occur unless someone is calling this function directly, outside of the tool
            raise NullPaddingTokenException("The current target slice must be padded to match the length of the loss slice, but the tokenizer's padding token ID is None.")
        
        if input_is_list_of_lists:
            for list_entry_num in range(0, len(target_ids_as_list)):
                #print(f"[get_padded_target_token_ids] Debug: target_ids_as_list[list_entry_num] = '{target_ids_as_list[list_entry_num]}' before padding.")
                while len_loss_slice > len(target_ids_as_list[list_entry_num]):
                    target_ids_as_list[list_entry_num].append(tokenizer.pad_token_id)
                #print(f"[get_padded_target_token_ids] Debug: target_ids_as_list[list_entry_num] = '{target_ids_as_list[list_entry_num]}' after padding.")
        else:
            while len_loss_slice > len(target_ids_as_list):
                target_ids_as_list.append(tokenizer.pad_token_id)
        result = target_ids_as_list
    
    if return_tensor:
        result = torch.tensor(result, device = target_ids.device)
    
    #print(f"[get_padded_target_token_ids] Debug: original_target_ids_length = {original_target_ids_length}, len(result) = {len(result)}, len(target_ids_as_list) = {len(target_ids_as_list)}, len_loss_slice = {len_loss_slice}, result = '{result}', target_ids_as_list = '{target_ids_as_list}'")
    
    return result

#def token_gradients(model, tokenizer, input_ids, input_slice, target_slice, loss_slice):
def token_gradients(model, tokenizer, input_ids, input_id_data):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_id_data.slice_data.control : slice
        The slice of the input sequence for which gradients need to be computed.
    input_id_data.slice_data.target : slice
        The slice of the input sequence to be used as targets.
    input_id_data.slice_data.loss : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in input_id_data.slice_data.control with respect to the loss.
    """

    #print_stats("token_gradients")
    #print("[token_gradients] Getting embedding weight matrix")
    #print(f"[token_gradients] Debug: model.model.embed_tokens = {model.model.embed_tokens}")
    embed_weights = get_embedding_matrix(model)
    #print(f"[token_gradients] Debug: embed_weights = {embed_weights}")
    #print_stats("token_gradients")

    #print(f"[token_gradients] Debug: embed_weights.dtype={embed_weights.dtype}")

    quantized_tensors = False
    if embed_weights is not None:
        if hasattr(embed_weights, "quantization_scheme"):
            quantized_tensors = True
        if embed_weights.data is not None and len(embed_weights.data) > 0:
            #print(f"[token_gradients] Debug: type(embed_weights.data) = {type(embed_weights.data)}")
            #print(f"[token_gradients] Debug: embed_weights.data = {embed_weights.data} with attributes: {dir(embed_weights.data)} and variables: {vars(embed_weights.data)}")
            #if hasattr(embed_weights.data, "qscheme"):
            #    print(f"[token_gradients] Debug: embed_weights.data.qscheme = {embed_weights.data.qscheme}")
            if embed_weights.data.is_quantized:
                quantized_tensors = True

    #print("[token_gradients] Getting one_hot")
    one_hot = None
    scales_value = None
    pczp_value = None
    if quantized_tensors:
        scales = embed_weights.data.q_per_channel_scales()
        pczp = embed_weights.data.q_per_channel_zero_points()
        scales_value = get_first_value_from_tensor(scales)
        pczp_value = int(get_first_value_from_tensor(pczp))
        #print(f"[token_gradients] Debug: scales = {scales}, scales_value = {scales_value}, type(scales_value) = {type(scales_value)}, pczp = {pczp}, pczp_value = {pczp_value}, type(pczp_value) = {type(pczp_value)}")
#        one_hot = torch.randint(0, 1, size=(input_ids[input_id_data.slice_data.control].shape[0],embed_weights.shape[0]), device=model.device, dtype=embed_weights.dtype)
        one_hot = create_new_quantized_tensor(0, (input_ids[input_id_data.slice_data.control].shape[0],embed_weights.shape[0]), model.device, embed_weights.data.dtype, scales_value, pczp_value)
    else:
        one_hot = torch.zeros(
            input_ids[input_id_data.slice_data.control].shape[0],
            embed_weights.shape[0],
            device=model.device,
            dtype=embed_weights.dtype
        )
    #print_stats("token_gradients")

    #print("[token_gradients] Getting one_hot scatter")
    one_hot_ones = None
    if quantized_tensors:
#        one_hot_ones = torch.randint(1, 2, size=(one_hot.shape[0],1), device=model.device, dtype=embed_weights.dtype)
        one_hot_ones = create_new_quantized_tensor(1, (one_hot.shape[0],1), model.device, embed_weights.data.dtype, scales_value, pczp_value)

    else:
        one_hot_ones = torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)

    one_hot.scatter_(
        1, 
        input_ids[input_id_data.slice_data.control].unsqueeze(1),
        one_hot_ones
    )
    #print_stats("token_gradients")

    #print("[token_gradients] one_hot.requires_grad_()")
    one_hot.requires_grad_()
    #print_stats("token_gradients")

    #print("[token_gradients] Getting input_embeds")
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    #print_stats("token_gradients")
    
    # now stitch it together with the rest of the embeddings
    #print("[token_gradients] Getting embeddings")
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    #print_stats("token_gradients")

    #print("[token_gradients] Getting full_embeds")
    full_embeds = torch.cat(
        [
            embeds[:,:input_id_data.slice_data.control.start,:], 
            input_embeds, 
            embeds[:,input_id_data.slice_data.control.stop:,:]
        ], 
        dim=1)
    #print_stats("token_gradients")

    #print("[token_gradients] converting full_embeds to float32 because that's what logits() expects")
    #full_embeds = full_embeds.to(torch.float32)
    #print_stats("token_gradients")

    #print(f"[token_gradients] Debug: full_embeds.dtype: {full_embeds.dtype}")
    
    #print("[token_gradients] Getting logits")
    logits = model(inputs_embeds=full_embeds).logits
    #print_stats("token_gradients")

    #print("[token_gradients] Getting targets")
    targets = input_ids[input_id_data.slice_data.target]
    #print_stats("token_gradients")
    
    # pad the target token IDs, if necessary
    targets = get_padded_target_token_ids(tokenizer, input_id_data.slice_data.loss, targets)
    # len_loss_slice = input_id_data.slice_data.loss.stop - input_id_data.slice_data.loss.start
    # if len_loss_slice > len(targets):
        # if tokenizer.pad_token_id is None:
            # # This should never occur unless someone is calling this function directly, outside of the tool
            # raise NullPaddingTokenException("The current target slice must be padded to match the length of the loss slice, but the tokenizer's padding token ID is None.")
        # targets_as_list = targets.tolist()
        # while len_loss_slice > len(targets_as_list):
            # targets_as_list.append(tokenizer.pad_token_id)
        # targets = torch.tensor(targets_as_list, device = targets.device)

    #print("[token_gradients] Getting loss")
    loss = nn.CrossEntropyLoss()(logits[0,input_id_data.slice_data.loss,:], targets)
    #print_stats("token_gradients")

    #print("[token_gradients] loss.backward()")
    loss.backward()
    #print_stats("token_gradients")
    
    if one_hot.grad is not None:
        #print("[token_gradients] Cloning one_hot.grad")
        grad = one_hot.grad.clone()
        #print_stats("token_gradients")

        #print("[token_gradients] Getting gradients")
        grad = grad / grad.norm(dim=-1, keepdim=True)
        #print_stats("token_gradients")
        return grad

    print("Error: one_hot.grad is None")
    return None

def sample_control(attack_params, adversarial_content_manager, current_adversarial_content, grad, number_of_candidates_to_generate, topk=256, not_allowed_tokens=None, random_seed = None):

    new_adversarial_token_ids = None

    if random_seed is not None:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    rand_int_generator = torch.Generator()
    rand_int_generator.seed()

    if grad is not None:
        if not_allowed_tokens is not None:
            grad[:, not_allowed_tokens.to(grad.device)] = numpy.infty

        top_indices = (-grad).topk(topk, dim=1).indices
        current_adversarial_content_token_ids_device = torch.tensor(current_adversarial_content.token_ids, device = attack_params.device).to(grad.device)

        original_adversarial_content_token_ids_device = current_adversarial_content_token_ids_device.repeat(number_of_candidates_to_generate, 1)

        #print(f"[sample_control] Debug: current_adversarial_content_token_ids_device = {current_adversarial_content_token_ids_device}, original_adversarial_content_token_ids_device = {original_adversarial_content_token_ids_device}, top_indices = {top_indices}")

        num_adversarial_tokens = len(current_adversarial_content_token_ids_device)

        new_token_pos = torch.arange(
            0, 
            num_adversarial_tokens, 
            num_adversarial_tokens / number_of_candidates_to_generate,
            device=grad.device
        ).type(torch.int64)
        num_rand_ints = number_of_candidates_to_generate
        #if top_indices.shape[0] < number_of_candidates_to_generate:
        # There's probably a better way to handle this, but I don't understand the low-level operation here well enough to implement that "better way" yet.
        #if top_indices.shape[0] < topk:
        if top_indices.shape[0] < num_adversarial_tokens:
            #print(f"Warning: top_indices.shape[0] ({top_indices.shape[0]}) is less than the current batch size ({number_of_candidates_to_generate}). The number of random integers will be decreased to that value. This usually indicates a problem with the tokens being processed.")
            #print(f"Warning: top_indices.shape[0] ({top_indices.shape[0]}) is less than the current topk value ({topk}). The number of top indices will be looped to create enough values. This usually indicates a problem with the tokens being processed.")
            print(f"Warning: the number of top token indices ({top_indices.shape[0]}) is less than the current number of adversarial content tokens ({num_adversarial_tokens}). The number of top indices will be looped to create enough values. This usually indicates a problem with the tokens being processed.")
            looped_values = []
            looped_value_number = 0
            #while len(looped_values) < topk:
            while len(looped_values) < num_adversarial_tokens:
                looped_values.append(top_indices[looped_value_number % top_indices.shape[0]].tolist())
                looped_value_number += 1
            top_indices = torch.tensor(looped_values, device = grad.device)
        #rand_ints = torch.randint(0, topk, (number_of_candidates_to_generate, 1), device=grad.device)
        # TKTK: replace this call to torch.randint with a random function that's *not* influenced by the PyTorch seed-setting operations that occur when --random-seed-comparisons is used, to avoid "randomly" selecting the same values over and over until all of the candidates have been filtered out due to being used previously.
        # TKTK: allow more than one token to be randomized:
        #       * Randomize exactly n tokens every time
        #       * Randomize between x and y tokens every time
        #       * Each token has an n% chance of being randomized every time
        #       Can probably handle this using most of the unfinished radiation garden code
        #rand_ints = torch.randint(0, topk, (num_rand_ints, 1), device = grad.device)
        rand_ints = torch.randint(0, topk, (num_rand_ints, 1), device = attack_params.device, generator = rand_int_generator).to(grad.device)
        #print(f"[sample_control] Debug: new_token_pos = {new_token_pos}, rand_ints = {rand_ints}")
        new_token_val = None
        top_indices_len_1 = top_indices.shape[0] - 1
        new_token_pos_in_bounds_values = []
        new_token_pos_out_of_bounds_values = []
        new_token_pos_values = new_token_pos.tolist()
        for i in range(0, len(new_token_pos_values)):
            if new_token_pos_values[i] > top_indices_len_1:
                new_token_pos_out_of_bounds_values.append(new_token_pos_values[i])
            else:
                new_token_pos_in_bounds_values.append(new_token_pos_values[i])
        if len(new_token_pos_out_of_bounds_values) > 0:
            #raise Exception(f"new_token_pos contained the following values, which are less than zero or greater than the upper bound of top_indices ({top_indices_len_1}): {new_token_pos_out_of_bounds_values}.")
            print(f"Warning: new_token_pos contained the following values, which are less than zero or greater than the upper bound of the list of top token indices ({top_indices_len_1}): {new_token_pos_out_of_bounds_values}. This usually indicates a problem with the tokens being processed.")
            #new_token_pos = torch.tensor(new_token_pos_in_bounds_values, device = grad.device)
            looped_values = []
            looped_value_number = 0
            while len(looped_values) < number_of_candidates_to_generate:
                looped_values.append(new_token_pos_in_bounds_values[looped_value_number % len(new_token_pos_in_bounds_values)])
                looped_value_number += 1
            new_token_pos = torch.tensor(looped_values, device = grad.device)
        # else:
            # new_token_val = torch.gather(
                # top_indices[new_token_pos], 1, 
                # rand_ints
            # )
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            rand_ints
        )


        #print(f"[sample_control] Debug: new_token_val = {new_token_val}")
        new_adversarial_token_ids = original_adversarial_content_token_ids_device.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        #print(f"[sample_control] Debug: new_adversarial_token_ids = {new_adversarial_token_ids}")

    if random_seed is not None:
        torch.manual_seed(attack_params.torch_manual_seed)
        torch.cuda.manual_seed_all(attack_params.torch_cuda_manual_seed_all)
    
    result = AdversarialContentList()
    
    if new_adversarial_token_ids is not None:
        #print(f"[sample_control] Debug: new_adversarial_token_ids = {new_adversarial_token_ids}")
        for i in range(new_adversarial_token_ids.shape[0]):
            #print(f"[sample_control] Debug: new_adversarial_token_ids[{i}] = {new_adversarial_token_ids[i]}")
            new_adversarial_token_ids_as_list = new_adversarial_token_ids[i].tolist()
            if AdversarialContent.token_list_contains_invalid_tokens(adversarial_content_manager.tokenizer, new_adversarial_token_ids_as_list):
                dummy = 1
                #print(f"[sample_control] Warning: adversarial_candidate '{new_adversarial_token_ids_as_list}' contains a token ID that is outside the valid range for this tokenizer (min = 0, max = {adversarial_content_manager.tokenizer.vocab_size}). The candidate will be ignored. This may indicate an issue with the attack code, or the tokenizer code.")
            else:
                new_candidate = AdversarialContent.from_token_ids(adversarial_content_manager.tokenizer, adversarial_content_manager.trash_fire_tokens, new_adversarial_token_ids_as_list)
                result.append_if_new(new_candidate)
    return result

def get_filtered_cands(attack_params, adversarial_content_manager, new_adversarial_content_list, previous_adversarial_values, filter_cand=True, current_adversarial_content = None):
    result = AdversarialContentList()
    filter_regex = attack_params.get_candidate_filter_regex()
    filtered_count = 0
    filtered_due_to_empty_string = []
    filtered_due_to_already_being_tested = []
    filtered_due_to_insufficient_token_count = []
    filtered_due_to_excessive_token_count = []
    filtered_due_to_nonmatching_token_count = []
    filtered_due_to_containing_newline_characters = []
    filtered_due_to_not_matching_regex = []
    filtered_due_to_repetitive_tokens = []
    filtered_due_to_repetitive_lines = []
    if new_adversarial_content_list is None:
        return result
    len_new_adversarial_content_list = len(new_adversarial_content_list.adversarial_content)
    for i in range(len_new_adversarial_content_list):
        #print(f"[get_filtered_cands] Debug: i = {i}")
        #print(f"[get_filtered_cands] Debug: new_adversarial_content_list.adversarial_content[i] = {new_adversarial_content_list.adversarial_content[i].get_short_description()}")
        adversarial_candidate = new_adversarial_content_list.adversarial_content[i].copy()
        if adversarial_candidate is not None and adversarial_candidate.as_string is not None:
            #adversarial_candidate_message_represenation = adversarial_candidate.adversarial_candidate.get_short_description()
            adversarial_candidate_message_represenation = adversarial_candidate.as_string
            #print(f"[get_filtered_cands] Debug: adversarial_candidate = '{adversarial_candidate.get_short_description()}', current_adversarial_content = '{current_adversarial_content.get_short_description()}', control_cand[i] = '{control_cand[i]}'")
            include_candidate = True
            # make sure the LLM sorcery hasn't accidentally introduced a token ID that's outside of the valid range
            if AdversarialContent.token_list_contains_invalid_tokens(adversarial_content_manager.tokenizer, adversarial_candidate.token_ids):
                    include_candidate = False
                    #print(f"[get_filtered_cands] Warning: adversarial_candidate '{adversarial_candidate.get_short_description()}' contains token ID {adversarial_candidate.token_ids[candidate_token_num]}, which is outside the valid range for this tokenizer (min = 0, max = {adversarial_content_manager.tokenizer.vocab_size}). The candidate will be ignored. This may indicate an issue with the attack code, or the tokenizer code.")
            if include_candidate and filter_cand:
                include_candidate = False
                
                if not adversarial_candidate.is_match(current_adversarial_content):
                    include_candidate = True
                else:
                    include_candidate = False
                    #print(f"[get_filtered_cands] Debug: rejecting candidate '{adversarial_candidate_message_represenation}' because it was equivalent to the current adversarial content value '{current_adversarial_content.get_short_description()}'.")
                    filtered_due_to_already_being_tested.append(adversarial_candidate)
                if include_candidate:
                    if adversarial_candidate_message_represenation.strip() == "":
                        include_candidate = False
                        #print(f"[get_filtered_cands] Debug: rejecting candidate '{adversarial_candidate_message_represenation}' because it is an empty string, or equivalent to an empty string.")
                        filtered_due_to_empty_string.append(adversarial_candidate)
                if include_candidate:
                    if previous_adversarial_values.contains_adversarial_content(adversarial_candidate):
                        include_candidate = False
                        #print(f"[get_filtered_cands] Debug: rejecting candidate '{adversarial_candidate_message_represenation}' because it was equivalent to a previous adversarial value.")
                        filtered_due_to_already_being_tested.append(adversarial_candidate)
                    #else:
                    #    print(f"[get_filtered_cands] Debug: candidate '{adversarial_candidate.get_short_description()}' is not equivalent to any previous adversarial values.")
                if include_candidate:
                    #token_input_ids = adversarial_content_manager.tokenizer(decoded_str, add_special_tokens=False).input_ids
                    if include_candidate:
                        
                        candidate_token_count = len(adversarial_candidate.token_ids)
                        current_adversarial_content_token_count = len(current_adversarial_content.token_ids)
                        if attack_params.candidate_filter_tokens_min is not None:
                            if candidate_token_count < attack_params.candidate_filter_tokens_min:
                                include_candidate = False
                                #print(f"[get_filtered_cands] Debug: rejecting candidate '{adversarial_candidate_message_represenation}' because its token count ({candidate_token_count}) was less than the minimum value specified ({attack_params.candidate_filter_tokens_min}).")
                                filtered_due_to_insufficient_token_count.append(adversarial_candidate)
                        if attack_params.candidate_filter_tokens_max is not None:
                            if candidate_token_count > attack_params.candidate_filter_tokens_max:
                                include_candidate = False
                                #print(f"[get_filtered_cands] Debug: rejecting candidate '{adversarial_candidate_message_represenation}' because its token count ({candidate_token_count}) was greater than the maximum value specified ({attack_params.candidate_filter_tokens_max}).")
                                filtered_due_to_excessive_token_count.append(adversarial_candidate)
                        if attack_params.attempt_to_keep_token_count_consistent:
                            if candidate_token_count != current_adversarial_content_token_count:
                                include_candidate = False
                                #print(f"[get_filtered_cands] Debug: rejecting candidate '{adversarial_candidate_message_represenation}' because its token count ({candidate_token_count}) was not equal to the length of '{current_adversarial_content.get_short_description()}' ({current_adversarial_content_token_count}).")
                                filtered_due_to_nonmatching_token_count.append(adversarial_candidate)

                    if include_candidate:
                        if attack_params.candidate_filter_newline_limit is not None:
                            newline_character_count = 0
                            for newline_character in ["\x0a", "\x0d"]:
                                if newline_character in adversarial_candidate.as_string:
                                    for current_char in adversarial_candidate.as_string:
                                        if current_char == newline_character:
                                            newline_character_count += 1
                            if newline_character_count > attack_params.candidate_filter_newline_limit:
                                include_candidate = False
                                #print(f"[get_filtered_cands] Debug: '{adversarial_candidate_message_represenation}' rejected due to presence of newline character(s)")
                                filtered_due_to_containing_newline_characters.append(adversarial_candidate)
                        if include_candidate and filter_regex is not None:
                            if filter_regex.search(adversarial_candidate.as_string):
                                dummy = 1
                                #print(f"[get_filtered_cands] Debug: '{adversarial_candidate_message_represenation}' represented as '{adversarial_candidate.as_string}' passed the regular expression filter")
                            else:
                                include_candidate = False
                                print(f"[get_filtered_cands] Debug: rejecting candidate '{adversarial_candidate_message_represenation}' because '{adversarial_candidate.as_string}' failed to pass the regular expression filter '{attack_params.token_filter_regex}'.")
                                filtered_due_to_not_matching_regex.append(adversarial_candidate)
                        if include_candidate and attack_params.candidate_filter_repetitive_tokens is not None and attack_params.candidate_filter_repetitive_tokens > 0:
                            token_counts = {}
                            already_notified_tokens = []
                            for c_token in token_input_ids:
                                t_count = 1
                                if c_token in token_counts:
                                    t_count = token_counts[c_token] + 1
                                    if t_count >= attack_params.candidate_filter_repetitive_tokens:
                                        include_candidate = False
                                        filtered_due_to_repetitive_tokens.append(adversarial_candidate)
                                        if c_token not in already_notified_tokens:
                                            already_notified_tokens.append(c_token)
                                            #print(f"[get_filtered_cands] Debug: '{adversarial_candidate_message_represenation}' rejected because it had more than {attack_params.candidate_filter_repetitive_tokens} occurrences of the token '{c_token}'")
                                token_counts[c_token] = t_count
                            #if include_candidate:
                            #    print(f"[get_filtered_cands] Debug: '{adversarial_candidate_message_represenation}' passed the repetitive token filter.")
                        if include_candidate and attack_params.candidate_filter_repetitive_lines is not None and attack_params.candidate_filter_repetitive_lines > 0:
                            candidate_lines = adversarial_candidate.as_string.splitlines()
                            token_counts = {}
                            already_notified_tokens = []
                            for c_line in candidate_lines:
                                t_count = 1
                                if c_line in token_counts:
                                    t_count = token_counts[c_line] + 1
                                    if t_count >= attack_params.candidate_filter_repetitive_lines:
                                        include_candidate = False
                                        filtered_due_to_repetitive_lines.append(adversarial_candidate)
                                        if c_line not in already_notified_tokens:
                                            already_notified_tokens.append(c_line)
                                            #print(f"[get_filtered_cands] Debug: '{adversarial_candidate_message_represenation}' rejected because it had more than {attack_params.candidate_filter_repetitive_lines} occurrences of the line '{c_line}'")
                                token_counts[c_line] = t_count
                            #if include_candidate:
                            #    print(f"[get_filtered_cands] Debug: '{adversarial_candidate_message_represenation}' passed the repetitive line filter.")
                            
                
            if include_candidate:
                if attack_params.candidate_replace_newline_characters is not None:
                    decoded_str = adversarial_candidate.as_string
                    decoded_str = decoded_str.replace("\n", attack_params.candidate_replace_newline_characters)
                    decoded_str = decoded_str.replace("\r", attack_params.candidate_replace_newline_characters)
                    if decoded_str != adversarial_candidate.as_string:
                        adversarial_candidate = AdversarialContent.from_string(adversarial_content_manager.tokenizer, adversarial_content_manager.trash_fire_tokens, decoded_str)
                #print(f"[get_filtered_cands] Debug: appending '{adversarial_candidate_message_represenation}' to candidate list.\n")
                result.append_if_new(adversarial_candidate)
            else:
                #print(f"[get_filtered_cands] Debug: not appending '{adversarial_candidate_message_represenation}' to candidate list because it was filtered out.\n")
                filtered_count += 1

    #print(f"[get_filtered_cands] Debug: control_cand = {control_cand}, cands = {cands}")

    if filter_cand:
        if len(result.adversarial_content) == 0:
            dummy = 1
            #print(f"[get_filtered_cands] Warning: no candidates found")
        else:
            # I *think* this step is supposed to append copies of the last entry in the list enough times to make the new list as long as the original list
            #cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
            # TKTK: try taking this out, because it seems weird to have to do this
            if len(result.adversarial_content) < len(new_adversarial_content_list.adversarial_content):
                while len(result.adversarial_content) < len(new_adversarial_content_list.adversarial_content):
                    result.adversarial_content.append(result.adversarial_content[-1].copy())
                    
            #print(f"[get_filtered_cands] Warning: {round(filtered_count / len(control_cand), 2)} control candidates were not valid")

    percent_passed = float(len(result.adversarial_content)) / float(len_new_adversarial_content_list)
    percent_rejected = float(filtered_count) / float(len_new_adversarial_content_list)
    if percent_rejected > attack_params.warn_on_filtered_candidate_percentage:
        filter_warning = f"Warning: {len(result.adversarial_content)}/{len_new_adversarial_content_list} ({percent_rejected:.0%}) of adversarial value candidates were filtered out during this iteration, which is greater than the warning threshold of {attack_params.warn_on_filtered_candidate_percentage:.0%}. This may be due to excessively strict or conflicting filtering options specified by the operator."
        len_filtered_due_to_empty_string = len(filtered_due_to_empty_string)
        if len_filtered_due_to_empty_string > 0:
            filter_warning = f"{filter_warning} {len_filtered_due_to_empty_string} candidate(s) were filtered out because they were equivalent to an empty string."

        len_filtered_due_to_already_being_tested = len(filtered_due_to_already_being_tested)
        if len_filtered_due_to_already_being_tested > 0:
            filter_warning = f"{filter_warning} {len_filtered_due_to_already_being_tested} candidate(s) were filtered out because they had already been tested in previous iterations."
        
        len_filtered_due_to_insufficient_token_count = len(filtered_due_to_insufficient_token_count)
        if len_filtered_due_to_insufficient_token_count > 0:
            filter_warning = f"{filter_warning} {len_filtered_due_to_insufficient_token_count} candidate(s) were filtered out because they had fewer than the minimum number of tokens specified by the operator ({attack_params.candidate_filter_tokens_min})."

        len_filtered_due_to_excessive_token_count = len(filtered_due_to_excessive_token_count)
        if len_filtered_due_to_excessive_token_count > 0:
            filter_warning = f"{filter_warning} {len_filtered_due_to_excessive_token_count} candidate(s) were filtered out because they had more than the maximum number of tokens specified by the operator ({attack_params.candidate_filter_tokens_max})."

        len_filtered_due_to_nonmatching_token_count = len(filtered_due_to_nonmatching_token_count)
        if len_filtered_due_to_nonmatching_token_count > 0:
            filter_warning = f"{filter_warning} {len_filtered_due_to_nonmatching_token_count} candidate(s) were filtered out because they had a different number of tokens than the current adversarial value, and the option to keep token count consistent is enabled."

        len_filtered_due_to_containing_newline_characters = len(filtered_due_to_containing_newline_characters)
        if len_filtered_due_to_containing_newline_characters > 0:
            filter_warning = f"{filter_warning} {len_filtered_due_to_containing_newline_characters} candidate(s) were filtered out because they contained more than the number of allowed newline characters."

        len_filtered_due_to_not_matching_regex = len(filtered_due_to_not_matching_regex)
        if len_filtered_due_to_not_matching_regex > 0:
            filter_warning = f"{filter_warning} {len_filtered_due_to_not_matching_regex} candidate(s) were filtered out because they did not match the regular expression '{attack_params.candidate_filter_regex}'."

        len_filtered_due_to_repetitive_tokens = len(filtered_due_to_repetitive_tokens)
        if len_filtered_due_to_repetitive_tokens > 0:
            filter_warning = f"{filter_warning} {len_filtered_due_to_repetitive_tokens} candidate(s) were filtered out because they had had more than the operator-specified number of repetitive tokens ({attack_params.candidate_filter_repetitive_tokens})."

        len_filtered_due_to_repetitive_lines = len(filtered_due_to_repetitive_lines)
        if len_filtered_due_to_repetitive_lines > 0:
            filter_warning = f"{filter_warning} {len_filtered_due_to_repetitive_lines} candidate(s) were filtered out because they had had more than the operator-specified number of repetitive lines ({attack_params.candidate_filter_repetitive_lines})."
        
        print(filter_warning)

    return result


def get_logits(*, model, tokenizer, input_ids, adversarial_content, adversarial_candidate_list = None, return_ids = False, batch_size=512):
    
    if adversarial_candidate_list is None or len(adversarial_candidate_list.adversarial_content) < 1:
        raise ValueError(f"adversarial_candidate_list must be an AdversarialContentList with at least 1 entry. Got empty array or null.")

    test_ids = None
    nested_ids = None

    number_of_adversarial_token_ids = len(adversarial_content.token_ids)

    max_len = number_of_adversarial_token_ids
    test_ids = []
    for i in range(0, len(adversarial_candidate_list.adversarial_content)):
        #tid = torch.tensor(tokenizer(adversarial_candidate_list.adversarial_content[i].token_ids, add_special_tokens=False).input_ids[:max_len], device=model.device)
        tid = torch.tensor(adversarial_candidate_list.adversarial_content[i].token_ids[:max_len], device=model.device)
        #tid = torch.tensor(adversarial_candidate_list.adversarial_content[i].token_ids, device=model.device)
        test_ids.append(tid)

    pad_tok = 0
    while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
        pad_tok += 1
    nested_ids = torch.nested.nested_tensor(test_ids)
    test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))

    decoded_test_ids = get_decoded_tokens(tokenizer, test_ids)
    #print(f"[get_logits] Debug: test_ids = '{test_ids}'\n decoded_test_ids = '{decoded_test_ids}'")

    if not(test_ids[0].shape[0] == number_of_adversarial_token_ids):
        raise ValueError((
            f"adversarial_candidate_list must have shape "
            f"(n, {number_of_adversarial_token_ids}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(0, number_of_adversarial_token_ids).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        result1 = forward(model=model, tokenizer = tokenizer, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        
        #print(f"[get_logits] Debug: returning result1 = '{result1}', ids = '{ids}', attn_mask = '{attn_mask}'")

        return result1, ids
    else:
        del locs, test_ids
        logits = forward(model=model, tokenizer = tokenizer, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        
        #print(f"[get_logits] Debug: returning logits = '{logits}', attn_mask = '{attn_mask}'")
        
        return logits
    

def forward(*, model, tokenizer, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        model_result = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        #model_result_decoded = get_decoded_tokens(tokenizer, model_result)
        #print(f"[forward] Debug: getting logits for model_result = '{model_result}', model_result_decoded = '{model_result_decoded}'")
        #print(f"[forward] Debug: getting logits for model_result = '{model_result}'")

        model_result_logits = model_result.logits

        logits.append(model_result_logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0)

# In this function, the logits returned by get_logits and forward are compared against the token IDs returned by get_logits
# ...which seems to correspond to the token IDs that represent the target output, repeated enough times to equal the length of the first entry in the list of candidate values? I think?
# If I understand the goal here, it's to treat the target tokens as coordinates and figure out how far away the candidate tokens are from them?
def target_loss(logits, ids, input_id_data, tokenizer):
    crit = nn.CrossEntropyLoss(reduction='none')
    # [blincoln] also corrected this to use the loss slice returned by get_prompt for consistency instead of redefining it here using the same logic as get_prompt
    #logits_sliced = logits[:,loss_slice,:]
    logits_sliced = logits[:,input_id_data.slice_data.loss,:]
    logits_sliced_transposed = logits_sliced.transpose(1,2)
    ids_sliced = ids[:,input_id_data.slice_data.target]
    
    # pad the target token IDs, if necessary
    ids_sliced = get_padded_target_token_ids(tokenizer, input_id_data.slice_data.loss, ids_sliced)
    # len_loss_slice = loss_slice.stop - loss_slice.start
    # if len_loss_slice > len(ids_sliced):
        # if tokenizer.pad_token_id is None:
            # # This should never occur unless someone is calling this function directly, outside of the tool
            # raise NullPaddingTokenException("The current target slice must be padded to match the length of the loss slice, but the tokenizer's padding token ID is None.")
        # targets_as_list = ids_sliced.tolist()
        # while len_loss_slice > len(targets_as_list):
            # targets_as_list.append(tokenizer.pad_token_id)
        # ids_sliced = torch.tensor(targets_as_list, device = targets.device)
    
    ids_sliced_decoded = get_decoded_tokens(tokenizer, ids_sliced)
    #print(f"[target_loss] Debug: calculating cross-entropy loss. logits_sliced = '{logits_sliced}', logits_sliced_transposed = '{logits_sliced_transposed}', ids_sliced = '{ids_sliced}', ids_sliced_decoded = '{ids_sliced_decoded}'")

    loss = crit(logits_sliced_transposed, ids_sliced)
    return loss.mean(dim=-1)

def get_missing_pad_token_names():
    result = [  "unk", 
                "bos",
                "eos" ]
    return result

def get_missing_pad_token_replacement(tokenizer, replacement_name):
    allowed_names = get_missing_pad_token_names()
    if replacement_name not in get_missing_pad_token_names(allowed_names):
        raise Exception(f"Unrecognized padding token replacement name: '{replacement_name}' - must be one of '{allowed_names}'")
    result = None
    if replacement_name == "bos":
        result = tokenizer.bos_token_id, tokenizer.bos_token
    if replacement_name == "eos":
        result = tokenizer.eos_token_id, tokenizer.eos_token
    if replacement_name == "unk":
        result = tokenizer.unk_token_id, tokenizer.unk_token
    return result

def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', dtype=torch.float16, trust_remote_code=True, ignore_mismatched_sizes=False, enable_hardcoded_tokenizer_workarounds = False, missing_pad_token_replacement = None, **kwargs):
    #print(f"[load_model_and_tokenizer] Debug: model_path = '{model_path}', tokenizer_path = '{tokenizer_path}', device = '{device}', dtype = {dtype}, trust_remote_code = {trust_remote_code}, ignore_mismatched_sizes = {ignore_mismatched_sizes}")

    #if ignore_mismatched_sizes:
    #    kwargs["ignore_mismatched_sizes"] = True

    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            **kwargs
        ).to(device).eval()
    
    
    tokenizer_path_to_load = model_path
    if tokenizer_path is not None:
        tokenizer_path_to_load = tokenizer_path
    
    #print(f"[load_model_and_tokenizer] Debug: tokenizer_path = '{tokenizer_path}', model_path = '{model_path}'")

    tokenizer = None
    
    #is_mamba = args.model_name.startswith("state-spaces/mamba-")
    #    if is_mamba:
    #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    #model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path_to_load,
            trust_remote_code=trust_remote_code,
            use_fast=False
        )
    except Exception as e:
        handled = False
        #if isinstance(e, ValueError):
        if 2 > 1:
            print(f"[load_model_and_tokenizer] Warning: unable to load standard tokenizer from '{tokenizer_path_to_load}', attempting to fall back to fast tokenizer. The exception thrown when loading the standard tokenizer was: {e}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path_to_load,
                    trust_remote_code=trust_remote_code,
                    use_fast=True
                )
                handled = True
            except Exception as e2:
                print(f"[load_model_and_tokenizer] Error loading both standard and fast tokenizers from '{tokenizer_path_to_load}': '{e}', '{e2}'")
                raise e        
        if not handled:
            print(f"[load_model_and_tokenizer] Error loading tokenizer from '{tokenizer_path_to_load}': '{e}'")
            raise e
    
    if enable_hardcoded_tokenizer_workarounds:
        if 'oasst-sft-6-llama-30b' in tokenizer_path_to_load:
            tokenizer.bos_token_id = 1
            tokenizer.unk_token_id = 0
        if 'guanaco' in tokenizer_path_to_load:
            tokenizer.eos_token_id = 2
            tokenizer.unk_token_id = 0
        if 'llama-2' in tokenizer_path_to_load:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'falcon' in tokenizer_path_to_load:
            tokenizer.padding_side = 'left'
            
    if not tokenizer.pad_token:
        if missing_pad_token_replacement is not None:
            tokenizer.pad_token_id, tokenizer.pad_token = get_missing_pad_token_replacement(tokenizer, missing_pad_token_replacement)
            print(f"[load_model_and_tokenizer] Warning: the tokenizer in '{tokenizer_path_to_load}' does not have a pad_token value defined. Using the alternative value '{missing_pad_token_replacement}' specified by the operator. If you encounter errors or unexpected results, consider specifying a different --missing-pad-token-replacement value on the command line.")
        else:
            print(f"[load_model_and_tokenizer] Warning: the tokenizer in '{tokenizer_path_to_load}' does not have a pad_token value defined. If you encounter errors or unexpected results, consider specifying a --missing-pad-token-replacement value on the command line.")
    
    return model, tokenizer
