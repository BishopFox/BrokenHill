import gc

import numpy as np
import psutil
import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_attacks import get_embedding_matrix, get_embeddings, get_decoded_token, get_decoded_tokens, get_encoded_token, get_encoded_tokens 

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

def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
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
#        one_hot = torch.randint(0, 1, size=(input_ids[input_slice].shape[0],embed_weights.shape[0]), device=model.device, dtype=embed_weights.dtype)
        one_hot = create_new_quantized_tensor(0, (input_ids[input_slice].shape[0],embed_weights.shape[0]), model.device, embed_weights.data.dtype, scales_value, pczp_value)
    else:
        one_hot = torch.zeros(
            input_ids[input_slice].shape[0],
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
        input_ids[input_slice].unsqueeze(1),
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
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
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
    targets = input_ids[target_slice]
    #print_stats("token_gradients")

    #print("[token_gradients] Getting loss")
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
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

def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    new_control_toks = None

    if grad is not None:
        if not_allowed_tokens is not None:
            grad[:, not_allowed_tokens.to(grad.device)] = np.infty

        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = control_toks.to(grad.device)

        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks

def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None, filter_regex = None, filter_repetitive_tokens = None, filter_repetitive_lines = None, filter_newline_limit = None, replace_newline_characters = None, attempt_to_keep_token_count_consistent = False, candidate_filter_tokens_min = None, candidate_filter_tokens_max = None):
    cands, filtered_count = [], 0
    if control_cand is None:
        return cands
    for i in range(control_cand.shape[0]):
        #print(f"[get_filtered_cands] Debug: i = {i}")
        #print(f"[get_filtered_cands] Debug: control_cand[i] = {control_cand[i]}")
        decoded_str = None
        try:
            #decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
            #decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=False)
            decoded_str = get_decoded_token(tokenizer, control_cand[i])
        except Exception as e:
            decoded_str = None
            decoded_tokens = get_decoded_tokens(tokenizer, control_cand[i].data)
            #print(f"[get_filtered_cands] Error: when calling get_decoded_token(tokenizer, control_cand[i]) with control_cand[i] = '{control_cand[i]}', decoded_tokens = '{decoded_tokens}': {e} - this may indicate an error in the attack code")            
        if decoded_str is not None:
            #print(f"[get_filtered_cands] Debug: decoded_str = '{decoded_str}', curr_control = '{curr_control}', control_cand[i] = '{control_cand[i]}'")
            include_candidate = True
            if filter_cand:
                include_candidate = False
                #if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                if decoded_str != curr_control:
                    include_candidate = True
                #else:
                    #print(f"[get_filtered_cands] Debug: rejecting candidate '{decoded_str}' because it was equivalent to the current control value '{curr_control}'.")
                token_input_ids = tokenizer(decoded_str, add_special_tokens=False).input_ids
                if include_candidate:
                    
                    len_temp_input_ids = len(token_input_ids)
                    len_control_cand_i = len(control_cand[i])
                    if candidate_filter_tokens_min is not None:
                        if len_temp_input_ids < candidate_filter_tokens_min:
                            include_candidate = False
                            #print(f"[get_filtered_cands] Debug: rejecting candidate '{decoded_str}' because the length of its input_ids ({len_temp_input_ids}) was less than the minimum value specified ({candidate_filter_tokens_min}).")
                    if candidate_filter_tokens_max is not None:
                        if len_temp_input_ids > candidate_filter_tokens_max:
                            include_candidate = False
                            #print(f"[get_filtered_cands] Debug: rejecting candidate '{decoded_str}' because the length of its input_ids ({len_temp_input_ids}) was greater than the maximum value specified ({candidate_filter_tokens_max}).")
                    if attempt_to_keep_token_count_consistent:
                        if len_temp_input_ids != len_control_cand_i:
                            include_candidate = False
                            #print(f"[get_filtered_cands] Debug: rejecting candidate '{decoded_str}' because the length of its input_ids ({len_temp_input_ids}) was not equal to the length of '{control_cand[i]}' ({len_control_cand_i}).")
                
                if include_candidate:
                
                    #print(f"[get_filtered_cands] Debug: appending '{decoded_str}' to candidate list because it passsed the filter")
                    
                    if filter_newline_limit is not None:
                        newline_character_count = 0
                        for newline_character in ["\x0a", "\x0d"]:
                            if newline_character in decoded_str:
                                for current_char in decoded_str:
                                    if current_char == newline_character:
                                        newline_character_count += 1
                        if newline_character_count > filter_newline_limit:
                            include_candidate = False
                            #print(f"[get_filtered_cands] Debug: '{decoded_str}' rejected due to presence of newline character(s)")
                    if include_candidate and filter_regex is not None:
                        if filter_regex.search(decoded_str):
                            dummy = 1
                            #print(f"[get_filtered_cands] Debug: '{decoded_str}' passsed the regular expression filter")
                        else:
                            include_candidate = False
                            #print(f"[get_filtered_cands] Debug: '{decoded_str}' failed the regular expression filter")
                    if include_candidate and filter_repetitive_tokens is not None and filter_repetitive_tokens > 0:
                        token_counts = {}
                        already_notified_tokens = []
                        for c_token in token_input_ids:
                            t_count = 1
                            if c_token in token_counts:
                                t_count = token_counts[c_token] + 1
                                if t_count >= filter_repetitive_tokens:
                                    include_candidate = False
                                    if c_token not in already_notified_tokens:
                                        already_notified_tokens.append(c_token)
                                        #print(f"[get_filtered_cands] Debug: '{decoded_str}' rejected because it had more than {filter_repetitive_tokens} occurrences of the line '{c_token}'")
                            token_counts[c_token] = t_count
                        #if include_candidate:
                        #    print(f"[get_filtered_cands] Debug: '{decoded_str}' passed the repetitive line filter.")
                    if include_candidate and filter_repetitive_lines is not None and filter_repetitive_lines > 0:
                        candidate_lines = decoded_str.splitlines()
                        token_counts = {}
                        already_notified_tokens = []
                        for c_line in candidate_lines:
                            t_count = 1
                            if c_line in token_counts:
                                t_count = token_counts[c_line] + 1
                                if t_count >= filter_repetitive_lines:
                                    include_candidate = False
                                    if c_line not in already_notified_tokens:
                                        already_notified_tokens.append(c_line)
                                        #print(f"[get_filtered_cands] Debug: '{decoded_str}' rejected because it had more than {filter_repetitive_lines} occurrences of the line '{c_line}'")
                            token_counts[c_line] = t_count
                        #if include_candidate:
                        #    print(f"[get_filtered_cands] Debug: '{decoded_str}' passed the repetitive line filter.")
                        
                
            if include_candidate:
                if replace_newline_characters is not None:
                    decoded_str = decoded_str.replace("\n", replace_newline_characters)
                    decoded_str = decoded_str.replace("\r", replace_newline_characters)
                if decoded_str in cands:
                    dummy = 1
                    #print(f"[get_filtered_cands] Debug: not appending '{decoded_str}' to candidate list because it was equivalent to another candidate.")
                else:
                    cands.append(decoded_str)
                    #print(f"[get_filtered_cands] Debug: appending '{decoded_str}' to candidate list.")
            else:
                #print(f"[get_filtered_cands] Debug: not appending '{decoded_str}' to candidate list because it was filtered out.")
                filtered_count += 1

    #print(f"[get_filtered_cands] Debug: control_cand = {control_cand}, cands = {cands}")

    if filter_cand:
        if len(cands) == 0:
            dummy = 1
            #print(f"[get_filtered_cands] Warning: no candidates found")
        else:
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
            #print(f"[get_filtered_cands] Warning: {round(filtered_count / len(control_cand), 2)} control candidates were not valid")
    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    
    if test_controls is None or len(test_controls) < 1:
        raise ValueError(f"test_controls must be a list of strings, got empty array or null")

    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
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
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits
    

def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0)

def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    return loss.mean(dim=-1)


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', dtype=torch.float16, trust_remote_code=True, ignore_mismatched_sizes=False, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            **kwargs
        ).to(device).eval()
    
    #print(f"[load_model_and_tokenizer] Debug: tokenizer_path = '{tokenizer_path}', model_path = '{model_path}'")
    
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
    tokenizer = None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
            use_fast=False
        )
    except Exception as e:
        handled = False
        if isinstance(e, ValueError):
            print(f"[load_model_and_tokenizer] Warning: unable to load standard tokenizer from '{tokenizer_path}', attempting to fall back to fast tokenizer.")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=trust_remote_code,
                    use_fast=True
                )
                handled = True
            except Exception as e2:
                print(f"[load_model_and_tokenizer] Error loading both standard and fast tokenizers from '{tokenizer_path}': '{e}', '{e2}'")
                raise e        
        if not handled:
            print(f"[load_model_and_tokenizer] Error loading tokenizer from '{tokenizer_path}': '{e}'")
            raise e
    
    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
