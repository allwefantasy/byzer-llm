# After vLLM > 0.2.7 , vLLM brings lora support, 
# Then the tokenizer of the model.engine is TokenizerGroup,
# you can get the original tokenizer by tokenizer.tokenizer
# or get lora_toeknizer by get_lora_tokenizer

def validate_args_engine_use_ray():
    try:
        from vllm.transformers_utils.tokenizer import TokenizerGroup
        return True
    except ImportError:
        return False



def get_local_tokenizer(llm,engine_args):    
    from vllm.engine.arg_utils import AsyncEngineArgs
    engine_args: AsyncEngineArgs = engine_args    

    if engine_args.engine_use_ray:  
        from vllm.transformers_utils.tokenizer import TokenizerGroup
        engine_configs = engine_args.create_engine_configs()    
        model_config = engine_configs[0]
        scheduler_config = engine_configs[3]
        lora_config = engine_configs[5]      
        init_kwargs = dict(
                enable_lora=bool(lora_config),
                max_num_seqs=scheduler_config.max_num_seqs,
                max_input_length=None,
                tokenizer_mode=model_config.tokenizer_mode,
                trust_remote_code=model_config.trust_remote_code,
                revision=model_config.tokenizer_revision)
        tokenizer: TokenizerGroup = TokenizerGroup(model_config.tokenizer, **init_kwargs)
        return tokenizer
    else:
        return llm.engine.tokenizer

def get_real_tokenizer(tokenizer):        
    is_tokenizer_group = hasattr(tokenizer,"get_lora_tokenizer")
    if is_tokenizer_group:
        final_tokenizer = tokenizer.tokenizer
    else:
        final_tokenizer = tokenizer
    return final_tokenizer    

