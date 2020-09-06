# Task 3 코드 분석

# Preprocess 전처리

```bash
$ python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json={path_dir}/data/simmc-fashion/fashion_train_dials.json \
    --output_path_predict={path_dir}/mm_dst/gpt2_dst/data/fashion/fashion_train_dials_predict.txt \
    --output_path_target={path_dir}/mm_dst/gpt2_dst/data/fashion/fashion_train_dials_target.txt \
    --output_path_special_tokens={path_dir}/mm_dst/gpt2_dst/data/fashion/special_tokens.json
    --len_context=2 \
    --use_multimodal_contexts=1 \
```

- `scripts/preprocess_input.py` : SIMMC(json 형식) 데이터셋을 GPT-2 모델에 input 으로 들어갈 수 있도록 형식을 바꾼다.  JSON → line-by-line representation of each turn !

```python
convert_json_to_flattened(
        input_path_json,
        output_path_predict,
        output_path_target,
        input_path_special_tokens=input_path_special_tokens,
        output_path_special_tokens=output_path_special_tokens,
        len_context=len_context,
        use_multimodal_contexts=use_multimodal_contexts)
```

# Train

```bash
$ python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir={path_dir}/save/fashion \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens={path_dir}/mm_dst/gpt2_dst/data/fashion/special_tokens.json \
    --do_train \
    --train_data_file={path_dir}/mm_dst/gpt2_dst/data/fashion/fashion_train_dials_target.txt \
    --do_eval \
    --eval_data_file={path_dir}/mm_dst/gpt2_dst/data/fashion/fashion_dev_dials_target.txt \
    --num_train_epochs=1 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    #--no_cuda
```

- `scripts/run_language_modeling` : huggingface/transformer 에 있는 파이썬 스크립트. GPT, BERT 같은 언어 모델에 데이터를 fine-tune 할때 쓰인다. 너무 길고 어렵기 때문에 분석은 생략..  😰

```bash
(base) sejin@nlpgpu6:~/simmc/mm_dst (master)$ ./run_train_gpt2.sh 
08/17/2020 13:35:58 - WARNING - __main__ -   Process rank: -1, device: cuda, n_gpu: 3, distributed training: False, 16-bits training: False
08/17/2020 13:35:59 - INFO - filelock -   Lock 139828123906176 acquired on /home/sejin/.cache/torch/transformers/4be02c5697d91738003fb1685c9872f284166aa32e061576bbe6aaeb95649fcf.db13c9bc9c7bdd738ec89e069621d88e05dc670366092d809a9cbcac6798e24e.lock
08/17/2020 13:35:59 - INFO - transformers.file_utils -   https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json not found in cache or force_download set to True, downloading to /home/sejin/.cache/torch/transformers/tmpmkqs6zld
Downloading: 100%|███████████████████████████████████████████████████████████████████| 665/665 [00:00<00:00, 582kB/s]
08/17/2020 13:36:00 - INFO - transformers.file_utils -   storing https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json in cache at /home/sejin/.cache/torch/transformers/4be02c5697d91738003fb1685c9872f284166aa32e061576bbe6aaeb95649fcf.db13c9bc9c7bdd738ec89e069621d88e05dc670366092d809a9cbcac6798e24e
08/17/2020 13:36:00 - INFO - transformers.file_utils -   creating metadata file for /home/sejin/.cache/torch/transformers/4be02c5697d91738003fb1685c9872f284166aa32e061576bbe6aaeb95649fcf.db13c9bc9c7bdd738ec89e069621d88e05dc670366092d809a9cbcac6798e24e
08/17/2020 13:36:00 - INFO - filelock -   Lock 139828123906176 released on /home/sejin/.cache/torch/transformers/4be02c5697d91738003fb1685c9872f284166aa32e061576bbe6aaeb95649fcf.db13c9bc9c7bdd738ec89e069621d88e05dc670366092d809a9cbcac6798e24e.lock
08/17/2020 13:36:00 - INFO - transformers.configuration_utils -   loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json from cache at /home/sejin/.cache/torch/transformers/4be02c5697d91738003fb1685c9872f284166aa32e061576bbe6aaeb95649fcf.db13c9bc9c7bdd738ec89e069621d88e05dc670366092d809a9cbcac6798e24e
08/17/2020 13:36:00 - INFO - transformers.configuration_utils -   Model config GPT2Config {
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "vocab_size": 50257
}

08/17/2020 13:36:00 - INFO - transformers.configuration_utils -   loading configuration file https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json from cache at /home/sejin/.cache/torch/transformers/4be02c5697d91738003fb1685c9872f284166aa32e061576bbe6aaeb95649fcf.db13c9bc9c7bdd738ec89e069621d88e05dc670366092d809a9cbcac6798e24e
08/17/2020 13:36:00 - INFO - transformers.configuration_utils -   Model config GPT2Config {
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "vocab_size": 50257
}
```

# Generate

```bash
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path={path_dir}/mm_dst/gpt2_dst/save/furniture/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file={path_dir}/mm_dst/gpt2_dst/data/furniture/furniture_devtest_dials_predict.txt \
    --path_output={path_dir}/mm_dst/gpt2_dst/results/furniture/furniture_devtest_dials_predicted.txt
```

- `scripts/run_generation.py` : 라이브러리의 자동 생성 모델을 사용한 텍스트 생성 (GPT/GPT-2/CTRL/Transfore-XL/XLNet)

```python
def main():
    ## 생략... arguments 로부터 modeltype, length, seed 등 설정 
    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    **model = model_class.from_pretrained(args.model_name_or_path)**
    model.to(args.device)
    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    results = []
    prompts = []
    if args.prompts_from_file:
        with open(args.prompts_from_file) as handle:
            prompts = handle.readlines()

    while True:
        if not prompts:
            prompts = [args.prompt if args.prompt else input("Model prompt >>> ")]
            if not args.prompt and (
                len(prompts) == 0
                or prompts[0].strip() == ''
                or prompts[0].lower() == 'quit'
            ):
                break  # break while True loop

        n_prompts = len(prompts)
        for i, prompt_text in enumerate(prompts):
            # Strip any trailing \n if provided
            prompt_text = prompt_text.strip('\n')

            # Different models need different input formatting and/or extra arguments
           
            **output_sequences = model.generate( # generate 
                input_ids=encoded_prompt,
                max_length=args.length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
            )**
            # .. 생략 
                # Decode text
                text = tokenizer.decode(
                    generated_sequence,
                    clean_up_tokenization_spaces=True
                )

                # Remove all text after the stop token
                text = text[: text.find(args.stop_token) if args.stop_token else None]

                # Add the prompt at the beginning of the sequence. Remove the
                # excess text that was used for pre-processing
                total_sequence = (
                    prompt_text + text[
                        len(tokenizer.decode(
                            encoded_prompt[0],
                            clean_up_tokenization_spaces=True
                        ))
                        :
                    ]
                )

                generated_sequences.append(total_sequence)
                print(total_sequence)

            results.append(generated_sequences)

        prompts = []
        if args.prompt or args.prompts_from_file:
            break  # break while True loop

```

# evaluate

```bash
python -m gpt2_dst.scripts.evaluate \
    --input_path_target={path_dir}/mm_dst/gpt2_dst/data/furniture/furniture_devtest_dials_target.txt \
    --input_path_predicted={path_dir}/mm_dst/gpt2_dst/results/furniture/furniture_devtest_dials_predicted.txt \
    --output_path_report={path_dir}/mm_dst/gpt2_dst/results/furniture/furniture_devtest_dials_report.json
```

- `scripts/evaluate.py` :  GPT-2 DST 모델 예측을 평가하기 위한 스크립트. 먼저, 라인별 문자열화된 DST output 을 통해 구문 분석한다. 그 후  DST Evaluation 스크립트를 실행하여 결과를 얻는다.

```python
# Parse input args ( input_path_target, input_path_predicted, output_path_report)
# Convert the data from the GPT-2 friendly format to JSON
	list_target = parse_flattened_results_from_file(input_path_target)
	list_predicted = parse_flattened_results_from_file(input_path_predicted)
# Evaluate
    report = evaluate_from_flat_list(list_target, list_predicted)
```

- `gpt2_dst/convert.py` :   SIMMC 데이터셋 (.JSON 형식)을 라인별 문자열 형식으로 바꾼 뒤  GPT-2 기반 DST 모델  입력으로 사용된다.

```python
def parse_flattened_result(to_parse):
    """
        Parse out the belief state from the raw text.
        Return an empty list if the belief state can't be parsed

        Input:
        - A single <str> of flattened result
          e.g. 'User: Show me something else => Belief State : DA:REQUEST ...'

        Output:
        - Parsed result in a JSON format, where the format is:
            [
                {
                    'act': <str>  # e.g. 'DA:REQUEST',
                    'slots': [
                        <str> slot_name,
                        <str> slot_value
                    ]
                }, ...  # End of a frame
            ]  # End of a dialog
    """
```

- `evaluate_dst.py`

```python
def evaluate_from_flat_list(d_true, d_pred):
    """
        <list>d_true and <list>d_pred are in the following format:
        (Each element represents a single turn, with (multiple) frames)
        [
            [
                {
                    'act': <str>,
                    'slots': [
                        [
                            SLOT_NAME, SLOT_VALUE
                        ], ...
                    ]
                },
                [End of a frame]
                ...
            ],
            [End of a turn]
            ...
        ]
    """
    # Count # corrects & # wrongs
    # Calculate metrics
    # 생략 
    return {
        'joint_accuracy': joint_accuracy,
        'act_rec': act_rec,
        'act_prec': act_prec,
        'act_f1': act_f1,
        'slot_rec': slot_rec,
        'slot_prec': slot_prec,
        'slot_f1': slot_f1,
    }
```