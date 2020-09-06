# Preprocessing for Task 1 & 2 분석

## Step 1 어시스턴트 API 추출하기

- `extract_actions.py` fashion 데이터에 관련해 input json 파일로 부터 action API 를 저장하고 save_path에 저장한다.

```python
def extract_actions(input_json_file, save_root, furniture_db, subtask):
    """Extract assistant API calls from keystrokes and NLU/NLG annotations.

    Args:
        input_json_file: JSON data file
        save_root: Folder to save the extracted API calls
        furniture_db: object wrapping the furniture database
        subtask: Single dominant or multiple actions
    """ ## 생략
def main(_):
    furniture_db = data_support.FurnitureDatabase(FLAGS.metadata_path)
    for input_json_file in FLAGS.json_path:
        extract_actions(
            input_json_file,
            FLAGS.save_root,
            furniture_db,
            FLAGS.subtask
        )
    furniture_db.shutdown()
```

- `extract_actions_fashion.py` fashion 데이터에 관련해 input json 파일로 부터 action API 를 저장하고 save_path에 저장한다.

```python
def extract_actions(input_json_file):
    """Extract action API for SIMMC fashion.

    Args:
        input_json_file: JSON data file to extraction actions
    """
    # Save extracted API calls.
def extract_info_attributes(round_datum):
    """Extract information attributes for current round using NLU annotations.

    Args:
        round_datum: Current round information

    Returns:
        get_attribute_matches: Information attributes
    """
def main(_):
    for input_json_file in FLAGS.json_path:
        extract_actions(input_json_file)
```

diag.json 파일로부터 API action들이 extract 되어 json 파일에 아래와 같이 저장되게 된다..

```json
[
    {
        "dialog_id": 7729,
        "turns": [
            {
                "turn_idx": 0,
                "relevant_apis_with_args": [
                    {
                        "api": "None",
                        "previousState": null,
                        "nextState": null,
                        "args": null
                    }
                ],
                "raw_action_with_args": [],
                "current_search_results": []
            },
            {
                "turn_idx": 1,
                "relevant_apis_with_args": [
                    {
                        "api": "SearchFurniture",
                        "previousState": {
                            "prefabInFocus": "",
                            "prefabsInCarousel": [],
                            "sharedPrefabInFocus": "",
                            "sharedPrefabsInCarousel": [],
                            "textPrefabInFocus": "",
                            "textPrefabsInCarousel": []
                        },
                        "nextState": {
                            "prefabInFocus": "",
                            "prefabsInCarousel": [
                                "1019705",
                                "763118",
                                "1122853"
                            ],
                            "sharedPrefabInFocus": "",
                            "sharedPrefabsInCarousel": [],
                            "textPrefabInFocus": "",
                            "textPrefabsInCarousel": []
                        },
                        "args": {
                            "furnitureType": "Bookcases",
                            "color": "",
                            "material": "",
                            "decorStyle": "",
                            "intendedRoom": "",
                            "minPrice": -1,
                            "maxPrice": -1
                        }
                    }
                ] // 생략
]
```

```bash
Reading: ../data/simmc_furniture/furniture_train_dials.json
Saving: ../data/simmc_furniture/furniture_train_dials_api_calls.json
Reading: ../data/simmc_furniture/furniture_dev_dials.json
Saving: ../data/simmc_furniture/furniture_dev_dials_api_calls.json
Reading: ../data/simmc_furniture/furniture_devtest_dials.json
Saving: ../data/simmc_furniture/furniture_devtest_dials_api_calls.json
Reading: ../data/simmc_furniture/furniture_train_dials.json
```

## Step 2 Vocabulary 추출하기

- `extract_vocabulary.py`: Extracts vocabulary for SIMMC dataset.

```python
def main(args):
    # Read the data, parse the datapoints.
    print("Reading: {}".format(args["train_json_path"]))
    with open(args["train_json_path"], "r") as file_id:
        train_data = json.load(file_id)
    dialog_data = train_data["dialogue_data"]

    counts = {}
    for datum in dialog_data:
        dialog_utterances = [
            ii[key] for ii in datum["dialogue"]
            for key in ("transcript", "system_transcript")
        ]
        dialog_tokens = [
            word_tokenize(ii.lower()) for ii in dialog_utterances
        ]
        for turn in dialog_tokens:
            for word in turn:
                counts[word] = counts.get(word, 0) + 1

    # Add <pad>, <unk>, <start>, <end>.
    counts["<pad>"] = args["threshold_count"] + 1
    counts["<unk>"] = args["threshold_count"] + 1
    counts["<start>"] = args["threshold_count"] + 1
    counts["<end>"] = args["threshold_count"] + 1

    word_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    words = [ii[0] for ii in word_counts if ii[1] >= args["threshold_count"]]
    vocabulary = {"word": words}
    # Save answers and vocabularies.
    print("Identified {} words..".format(len(words)))
    print("Saving dictionary: {}".format(args["vocab_save_path"]))
    with open(args["vocab_save_path"], "w") as file_id:
        json.dump(vocabulary, file_id)
```

```bash
Identified 2473 words..
Saving dictionary: ../data/simmc_furniture/furniture_vocabulary.json
```

## Step 3 Asset 를 읽고 Embedding 하기

- `embed_fashion_assets.py` : Globe 임베딩을 결합하여 패션 에셋 임베딩을 생성함.

    ```python
    # Attributes to encode.
    EMBED_ATTRIBUTES = ["type", "color", "embellishments", "pattern"]
    def main(args):
        with open(args["input_asset_file"], "r") as file_id:
            assets = json.load(file_id)
        # Select and embed only the top attributes. 코드 생략...
        # Vocabulary for each field.
        vocabulary = {key: {} for key in EMBED_ATTRIBUTES}
        for asset in cleaned_assets:
            for attr in EMBED_ATTRIBUTES:
                attr_val = asset.get(attr, [])
                for val in attr_val:
                    vocabulary[attr][val] = vocabulary[attr].get(val, 0) + 1
        # Embedding for each item. 코드 생략 
        for asset in cleaned_assets:
            embed_vector = []
            for attr in EMBED_ATTRIBUTES:
                if attr in asset and len(asset[attr]) > 0:
                    attr_val = asset[attr]
                    feature_vector = np.stack(
                        [nlp(val).vector for val in attr_val]
                    ).mean(0)
                else:
                    feature_vector = zero_features
                embed_vector.append(feature_vector)
            embeddings.append(np.concatenate(embed_vector))
            id_list.append(asset["id"])
        embeddings = np.stack(embeddings)
        print("Saving embeddings: {}".format(args["embed_path"]))
        np.save(
            args["embed_path"],
            {
                "asset_id": id_list,
                "embedding": embeddings,
                "asset_feature_size": embeddings.shape[1],
            },
        )
    ```

    - `embed_furniture_assets.py` : Create furniture asset embeddings by concatenating attribute Glove embeddings.

    ```python
    # Attributes to encode.
    EMBED_ATTRIBUTES = [
        "class_name", "color", "decor_style", "intended_room", "material"
    ]
    def main(args):
        assets = data_support.read_furniture_metadata(args["input_csv_file"])
        cleaned_assets = []
        # Quick fix dictionary.
        correction = {
            "['Traditional', 'Modern'']": "['Traditional', 'Modern']",
            "[Brown']": "['Brown']",
        }
        for _, asset in assets.items():
            clean_asset = {}
            for key in EMBED_ATTRIBUTES:
                val = asset[key]
                val = correction.get(val, val).lower()
                val = ast.literal_eval(val) if "[" in val else val
                clean_asset[key] = val if isinstance(val, list) else [val]
            clean_asset["id"] = int(asset["obj"].split("/")[-1].strip(".zip"))
            cleaned_assets.append(clean_asset)

        # Vocabulary for each field.
        # Embedding for each item. 생략
       
    ```

    ```bash
    Reading: ../data/simmc_furniture/furniture_metadata.csv
    Saving embeddings: ../data/simmc_furniture/furniture_asset_embeds.npy
    ```

    ## Step 4 Convert all the splits(train,dev,test) into npy files for dataloader.

    - `build_multimodal_inputs.py` : JSON file 로부터 train/evaluate 에 쓰일 dictionary of multimodal inputs 생성.

    ```python
    def build_multimodal_inputs(input_json_file):
        # Read the raw data.
        print("Reading: {}".format(input_json_file))
        with open(input_json_file, "r") as file_id:
            data = json.load(file_id)
        # Read action supervision.
        print("Reading action supervision: {}".format(FLAGS.action_json_path))
        with open(FLAGS.action_json_path, "r") as file_id:
            extracted_actions = json.load(file_id)
        # Convert into a dictionary.
        extracted_actions = {ii["dialog_id"]: ii for ii in extracted_actions}
        # Obtain maximum dialog length.
        # 코드 생략 
        # Setup datastructures for recoding utterances, actions, action supervision,
        # carousel states, and outputs.
        action_info = {
            "action": np.full((num_dialogs, max_dialog_len), "None", dtype="object_"),
            "action_supervision": copy.deepcopy(empty_action_list),
            "carousel_state": copy.deepcopy(empty_action_list),
            "action_output_state": copy.deepcopy(empty_action_list)
        }
        # Compile dictionaries for user and assitant utterances separately.
        utterance_dict = {"user": {}, "assistant": {}}
        action_keys = ("action",)
        if FLAGS.domain == "furniture":
            action_keys += ("carousel_state", "action_output_state")
        elif FLAGS.domain == "fashion":
            task_mapping = {ii["task_id"]: ii for ii in data["task_mapping"]}
            dialog_image_ids = {
                "memory_images": [], "focus_images": [], "database_images": []
            }

        # If retrieval candidates file is available, encode the candidates.
        if FLAGS.retrieval_candidate_file:
            print("Reading retrieval candidates: {}".format(
                FLAGS.retrieval_candidate_file)
            )
            with open(FLAGS.retrieval_candidate_file, "r") as file_id:
                candidates_data = json.load(file_id)
            candidate_pool = candidates_data["system_transcript_pool"]
            candidate_ids = candidates_data["retrieval_candidates"]
            candidate_ids = {ii["dialogue_idx"]: ii for ii in candidate_ids}

            def get_candidate_ids(dialog_id, round_id):
                """Given the dialog_id and round_id, get the candidates. 코드 생략"""
            # Read the first dialog to get number of candidates.
            **encoded_candidates = np.full(
                (num_dialogs, max_dialog_len, num_candidates), -1, dtype=np.int32
            )**

        for datum_id, datum in enumerate(data["dialogue_data"]):
            dialog_id = datum["dialogue_idx"]
            dialog_ids[datum_id] = dialog_id
            # Get action supervision.
            dialog_action_data = extracted_actions[dialog_id]["actions"]
            # Record images for fashion.
            if FLAGS.domain == "fashion":
                # Assign random task if not found (1-2 dialogs).
                if "dialogue_task_id" not in datum:
                    print("Dialog task id not found, using 1874 (random)!")
                task_info = task_mapping[datum.get("dialogue_task_id", 1874)]
                for key in ("memory_images", "database_images"):
                    dialog_image_ids[key].append(task_info[key])
                dialog_image_ids["focus_images"].append(
                    extracted_actions[dialog_id]["focus_images"]
                )

            for round_id, round_datum in enumerate(datum["dialogue"]):
                for key, speaker in (  ## speaker 에 따라 encoding 
                    ("transcript", "user"), ("system_transcript", "assistant")
                ):
                    utterance_clean = round_datum[key].lower().strip(" ")
                    speaker_pool = utterance_dict[speaker]
                    if utterance_clean not in speaker_pool:
                        speaker_pool[utterance_clean] = len(speaker_pool)
                    encoded_dialogs[speaker][datum_id, round_id] = (
                        speaker_pool[utterance_clean]
                    )

                # Record action related keys.
                action_datum = dialog_action_data[round_id]
                cur_action_supervision = action_datum["action_supervision"]
                if FLAGS.domain == "furniture":
                    if cur_action_supervision is not None:
                        # Retain only the args of supervision.
                        cur_action_supervision = cur_action_supervision["args"]

                action_info["action_supervision"][datum_id][round_id] = (
                    cur_action_supervision
                )
                for key in action_keys:
                    action_info[key][datum_id][round_id] = action_datum[key]
                action_counts[action_datum["action"]] += 1
        support.print_distribution(action_counts, "Action distribution:")

        # Record retrieval candidates, if path is provided. 생략 
        # Sort utterance list for consistency.
        # Convert the pools into matrices.
        # If token-wise encoding is to be used.
        print("Vocabulary: {}".format(FLAGS.vocab_file))
        if not FLAGS.pretrained_tokenizer:
            with open(FLAGS.vocab_file, "r") as file_id:
                vocabulary = json.load(file_id)
            mm_inputs["vocabulary"] = vocabulary
            word2ind = {word: index for index, word in enumerate(vocabulary["word"])}

            mm_inputs["user_sent"], mm_inputs["user_sent_len"] = convert_pool_matrices(
                utterance_list["user"], word2ind
            )
            mm_inputs["assist_sent"], mm_inputs["assist_sent_len"] = convert_pool_matrices(
                utterance_list["assistant"], word2ind
            )
            # Token aliases.
            pad_token = word2ind["<pad>"]
            start_token = word2ind["<start>"]
            end_token = word2ind["<end>"]
        else:
            # Use pretrained BERT tokenizer. 생략.  pad,start,end 토큰이 바뀜 
           
        # Get the input and output version for RNN for assistant_sent.
        extra_slice = np.full((len(mm_inputs["assist_sent"]), 1), start_token, np.int32)
        mm_inputs["assist_in"] = np.concatenate(
            [extra_slice, mm_inputs["assist_sent"]], axis=1
        )
        extra_slice.fill(pad_token)
        mm_inputs["assist_out"] = np.concatenate(
            [mm_inputs["assist_sent"], extra_slice], axis=1
        )
        for ii in range(len(mm_inputs["assist_out"])):
            mm_inputs["assist_out"][ii, mm_inputs["assist_sent_len"][ii]] = end_token
        mm_inputs["assist_sent_len"] += 1

        # Save the memory and dataset image_ids for each instance.
        if FLAGS.domain == "fashion":
            mm_inputs.update(dialog_image_ids)

        # Save the retrieval candidates.
        if FLAGS.retrieval_candidate_file:
            mm_inputs["retrieval_candidates"] = encoded_candidates

        # Save the dialogs by user/assistant utterances.
        mm_inputs["user_utt_id"] = encoded_dialogs["user"]
        mm_inputs["assist_utt_id"] = encoded_dialogs["assistant"]
        mm_inputs["dialog_len"] = dialog_lens
        mm_inputs["dialog_id"] = dialog_ids
        mm_inputs["paths"] = {
            "data": FLAGS.json_path,
            "action": FLAGS.action_json_path,
            "retrieval": FLAGS.retrieval_candidate_file,
            "vocabulary": FLAGS.vocab_file
        }
        return mm_inputs

    ```

    ```bash
    Reading: ../data/simmc_furniture/furniture_train_dials.json
    Reading action supervision: ../data/simmc_furniture/furniture_train_dials_api_calls.json
    Reading retrieval candidates: ../data/simmc_furniture/furniture_train_dials_retrieval_candidates.json
    Action distribution:
    	None             [34%]: 9908
    	SpecifyInfo      [23%]: 6728
    	SearchFurniture  [18%]: 5149
    	AddToCart        [9%]: 2750
    	FocusOnFurniture [7%]: 2125
    	Rotate           [6%]: 1842
    	NavigateCarousel [2%]: 711

    Vocabulary: ../data/simmc_furniture/furniture_vocabulary.json
    100%|████████████████████████████████████████████████████████████████████████████████████████| 6584/6584 [00:00<00:00, 7697.06it/s]
    6584it [00:00, 318611.09it/s]
    100%|████████████████████████████████████████████████████████████████████████████████████████| 6631/6631 [00:00<00:00, 7604.47it/s]
    6631it [00:00, 309556.68it/s]
    ```

    ## Step 5 : multimodal input 으로 부터 attributes 들 추출하기

    - `extract_attribute_vocabulary.py`

    ```python
    def extract_action_attributes(args):
        # Read the data, parse the datapoints. 생략
        # Get action attributes.
        attr_vocab = {}
        for ii in range(num_instances):
            for jj in range(num_rounds):
                cur_action = actions[ii, jj]
                if cur_action == "None":
                    continue
                if cur_action not in attr_vocab:
                    if args[DOMAIN] == FURNITURE:
                        attr_vocab[cur_action] = collections.defaultdict(dict)
                    elif args[DOMAIN] == FASHION:
                        attr_vocab[cur_action] = collections.defaultdict(
                            lambda: collections.defaultdict(lambda: 0)
                        )

                cur_super = data["action_supervision"][ii][jj]
                if cur_super is None:
                    continue
                for key, val in cur_super.items():
                    if args[DOMAIN] == FURNITURE:
                        if key in EXCLUDE_KEYS_FURNITURE:
                            continue
                        if isinstance(val, list):
                            val = tuple(val)
                        new_count = attr_vocab[cur_action][key].get(val, 0) + 1
                        **attr_vocab[cur_action][key][val] = new_count**

                    elif args[DOMAIN] == FASHION:
                        if key in EXCLUDE_KEYS_FASHION:
                            continue
                        if isinstance(val, list):
                            val = tuple(val)
                            for vv in val:
                                # If vv not in INCLUDE_ATTRIBUTES_FASHION,
                                # assign it to "other."
                                if vv not in INCLUDE_ATTRIBUTES_FASHION:
                                    vv = "other"
                                attr_vocab[cur_action][key][vv] += 1
                        else:
                            # If val not in INCLUDE_ATTRIBUTES_FASHION,
                            # assign it to other.
                            if val not in INCLUDE_ATTRIBUTES_FASHION:
                                val = "other"
                            attr_vocab[cur_action][key][val] += 1

        attr_vocab = {
            key: sorted(val)
            for attr_values in attr_vocab.values()
            for key, val in attr_values.items()
        }
    ```

    Output for $DOMAIN = furniture

    ```bash
    Saving multimodal inputs: ../data/simmc_furniture/furniture_devtest_dials_mm_inputs.npy
    {'furnitureType': ['Accent Chairs', 'Area Rugs', 'Bookcases', 'Coffee & Cocktail Tables', 'Dining Chairs', 'Dining Tables', 'End Tables', 'Kitchen Islands', 'Office Chairs', 'Ottomans', 'Sofas', 'Table Lamps', 'Teen Bookcases'], 'color': ['', 'Beige', 'Black', 'Blue', 'Brown', 'Gray', 'Green', 'Purple', 'Red', 'White', 'Yellow'], 'matches': ['color', 'dimensions', 'info', 'material', 'price'], 'navigate_direction': ['Here', 'Next', 'Previous'], 'position': ['center', 'left', 'right'], 'direction': ['back', 'down', 'front', 'left', 'right', 'up']}
    Saving attribute dictionary: ../data/simmc_furniture/furniture_attribute_vocabulary.json
    ```