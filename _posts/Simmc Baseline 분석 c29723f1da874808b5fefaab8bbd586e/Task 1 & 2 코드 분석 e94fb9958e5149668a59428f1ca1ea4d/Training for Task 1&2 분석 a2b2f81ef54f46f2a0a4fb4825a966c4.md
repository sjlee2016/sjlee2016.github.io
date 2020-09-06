# Training for Task 1&2 분석

- `train_simmc_agent.py`: Trains SIMMC baselines

```python
# 전체 플로우를 이해하기 위해 중요하다고 생각되는 부분들만.. some parts of the code are omitted 
args = options.read_command_line() # Arguments
args.update(train_loader.get_data_related_arguments()) # Dataloader
wizard = models.Assistant(args) # Model
wizard.train()
optimizer = torch.optim.**Adam**(wizard.parameters(), args["learning_rate"]) # Adam Optimizer.
smoother = support.ExponentialSmoothing() # Training iterations Smoothing 이란 말 처럼,  급격한 변화에 대해서는 일정부분 이상은 예외치로 보며, 뒤늦게 따라가는 경향이 있음
num_iters_per_epoch = train_loader.num_instances / args["batch_size"]
print("Number of iterations per epoch: {:.2f}".format(num_iters_per_epoch))
eval_dict = {}
best_epoch = -1
# first_batch = None
for iter_ind, batch in enumerate(train_loader.get_batch()):
    epoch = iter_ind / num_iters_per_epoch
    batch_loss = wizard(batch)
    batch_loss_items = {key: val.item() for key, val in batch_loss.items()}
    **losses = smoother.report(batch_loss_items)**
    optimizer.zero_grad() # Optimization steps.
    batch_loss["total"].backward()
    torch.nn.utils.clip_grad_value_(wizard.parameters(), 1.0)
    **optimizer.step()**
    # Perform evaluation, every X number of epochs.
    if (
        val_loader
        and int(epoch) % args["eval_every_epoch"] == 0
        and (iter_ind == math.ceil(int(epoch) * num_iters_per_epoch))
    ):  
        **eval_dict[int(epoch)], eval_outputs = evaluation.evaluate_agent(
            wizard, val_loader, args
        )**
        # Print the best epoch so far.
        best_epoch, best_epoch_dict = support.sort_eval_metrics(eval_dict)[0]
        print("\nBest Val Performance: Ep {}".format(best_epoch))
        for item in best_epoch_dict.items():
            print("\t{}: {:.2f}".format(*item))

    # Save the model every epoch. .. 코드는 생략 
```

## Step 1 Arguments 저장

- `options.py`커멘드라인을 통해 프로그램에 사용될 global variables 값을 지정해주는 코드. 아래와 같이 train_simmc_agent 코드 처음 부분에 이와 같이 사용된다.

```python
# Arguments. 
args = options.read_command_line()
```

  저장되는 Arguments

```python
1. "--train_data_path", required=True, help="Path to compiled training data"
2. "--eval_data_path", default=None, help="Path to compiled evaluation data"
3.  "--snapshot_path", default="checkpoints/", help="Path to save checkpoints"
4. "--metainfo_path", default="data/furniture_metainfo.json", help="Path to file containing metainfo",
5. "--attr_vocab_path",  default="data/attr_vocab_file.json", help="Path to attribute vocabulary file",
6. "--domain", required=True, choices=["furniture", "fashion"],  help="Domain to train the model on",
7. "--asset_embed_path", default="data/furniture_asset_path.npy", help="Path to asset embeddings",
# Specify encoder/decoder flags.
# Model hyperparameters.
8. "--encoder", required=True,
        choices=[
            "history_agnostic",
            "history_aware",
            "pretrained_transformer",
            "hierarchical_recurrent",
            "memory_network",
            "tf_idf",
        ],
        help="Encoder type to use for text",
    )
9.  "--text_encoder",
        required=True,
        choices=["lstm", "transformer"],
        help="Encoder type to use for text",
    )
10. "--word_embed_size", default=128, type=int, help="size of embedding for text"
11. "--hidden_size",default=128, type=int,
        help=(
            "Size of hidden state in LSTM/transformer."
            "Must be same as word_embed_size for transformer"
        ),
    )
# Parameters for transformer text encoder.  
12. "--num_heads_transformer", default=-1, type=int, help="Number of heads in the transformer"
13. "--num_layers_transformer", default=-1, type=int, help="Number of layers in the transformer",
14. "--hidden_size_transformer",  default=2048, type=int, help="Hidden Size within transformer",
15. "--num_layers", default=1, type=int, help="Number of layers in LSTM"
16. "--use_action_attention", dest="use_action_attention", action="store_true",
        default=False, help="Use attention over all encoder statesfor action"
17. "--use_action_output",  dest="use_action_output", action="store_true", default=False,
        help="Model output of actions as decoder memory elements"
18. "--use_multimodal_state", default=False, help="Use multimodal state for action prediction (fashion)",
19. "--use_bahdanau_attention",  default=False, help="Use bahdanau attention for decoder LSTM",
20. "--skip_retrieval_evaluation" default=True, help="Evaluation response generation through retrieval"
21. "--skip_bleu_evaluation", default=True,  help="Use beamsearch to evaluate BLEU score"
22. "--max_encoder_len", default=24, type=int, help="Maximum encoding length for sentences",
23. "--max_history_len", default=100, type=int, help="Maximum encoding length for history encoding",
24. "--max_decoder_len", default=26, type=int, help="Maximum decoding length for sentences",
25. "--max_rounds",  default=30,  type=int, help="Maximum number of rounds for the dialog",
26. "--share_embeddings", default=True, help="Encoder/decoder share emebddings",
27. "--batch_size", default=30,  type=int,  help="Training batch size (adjust based on GPU memory)",
28. "--learning_rate", default=1e-3, type=float, help="Learning rate for training"
29."--dropout", default=0.2, type=float, help="Dropout"
30. "--num_epochs",  default=20, type=int, help="Maximum number of epochs to run training",
31. "--eval_every_epoch", default=1,ype=int, help="Number of epochs to evaluate every",
32. "--save_every_epoch", default=-1, type=int, help="Epochs to save the model every, -1 does not save",
33."--save_prudently", default=False,  help="Save checkpoints prudently (only best models)",
34."--gpu_id", type=int, default=-1, help="GPU id to use, -1 for CPU"
```

## Step 2 DataLoaders

- `loaders/`: data를 네트워크 입력으로 사용하기 위해 사전에 정리를 해주는 dataloaders

    ```python
    # Dataloader defined in train_simmc_agent
    dataloader_args = {
        "single_pass": False,
        "shuffle": True,
        "data_read_path": args["train_data_path"],
        "get_retrieval_candidates": False
    }
    dataloader_args.update(args)
    train_loader = loaders.DataloaderSIMMC(dataloader_args)
    args.update(train_loader.get_data_related_arguments())
    # Initiate the loader for val (DEV) data split.
    if args["eval_data_path"]:
        dataloader_args = {
            "single_pass": True,
            "shuffle": False,
            "data_read_path": args["eval_data_path"],
            "get_retrieval_candidates": args["retrieval_evaluation"]
        }
        dataloader_args.update(args)
        val_loader = loaders.DataloaderSIMMC(dataloader_args)
    else:
        val_loader = None
    ```

    - `loader_base.py` : Parent class for data loaders

    ```python
    class LoaderParent:
        def __init__(self):
            # Assert the presence of mandatory attributes to setup prefetch daemon.
        def load_one_batch(self, sample_ids):
            # Load one batch given the sample indices.
        def _setup_prefetching(self): # Prefetches batches to save time.
        def get_batch(self): # Batch generator depending on train/eval mode.
        def _run_prefetch(self): # Load batch from file
        def _ship_torch_batch(self, batch): #  Ship a batch in PyTorch.
           # Useful for cross-package dataloader.
        def _ship_helper(self, numpy_array): # Helper to ship numpy arrays to torch.
        def compute_idf_features(self): # Computes idf scores based on train set.
        def compute_tf_features(self, utterances, utterance_lens): # Compute TF features for either train/val/test set.
        def get_data_related_arguments(self):
           # Get data related arguments like vocab_size, etc.
        def numpy(batch_torch): # Convert a batch into numpy arrays.
        def num_instances(self): # Number of instances in the dataloader.
    ```

    - `loader_simmc.py` : Dataloader for SIMMC Dataset.

    ```python
    class DataloaderSIMMC(loaders.LoaderParent):
        """Loads data for SIMMC datasets.
        """
        def __init__(self, params): # if encoder is not pretrained transformer 
                                    # it adjust the tokens to '<start>'..'<end>'
                                    # if the model is BERT, the tokens are adjusted to 
                                    # [start] and [end] 
            # Read the metainfo and attribute vocabulary for the dataset.
            # Encode attribute supervision according to the domain (furniture/fashion) 
            # prepare embeddings for assets 
            # Additional data constructs (post-processing).

        def load_one_batch(self, sample_ids): 
          # Loads a batch, given the sample ids.
          # sets dialog length and id and max length, user utt id
          # .. 생략 .. 
    ```

    - `loader_vocabulary.py` : Loads vocabulary and performs additional text processing.

    ```python
    class Vocabulary:
        def __init__(self, vocabulary_path=None, immutable=False, verbose=True):
            """Initialize the vocabulary object given a path, else empty object.
        Args:
          vocabulary_path: List of words in a text file, one in each line.
          immutable: Once initialized, no new words can be added.
        """
        def __contains__(self, key):
            """Check if a word is contained in a vocabulary.
          """
        def _setup_vocabulary(self):
            """Sets up internal dictionaries.
        """
            # Check whether <unk>,<start>,<end> and <pad> are part of the word list.
            # Else add them.
        def add_new_word(self, *new_words):
            """Adds new words to an existing vocabulary object."""
        def word(self, index):
            """Returns the word given the index."""
        def index(self, word, unk_default=False):
            """Returns the index given the word."""
        def set_vocabulary_state(self, state):
            """Given a state (list of words), setup the vocabulary object state."""
        def get_vocabulary_state(self):
            """Returns the vocabulary state (deepcopy).
        Returns:
          Deepcopy of list of words.
        """
        def get_tensor_string(self, tensor):
            """Converts a tensor into a string after decoding it using vocabulary.
        """
        @property
        def vocab_size(self):
            return len(self._words)
    ```

    ## Step 3 : Model

    ![Training%20for%20Task%201&2%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%E1%86%A8%20a2b2f81ef54f46f2a0a4fb4825a966c4/Untitled.png](Training%20for%20Task%201&2%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%E1%86%A8%20a2b2f81ef54f46f2a0a4fb4825a966c4/Untitled.png)

    - `models/`: Model files

        ```python
        wizard = models.Assistant(args)
        wizard.train()
        ```

        - `assistant.py`: SIMMC Assistant Wrapper Class

        ```python
        class Assistant(nn.Module):
            def __init__(self, params): # 중요하다 생각되는 부분 빼고 생략 
                self.encoder = encoders.ENCODER_REGISTRY[params["encoder"]](params)
                self.decoder = models.GenerativeDecoder(params) # 인코더 디코더 지정 
                # encoder 로 transformer 사용시  bert word_embedding 사용 
                self.action_executor = models.ActionExecutor(params)  # 어시스턴트의 API를 예측한다  
                # Initialize weights.
                # Sharing word embeddings across encoder and decoder.
            def forward(self, batch, mode=None): # batch 와 mode를 input 으로 받고 forward propagation
                  # 여기서 mode는 training/evaluate 시 None. Text generation 시   BEAMSEARCH / SAMPLE / MAX
                  ## 생략..
        			  outputs = self.encoder(batch)
                action_output = self.action_executor(batch, outputs)
                outputs.update(action_output)
                decoder_output = self.decoder(batch, outputs)
                # If evaluating by retrieval, construct fake batch for each candidate.
                # Inputs from batch used in decoder:
                #   assist_in, assist_out, assist_in_len, assist_mask
                if self.params["retrieval_evaluation"] and not self.training:
                    option_scores = []
                    batch_size, num_rounds, num_candidates, _ = batch["candidate_in"].shape
                    replace_keys = ("assist_in", "assist_out", "assist_in_len", "assist_mask")
                    for ii in range(num_candidates):
                        for key in replace_keys:
                            new_key = key.replace("assist", "candidate")
                            batch[key] = batch[new_key][:, :, ii]
                        decoder_output = self.decoder(batch, outputs)
                        log_probs = torch_support.unflatten(
                            decoder_output["loss_token"], batch_size, num_rounds
                        )
                        option_scores.append(-1 * log_probs.sum(-1))
                    option_scores = torch.stack(option_scores, 2)
                    outputs["candidate_scores"] = [
                        {
                            "dialog_id": batch["dialog_id"][ii].item(),
                            "candidate_scores": [
                                list(option_scores[ii, jj].cpu().numpy())
                                for jj in range(batch["dialog_len"][ii])
                            ]
                        }
                        for ii in range(batch_size)
                    ]

                # Local aliases.
                loss_token = decoder_output["loss_token"]
                pad_mask = decoder_output["pad_mask"]
                if self.training:
                    loss_token = loss_token.sum() / (~pad_mask).sum().item()
                    loss_action = action_output["action_loss"]
                    loss_action_attr = action_output["action_attr_loss"]
                    loss_total = loss_action + loss_token + loss_action_attr
                    return {
                        "token": loss_token,
                        "action": loss_action,
                        "action_attr": loss_action_attr,
                        "total": loss_total,
                    }
                else:  
                    outputs.update(
                        {"loss_sum": loss_token.sum(), "num_tokens": (~pad_mask).sum()}
                    )
                    return outputs
        ```

        Encoder

        - `encoders/`: Different types of encoders

            ![Training%20for%20Task%201&2%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%E1%86%A8%20a2b2f81ef54f46f2a0a4fb4825a966c4/Untitled%201.png](Training%20for%20Task%201&2%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%E1%86%A8%20a2b2f81ef54f46f2a0a4fb4825a966c4/Untitled%201.png)

            - `history_agnostic.py` :  HAE와 동일하지만 LSTM 대신 transformer를 사용한 모델
            - `hierarchical_recurrent.py` : 사용자와 에이전트의 과거 대화 내용을 기억하면서 현재 입력에 대한 응답을 산출
            - `memory_network.py` : 유저의 내용을 기억하면서 인코딩하는 모델
            - `tf_idf_encoder.py` : 단어 빈도를 사용해서 인코딩하는 모델

        Response Generator

        - `decoder.py`: Response decoder, language model with LSTM or Transformers

        ```python
        class GenerativeDecoder(nn.Module):
            def __init__(self, params): # set context encoder, word embedding, text encoder 
            def _generate_no_peek_mask(self, size): # Generates square masks for transformers to avoid peeking.
            def forward(self, batch, encoder_output):
                """
               Args:
                    batch: Dict of batch variables.
                    encoder_output: Dict of outputs from the encoder.

                Returns:
                    decoder_outputs: Dict of outputs from the forward pass.
                """
        ```

        - `carousel_embedder.py`: Learns multimodal embedding for furniture

        ```python
        class CarouselEmbedder(nn.Module):
            def __init__(self, params): # initialize embedding network, mask, attention
            def forward(self, carousel_state, encoder_state, encoder_size):
                """Carousel Embedding.

                Args:
                    carousel_state: State of the carousel
                    encoder_state: State of the encoder
                    encoder_size: (batch_size, num_rounds)

                Returns:
                    new_encoder_state:
                """ 
                for inst_id in range(batch_size):
                    for round_id in range(num_rounds):
                        round_datum = carousel_state[inst_id][round_id]
                        if round_datum is None:
                            carousel_features = self.none_features
                            carousel_sizes.append(1)
                        elif "focus" in round_datum:
                            carousel_features = torch.cat(
                                [round_datum["focus"], self.carousel_pos["focus"]]
                            ).unsqueeze(0)
                            carousel_features = torch.cat(
                                [carousel_features, self.empty_feature, self.empty_feature],
                                dim=0,
                            )
                            carousel_sizes.append(1)
                        elif "carousel" in round_datum:
                            carousel_size = len(round_datum["carousel"])
                            if carousel_size < 3:
                                all_embeds = torch.cat(
                                    [round_datum["carousel"]]
                                    + self.occupancy_embeds[carousel_size],
                                    dim=0,
                                )
                            else:
                                all_embeds = round_datum["carousel"]
                            all_states = self.occupancy_states[carousel_size]
                            carousel_features = torch.cat([all_embeds, all_states], -1)
                            carousel_sizes.append(carousel_size)
                        # Project into same feature shape.
                        carousel_features = self.carousel_embed_net(carousel_features)
                        carousel_states.append(carousel_features)
                # Shape: (L,N,E)
                carousel_states = torch.stack(carousel_states, dim=1)
                # Mask: (N,S)
                carousel_len = self.host.LongTensor(carousel_sizes)
                query = encoder_state.unsqueeze(0)
                attended_query, attented_wts = self.carousel_attend(
                    query,
                    carousel_states,
                    carousel_states,
                    key_padding_mask=self.carousel_mask[carousel_len - 1],
                )
                carousel_encode = torch.cat([attended_query.squeeze(0), encoder_state], dim=-1)
                return carousel_encode
        ```

        - `user_memory_embedder.py`: Learns multimodal embedding for fashion

        ```python
        class UserMemoryEmbedder(nn.Module):
            def forward(self, multimodal_state, encoder_state, encoder_size):
                """Multimodal Embedding.

                Args:
                    multimodal_state: Dict with memory, database, and focus images
                    encoder_state: State of the encoder
                    encoder_size: (batch_size, num_rounds)

                Returns:
                    multimodal_encode: Encoder state with multimodal information
                """
            def _setup_category_states(self):
                """Setup category states (focus + memory images).
                """
                # NOTE: Assumes three memory images; make it adaptive later.
                self.category_state = torch.stack(
                    [
                        self.category_embeds["focus"],
                        self.category_embeds["memory"],
                        self.category_embeds["memory"],
                        self.category_embeds["memory"],
                    ],
                    dim=0,
                ).unsqueeze(0) 
        ```

        - `positional_encoding.py`: Positional encoding unit for transformers
        - `self_attention.py`: Self attention model unit
        - `{fashion|furniture}_model_metainfo.json`: API 와 attribute에 대한 정보를 담고 있는 json 파일

        ```bash
        {
          "actions": [
            {
                "id": 0,
                "name": "SearchFurniture",
                "attributes": ["color", "furnitureType"]
            },
            {
                "id": 1,
                "name": "SpecifyInfo",
                "attributes": ["matches"]
            },
            {
                "id": 2,
                "name": "FocusOnFurniture",
                "attributes": ["position"]
            },
            {
                "id": 3,
                "name": "Rotate",
                "attributes": ["direction"]
            },
            {
                "id": 4,
                "name": "NavigateCarousel",
                "attributes": ["navigate_direction"]
            },
            {
                "id": 5, 
                "name": "AddToCart",
                "attributes": []
            },
            {
                "id": 6, 
                "name": "None",
                "attributes": []
            }
          ]
        }
        ```

        - `action_executor.py`: Executes the actions and predicts action attributes for SIMMC.

        ```python
        def forward(self, batch, prev_outputs):
                """Forward pass a given batch.
                Args:
                    batch: Batch to forward pass
                    prev_outputs: Output from previous modules.

                Returns:
                    outputs: Dict of expected outputs
                """
                # Predict and execute actions.
                action_logits = self.action_net(encoder_state)
                dialog_mask = batch["dialog_mask"]
                batch_size, num_rounds = dialog_mask.shape
                loss_action = self.criterion(action_logits, batch["action"].view(-1))
                loss_action.masked_fill_((~dialog_mask).view(-1), 0.0)
                loss_action_sum = loss_action.sum() / dialog_mask.sum().item()
                outputs["action_loss"] = loss_action_sum
                if not self.training:
                    # Check for action accuracy.
                    action_logits = support.unflatten(action_logits, batch_size, num_rounds)
                    actions = action_logits.argmax(dim=-1)
                    action_logits = nn.functional.log_softmax(action_logits, dim=-1)
                    action_list = self.action_map.get_vocabulary_state()
                    # Convert predictions to dictionary.
                    action_preds_dict = [
                        {
                            "dialog_id": batch["dialog_id"][ii].item(),
                            "predictions": [
                                {
                                    "action": self.action_map.word(actions[ii, jj].item()),
                                    "action_log_prob": {
                                        action_token: action_logits[ii, jj, kk].item()
                                        for kk, action_token in enumerate(action_list)
                                    },
                                    "attributes": {}
                                }
                                for jj in range(batch["dialog_len"][ii])
                            ]
                        }
                        for ii in range(batch_size)
                    ]
                    outputs["action_preds"] = action_preds_dict
                else:
                    actions = batch["action"]

                # Run classifiers based on the action, record supervision if training.
                if self.training:
                    assert (
                        "action_super" in batch
                    ), "Need supervision to learn action attributes"
                attr_logits = collections.defaultdict(list)
                attr_loss = collections.defaultdict(list)
                **encoder_state_unflat = support.unflatten(
                    encoder_state, batch_size, num_rounds
                )**

                host = torch.cuda if self.params["use_gpu"] else torch
                for inst_id in range(batch_size):
                    for round_id in range(num_rounds):
                        # Turn out of dialog length.
                        if not dialog_mask[inst_id, round_id]:
                            continue

                        cur_action_ind = actions[inst_id, round_id].item()
                        cur_action = self.action_map.word(cur_action_ind)
                        cur_state = encoder_state_unflat[inst_id, round_id]
                        supervision = batch["action_super"][inst_id][round_id]
                        # If there is no supervision, ignore and move on to next round.
                        if supervision is None:
                            continue

                        # Run classifiers on attributes.
                        # Attributes overlaps completely with GT when training.
                        if self.training:
                            classifier_list = self.action_metainfo[cur_action]["attributes"]
                            if self.params["domain"] == "furniture":
                                for key in classifier_list:
                                    cur_gt = (
                                        supervision.get(key, None)
                                        if supervision is not None
                                        else None
                                    )
                                    new_entry = (cur_state, cur_gt, inst_id, round_id)
                                    attr_logits[key].append(new_entry)
                            elif self.params["domain"] == "fashion":
                                for key in classifier_list:
                                    cur_gt = supervision.get(key, None)
                                    gt_indices = host.FloatTensor(
                                        len(self.attribute_vocab[key])
                                    ).fill_(0.)
                                    gt_indices[cur_gt] = 1
                                    new_entry = (cur_state, gt_indices, inst_id, round_id)
                                    attr_logits[key].append(new_entry)
                            else:
                                raise ValueError("Domain neither of furniture/fashion!")
                        else:
                            classifier_list = self.action_metainfo[cur_action]["attributes"]
                            action_pred_datum = action_preds_dict[
                                inst_id
                            ]["predictions"][round_id]
                            if self.params["domain"] == "furniture":
                                # Predict attributes based on the predicted action.
                                for key in classifier_list:
                                    classifier = self.classifiers[key]
                                    model_pred = classifier(cur_state).argmax(dim=-1)
                                    attr_pred = self.attribute_vocab[key][model_pred.item()]
                                    action_pred_datum["attributes"][key] = attr_pred
                            elif self.params["domain"] == "fashion":
                                # Predict attributes based on predicted action.
                                for key in classifier_list:
                                    classifier = self.classifiers[key]
                                    model_pred = classifier(cur_state) > 0.5
                                    attr_pred = [
                                        self.attribute_vocab[key][index]
                                        for index, ii in enumerate(model_pred)
                                        if ii
                                    ]
                                    action_pred_datum["attributes"][key] = attr_pred
                            else:
                                raise ValueError("Domain neither of furniture/fashion!")

                # Compute losses if training, else predict.
                if self.training:
                    for key, values in attr_logits.items():
                        classifier = self.classifiers[key]
                        prelogits = [ii[0] for ii in values if ii[1] is not None]
                        if not prelogits:
                            continue
                        logits = classifier(torch.stack(prelogits, dim=0))
                        if self.params["domain"] == "furniture":
                            gt_labels = [ii[1] for ii in values if ii[1] is not None]
                            gt_labels = host.LongTensor(gt_labels)
                            attr_loss[key] = self.criterion_mean(logits, gt_labels)
                        elif self.params["domain"] == "fashion":
                            gt_labels = torch.stack(
                                [ii[1] for ii in values if ii[1] is not None], dim=0
                            )
                            attr_loss[key] = self.criterion_multi(logits, gt_labels)
                        else:
                            raise ValueError("Domain neither of furniture/fashion!")

                    total_attr_loss = host.FloatTensor([0.0])
                    if len(attr_loss.values()):
                        total_attr_loss = sum(attr_loss.values()) / len(attr_loss.values())
                    outputs["action_attr_loss"] = total_attr_loss

                # Obtain action outputs as memory cells to attend over.
                if self.params["use_action_output"]:
                    if self.params["domain"] == "furniture":
                        encoder_state_out = self.action_output_embed(
                            batch["action_output"],
                            encoder_state_old,
                            batch["dialog_mask"].shape[:2],
                        )
                    elif self.params["domain"] == "fashion":
                        multimodal_state = {}
                        for ii in ["memory_images", "focus_images"]:
                            multimodal_state[ii] = batch[ii]
                        # For action output, advance focus_images by one time step.
                        # Output at step t is input at step t+1.
                        feature_size = batch["focus_images"].shape[-1]
                        zero_tensor = host.FloatTensor(batch_size, 1, feature_size).fill_(0.)
                        multimodal_state["focus_images"] = torch.cat(
                            [batch["focus_images"][:, 1:, :], zero_tensor], dim=1
                        )
                        encoder_state_out = self.multimodal_embed(
                            multimodal_state, encoder_state_old, batch["dialog_mask"].shape[:2]
                        )
                    else:
                        raise ValueError("Domain neither furniture/fashion!")
                    outputs["action_output_all"] = encoder_state_out

                outputs.update(
                    {"action_logits": action_logits, "action_attr_loss_dict": attr_loss}
                )
                return outputs
        ```

        ## Step 4 : Evaluate

    - `eval_simmc_agent.py`  Furniture/Fashion 데이터셋에 대해 SIMMC agent를 BLEU score, retrieval score,  perplexity, action prediction 등을 계산하여 evaluate 한다

    ```python
    def evaluate_agent(wizard, val_loader, args):
        """Evaluate a SIMMC agent given a dataloader.

        Args:
            wizard: SIMMC model
            dataloader: Dataloader to use to run the model on
            args: Arguments for evaluation
        Output : eval_dict(BLEU,retireval score, perplexity,action_accuracy,action_perplexity, action_attribute_accuracy)
                eval_outputs(action_prediction, model_responses)
        """
        total_iters = int(val_loader.num_instances / args["batch_size"])
        # Turn autograd off for evaluation -- light-weight and faster.
        with torch.no_grad():
            wizard.eval()
            matches = []
            for batch in progressbar(val_loader.get_batch(), total=int(total_iters)):
                if args["bleu_evaluation"]:
                    mode = {"next_token": "ARGMAX", "beam_size": 5}
                else:
                    mode = None
                batch_outputs = wizard(batch, mode)
                # Stringify model responses.
                if args["bleu_evaluation"]:
                    batch_outputs["model_response"] = (
                        val_loader.stringify_beam_outputs(
                            batch_outputs["beam_output"], batch
                        )
                    )
                    # Remove beam output to avoid memory issues.
                    del batch_outputs["beam_output"]
                matches.append(batch_outputs)
        wizard.train()

        # Compute perplexity.
        total_loss_sum = sum(ii["loss_sum"].item() for ii in matches)
        num_tokens = sum(ii["num_tokens"].item() for ii in matches)
        avg_loss_eval = total_loss_sum / num_tokens

        # Compute BLEU score.
        if args["bleu_evaluation"]:
            model_responses = [jj for ii in matches for jj in ii["model_response"]]
            bleu_score = val_loader.evaluate_response_generation(model_responses)
        else:
            model_responses = None
            bleu_score = -1.

        # Evaluate retrieval score.
        if args["retrieval_evaluation"]:
            candidate_scores = [jj for ii in matches for jj in ii["candidate_scores"]]
            retrieval_metrics = val_loader.evaluate_response_retrieval(candidate_scores)
            print(retrieval_metrics)
        else:
            retrieval_metrics = {}

        # Evaluate action prediction.
        action_predictions = [jj for ii in matches for jj in ii["action_preds"]]
        action_metrics = val_loader.evaluate_action_prediction(action_predictions)
        print(action_metrics["confusion_matrix"])
        print_str = (
            "\nEvaluation\n\tLoss: {:.2f}\n\t"
            "Perplexity: {:.2f}\n\tBLEU: {:.2f}\n\t"
            "Action: {:.2f}\n\t"
            "Action Perplexity: {:.2f}\n\t"
            "Action Attribute Accuracy: {:.2f}"
        )
        print(
            print_str.format(
                avg_loss_eval,
                math.exp(avg_loss_eval),
                bleu_score,
                100 * action_metrics["action_accuracy"],
                action_metrics["action_perplexity"],
                100 * action_metrics["attribute_accuracy"]
            )
        )
        # Save the results to a file.
        eval_dict = {
            "loss": avg_loss_eval,
            "perplexity": math.exp(avg_loss_eval),
            "bleu": bleu_score,
            "action_accuracy": action_metrics["action_accuracy"],
            "action_perplexity": action_metrics["action_perplexity"],
            "action_attribute": action_metrics["attribute_accuracy"]
        }
        eval_dict.update(retrieval_metrics)
        eval_outputs = {
            "model_actions": action_predictions,
            "model_responses": model_responses
        }
        return eval_dict, eval_outputs
    ```