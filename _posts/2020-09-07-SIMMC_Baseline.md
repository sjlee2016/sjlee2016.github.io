# Simmc Baseline 분석

# Data

### /data folder structure.

여기서 domain은 simmc_furniture | simmc_fashion

```
[Main Data]
- full dialogs: ./{domain}/{train|dev|devtest|test}_dials.json
- list of dialog IDs per split: ./{domain}/{train|dev|devtest|test}_dialog_ids

[Metadata]
- Fashion metadta: ./simmc_fashion/fashion_metadata.json
- Furniture metadata: ./simmc_furniture/furniture_metadata.csv
- images: ./simmc-furniture/figures/{object_id}.png
```

### Data format

```json
{
  "split": support.extract_split_from_filename(json_path),
  "version": 1.0,
  "year": 2020,
  "domain": FLAGS.domain,
  "dialogue_data": [
  {
    “dialogue”: [
      {
        “belief_state”: [
          {
            “act”: <str>,
            “slots”: [
              [ <str> slot_name, <str> slot_value  ], // end of a slot name-value pair
              ...
            ]
          }, // end of an act-slot pair
          ...
        ],
        “domain”: <str>,
        “raw_assistant_keystrokes”: <dict>,
        “state_graph_{idx}”: <dict>,
        “syste_belief_state”: <dict>,
        “system_transcript”: <str>,
        “system_transcript_annotated”: <str>,
        “transcript”: <str>,
        “transcript_annotated”: <str>,
        “turn_idx”: <int>,
        “turn_label”: [ <dict> ],
        “visual_objects”: <dict> .
      }, // end of a turn (always sorted by turn_idx)
      ...
    ],
    “dialogue_coref_map”: {
      // map from object_id to local_id re-indexed for each dialog
      <str>: <int>
    },
    “dialogue_idx”: <int>,
    “domains”: [ <str> ]
  }
]
}

```

데이터는 각 subtask에 대해 각 데이터 리더/사전 처리 스크립트를 사용하여 처리됨 

*Note 

`visual_objects` : 해당 턴에서 유저에게 display 되는 오브젝트 리스트와 해당 오브젝트의 attributes. 

`state_graph_{idx}` 대화 히스토리와 멀티모달 컨텍스트를 각각 다른 단계(예: 항목 표시, 정보 제공 보조자, 사용자 기본 설정 제공 등)로 나타낸 그래프

- state_graph_0: 유저의 말 이전 state
- state_graph_1: 유저의 말 이후 바뀐 state
- state_graph_2: 어시스턴트의 액션과 말 후에 바뀐 final state

`raw_assistant_keystrokes` : 데이터를 수집하는 동안 Unity 인터페이스를 사용하여 Human Assistant(마법사)가 수행한 원시 UI 상호 작용

### Metadata

```
<fashion_metadata.json>
{
    <int> object_id: {
        “metadata”: {dict},
        “url”: <str> source image
    }, // end of an object
}

<furniture_metadata.csv>
columns:
- product_name
- product_description
- product_thumbnail_image_url
- material
- color
- obj ({object_id}.zip)
...
```

# Sub-Task #1: Multimodal Assistant API Prediction (/mm_action_prediction)

어시스턴트 API 를 다이어로그 히스토리, 멀티모덜 콘텍스트, 유저 utterance를 통해 예측하는 테스크.  예를 들어, 유저가 가구 가격에 대한 정보를  물어볼때 specifyInfo API 와 price argument를 어시스턴트가 사용해야 한다는것을 예측한다. 

# Sub-Task #2: Multimodal Assistant Response Generation

어시스턴트의 response 를 다이어로그 히스토리, 멀티모덜 콘텍스트, 어시스턴트 API call 과 현재 utterance를 통해 생성하는 테스크. 

[Task 1 & 2 코드 분석](Simmc%20Baseline%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%E1%86%A8%20c29723f1da874808b5fefaab8bbd586e/Task%201%20&%202%20%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%E1%86%A8%20e94fb9958e5149668a59428f1ca1ea4d.md)

# Sub-Task #3: Multimodal Dialog State Tracking (/mm-dst)

The Multimodal Dialog State Tracking (MM-DST) task involves systematically **tracking the attributes of dialog act labels** cumulative across **multiple turns**. Multimodal belief states at each turn should encode sufficient information for handling user utterances in the downstream dialog components (e.g. Dialog Policy).

[Task 3 코드 분석](Simmc%20Baseline%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%E1%86%A8%20c29723f1da874808b5fefaab8bbd586e/Task%203%20%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%E1%86%A8%20237b9c1ac6d14b9a890d5ea6c7ac7e2c.md)