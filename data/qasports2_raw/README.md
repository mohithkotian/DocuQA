---
configs:
- config_name: all
  data_files:
  - split: train
    path:
    - "basketball/train/data.parquet"
    - "football/train/data.parquet"
    - "americanfootball/train/data.parquet"
    - "cricket/train/data.parquet"
    - "hockey/train/data.parquet"
    - "golf/train/data.parquet"
    - "rugbyunion/train/data.parquet"
    - "rugby/train/data.parquet"
    - "baseball/train/data.parquet"
    - "martialarts/train/data.parquet"
    - "boxing/train/data.parquet"
    - "mixedmartialarts/train/data.parquet"
    - "fitness/train/data.parquet"
    - "cycling/train/data.parquet"
    - "badminton/train/data.parquet"
    - "gymnastics/train/data.parquet"
    - "handball/train/data.parquet"
    - "skiing/train/data.parquet"
    - "horse_racing/train/data.parquet"
    - "f1/train/data.parquet"
  - split: test
    path:
    - "basketball/test/data.parquet"
    - "football/test/data.parquet"
    - "americanfootball/test/data.parquet"
    - "cricket/test/data.parquet"
    - "hockey/test/data.parquet"
    - "golf/test/data.parquet"
    - "rugbyunion/test/data.parquet"
    - "rugby/test/data.parquet"
    - "baseball/test/data.parquet"
    - "martialarts/test/data.parquet"
    - "boxing/test/data.parquet"
    - "mixedmartialarts/test/data.parquet"
    - "fitness/test/data.parquet"
    - "cycling/test/data.parquet"
    - "badminton/test/data.parquet"
    - "gymnastics/test/data.parquet"
    - "handball/test/data.parquet"
    - "skiing/test/data.parquet"
    - "horse_racing/test/data.parquet"
    - "f1/test/data.parquet"
  - split: validation
    path:
    - "basketball/validation/data.parquet"
    - "football/validation/data.parquet"
    - "americanfootball/validation/data.parquet"
    - "cricket/validation/data.parquet"
    - "hockey/validation/data.parquet"
    - "golf/validation/data.parquet"
    - "rugbyunion/validation/data.parquet"
    - "rugby/validation/data.parquet"
    - "baseball/validation/data.parquet"
    - "martialarts/validation/data.parquet"
    - "boxing/validation/data.parquet"
    - "mixedmartialarts/validation/data.parquet"
    - "fitness/validation/data.parquet"
    - "cycling/validation/data.parquet"
    - "badminton/validation/data.parquet"
    - "gymnastics/validation/data.parquet"
    - "handball/validation/data.parquet"
    - "skiing/validation/data.parquet"
    - "horse_racing/validation/data.parquet"
    - "f1/validation/data.parquet"
  default: true
- config_name: basketball
  data_files:
  - split: train
    path: "basketball/train/data.parquet"
  - split: test
    path: "basketball/test/data.parquet"
  - split: validation
    path: "basketball/validation/data.parquet"
- config_name: football
  data_files:
  - split: train
    path: "football/train/data.parquet"
  - split: test
    path: "football/test/data.parquet"
  - split: validation
    path: "football/validation/data.parquet"
- config_name: americanfootball
  data_files:
  - split: train
    path: "americanfootball/train/data.parquet"
  - split: test
    path: "americanfootball/test/data.parquet"
  - split: validation
    path: "americanfootball/validation/data.parquet"
- config_name: cricket
  data_files:
  - split: train
    path: "cricket/train/data.parquet"
  - split: test
    path: "cricket/test/data.parquet"
  - split: validation
    path: "cricket/validation/data.parquet"
- config_name: hockey
  data_files:
  - split: train
    path: "hockey/train/data.parquet"
  - split: test
    path: "hockey/test/data.parquet"
  - split: validation
    path: "hockey/validation/data.parquet"
- config_name: golf
  data_files:
  - split: train
    path: "golf/train/data.parquet"
  - split: test
    path: "golf/test/data.parquet"
  - split: validation
    path: "golf/validation/data.parquet"
- config_name: rugbyunion
  data_files:
  - split: train
    path: "rugbyunion/train/data.parquet"
  - split: test
    path: "rugbyunion/test/data.parquet"
  - split: validation
    path: "rugbyunion/validation/data.parquet"
- config_name: rugby
  data_files:
  - split: train
    path: "rugby/train/data.parquet"
  - split: test
    path: "rugby/test/data.parquet"
  - split: validation
    path: "rugby/validation/data.parquet"
- config_name: baseball
  data_files:
  - split: train
    path: "baseball/train/data.parquet"
  - split: test
    path: "baseball/test/data.parquet"
  - split: validation
    path: "baseball/validation/data.parquet"
- config_name: martialarts
  data_files:
  - split: train
    path: "martialarts/train/data.parquet"
  - split: test
    path: "martialarts/test/data.parquet"
  - split: validation
    path: "martialarts/validation/data.parquet"
- config_name: boxing
  data_files:
  - split: train
    path: "boxing/train/data.parquet"
  - split: test
    path: "boxing/test/data.parquet"
  - split: validation
    path: "boxing/validation/data.parquet"
- config_name: mixedmartialarts
  data_files:
  - split: train
    path: "mixedmartialarts/train/data.parquet"
  - split: test
    path: "mixedmartialarts/test/data.parquet"
  - split: validation
    path: "mixedmartialarts/validation/data.parquet"
- config_name: fitness
  data_files:
  - split: train
    path: "fitness/train/data.parquet"
  - split: test
    path: "fitness/test/data.parquet"
  - split: validation
    path: "fitness/validation/data.parquet"
- config_name: cycling
  data_files:
  - split: train
    path: "cycling/train/data.parquet"
  - split: test
    path: "cycling/test/data.parquet"
  - split: validation
    path: "cycling/validation/data.parquet"
- config_name: badminton
  data_files:
  - split: train
    path: "badminton/train/data.parquet"
  - split: test
    path: "badminton/test/data.parquet"
  - split: validation
    path: "badminton/validation/data.parquet"
- config_name: gymnastics
  data_files:
  - split: train
    path: "gymnastics/train/data.parquet"
  - split: test
    path: "gymnastics/test/data.parquet"
  - split: validation
    path: "gymnastics/validation/data.parquet"
- config_name: handball
  data_files:
  - split: train
    path: "handball/train/data.parquet"
  - split: test
    path: "handball/test/data.parquet"
  - split: validation
    path: "handball/validation/data.parquet"
- config_name: skiing
  data_files:
  - split: train
    path: "skiing/train/data.parquet"
  - split: test
    path: "skiing/test/data.parquet"
  - split: validation
    path: "skiing/validation/data.parquet"
- config_name: horse_racing
  data_files:
  - split: train
    path: "horse_racing/train/data.parquet"
  - split: test
    path: "horse_racing/test/data.parquet"
  - split: validation
    path: "horse_racing/validation/data.parquet"
- config_name: f1
  data_files:
  - split: train
    path: "f1/train/data.parquet"
  - split: test
    path: "f1/test/data.parquet"
  - split: validation
    path: "f1/validation/data.parquet"
pretty_name: QASports2
license: mit
task_categories:
- question-answering
task_ids:
- extractive-qa
language:
- en
multilinguality:
- monolingual
source_datasets:
- extended|wikipedia
tags:
- sports
- open-domain-qa
- extractive-qa
size_categories:
- 1M<n<10M
---

### Dataset Summary

QASports is the first large sports-themed question answering dataset counting over 1 million questions and answers about 124k preprocessed wiki pages, using as documents the wiki of 20 of the most popular sports in the world, like Soccer, American Football, Basketball, Cricket, and so on. Each sport can be downloaded individually as a subset, with the train, test and validation splits, or all subsets can be downloaded together.

- 🔧 Processing scripts: https://github.com/leomaurodesenv/qasports-dataset-scripts/

### Supported Tasks and Leaderboards

Extractive Question Answering.

### Languages

English.

## Dataset Structure

### Data Instances

A sample looks as follows.
```
{
    "answer": {
        "offset": [42,44],
        "text": "16"
    },
    "context": "The following is a list of squads for all 16 national teams competing at the Copa América Centenario. Each national team had to submit a squad of 23 players, 3 of whom must be goalkeepers. The provisional squads were announced on 4 May 2016. A final selection was provided to the organisers on 20 May 2016.",
    "qa_id": "61200579912616854316543272456523433217",
    "question": "How many national teams competed at the Copa América Centenario?",
    "context_id": "171084087809998484545703642399578583178",
    "context_title": "Copa América Centenario squads | Football Wiki | Fandom",
    "url": "https://football.fandom.com/wiki/Copa_Am%C3%A9rica_Centenario_squads"	
}
```
### Data Fields

The data fields are the same among all splits.
- `qa_id`: `[string]` question-answering id.
- `context_id`: `[string]` context id.
- `context_title`: `[string]` context title.
- `context_url`: `[string]` context URL.
- `context_categories`: `[string]` categories. Split by comma.
- `context`: `[string]` context text.
- `question`: `[string]` question.
- `answers`: `[dictionary]` containing:
  - `text`: `[string]` answer text.
  - `offset`: `[list]` containing:
    - `[int32]` with offset start and end.
