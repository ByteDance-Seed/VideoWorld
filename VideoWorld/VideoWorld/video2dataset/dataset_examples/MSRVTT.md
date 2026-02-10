# [MSR-VTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)

MSR-VTT is a benchmark dataset to test video-text understanding, it has 20 captions per video and can be used to evaluate the retrieval capabilities of models. It has multiple variants (which are subsets of the original 10k videos) described [here](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/msrvtt).

## Download the videos and metadata

MSR-VTT is small enough (10k samples) where the videos and metadata are hosted as a .zip file [here](https://www.mediafire.com/folder/h14iarbs62e7p/shared). Download the files and unzip them

## Preprocess into video2dataset input format

Before you can create a video2dataset dataset you need to create the correct input format. Here's the script that does that for the test split:

```python3
import json
import pandas as pd

with open('test_videodatainfo.json', 'r') as f:
    data = json.load(f)

df_videos = pd.DataFrame(data['videos'])
df_sentences = pd.DataFrame(data['sentences'])

df = df_videos

vid_ids = df_videos["video_id"].tolist()
caps = {}
for vid_id in vid_ids:
    sents = df_sentences[df_sentences["video_id"] == vid_id]["caption"].tolist()
    caps[vid_id] = sents

def add_caption_columns(row, captions_dict):
    video_id = row['video_id']
    captions = captions_dict.get(video_id, [''] * 20)
    for i, caption in enumerate(captions):
        row[f'caption_{i}'] = caption
    return row

df = df.apply(add_caption_columns, axis=1, args=(caps,))

df['location'] = "/path/to/TestVideo/" + df["video_id"].astype(str) + ".mp4"

df.to_parquet("msr-vtt-test.parquet")
```

## Create the dataset

This video2dataset takes the mp4s unzipped in TestVideo and organizes them into a dataset that can be easily read with our dataloader:

```bash
#!/bin/bash

video2dataset --url_list="msr-vtt-test.parquet" \
        --input_format="parquet" \
        --output-format="webdataset" \
        --output_folder="/fsx/iejmac/datasets/msr-vtt/dataset" \
        --url_col="location" \
        --save_additional_columns='[video_id,category,id,caption_0,caption_1,caption_2,caption_3,caption_4,caption_5,caption_6,caption_7,caption_8,caption_9,caption_10,caption_11,caption_12,caption_13,caption_14,caption_15,caption_16,caption_17,caption_18,caption_19]' \
```
