# [Kinetics](https://www.deepmind.com/open-source/kinetics)

Kinetics is a large video classification dataset. It has multiple variants (400, 600, 700, etc.) but I will only show an example for K400.

## Preparation

Kinetics is a youtube dataset however many of the videos have gone private, been removed, and are no longer available. To keep the dataset intact people host it on S3 so we will make use of that to get the mp4s onto your machine first, then process them and store in webdataset format using video2dataset. For downloading we can use the [kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset) repo. Follow their [instruction](https://github.com/cvdfoundation/kinetics-dataset#kinetics-400) on how to download Kinetics400.


After this we need to create a table to input into video2dataset. Their annotations csv files are a good base, we just need to add the location of the mp4s. We can do that with the following code:

```python3
import pandas as pd
df = pd.read_csv("path/to/train.csv")
mp4_path = "/path/to/kinetics-dataset/k400/train"  # this is where the mp4s are stored after extraction
df['path'] = mp4_path + "/" + df['youtube_id'] + "_" + df['time_start'].astype(str).str.zfill(6) + "_" + df['time_end'].astype(str).str.zfill(6) + ".mp4"
df.to_parquet("k400_train.parquet")
```

## Create the dataset using video2dataset

This simple script can take all those mp4s and organize them into shards and preprocess them according to your specified config (I didn't apply any transformations for my usage)

```bash
#!/bin/bash

video2dataset --url_list="k400_train.parquet" \
        --input_format="parquet" \
        --output-format="webdataset" \
        --output_folder="dataset" \
        --url_col="path" \
        --caption_col="label" \
        --save_additional_columns='[youtube_id,time_start,time_end,split,is_cc]' \
        --config=default \
```
