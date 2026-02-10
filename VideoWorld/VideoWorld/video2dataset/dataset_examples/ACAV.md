# [ACAV100M](https://acav100m.github.io)
ACAV100M is a dataset of 100M videos with high audio-visual correspondance.

## Download the metdata
Go to the [downloads](https://acav100m.github.io/#downloads) section of the dataset page and download the subset you wish. This just has the youtube ID, start time, and end time. For the current version of video2dataset you need to preprocess this file a bit:
1. Create a column with the link instead of just the ID: f"youtube.com/watch?v={youtubeID}"
```python
df["video_link"] = df.apply(lambda row: f"https://youtube.com/watch?v={row['videoID']}", axis=1)
```
2. Create a column for the timespan consistent with video2dataset expected clip format: [[start, end]]
```python
df["clips"] = df.apply(lambda row: [[row['start'], row['end']]], axis=1)
```

## Create the appropriate video2dataset config

Since we only want to download 10 second clips we can allow for more samples per shard than usual. Also since this is a massive dataset we want to use a different distributor like pyspark. We also want to specify that we should download the metadata of the video and include that into the json file of each sample using the `yt_metadata_args`.

```yaml
subsampling: {}

reading:
    yt_args:
        download_size: 360
        download_audio_rate: 44100
        yt_metadata_args:
            get_info: True
    timeout: 60
    sampler: null

storage:
    number_sample_per_shard: 2000
    captions_are_subtitles: False

distribution:
    processes_count: 96
    thread_count: 48
    subjob_size: 10000
    distributor: "pyspark"
```

## Download the videos with video2dataset

This command runs video2dataset on the input table and saves the video clips along with the metadata in the webdataset format.

```python
video2dataset(
    url_list="ACAV100M_clip_unique.parquet",
    output_folder="output_folder",
    output_format="webdataset",
    input_format="parquet",
    url_col="video_link",
    caption_col="title",
    clip_col="clips",
    save_additional_columns=["videoID", "start", "end"],
    enable_wandb=True,
    config="path/to/config.yaml"
)
```

## Performance
The job ran on 64 nodes with 48 CPU cores each and processed the dataset in 7 hours. This wasn't the entire ACAV100M dataset but instead a 60M subset of only the unique videos in the dataset. The title and description of the videos were pre-extracted and placed in the parquet file along with the description and also the timespan of the clip was processed into the format that video2dataset expects. [Here's](https://wandb.ai/iejmac/video2dataset/runs/3kon2409?workspace=user-iejmac) a link to the wandb logs from the processing of the dataset. We were able to get 0.8 vid/s/core which comes out to approximately 100MB/s/core.
