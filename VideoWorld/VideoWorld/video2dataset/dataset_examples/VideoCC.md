# [VideoCC](https://github.com/google-research-datasets/videoCC-data)

VideoCC is a video-text dataset with ~10M samples created by starting from an image-text dataset and retrieving videos with frames similar to images in that dataset.

## Download the metadata

Go to [this section](https://github.com/google-research-datasets/videoCC-data#data-format-for-videocc) of their README and download the CSV file. It will need some simple processing which can be done with this code snippet:
```python3
import pandas as pd

df = pd.read_csv("video_cc_public.csv")
df.columns = ["video_url", "start", "end", "caption"]
df["video_url"] = df["video_url"].apply(lambda x: f"https://www.youtube.com/watch?v={x}")
df['start'] = df['start'] / 1_000_000
df['end'] = df['end'] / 1_000_000
df.to_parquet("video_cc.parquet")
```

## Create the config

We want to perform cut detection and clipping based on those cuts so we need to specify some subsamplers. We also want to extract compression metadata so we add the FFProbeSubsampler.

```yaml
subsampling:
    CutDetectionSubsampler:
        cuts_are_clips: True
        args:
            cut_detection_mode: "all"
            framerates: null
            threshold: 11.5
            min_scene_len: 15
    ClippingSubsampler:
        args:
            min_length: 3.0
            max_length: 20.0
            max_length_strategy: "all"
            precision: "keyframe_adjusted"
    FFProbeSubsampler:
        args:
            extract_keyframes: False

reading:
    yt_args:
        download_size: 360
        download_audio_rate: 44100
        yt_metadata_args:
            writesubtitles:  'all'
            subtitleslangs: ['en']
            writeautomaticsub: True
            get_info: True
    timeout: 60
    sampler: null

storage:
    number_sample_per_shard: 1000
    oom_shard_count: 5
    captions_are_subtitles: False

distribution:
    processes_count: 48
    thread_count: 48
    subjob_size: 1000
    distributor: "slurm"
    distributor_args:
        partition: "cpu64"
        n_nodes: 50
        account: "laion"
        cache_path: "/fsx/home-iejmac/.slurm_cache"
```

## Download and process the videos using video2dataset:

The dataset can be downloaded with cut detection and clipping using the following python code:

```python3
video2dataset(
    url_list='video_cc.parquet',
    input_format='parquet',
    output_format='webdataset',
    output_folder='dataset',
    url_col="video_url",
    encode_formats={"video": "mp4"},
    stage="download",
    config="path/to/config.yaml"
)
```

## Performance

NOTE: VideoCC is a youtube-based dataset so it will be slower to download than mp4-based ones. On a single cpu16 (16 core) EC2 instance (c6i-4xlarge) the entirety of VideoCC (~5M samples) can be downloaded in ~220h. It achieves 5 video/s (0.31 videos/s/core) or 87 Mb/s and I highly recommend paralellizing this over many nodes. This means the cost to download VideoCC comes out to ~0.68$/hr * 220h = 150$
