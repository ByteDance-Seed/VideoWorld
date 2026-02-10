# API

The module exposes a single function `video2dataset` which takes the same arguments as the command line tool:

## Core Args

```
url_list: list of input urls - can be any of the supported input formats
    (csv, parquet, braceexpand tar paths etc.)
output_folder: Desired location of output dataset (default = "dataset")
output_format: Format of output dataset, can be (default = "files")
    - files, samples saved in subdirectory for each shard (useful for debugging)
    - webdataset, samples saved in tars (useful for efficient loading)
    - parquet, sampels saved in parquet (as bytes)
    - tfrecord, samples saved in tfrecord (as bytes)
    - dummy, does not save (useful for benchmarks)
input_format: Format of the input, can be (default = "csv")
    - txt, text file with a url in each line
    - csv, csv file with urls, (and captions + metadata)
    - tsv, tsv - || -
    - tsv.gz, - || - but compressed gzip
    - json, loads urls and metadata as json
    - parquet, loads urls and metadata as parquet
    - webdataset, usually braceexpand format of mutliple paths to tars to re-process
encode_formats: Dict that specifies what extension each modality should use (default = "{'video': 'mp4'}")
    f.e. {"video": "mp4", "audio": "m4a"}
stage: String that tells video2dataset what stage of processing is being performed. Can be (default = 'download')
    WARNING: To be depracated soon (this information should be deduced based on config)
    - download, when input is some tabular format and data must be downloaded first
    - subset, tar files are already written and we would like to re-process (input_format == "webdataset")
    - optical_flow, tar files are written and we woudl like to compute optical_flow and save to md shards
url_col: Column in input (if has columns) that contains the url (default = "url")
caption_col: Column in input (if has columns) that contains captions (to be written as txt) (default = None)
clip_col: Column in input (if has columns) that contains timeframes of clips for how to split video (default = None)
save_additional_columns: List of column names to save to json component of a sample (defualt = None)
enable_wandb: Whether or not to log info to wandb (default = False)
wandb_project: Name of wandb project to log runs to (default = "video2dataset")
incremental_mode: Decides how to handle restarting, Can be (default = "incremental")
    - incremental, checks which shards are done and skips those
    - overwrite, deletes and reprocesses shards as it goes
max_shard_retry: Maximum amount of attempts to retry a failed shard (default = 1)
tmp_dir: Path to temporary directory on your file system (default = "/tmp")
config: Path to your config of choice or the config itself (more info on configs in API doc) (default = "default")
```

## Config

Any video2dataset configs is divided into 4 components which describe the following:
- subsampling: how the data should be transformed on its way from input to output
- reading: how the data should be read (from the internet or local FS)
- storage: how the data should be stored
- distribution: how processing should be distributed among compute resources

Lets look at an example and go over where we can find information on how to fill out a config:

```yaml
subsampling:
    ResolutionSubsampler:
        args:
            video_size: 224
            resize_mode: "scale,crop,pad"
    FrameSubsampler:
        args:
           frame_rate: 5
    CutDetectionSubsampler:
        cuts_are_clips: True
        args:
            cut_detection_mode: "all"
            framerates: null
            threshold: 27
            min_scene_len: 15
    ClippingSubsampler:
        args:
            min_length: 4.0
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
        yt_metadata_args: null
    timeout: 60
    sampler: null

storage:
    number_sample_per_shard: 1000
    oom_shard_count: 5
    captions_are_subtitles: False

distribution:
    processes_count: 16
    thread_count: 32
    subjob_size: 1000
    distributor: "multiprocessing"
```

### Subsampling

This component of the config will contain a dict with subsampler names and their respective parameters. Each subsampler specification is formatted as such:

```yaml
SubsamplerName:
    ninit_arg1: "test"
    ...
    ninit_argn: "test"
    args:
        init_arg1: 1
        ...
        init_argn: n
```

The arguments within each subsampler are split into 2 groups:
- initialization args: these are the arguments inside the "args" dict of each subsampler specification. If this is not present the subsampler will not be initialized.
- non-initialization args: the video2dataset worker will use these to perform operations outside the subsampler. They can be used to specify processes even without the presenece of the initialized subsampler. A good example is the `cuts_are_clips` parameter in the `CutDetectionSubsampler`. If we perform a stage where we detect cuts in the input videos and place those cuts in the metadata but do not clip the videos themselves we can simply perform this clipping in a subsequent stage without the need for re-computing those cuts. We would do this by *just* passing in the CutDetectionSubsampler specification without the *args* but adding `cuts_are_clips`. video2dataset would then just extract the cut information from the metadata and use that for clipping. It would look something like this:
```yaml
...
    CutDetectionSubsampler:
        cuts_are_clips: True
    ClippingSubsampler:
        args:
            min_length: 4.0
            max_length: 20.0
            max_length_strategy: "all"
            precision: "keyframe_adjusted"
...
```

To check what values can be present in a specification dict of a given subsampler check the documentation in the [docstring of the subsampler](https://github.com/iejMac/video2dataset/tree/main/video2dataset/subsamplers).

### Reading

The reading component of the config informs video2dataset the preferred options for reading data from the specified source. Here is a maxed out reading specification (it has all possible options specified):

```yaml
reading:
    yt_args:
        download_size: 360
        yt_metadata_args:
            writesubtitles: 'all'
            subtitleslangs: ['en']
            writeautomaticsub: True
            get_info: True
    dataloader_args:
        resize_size: 16
        decoder_kwargs:
            n_frames: null
            fps: 2
            num_threads: 8
            return_bytes: True
    timeout: 60
    sampler: null
```

Options:

```
yt_args: arguments used by the YtDlpDownloader, see the docstring of that class in data_reader.py for
    an explanation on what they do.
dataloader_args: arguments passed to the dataloader which will be used to load frames for stages that
    need them (f.e. optical flow). Follow dataloader documentation for that
timeout: tells video2dataset the maximum time to consider downloading a video.
sampler: a class that samples shards from the input (f.e. used by slurm distributor to tell workers 
    which shards to work on)
```

### Storage

The storage specification tells video2dataset how to store shards/samples. Here are the possible arguments:
```
number_sample_per_shard: how many samples should be in each shard
oom_shard_count: the order of magnitude of the number of shards, used only to decide what zero padding
    to use to name the shard files
captions_are_subtitles: If subtitles are present in metadata we can clip the video according
    to those by setting this parameter to True (TODO: this could be a ClippingSubsapler arg along with `cuts_are_clips`)
```

### Distribution

The distribution specification tells video2dataset how to parallelize processing at multiple levels. There are 4 required arguments:

```
processes_count: The number of processes used for downloading the dataset at the shard level. This is important to be high for performance.
thread_count: The number of threads to use for processing samples (sample level).
subjob_size: the number of shards to download in each subjob supporting it, a subjob can be a pyspark job for example
distributor: the type of distribution used, can be
    - multiprocessing, uses multiprocessing pool to spawn processes
    - spark, use a pyspark session to create workers on a spark cluster (see details in `examples/distributed_spark.md`)
    - slurm, use slurm to distribute processing to multiple slurm workers
```

On top of these args some distributors like slurm need additional arguments. You can check the docstring of the distributor to see what needs to be specified and then fill in the `distributor_args` entry in the config. Here's an example (currently only slurm requires this):

```yaml
distribution:
    processes_count: 1
    thread_count: 8
    subjob_size: 1000
    distributor: "slurm"
    distributor_args:
        cpus_per_task: 96
        job_name: "v2ds"
        partition: "g80"
        n_nodes: 2
        gpus_per_node: 8
        account: "stablediffusion"
        tasks_per_node: 1
```

### Examples

We provide 3 example configs which can be used by setting the config parameter of video2dataset to the string:
- `default` - This performs no subsampling and attempts to download videos at either native resolution or close to 360p for youtube videos.
- `downsample_ml` - performs resolution downsampling to 224x224 via a scale, center crop, and pad for smaller videos, downsamples FPS to 5, detects cuts, clips the videos according to those cuts and saves compression metadata from FFProbe.
- `optical_flow` - takes a created tar dataset and computes the optical flow for the videos in it

If you want to create your own config based on these you can copy them from video2dataset/configs
