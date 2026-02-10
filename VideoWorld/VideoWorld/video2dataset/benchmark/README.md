# video2dataset Benchmark Suite
The code in here can be used to benchmark the performance of video2dataset components (and end2end) on different hardware/data conditions

## Subsamplers

Test out the performance of video2dataset subsamplers by configuring the ```subsamplers_config.yaml``` file with the subsamplers you want to benchmark along with the parameters (to form a parameter grid to test over). You can also choose a set of videos to test on (in case thats relevant for your use case). The script will output a JSON file with the information of the system performing the benchmark and a metrics dict for each parameter set in the input grid.

For subsampler_config:
```yaml
video_set:
  - path: "dataset/mp4/{00000..00009}.tar"
subsamplers:
  - name: ResolutionSubsampler
    parameters:
      - video_size: [360, 60]
        resize_mode: ["scale"]
```

The benchmark will output:
```json
{
    "system_info": {
        "platform": "Linux",
        "cpu_count": 96,
        "cpu_info": "x86_64",
        "gpu_info": "NVIDIA A100-SXM4-80GB",
        "gpu_count": 8
    },
    "configs_and_metrics": [
        {
            "name": "ResolutionSubsampler",
            "config": {
                "video_size": 360,
                "resize_mode": "scale"
            },
            "metrics": {
                "time": 111.43327045440674,
                "samples": 100,
                "bytes": 236045214
            }
        },
        {
            "name": "ResolutionSubsampler",
            "config": {
                "video_size": 60,
                "resize_mode": "scale"
            },
            "metrics": {
                "time": 29.380290031433105,
                "samples": 100,
                "bytes": 236045214
            }
        }
    ]
}
```
