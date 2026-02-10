# Contribution guide

video2dataset open contributions to add new features, improve efficiency or improve the code health.

## How to validate your changes ?

Before merging a change (especially for non trivial changes), we ask to:

* make sure the linting is passing, you can run `make black` and `make lint` locally and then check the status in a PR
* make sure the existing tests are passing, you can run `make test` locally and then check the status in a PR
* add new tests for new features or for bug fixes
* run manually an efficiency test. video2dataset must remain fast so this is important

## Efficiency test

To test the efficiency of video2dataset, you can follow [this example to download webvid](dataset_examples/WebVid.md)

Using 16 processes with 16 threads each is particularly important to check the speed. Enabling wandb is also important.

You can run with only the `results_2M_val` to reduce the run time of this test.

You should observe 14.4 videos/s/core in wandb.

Please post the wandb link in the PR to show this is working. It will make it faster for the reviewer to merge the PR.


