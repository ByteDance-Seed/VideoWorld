# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup, find_packages
from pathlib import Path
import os

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    def _read_reqs(relpath):
        fullpath = os.path.join(os.path.dirname(__file__), relpath)
        with open(fullpath) as f:
            return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]

    REQUIREMENTS = _read_reqs("requirements.txt")

    setup(
        name="video2dataset",
        packages=find_packages(),
        include_package_data=True,
        version="1.3.0",
        license="MIT",
        description="Easily create large video dataset from video urls",
        long_description=long_description,
        long_description_content_type="text/markdown",
        entry_points={"console_scripts": ["video2dataset=video2dataset.main:main"]},
        author="Maciej Kilian",
        author_email="kilianmaciej6@gmail.com",
        url="https://github.com/iejMac/video2dataset",
        data_files=[
            (
                ".",
                [
                    "README.md",
                    "video2dataset/configs/caption.yaml",
                    "video2dataset/configs/default.yaml",
                    "video2dataset/configs/downsample_ml.yaml",
                    "video2dataset/configs/optical_flow.yaml",
                ],
            )
        ],
        keywords=["machine learning"],
        install_requires=REQUIREMENTS,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
        ],
    )
