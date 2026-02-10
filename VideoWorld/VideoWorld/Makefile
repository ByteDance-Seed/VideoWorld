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
SHELL=/bin/bash
PROJECT_NAME=falcon
PROJECT_PATH=${PROJECT_NAME}/
LINT_PATHS=${PROJECT_PATH} tools/ configs/ tokenizer/ setup.py

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

pytest:
	$(call check_install, pytest)
	$(call check_install, pytest_cov)
	pytest tests --cov ${PROJECT_PATH} --durations 0 -v --cov-report term-missing


lint:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)
	flake8 ${LINT_PATHS} --count --show-source --statistics

format:
	# sort imports
	$(call check_install, isort)
	isort ${LINT_PATHS}
	# reformat using yapf
	$(call check_install, yapf)
	yapf -ir ${LINT_PATHS}

check-codestyle:
	$(call check_install, isort)
	$(call check_install, yapf)
	# isort --check ${LINT_PATHS} && yapf -r -d ${LINT_PATHS}
	yapf -r -d ${LINT_PATHS}

check-docstyle:
	$(call check_install, pydocstyle)
	pydocstyle ${PROJECT_PATH} --convention=google

doc:
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	cd docs && make html && cd _build/html && python3 -m http.server

spelling:
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)
	cd docs && make spelling SPHINXOPTS="-W"

clean:
	cd docs && make clean

commit-checks: format lint check-docstyle


# PyPI package release
release:
	$(call check_install, twine)
	python setup.py sdist bdist_wheel
	twine -r devpi-dev upload dist/*

nightly-upload:
	$(call check_install, twine)
	./scripts/nightly_upload.py ${PROJECT_NAME}

.PHONY: clean spelling doc lint format check-codestyle check-docstyle commit-checks
