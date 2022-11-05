# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""Dialog AMR dataset."""
import json
import datasets
logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """
PubMed articles.

There are three features:
  - src: source dialog.
  - tok_ali: source bpe alignment.
  - edge_mask: source bpe alignment.
  - edge_label: source bpe alignment.
  - amr: source AMR graph.
"""


_SRC = "src"
_SRC_ALI = "src_ali"
_REL_MASK = "rel_mask"
_REL_LABEL = "rel_label"
_AMR = "amr"

def adjlst_to_adjmat(inp_str):
    inp_lst = [itm.split(" ") for itm in inp_str.split("\t")]
    lenh = len(inp_lst)
    adj_mat = [[0 for _ in range(lenh)] for _ in range(lenh)]
    for ridx, itm in enumerate(inp_lst):
        if len(itm):
            for cidx in itm:
                if len(cidx):
                    adj_mat[ridx][int(cidx)] = 1
    return adj_mat
	

class AMRData(datasets.GeneratorBasedBuilder):
    """AMR Dataset."""

    # Version 1.2.0 expands coverage, includes ids, and removes web contents.
    VERSION = datasets.Version("1.1.0")
    #'''
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _SRC: datasets.Value("string"),
                    _SRC_ALI: datasets.Value("string"),
                    _REL_MASK: datasets.Value("string"),
                    _REL_LABEL: datasets.Value("string"),
                    _AMR: datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )
    #'''
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        train_path = self.config.data_files["train"]
        dev_path = self.config.data_files["validation"]
        test_path = self.config.data_files["test"]
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_path}
            ),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        logger.info("generating examples from = %s", filepath[0])
        # print("generating examples from = %s", filepath)
        with open(filepath[0], "r", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                json_dict = json.loads(line)
                src = json_dict["src"]
                src_ali = json_dict["src_ali"]
                rel_mask = json_dict["rel_mask"]
                rel_label = json_dict["rel_label"]
                amr = json_dict["amr"]
                yield idx, {_SRC: src, _SRC_ALI: src_ali, _REL_MASK: rel_mask, _REL_LABEL: rel_label, _AMR: amr}
