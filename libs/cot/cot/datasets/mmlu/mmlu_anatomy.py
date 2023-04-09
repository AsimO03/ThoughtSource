# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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


import csv

import os
import re
from typing import Dict, List, Tuple

import datasets

from cot.utils import schemas
from cot.utils.configs import ThoughtSourceConfig

_CITATION = """\
@article{hendryckstest2021,
      title={Measuring Massive Multitask Language Understanding},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
      journal={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2021}
    }
"""
_DATASETNAME = "mmlu_anatomy"

_DESCRIPTION = """\
anatomy subset of tasksource/mmlu
"""

_HOMEPAGE = "https://github.com/hendrycks/test"

_LICENSE = ""
_URLS = {
    _DATASETNAME: "https://www.dropbox.com/s/nv4z13trkpq80bj/mmlu.tar?dl=1"
}


_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class mmluDataset(datasets.GeneratorBasedBuilder):
    """MC test consisting of anatomy questions"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="source", 
            version=SOURCE_VERSION,
            description="mmlu_anatomy source schema",
            schema="source",
            subset_id="mmlu_anatomy",
            # description=f"Hendrycks Test Subject {sub}"
        ),
        # for sub in _SUBJECTS
         ThoughtSourceConfig(
            name="thoughtsource",
            version=BIGBIO_VERSION,
            description="mmlu_anatomy thoughtsource schema",
            schema="thoughtsource",
            subset_id="mmlu_anatomy",
        ),
    ]

    DEFAULT_CONFIG_NAME = "thoughtsource"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "choices": datasets.Value("string"),
                    "explanation": [datasets.Value("string")],
                }
            )
        elif self.config.schema == "thoughtsource":
            features = schemas.cot_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION, 
            features=features, 
            homepage=_HOMEPAGE, 
            license=_LICENSE,
            citation=_CITATION, 
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "filepath": os.path.join(
                        data_dir,
                        "data", 
                        "dev",
                        "anatomy_dev.csv",
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "filepath": os.path.join(
                        data_dir,
                        "data", 
                        "test",
                        "anatomy_test.csv",
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                    "filepath": os.path.join(
                        data_dir,
                        "data", 
                        "val",
                        "anatomy_val.csv",
                    )
                },
            ),
            
        ]
   
    def _generate_examples(self, filepath, split) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            with open(filepath, "r") as infile:
                for key, example in enumerate(self._generate_parsed_documents(infile)):
                    yield key, example

        elif self.config.schema == "thoughtsource":
            with open(filepath, "r") as infile:
                for key, example in enumerate(self._generate_parsed_documents(infile)):
                    yield key, self._source_to_thoughtsource(example, split)

    def _generate_parsed_documents(self, fstream):
        question_object = list(self._generate_raw_documents(fstream))
        id_counter = 0
        for quest in self._process_questions(question_object):
            question_id = id_counter
            id_counter += 1
            question = quest['question']
            choices = quest['choice_a'], quest['choice_b'], quest['choice_c'], quest['choice_d']
            answer = quest['answer']
            explanations = [re.search(r".*(?= \(.*\) \(.*\))", x).group() for x in quest['explanation'] if "No UUID specified" not in x]

            yield {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "choices": choices,
                "explanation": explanations,
            }

    def _generate_raw_documents(self, fstream):
        raw_document = []
        for line in fstream:
            if line.strip():
                raw_document.append(line.strip())
            elif raw_document:
                yield raw_document
                raw_document = []
        if raw_document:
            yield raw_document

    def _source_to_thoughtsource(self, example, split):
        cot = example["explanation"]

        # resolve ( tree ; plant ) synsets
        pattern = r"\((.*?) ; (.*?)\)"
        for idx in range(len(cot)):
            match = re.search(pattern, cot[idx])
            while match:
                cot[idx] = cot[idx][: match.span()[0]] + match.group(1) + cot[idx][match.span()[1] :]
                match = re.search(pattern, cot[idx])

        cot = [x.capitalize() for x in cot]
        cot = [x + "." if x[-1] not in [".", "!", "?"] else x for x in cot]

        example_ = {
            "id": "mmlu_anatomy_" + split + "_" + str(example["question_id"]),
            "ref_id": "",
            "question": example["question"],
            "type": "multiplechoice",
            "choices": example["choices"],
            "context": "",
            "cot": cot,
            "answer": [example["answer"]],
            "feedback": [],
            "generated_cot": [],
        }
        return example_
    
    def _process_questions(self, raw_data):
        questions = []
        for question_str in raw_data[0]:
            question_parts = question_str.split(',')
            question = question_parts[0]
            choice_a = question_parts[1]
            choice_b = question_parts[2]
            choice_c = question_parts[3]
            choice_d = question_parts[4]
            answer = question_str[-1]
            # Map answer to the corresponding choice
            if answer == 'A':
                answer = choice_a
            elif answer == 'B':
                answer = choice_b
            elif answer == 'C':
                answer = choice_c
            elif answer == 'D':
                answer = choice_d
            else:
                # Handle invalid answers
                answer = None
            question = {
                'question': question,
                'choice_a': choice_a,
                'choice_b': choice_b,
                'choice_c': choice_c,
                'choice_d': choice_d,
                'answer': answer,
                'explanation': []
            }
            questions.append(question)
        return questions