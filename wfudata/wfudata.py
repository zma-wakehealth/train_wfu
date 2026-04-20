# custom dataset script to handle the input xml data
# structure of this class (_info, _split_generators and _generate_examples) refers to https://huggingface.co/docs/datasets/v3.6.0/en/dataset_script

import os
import xml.etree.ElementTree as ET
from itertools import count

import datasets

_DESCRIPTION = "WFU hand-tagged de-identification dataset (XML format)"

# switch to BIO format
BASE_TYPES = [
    'AGE', 'DATE', 'EMAIL', 'HOSPITAL', 'IDNUM', 'INITIALS',
    'IPADDRESS', 'LOCATION', 'NAME', 'OTHER', 'PHONE', 'URL', 'NORMAL'
]

CLASS_NAMES = ['O']
for base_type in BASE_TYPES:
    CLASS_NAMES.append(f"B-{base_type}")
    CLASS_NAMES.append(f'I-{base_type}')

LABEL2ID = {label: i for i, label in enumerate(CLASS_NAMES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

class I2B2WFUDataset(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="deid", version=datasets.Version("1.0.0"))
    ]

    DEFAULT_CONFIG_NAME = "deid"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "filename": datasets.Value("string"),
                "text": datasets.Value("string"),
                'label': datasets.ClassLabel(num_classes=len(CLASS_NAMES), names=CLASS_NAMES),
                "phi": datasets.Sequence({
                    "id": datasets.Value("string"),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                    # store the base type string
                    "type": datasets.Value("string"),
                    "tag_text": datasets.Value("string")
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir or "wfudata"

        train_dir = os.path.join(data_dir, "training_wfu")
        test_dir = os.path.join(data_dir, "testing_wfu")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"dirnames": [train_dir]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"dirnames": [test_dir]},
            ),
        ]

    def _get_text(self, root):
        node = root.find("TEXT")
        if node is None or node.text is None:
            return ""
        return node.text

    def _parse_tags(self, root):
        tags_node = root.find("TAGS")
        if tags_node is None:
            return []

        phis = []
        for tag in tags_node:
            try:
                start = int(tag.attrib["start"])
                end = int(tag.attrib["end"])
                tag_type = tag.attrib["TYPE"]
                tag_text = tag.attrib["text"]

                # Skip unknown labels safely
                if tag_type not in BASE_TYPES:
                    continue

                phis.append({
                    "id": tag.attrib.get("id", ""),
                    "start": start,
                    "end": end,
                    "type": tag_type,
                    "tag_text": tag_text
                })

            except KeyError:
                # Skip malformed tag entries
                continue
            except ValueError:
                # Skip invalid integer conversions
                continue

        return phis

    def _parse_xml(self, filepath):
        root = ET.parse(filepath).getroot()
        text = self._get_text(root)
        phis = self._parse_tags(root)
        return text, phis

    def _generate_examples(self, dirnames):
        uid = count(0)

        for dirname in dirnames:
            if not os.path.exists(dirname):
                raise FileNotFoundError(f"Directory not found: {dirname}")

            filenames = sorted(os.listdir(dirname))

            for filename in filenames:
                if not filename.endswith(".xml"):
                    continue

                filepath = os.path.join(dirname, filename)

                try:
                    text, phis = self._parse_xml(filepath)
                    # add a check to make sure the index is correct
                    for phi in phis:
                        # if '&quot' in phi['tag_text']:
                        if '&apos' in phi['tag_text']:
                            print(phi)
                            print('===', text[max(phi['start']-10,0):min(phi['end']+10,len(text))])
                        if text[phi['start']:phi['end']] != phi['tag_text']:
                            print(f"!!! ERROR {phi} {text[phi['start']:phi['end']]}")
                except ET.ParseError:
                    # Skip corrupted XML files
                    continue

                yield next(uid), {
                    "filename": filename,
                    "text": text,
                    "phi": phis,
                }

# some simple sanity check
if (__name__ == '__main__'):
    from datasets import load_dataset

    ds = load_dataset('./wfudata.py', data_dir='../wfudata', trust_remote_code=True)

    print(ds)
    print(ds['train'][2])