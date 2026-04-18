import os
import xml.etree.ElementTree as ET
from itertools import count

import datasets

_DESCRIPTION = "WFU hand-tagged de-identification dataset (XML format)"

CLASS_NAMES = [
    'AGE', 'DATE', 'EMAIL', 'HOSPITAL', 'IDNUM', 'INITIALS',
    'IPADDRESS', 'LOCATION', 'NAME', 'OTHER', 'PHONE', 'URL', 'NORMAL'
]

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
                "phi": datasets.Sequence({
                    "id": datasets.Value("string"),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                    "type": datasets.ClassLabel(names=CLASS_NAMES),
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

                # Skip unknown labels safely
                if tag_type not in LABEL2ID:
                    continue

                phis.append({
                    "id": tag.attrib.get("id", ""),
                    "start": start,
                    "end": end,
                    "type": tag_type,
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
                except ET.ParseError:
                    # Skip corrupted XML files
                    continue

                yield next(uid), {
                    "filename": filename,
                    "text": text,
                    "phi": phis,
                }
