import datasets
import os
import xml.etree.ElementTree as et
from itertools import count
import numpy as np

_DESCRIPTION = """
This is from the wfu hand tag data
"""

_class_names = [
    'AGE',
    'DATE',
    'EMAIL',
    'HOSPITAL',
    'IDNUM',
    'INITIALS',
    'IPADDRESS',
    'LOCATION',
    'NAME',
    'OTHER',
    'PHONE',
    'URL',
    'NORMAL'
]

id2label, label2id = {}, {}
for id, label in enumerate(_class_names):
    id2label[id] = label
    label2id[label] = id

normal_l = label2id['NORMAL']
useful_class_names = _class_names[:]
useful_class_names.remove('NORMAL')

class i2b2_2014_Dataset(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name='deid')
    ]

    DEFAULT_CONFIG_NAME = "deid"

    def _info(self):
        return datasets.DatasetInfo(
            description = _DESCRIPTION,
            features = datasets.Features({
                'filename': datasets.Value('string'),
                'text': datasets.Value('string'),
                'label': datasets.ClassLabel(num_classes=len(_class_names), names=_class_names),
                'phi': datasets.Sequence({
                    'ids': datasets.Value('string'),
                    'offsets': datasets.Sequence(datasets.Value('int32'), length=2), # the sequence seems just bracket, so do not do sequence([])
                    'types': datasets.Value('string')
                })
            }))

    def _split_generators(self, dl_manager):
        path = os.path.join(os.getcwd(), 'wfudata/')
        train_dirnames = [os.path.join(path, 'training_wfu')]
        test_dirnames = [os.path.join(path, 'testing_wfu')]

        # remember the gen_kwargs needs to match the one in _generate_examples
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'dirnames': train_dirnames}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'dirnames': test_dirnames}),
        ]
    
    def _generate_examples(self, dirnames):
        uid = count(0)
        for dirname in dirnames:
            filenames = sorted(os.listdir(dirname))
            for filename in filenames:
                xmldoc = et.parse(os.path.join(dirname,filename)).getroot()
                text = xmldoc.findall("TEXT")[0].text
                phis = []
                for line in xmldoc.findall("TAGS")[0]:
                    phi = {
                        'ids': line.attrib['id'],
                        'offsets': [np.int32(line.attrib['start']), np.int32(line.attrib['end'])],
                        'types': line.attrib['TYPE']
                    }
                    phis.append(phi)
                yield next(uid), {
                    'filename': filename,
                    'text': text,
                    'phi': phis
                }