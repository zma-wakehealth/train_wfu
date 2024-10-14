from lxml import etree
import pandas as pd
import json
import argparse
from sklearn.model_selection import train_test_split
import os
import numpy as np
import shutil

# see test_xml.ipynb for this list
# types_to_include = {
#  'AGE':0,
#  'DATE':1,
#  'EMAIL':2,
#  'HOSPITAL':3,
#  'IDNUM':4,
#  'INITIALS':5,
#  'IPADDRESS':6,
#  'LOCATION':7,
#  'NAME':8,
#  'OTHER':9,
#  'PHONE':10,
#  'URL':11
# }

types_to_include = {
 'AGE':0,
 'DATE':1,
 'EMAIL':2,
 'HOSPITAL':3,
 'IDNUM':4,
 'INITIALS':5,
 'LOCATION':6,
 'NAME':7,
 'OTHER':8,
 'PHONE':9,
 'URL':10
}

def prepare_tag(id, start, end, text, etype):
    ''' prepare the tag line for the xml file '''
    element = etree.Element(etype, id='P'+str(id), start=str(start), end=str(end), text=text, TYPE=etype, comment='')
    return element

def prettyprint(element, **kwargs):
    xml = etree.tostring(element, pretty_print=True, **kwargs)
    print(xml.decode(), end='')

if (__name__ == '__main__'):

    n_splits = 5

    # raw data is in this csv file
    jsons = pd.read_csv('./wfudata/wfu_annotated.csv', usecols=['JSON_DATA', 'TEXT_CLASS'])

    # # this bits show we need to get rid of IPADDRESS
    # for seed in [42]:
    #     jsons = jsons.sample(frac=1.0, replace=False, random_state=seed, ignore_index=True)
    #     intervals = [int(np.round(x)) for x in np.linspace(0, len(jsons), n_splits+1)]
    #     for cv in range(n_splits):
    #         counts = np.zeros(len(types_to_include), dtype=np.int32)
    #         i, j = intervals[cv], intervals[cv+1]
    #         records = pd.concat([jsons[:i], jsons[j:]], axis=0).copy()
    #         for record in records['JSON_DATA']:
    #             record = json.loads(record)
    #             for x in record['asets']:
    #                 if x['type'] in types_to_include and len(x['annots']) > 0:
    #                     counts[types_to_include[x['type']]] += 1
    #         print(counts)

    # get the intervals for cv
    jsons = jsons.sample(frac=1.0, replace=False, random_state=42, ignore_index=True)
    intervals = [int(np.round(x)) for x in np.linspace(0, len(jsons), n_splits+1)]
    for cv in range(n_splits):
        i, j = intervals[cv], intervals[cv+1]

        # do train test split
        for split in ['train', 'test']:
            if split == 'train':
                records = pd.concat([jsons[:i], jsons[j:]], axis=0).copy()
                output_dir = f'wfudata_fold_{cv}/training_wfu'

                # do some counting do double check there's no empty class
                counts = np.zeros(len(types_to_include), dtype=np.int32)
                for record in records['JSON_DATA']:
                    record = json.loads(record)
                    for x in record['asets']:
                        if x['type'] in types_to_include and len(x['annots']) > 0:
                            counts[types_to_include[x['type']]] += 1
                print(f'cv={cv}, counts={counts}')

            else:
                records = jsons[i:j].copy()
                output_dir = f'wfudata_fold_{cv}/testing_wfu'

            # need to replace the path in the .py file for the dataset to load
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            with open('wfudata_fold.py', 'r') as fid:
                filedata = fid.read()
            filedata = filedata.replace('wfudata', f'wfudata_fold_{cv}')
            with open(f'wfudata_fold_{cv}/wfudata_fold_{cv}.py', 'w') as fid:
                fid.write(filedata)

            # load the json file
            # for k, record in enumerate(records):
            for k in range(len(records)):
                record = records.iloc[k]['JSON_DATA']
                output_file = os.path.join(output_dir, '{:05d}.xml'.format(k))
                record = json.loads(record)
                root = etree.Element('wfu')

                # prepare the TEXT tag
                text = etree.SubElement(root, 'TEXT')
                text.text = etree.CDATA(record['signal'])

                # loop over each type for the span
                tags = etree.SubElement(root, 'TAGS')
                elements = []
                for x in record['asets']:
                    if x['type'] in types_to_include:
                        for annot in x['annots']:
                            start = annot[0]
                            end = annot[1]
                            elements.append(prepare_tag(len(elements), start, end, record['signal'][start:end], x['type']))
                # need to sort to make sure they are in orders
                for x in sorted(elements, key=lambda x: int(x.attrib['start'])):
                    tags.append(x)

                # record the text_class
                text_class = etree.SubElement(root, 'TEXT_CLASS')
                text_class.text = records.iloc[k]['TEXT_CLASS']

                # take a look
                # prettyprint(root)
                with open(output_file, 'w') as fid:
                    xml = etree.tostring(root, pretty_print=True)
                    fid.write(xml.decode())
