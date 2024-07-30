from lxml import etree
import pandas as pd
import json
import argparse
from sklearn.model_selection import train_test_split
import os

# see test_xml.ipynb for this list
types_to_include = set([
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
])

def prepare_tag(id, start, end, text, etype):
    ''' prepare the tag line for the xml file '''
    element = etree.Element(etype, id='P'+str(id), start=str(start), end=str(end), text=text, TYPE=etype, comment='')
    return element

def prettyprint(element, **kwargs):
    xml = etree.tostring(element, pretty_print=True, **kwargs)
    print(xml.decode(), end='')

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', help = 'choose between train or test split', required=True)
    args = parser.parse_args()

    # raw data is in this csv file
    jsons = pd.read_csv('./wfudata/wfu_annotated.csv')['JSON_DATA']

    # get the train or test split
    if args.split == 'train':
        records, _ = train_test_split(jsons, test_size=0.2, random_state=42)
        output_dir = 'wfudata/training_wfu'
    else:
        _, records = train_test_split(jsons, test_size=0.2, random_state=42)
        output_dir = 'wfudata/testing_wfu'
    os.makedirs(output_dir, exist_ok=True)

    # load the json file
    for k, record in enumerate(records):
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

        # take a look
        # prettyprint(root)
        with open(output_file, 'w') as fid:
            xml = etree.tostring(root, pretty_print=True)
            fid.write(xml.decode())
