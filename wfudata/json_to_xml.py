import os
import json
import html
import argparse
import pandas as pd
from lxml import etree
from sklearn.model_selection import train_test_split
from ner_utils import normalize_text
import shutil

# NER labels to include in the XML conversion
TYPES_TO_INCLUDE = {
    'AGE', 'DATE', 'EMAIL', 'HOSPITAL', 'IDNUM', 'INITIALS',
    'IPADDRESS', 'LOCATION', 'NAME', 'OTHER', 'PHONE', 'URL', 'ADDRESS'
}

def get_mapping_offset(original_idx, raw_text, replace_newlines=True):
    """
    Calculates the new index of a character after the text has been normalized.
    """
    prefix = raw_text[:original_idx]
    clean_prefix = normalize_text(prefix, replace_newlines=replace_newlines)
    return len(clean_prefix)

def trim_span_end(start, end, raw_text):
    """
    If a span accidentally caught the first part of an HTML entity 
    (like '&apos') because the annotator wanted to exclude the "'s", 
    this shrinks the span backward to drop the HTML junk.
    """
    # 1. Get the text currently inside the span
    span_text = raw_text[start:end]
    
    # List of common clinical entities that might be trailing
    fragments = ('&apos', '&amp', '&quot', '&lt', '&gt', '&nbsp')
    
    # 2. Check if the span ends with any of these fragments
    for frag in fragments:
        if span_text.endswith(frag):
            # Shrink the end index by the length of the fragment
            return end - len(frag)
            
    # Optional: What if they accidentally highlighted the whole '&apos;'?
    # e.g., the raw span is "Austin&apos;" but you still just want "Austin"
    full_fragments = [f + ';' for f in fragments]
    for frag in full_fragments:
        if span_text.endswith(frag):
            return end - len(frag)
            
    return end

def prepare_tag(tag_id, start, end, text, etype):
    """Creates an lxml Element for a PHI tag."""
    return etree.Element(
        etype, 
        id='P'+str(tag_id), 
        start=str(start), 
        end=str(end), 
        text=text, 
        TYPE=etype, 
        comment=''
    )

def process_records(records, target_dir):
    """Processes a list of JSON records and saves them as XML files."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=False)
    
    for k, record_str in enumerate(records):
        record = json.loads(record_str)
        raw_signal = record['signal']
        
        # 1. Normalize the full text for the <TEXT> block
        clean_signal = normalize_text(raw_signal)
        
        # 2. Build XML structure
        root = etree.Element('wfu')
        text_node = etree.SubElement(root, 'TEXT')
        text_node.text = etree.CDATA(clean_signal)
        
        tags_node = etree.SubElement(root, 'TAGS')
        elements = []
        
        # 3. Process annotations
        for asset in record['asets']:
            if asset['type'] in TYPES_TO_INCLUDE:
                for annot in asset['annots']:
                    orig_start, orig_end = annot[0], annot[1]
                    
                    # Trim partial HTML entities from the span
                    fixed_end = trim_span_end(orig_start, orig_end, raw_signal)
                    
                    # Map original indices to normalized indices
                    new_start = get_mapping_offset(orig_start, raw_signal)
                    new_end = get_mapping_offset(fixed_end, raw_signal)
                    
                    # Extract the cleaned tag text
                    tag_text = clean_signal[new_start:new_end]
                    
                    elements.append(prepare_tag(len(elements), new_start, new_end, tag_text, asset['type']))

        # 4. Sort tags by start position and append to XML
        for x in sorted(elements, key=lambda x: int(x.attrib['start'])):
            tags_node.append(x)

        # 5. Save to disk
        output_file = os.path.join(target_dir, f'{k:05d}.xml')
        with open(output_file, 'wb') as f:
            f.write(etree.tostring(root, pretty_print=True, encoding='utf-8', xml_declaration=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert WFU JSON data to Cleaned XML")
    parser.add_argument('-i', '--input-csv', default='./wfudata/wfu_annotated.csv', help='Path to raw CSV')
    parser.add_argument('-o', '--output-dir', default='./wfudata', help='Base directory for output XMLs')
    args = parser.parse_args()

    # Load raw data
    df = pd.read_csv(args.input_csv)
    jsons = df['JSON_DATA'].tolist()

    # Perform 80/20 Split
    train_records, test_records = train_test_split(jsons, test_size=0.2, random_state=42)

    print(f"Processing {len(train_records)} training records...")
    process_records(train_records, os.path.join(args.output_dir, 'training_wfu'))

    print(f"Processing {len(test_records)} testing records...")
    process_records(test_records, os.path.join(args.output_dir, 'testing_wfu'))

    print("Conversion complete.")
