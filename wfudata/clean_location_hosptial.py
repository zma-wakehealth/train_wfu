import json
import os
import csv
import shutil
import sys
from datetime import datetime

# --- Configuration ---
CONTEXT = 80
WINDOW = 80
INPUT_CSV = "data/wfu_annotated.csv"
OUTPUT_CSV = "data/wfu_annotated_cleaned_location_hospital.csv"  # Now saves to a separate file

csv.field_size_limit(9999999)

VALID_ACTIONS = {
    "l": "LOCATION",
    "h": "HOSPITAL",
    "a": "ADDRESS",
    "n": "NAME",
    "d": "DELETE",
    "s": "SKIP",
    "m": "MERGE",
    "x": "SPLIT",
    "p": "PHONE"
}

EDITABLE_TYPES = {"LOCATION", "HOSPITAL"}

# -----------------------------
# Helpers
# -----------------------------

def get_context(text, start, end, window=CONTEXT):
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:start] + "[" + text[start:end] + "]" + text[end:right]

def find_nearby_entities(data, start, end, window=WINDOW):
    nearby = []
    for aset in data.get("asets", []):
        if not aset.get("hasSpan"):
            continue
        etype = aset.get("type")
        # these are not useful tags
        if etype == 'SEGMENT' or etype == 'lex' or etype == 'zone':
            continue
        for s, e in aset.get("annots", []):
            # Check if this span is within the 'nearby' window
            if e >= start - window and s <= end + window:
                nearby.append((etype, s, e))
    return nearby

def list_merge_candidates(signal, entities, exclude_span):
    indexed = []
    print("\nNearby entities (eligible for merge):")
    for etype, s, e in entities:
        if (s, e) == exclude_span:
            continue
        text = signal[s:e].replace("\n", " ")
        idx = len(indexed)
        print(f"  [{idx}] {etype:<10} {s}-{e}: {text}")
        indexed.append((etype, s, e))
    return indexed

def add_annotation(data, label, span):
    for aset in data["asets"]:
        if aset.get("type") == label:
            aset["annots"].append(span)
            return
    data["asets"].append({
        "type": label,
        "hasSpan": True,
        "attrs": [],
        "annots": [span],
    })

def remove_specific_spans(data, spans_to_remove):
    for aset in data["asets"]:
        if not aset.get("hasSpan"):
            continue
        aset["annots"] = [
            a for a in aset["annots"]
            if (a[0], a[1]) not in spans_to_remove
        ]

# -----------------------------
# Core logic
# -----------------------------

def process_json_content(data, text_id):
    signal = data.get("signal", "")
    modified = False

    for aset in data.get("asets", []):
        if aset.get("type") not in EDITABLE_TYPES:
            continue

        original_annots = list(aset.get("annots", []))
        aset["annots"] = []

        for span in original_annots:
            start, end = span[0], span[1]

            print("\n" + "=" * 80)
            print(f"TEXT_ID: {text_id} | Type: {aset['type']}")
            print(f"Span: {start}-{end} | Text: '{signal[start:end]}'")
            print("-" * 40)
            print("Context:")
            print(get_context(signal, start, end))

            nearby = find_nearby_entities(data, start, end)
            for etype, s, e in nearby:
                print(f"Nearby tag: {etype}, start={s} end={e}")
            
            action = input(
                "\nAction: [l]oc | [h]osp | [a]ddress | [n]ame | [p]hone | [m]erge | s[x]plit | [d]elete | [s]kip : "
            ).strip().lower()

            if action not in VALID_ACTIONS:
                aset["annots"].append(span)
                continue

            choice = VALID_ACTIONS[action]

            if choice == "SKIP":
                aset["annots"].append(span)

            elif choice == "DELETE":
                modified = True
                continue

            elif choice in ("LOCATION", "HOSPITAL", "NAME", "ADDRESS", "PHONE"):
                modified = True
                add_annotation(data, choice, span)

            elif choice == "MERGE":
                candidates = list_merge_candidates(signal, nearby, (start, end))
                if not candidates:
                    print("No nearby entities. Keeping original.")
                    aset["annots"].append(span)
                    continue

                sel = input("Indices to merge (e.g., 0,1): ").strip()
                try:
                    idxs = [int(i) for i in sel.split(",")]
                    selected = [candidates[i] for i in idxs]
                    selected.append((aset["type"], start, end))
                    
                    merged_start = min(s for _, s, _ in selected)
                    merged_end = max(e for _, _, e in selected)

                    final = input("Final label [l/h/a/n/p]: ").strip().lower()
                    if final not in ("l", "h", "a", "n", "p"):
                        raise ValueError("Invalid label")

                    remove_specific_spans(data, {(s, e) for _, s, e in selected})
                    add_annotation(data, VALID_ACTIONS[final], [merged_start, merged_end])
                    modified = True
                except:
                    print("Error in merge selection. Skipping.")
                    aset["annots"].append(span)

            elif choice == "SPLIT":
                text = signal[start:end]
                while True:
                    print(f"\nFull entity text: '{text}'")
                    print("Enter relative ranges (e.g. 0:10,12:20) or 'c' to cancel split:")
                    ranges_input = input("Ranges: ").strip().lower()

                    if ranges_input == 'c':
                        aset["annots"].append(span)
                        break

                    try:
                        parts = []
                        for r in ranges_input.split(","):
                            a, b = map(int, r.split(":"))
                            parts.append((a, b))
                        
                        print("\nPreview of fragments:")
                        for i, (a, b) in enumerate(parts):
                            print(f"  [{i}] '{text[a:b]}'")
                        
                        confirm = input("\nLooks correct? [y]es | [r]etry | [c]ancel: ").strip().lower()
                        
                        if confirm == 'r':
                            continue
                        if confirm == 'c':
                            aset["annots"].append(span)
                            break
                        if confirm == 'y':
                            remove_specific_spans(data, {(start, end)})
                            for a, b in parts:
                                frag = text[a:b].strip()
                                lbl = input(f"Label for '{frag}' [l/h/a/n]: ").strip().lower()
                                if lbl in ("l", "h", "a", "n"):
                                    add_annotation(data, VALID_ACTIONS[lbl], [start + a, start + b])
                            modified = True
                            break
                    except Exception as e:
                        print(f"Error parsing ranges ({e}). Please try again.")

    return modified, data

# -----------------------------
# Driver
# -----------------------------

def get_processed_ids(filepath):
    """Checks the output file to see which TEXT_IDs have already been completed."""
    if not os.path.exists(filepath):
        return set()
    processed = set()
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["TEXT_ID"])
    return processed

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    # 1. Resuming: Find out what's already done
    processed_ids = get_processed_ids(OUTPUT_CSV)
    print(f"Resuming... {len(processed_ids)} records already found in {OUTPUT_CSV}.")

    # 2. Open input and output files
    with open(INPUT_CSV, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as f_out:
        
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)

        # If it's a new file, write the header
        if os.path.getsize(OUTPUT_CSV) == 0:
            writer.writeheader()

        # 3. Stream process row by row
        try:
            for row in reader:
                text_id = row.get("TEXT_ID", "Unknown")
                
                # Skip if already in the output CSV
                if text_id in processed_ids:
                    continue

                try:
                    json_content = json.loads(row["JSON_DATA"])
                except:
                    # If JSON is corrupted, save it as is and move on
                    writer.writerow(row)
                    f_out.flush()
                    continue

                is_changed, updated_json = process_json_content(json_content, text_id)
                
                if is_changed:
                    row["JSON_DATA"] = json.dumps(updated_json)
                    row["UPDATE_DATE"] = datetime.now().strftime("%Y-%m-%d")
                    print(f"✅ Modified {text_id}")
                else:
                    print(f"⏭ No changes to {text_id}")

                # Save immediately to disk
                writer.writerow(row)
                f_out.flush() # Forces the write so data isn't stuck in a buffer

        except KeyboardInterrupt:
            print("\n\nProcess interrupted by user. Your progress has been saved.")

    print(f"\nSession finished. Check {OUTPUT_CSV} for results.")

if __name__ == "__main__":
    main()
