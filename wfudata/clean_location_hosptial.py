import json
import os
import csv
import shutil
import sys
from datetime import datetime

# --- Configuration ---
CONTEXT = 40
WINDOW = 60
INPUT_CSV = "wfu_data.csv"

# Label mapping for interactive input
VALID_ACTIONS = {
    "l": "LOCATION",
    "h": "HOSPITAL",
    "a": "ADDRESS",  # New class for physical addresses
    "n": "NAME",
    "d": "DELETE",
    "s": "SKIP",
    "m": "MERGE",
    "x": "SPLIT",
}

# The types we want to audit/fix
EDITABLE_TYPES = {"LOCATION", "HOSPITAL"}

# -----------------------------
# Helpers
# -----------------------------

def get_context(text, start, end, window=CONTEXT):
    """Provides a snippet of text around the span for visual verification."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:start] + "[" + text[start:end] + "]" + text[end:right]

def find_nearby_entities(data, start, end, window=WINDOW):
    """Finds other annotations within a certain window to help with merge decisions."""
    nearby = []
    for aset in data.get("asets", []):
        if not aset.get("hasSpan"):
            continue
        etype = aset.get("type")
        for s, e, *_ in aset.get("annots", []):
            # Check if this span is within the 'nearby' window
            if e >= start - window and s <= end + window:
                nearby.append((etype, s, e))
    return nearby

def list_merge_candidates(signal, entities, exclude_span):
    """Prints nearby entities and returns a list for index-based selection."""
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
    """Safely adds a span to the correct label set in the JSON structure."""
    for aset in data["asets"]:
        if aset.get("type") == label:
            aset["annots"].append(span)
            return
    # Create new aset if label doesn't exist
    data["asets"].append({
        "type": label,
        "hasSpan": True,
        "attrs": [],
        "annots": [span],
    })

def remove_specific_spans(data, spans_to_remove):
    """Removes spans from any label category (used during merge/split)."""
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
    """
    Loops through relevant annotations in a single JSON object.
    Returns (modified_bool, updated_json_dict).
    """
    signal = data.get("signal", "")
    modified = False

    # We iterate through a copy of asets because we might modify them in-place
    for aset in data.get("asets", []):
        if aset.get("type") not in EDITABLE_TYPES:
            continue

        original_annots = list(aset.get("annots", []))
        # Reset current list; we will re-add them based on user action
        aset["annots"] = []

        for span in original_annots:
            start, end = span[0], span[1]

            print("\n" + "=" * 80)
            print(f"TEXT_ID: {text_id} | Current label: {aset['type']}")
            print(f"Span: {start}-{end} | Text: '{signal[start:end]}'")
            print("-" * 40)
            print("Context:")
            print(get_context(signal, start, end))

            nearby = find_nearby_entities(data, start, end)
            
            action = input(
                "\nAction: [l]oc | [h]osp | [a]ddress | [n]ame | [m]erge | s[x]plit | [d]elete | [s]kip : "
            ).strip().lower()

            if action not in VALID_ACTIONS:
                # Default to keeping the original if input is garbled
                aset["annots"].append(span)
                continue

            choice = VALID_ACTIONS[action]

            if choice == "SKIP":
                aset["annots"].append(span)

            elif choice == "DELETE":
                modified = True
                continue

            elif choice in ("LOCATION", "HOSPITAL", "NAME", "ADDRESS"):
                modified = True
                add_annotation(data, choice, span)

            elif choice == "MERGE":
                candidates = list_merge_candidates(signal, nearby, (start, end))
                if not candidates:
                    print("No nearby entities found. Skipping merge.")
                    aset["annots"].append(span)
                    continue

                sel = input("Indices to merge (e.g., 0,1): ").strip()
                try:
                    idxs = [int(i) for i in sel.split(",")]
                    selected = [candidates[i] for i in idxs]
                except:
                    aset["annots"].append(span)
                    continue

                selected.append((aset["type"], start, end))
                merged_start = min(s for _, s, _ in selected)
                merged_end = max(e for _, _, e in selected)

                final = input("Final label [l/h/a/n]: ").strip().lower()
                if final not in ("l", "h", "a", "n"):
                    aset["annots"].append(span)
                    continue

                remove_specific_spans(data, {(s, e) for _, s, e in selected})
                add_annotation(data, VALID_ACTIONS[final], [merged_start, merged_end])
                modified = True

            elif choice == "SPLIT":
                text = signal[start:end]
                print(f"\nEntity text: {text}")
                ranges = input("Relative ranges (e.g. 0:10,12:20): ").strip()

                try:
                    remove_specific_spans(data, {(start, end)})
                    for r in ranges.split(","):
                        a, b = map(int, r.split(":"))
                        frag = text[a:b].strip()
                        lbl = input(f"Label for '{frag}' [l/h/a/n]: ").strip().lower()
                        if lbl in ("l", "h", "a", "n"):
                            add_annotation(data, VALID_ACTIONS[lbl], [start + a, start + b])
                    modified = True
                except:
                    print("Split failed. Keeping original.")
                    aset["annots"].append(span)

    return modified, data

# -----------------------------
# Driver
# -----------------------------

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    # 1. Create a backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{INPUT_CSV}.{timestamp}.bak"
    shutil.copy(INPUT_CSV, backup_path)
    print(f"Backup created: {backup_path}")

    # 2. Read existing data
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # 3. Process rows
    modified_count = 0
    try:
        for row in rows:
            text_id = row.get("TEXT_ID", "Unknown")
            try:
                json_content = json.loads(row["JSON_DATA"])
            except Exception as e:
                print(f"Skipping {text_id}: JSON parse error.")
                continue

            is_changed, updated_json = process_json_content(json_content, text_id)
            
            if is_changed:
                row["JSON_DATA"] = json.dumps(updated_json)
                modified_count += 1
                # Optional: Update the UPDATE_DATE field to today
                row["UPDATE_DATE"] = datetime.now().strftime("%Y-%m-%d")

    except KeyboardInterrupt:
        print("\n\nProcess interrupted. Saving progress made so far...")

    # 4. Save back to CSV
    with open(INPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFinished. Updated {modified_count} rows in {INPUT_CSV}.")

if __name__ == "__main__":
    main()
