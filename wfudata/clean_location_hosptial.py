import json
import os
import csv
import sys
from datetime import datetime

# --- Configuration ---
CONTEXT = 100
WINDOW = 100
INPUT_CSV = "data/wfu_annotated_cleaned_location_hospital_resampled.csv"
OUTPUT_CSV = "data/wfu_annotated_cleaned_location_hospital_v2.csv"

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

EDITABLE_TYPES = {"LOCATION", "HOSPITAL", "ADDRESS"}

# -----------------------------
# Helpers
# -----------------------------

def get_context(text, start, end, window=CONTEXT):
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:start] + "[" + text[start:end] + "]" + text[end:right]

def flatten_spans(data):
    spans = []
    for aset in data.get("asets", []):
        if not aset.get("hasSpan"):
            continue
        etype = aset["type"]
        if etype in ('SEGMENT', 'lex', 'zone'):
            continue
        for annot in aset.get("annots", []):
            spans.append({
                "type": etype,
                "start": annot[0],
                "end": annot[1]
            })
    return spans

def rebuild_asets(spans):
    aset_dict = {}
    for s in spans:
        etype = s["type"]
        if etype not in aset_dict:
            aset_dict[etype] = {
                "type": etype,
                "hasSpan": True,
                "attrs": [],
                "annots": []
            }
        aset_dict[etype]["annots"].append([s["start"], s["end"]])
    return list(aset_dict.values())

def find_nearby(spans, start, end, window=WINDOW):
    nearby = []
    for s in spans:
        if s["end"] >= start - window and s["start"] <= end + window:
            nearby.append(s)
    return nearby

def get_overlaps(spans, new_start, new_end, exclude):
    overlaps = []
    for s in spans:
        if (s["start"], s["end"]) == exclude:
            continue
        if max(s["start"], new_start) < min(s["end"], new_end):
            overlaps.append(s)
    return overlaps

# -----------------------------
# Core logic (FLATTEN VERSION)
# -----------------------------

def process_json_content(data, text_id):
    signal = data.get("signal", "")
    spans = flatten_spans(data)

    # sort helps mental consistency
    spans.sort(key=lambda x: x["start"])

    new_spans = []
    used_spans = set()
    modified = False

    for i, span in enumerate(spans):
        start, end = span["start"], span["end"]
        etype = span["type"]

        if (start, end) in used_spans:
            continue

        if etype not in EDITABLE_TYPES:
            new_spans.append(span)
            continue

        print("\n" + "=" * 80)
        print(f"TEXT_ID: {text_id} | Type: {etype}")
        print(f"Span: {start}-{end} | Text: '{signal[start:end]}'")
        print("-" * 40)
        print(get_context(signal, start, end))

        action = input(
            "\nAction: [l]oc | [h]osp | [a]ddress | [n]ame | [p]hone | [m]erge | s[x]plit | [d]elete | [s]kip : "
        ).strip().lower()

        if action not in VALID_ACTIONS:
            new_spans.append(span)
            continue

        choice = VALID_ACTIONS[action]

        # ------------------------
        # SKIP
        # ------------------------
        if choice == "SKIP":
            new_spans.append(span)

        # ------------------------
        # DELETE
        # ------------------------
        elif choice == "DELETE":
            modified = True
            used_spans.add((start, end))
            continue

        # ------------------------
        # RELABEL
        # ------------------------
        elif choice in ("LOCATION", "HOSPITAL", "NAME", "ADDRESS", "PHONE"):
            new_spans.append({
                "type": choice,
                "start": start,
                "end": end
            })
            used_spans.add((start, end))
            modified = True

        # ------------------------
        # MERGE (range-based)
        # ------------------------
        elif choice == "MERGE":
            try:
                while True:
                    rel_input = input("Relative range (e.g., -10:30): ").strip()
                    try:
                        rel_s, rel_e = map(int, rel_input.split(":"))
                        new_start = max(0, start + rel_s)
                        new_end = min(len(signal), start + rel_e)

                        print(f"Preview: '{signal[new_start:new_end]}'")
                        confirm = input("Correct or not? [y/n]: ").strip().lower()
                        if confirm == 'y':
                            break
                    except Exception as e:
                        print(f"Merge error: {e}")
                        continue

                overlaps = get_overlaps(spans, new_start, new_end, (start, end))

                if overlaps:
                    print("\nOverlaps:")
                    for o in overlaps:
                        print(f"{o['type']} {o['start']}-{o['end']} '{signal[o['start']:o['end']]}'")

                    confirm = input("Delete overlaps? [y/n]: ").strip().lower()
                    if confirm != "y":
                        new_spans.append(span)
                        continue

                    for o in overlaps:
                        used_spans.add((o["start"], o["end"]))

                final = input("Final label [l/h/a/n/p]: ").strip().lower()
                if final not in ("l", "h", "a", "n", "p"):
                    raise ValueError

                new_spans.append({
                    "type": VALID_ACTIONS[final],
                    "start": new_start,
                    "end": new_end
                })

                used_spans.add((start, end))
                used_spans.add((new_start, new_end))
                modified = True

            except:
                print("Merge error.")
                new_spans.append(span)

        # ------------------------
        # SPLIT
        # ------------------------
        elif choice == "SPLIT":
            text = signal[start:end]

            while True:
                ranges_input = input("Ranges (e.g. 0:5,6:10) or 'c': ").strip()

                if ranges_input == 'c':
                    new_spans.append(span)
                    break

                try:
                    parts = []
                    for r in ranges_input.split(","):
                        a, b = map(int, r.split(":"))
                        a = max(0, min(len(text), a))
                        b = max(0, min(len(text), b))
                        parts.append((a, b))

                    for i, (a, b) in enumerate(parts):
                        print(f"[{i}] '{text[a:b]}'")

                    confirm = input("[y]es/[r]etry/[c]ancel: ").strip()

                    if confirm == 'r':
                        continue
                    if confirm == 'c':
                        new_spans.append(span)
                        break

                    if confirm == 'y':
                        for a, b in parts:
                            lbl = input("Label [l/h/a/n/p]: ").strip().lower()
                            if lbl in ("l", "h", "a", "n", "p"):
                                new_spans.append({
                                    "type": VALID_ACTIONS[lbl],
                                    "start": start + a,
                                    "end": start + b
                                })

                        used_spans.add((start, end))
                        modified = True
                        break

                except Exception as e:
                    print(f"Split error: {e}")

    # rebuild clean structure
    data["asets"] = rebuild_asets(new_spans)

    return modified, data

# -----------------------------
# Driver (same)
# -----------------------------

def get_processed_ids(filepath):
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
        print("Missing input.")
        return

    processed_ids = get_processed_ids(OUTPUT_CSV)

    with open(INPUT_CSV, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)

        if os.path.getsize(OUTPUT_CSV) == 0:
            writer.writeheader()

        try:
            for row in reader:
                tid = row["TEXT_ID"]
                if tid in processed_ids:
                    continue

                try:
                    data = json.loads(row["JSON_DATA"])
                except:
                    writer.writerow(row)
                    continue

                changed, updated = process_json_content(data, tid)

                if changed:
                    row["JSON_DATA"] = json.dumps(updated)
                    row["UPDATE_DATE"] = datetime.now().strftime("%Y-%m-%d")

                writer.writerow(row)
                f_out.flush()

        except KeyboardInterrupt:
            print("Interrupted.")

    print("Done.")

if __name__ == "__main__":
    main()
