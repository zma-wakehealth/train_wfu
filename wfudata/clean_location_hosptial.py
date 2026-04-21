import json
import os
import shutil
import sys

CONTEXT = 40
WINDOW = 60

VALID_ACTIONS = {
    "l": "LOCATION",
    "h": "HOSPITAL",
    "n": "NAME",
    "d": "DELETE",
    "s": "SKIP",
    "m": "MERGE",
    "x": "SPLIT",
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
        for s, e, *_ in aset.get("annots", []):
            if e >= start - window and s <= end + window:
                nearby.append((etype, s, e))
    return nearby

def print_nearby_entities(signal, entities, exclude_span=None):
    seen = set()
    for etype, s, e in sorted(entities, key=lambda x: (x[1], x[2])):
        if exclude_span and (s, e) == exclude_span:
            continue
        key = (etype, s, e)
        if key in seen:
            continue
        text = signal[s:e].replace("\n", " ")
        print(f"  {etype:<10}: {text} ({s}-{e})")
        seen.add(key)

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

def merge_spans(spans):
    return min(s for _, s, _ in spans), max(e for _, _, e in spans)

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
# Core processing
# -----------------------------

def process_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

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
            print(f"File: {path}")
            print(f"Current label: {aset['type']}")
            print(f"Span: {start}-{end}")
            print("Context:")
            print(get_context(signal, start, end))

            nearby = find_nearby_entities(data, start, end)
            print("\nNearby tagged entities:")
            if nearby:
                print_nearby_entities(signal, nearby, (start, end))
            else:
                print("  (none)")

            action = input(
                "\nAction: [l]ocation | [h]ospital | [n]ame | "
                "[m]erge | s[x]plit | [d]elete | [s]kip : "
            ).strip().lower()

            if action not in VALID_ACTIONS:
                aset["annots"].append(span)
                continue

            choice = VALID_ACTIONS[action]

            # ---- BASIC ----
            if choice == "SKIP":
                aset["annots"].append(span)

            elif choice == "DELETE":
                modified = True
                continue

            elif choice in ("LOCATION", "HOSPITAL", "NAME"):
                modified = True
                add_annotation(data, choice, span)

            # ---- MERGE ----
            elif choice == "MERGE":
                candidates = list_merge_candidates(signal, nearby, (start, end))
                if not candidates:
                    aset["annots"].append(span)
                    continue

                sel = input("Indices to merge (comma-separated): ").strip()
                try:
                    idxs = [int(i) for i in sel.split(",")]
                    selected = [candidates[i] for i in idxs]
                except Exception:
                    aset["annots"].append(span)
                    continue

                selected.append((aset["type"], start, end))
                merged_span = merge_spans(selected)

                final = input("Final label [l/h/n]: ").strip().lower()
                if final not in ("l", "h", "n"):
                    aset["annots"].append(span)
                    continue

                remove_specific_spans(
                    data,
                    {(s, e) for _, s, e in selected}
                )
                add_annotation(data, VALID_ACTIONS[final], list(merged_span))
                modified = True

            # ---- SPLIT ----
            elif choice == "SPLIT":
                text = signal[start:end]
                print("\nFull entity text:\n" + text.replace("\n", " "))

                print("\nEnter relative ranges (e.g. 0:34,36:72)")
                ranges = input("Ranges: ").strip()

                try:
                    parts = []
                    for r in ranges.split(","):
                        a, b = r.split(":")
                        parts.append((int(a), int(b)))
                except Exception:
                    aset["annots"].append(span)
                    continue

                remove_specific_spans(data, {(start, end)})

                for a, b in parts:
                    frag = text[a:b].strip()
                    if not frag:
                        continue
                    lbl = input(f"Label for '{frag}' [l/h/n]: ").strip().lower()
                    if lbl not in ("l", "h", "n"):
                        continue
                    add_annotation(
                        data,
                        VALID_ACTIONS[lbl],
                        [start + a, start + b]
                    )
                modified = True

        # preserve remaining annotations
        # (only LOCATION/HOSPITAL ones were filtered)
        aset["annots"].extend(aset["annots"])

    return modified, data

# -----------------------------
# Driver
# -----------------------------

def main(input_dir):
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))

    for fname in files:
        path = os.path.join(input_dir, fname)
        backup = path + ".bak"

        if not os.path.exists(backup):
            shutil.copy(path, backup)

        modified, data = process_file(path)

        if modified:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"✅ Saved changes to {fname}")
        else:
            print(f"⏭ No changes to {fname}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_loc_hosp_interactive.py <json_dir>")
        sys.exit(1)
    main(sys.argv[1])
