#!/usr/bin/env python3
"""
make_brca_idc_ilc_csv.py

Enhanced script to create TCGA-BRCA IDC/ILC dataset matching M3amba paper specifications.
Includes options for exact paper matching and proper train/val/test splits.
python make_brca_idc_ilc_csv.py   --wsi-dir /vol/research/datasets/pathology/tcga/tcga-brca/WSIs/   --out brca_idc_ilc_pure.csv   --one-slide-per-patient   --primary-only   --require-invasive   --slide-id-regex '.*-01Z-00-.*'   --create-splits   --split-seed 42

"""

import os, re, csv, json, time, math, argparse, sys
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ----------------------------- parsing helpers -----------------------------
UUID_RE = re.compile(r"\b([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})\b")

def patient_from_slide_id(slide_id: str) -> str:
    parts = slide_id.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else slide_id

def extract_dx_num(slide_id: str) -> int:
    m = re.search(r"-DX(\d+)$", slide_id, flags=re.IGNORECASE)
    if not m: return 99
    try: return int(m.group(1))
    except ValueError: return 99

# MODIFIED: This function now parses two different IDs from the filename.
def scan_svs(root: str) -> List[Tuple[str, str, str, str]]:
    items = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".svs"):
                # The full base name (e.g., TCGA-XXX.YYYY) is the new 'slide_id'
                long_id = fn.rsplit(".", 1)[0]
                # The short base name (e.g., TCGA-XXX) is the new 'slide_id1'
                short_id = fn.split(".", 1)[0]
                items.append((os.path.join(dp, fn), fn, long_id, short_id))
    items.sort(key=lambda x: x[1].lower())
    return items

def dedupe_one_slide_per_patient(items: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
    best: Dict[str, Tuple[str, str, str, str, int]] = {}
    for full_path, fn, long_id, short_id in items:
        pid = patient_from_slide_id(short_id)
        dx = extract_dx_num(short_id)
        cur = best.get(pid)
        if not cur or (dx < cur[4]) or (dx == cur[4] and fn < cur[1]):
            best[pid] = (full_path, fn, long_id, short_id, dx)
    return [(fp, fn, lid, sid) for (fp, fn, lid, sid, _) in best.values()]

# ------------------------- diagnosis → label helpers ------------------------
IDC_CODES_BASE: Set[int] = {8500, 8521}
ILC_CODES_BASE: Set[int] = {8520}
AMBIG_CODES_BASE: Set[int] = {8522, 8523, 8524}

AMBIG_TEXT = ["mixed","ductal and lobular","lobular and ductal","tubulolobular","tubulo-lobular","and lobular","and ductal"]

def parse_morph_code(morph: Optional[str]) -> Optional[int]:
    if morph is None: return None
    s = str(morph).strip().lower()
    for suf in ("/3","/2","/1"): s = s.replace(suf,"")
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 4:
        try: return int(digits[:4])
        except ValueError: return None
    return None

def is_invasive_behavior(morph: Optional[str]) -> bool:
    s = str(morph).strip().lower() if morph else ""
    return "/3" in s

def text_to_label(primary: Optional[str]) -> Optional[str]:
    if not primary: return None
    s = primary.strip().lower()
    if any(k in s for k in AMBIG_TEXT): return None
    if "duct" in s: return "IDC"
    if "lobul" in s: return "ILC"
    return None

def choose_diagnosis_pool(diagnoses: List[dict], primary_only: bool) -> List[dict]:
    pool = diagnoses or []
    if not primary_only: return pool
    prim = [d for d in pool if str(d.get("classification_of_tumor") or "").strip().lower() == "primary"]
    if prim: return prim
    with_days = [(d, d.get("days_to_diagnosis")) for d in pool if d.get("days_to_diagnosis") is not None]
    if with_days:
        return [min(with_days, key=lambda x: x[1])[0]]
    return []

def resolve_label_from_diags(
    diagnoses: List[dict],
    ilc_include_mixed: bool,
    primary_only: bool,
    morphology_only: bool,
    require_invasive: bool
) -> Optional[str]:
    pool = choose_diagnosis_pool(diagnoses, primary_only=primary_only)
    if require_invasive:
        pool = [d for d in pool if is_invasive_behavior(d.get("morphology"))]

    IDC_CODES = set(IDC_CODES_BASE)
    ILC_CODES = set(ILC_CODES_BASE)
    AMBIG_CODES = set(AMBIG_CODES_BASE)
    
    if ilc_include_mixed:
        ILC_CODES.update({8522, 8524})
        AMBIG_CODES.discard(8522)
        AMBIG_CODES.discard(8524)

    has_idc = has_ilc = False
    fallback: Optional[str] = None

    for d in pool:
        code = parse_morph_code(d.get("morphology"))
        if code is not None:
            if code in IDC_CODES: has_idc = True
            elif code in ILC_CODES: has_ilc = True
            elif code in AMBIG_CODES: return None
        if not morphology_only and fallback is None:
            t = text_to_label(d.get("primary_diagnosis"))
            if t: fallback = t

    if has_idc and has_ilc: return None
    if has_idc: return "IDC"
    if has_ilc: return "ILC"
    return fallback if not morphology_only else None

# ------------------------------ GDC API calls ------------------------------
def _gdc_post(url: str, payload: dict, timeout: int = 60) -> Optional[dict]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type":"application/json","Accept":"application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

def fetch_cases_by_slide_ids(slide_ids: List[str], project: str, batch_size: int, retries: int, sleep_ms: int, quiet: bool) -> Dict[str, dict]:
    base = "https://api.gdc.cancer.gov/cases"
    fields = ",".join([
        "case_id","submitter_id","project.project_id",
        "samples.portions.slides.submitter_id",
        "files.file_id","files.file_name","files.data_type","files.experimental_strategy",
        "diagnoses.morphology","diagnoses.primary_diagnosis",
        "diagnoses.classification_of_tumor","diagnoses.days_to_diagnosis"
    ])
    out: Dict[str, dict] = {}
    it = range(0, len(slide_ids), batch_size)
    if tqdm and not quiet:
        it = tqdm(it, total=math.ceil(len(slide_ids)/batch_size), desc="GDC cases by slide_id", unit="batch")
    for i in it:
        chunk = slide_ids[i:i+batch_size]
        filters = {"op":"and","content":[
            {"op":"in","content":{"field":"project.project_id","value":[project]}},
            {"op":"in","content":{"field":"samples.portions.slides.submitter_id","value":chunk}},
        ]}
        payload = {"filters":filters, "fields":fields, "format":"JSON", "size": len(chunk)}
        attempt = 0
        while attempt <= retries:
            doc = _gdc_post(base, payload)
            if doc:
                for h in ((doc.get("data") or {}).get("hits") or []):
                    diags = [{
                        "morphology": d.get("morphology"),
                        "primary_diagnosis": d.get("primary_diagnosis"),
                        "classification_of_tumor": d.get("classification_of_tumor"),
                        "days_to_diagnosis": d.get("days_to_diagnosis"),
                    } for d in (h.get("diagnoses") or [])]
                    files = [{
                        "file_name": f.get("file_name"),
                        "file_id": (f.get("file_id") or "").lower(),
                        "data_type": f.get("data_type"),
                        "experimental_strategy": f.get("experimental_strategy"),
                    } for f in (h.get("files") or [])]
                    for samp in (h.get("samples") or []):
                        for por in (samp.get("portions") or []):
                            for sl in (por.get("slides") or []):
                                sid = sl.get("submitter_id")
                                if sid:
                                    out[sid] = {"case_id": h.get("case_id"),
                                                "submitter_id": h.get("submitter_id"),
                                                "diagnoses": diags,
                                                "files": files}
                break
            attempt += 1
            time.sleep(sleep_ms/1000.0)
    return out

# ------------------------- Dataset balancing and splitting ------------------------
def balance_to_paper_counts(rows: List[List[str]], target_idc: int = 749, target_ilc: int = 203, seed: int = 42) -> List[List[str]]:
    """Balance dataset to match paper's IDC/ILC counts"""
    random.seed(seed)
    
    idc_rows = [r for r in rows if r[3] == "IDC"]
    ilc_rows = [r for r in rows if r[3] == "ILC"]
    
    if len(idc_rows) > target_idc:
        idc_rows = random.sample(idc_rows, target_idc)
    if len(ilc_rows) > target_ilc:
        ilc_rows = random.sample(ilc_rows, target_ilc)
    
    return idc_rows + ilc_rows

def create_splits(rows: List[List[str]], train_ratio: float = 0.65, val_ratio: float = 0.10, 
                  test_ratio: float = 0.25, seed: int = 42, stratify: bool = True) -> Tuple[List, List, List]:
    """Create train/val/test splits with optional stratification"""
    random.seed(seed)
    np.random.seed(seed)
    
    if stratify:
        label_groups = defaultdict(list)
        for row in rows:
            label_groups[row[3]].append(row)
        
        train, val, test = [], [], []
        for label, group_rows in label_groups.items():
            random.shuffle(group_rows)
            n = len(group_rows)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train.extend(group_rows[:n_train])
            val.extend(group_rows[n_train:n_train + n_val])
            test.extend(group_rows[n_train + n_val:])
    else:
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train = rows[:n_train]
        val = rows[n_train:n_train + n_val]
        test = rows[n_train + n_val:]
    
    return train, val, test

# ----------------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build TCGA-BRCA IDC/ILC dataset matching M3amba paper.")
    ap.add_argument("--wsi-dir", required=True)
    ap.add_argument("--out", default="brca_idc_ilc.csv")
    ap.add_argument("--project", default="TCGA-BRCA")
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep-ms", type=int, default=250)
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--one-slide-per-patient", action="store_true")
    ap.add_argument("--ilc-include-mixed", action="store_true")
    ap.add_argument("--morphology-only", action="store_true")
    ap.add_argument("--require-invasive", action="store_true")
    ap.add_argument("--slide-id-regex", default=None)
    ap.add_argument("--dump-code-hist", action="store_true")
    ap.add_argument("--match-paper-counts", action="store_true", help="Balance to match paper's 749 IDC / 203 ILC")
    ap.add_argument("--create-splits", action="store_true", help="Create train/val/test splits")
    ap.add_argument("--split-seed", type=int, default=42, help="Random seed for splits")
    ap.add_argument("--recover-ilc", action="store_true", help="Try to recover pure ILC from ambiguous cases")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.wsi_dir):
        print(f"Not a directory: {args.wsi_dir}", file=sys.stderr)
        sys.exit(2)

    if not args.quiet: 
        print(f"Scanning for .svs under: {args.wsi_dir}")
    items = scan_svs(args.wsi_dir)
    if not args.quiet: 
        print(f"Found {len(items)} .svs files.")
    if not items: 
        sys.exit(3)

    if args.slide_id_regex:
        rgx = re.compile(args.slide_id_regex)
        before = len(items)
        items = [it for it in items if rgx.match(it[2])]
        if not args.quiet:
            print(f"Slide-id-regex filter: reduced {before} → {len(items)} slides (pattern: {args.slide_id_regex})")

    if args.one_slide_per_patient:
        before = len(items)
        items = dedupe_one_slide_per_patient(items)
        if not args.quiet:
            print(f"One-slide-per-patient: reduced {before} → {len(items)} slides.")

    # MODIFIED: Use the short_id for querying the GDC API
    gdc_query_ids = sorted({short_id for (_, _, _, short_id) in items})
    sid_to_case = fetch_cases_by_slide_ids(
        gdc_query_ids, project=args.project, batch_size=args.batch_size,
        retries=args.retries, sleep_ms=args.sleep_ms, quiet=args.quiet
    )

    kept_idc = kept_ilc = skipped_amb = skipped_noinfo = 0
    not_found = []
    rows: List[List[str]] = []
    code_hist: Dict[int,int] = {}
    ambiguous_rows = []
    ambiguous_details = []

    it = items if (args.quiet or not tqdm) else tqdm(items, desc="Resolving & labeling", unit="slide")
    # MODIFIED: Unpack four items from the tuple now, including long and short IDs
    for full_path, fn, slide_id, slide_id1 in it:
        # MODIFIED: Use the short_id (now slide_id1) to get case info and patient ID
        info = sid_to_case.get(slide_id1)
        patient_id = patient_from_slide_id(slide_id1)
        
        if not info:
            not_found.append(slide_id1)
            continue

        raw_diags = info.get("diagnoses") or []
        pool = choose_diagnosis_pool(raw_diags, primary_only=args.primary_only)
        if args.require_invasive:
            pool = [d for d in pool if is_invasive_behavior(d.get("morphology"))]

        codes = set()
        morph_codes = []
        primary_diag_texts = []
        for d in pool:
            c = parse_morph_code(d.get("morphology"))
            if c is not None: 
                codes.add(c)
                morph_codes.append(str(c))
            pd = d.get("primary_diagnosis")
            if pd:
                primary_diag_texts.append(pd)
        for c in codes:
            code_hist[c] = code_hist.get(c,0)+1

        label = resolve_label_from_diags(
            raw_diags,
            ilc_include_mixed=args.ilc_include_mixed,
            primary_only=args.primary_only,
            morphology_only=args.morphology_only,
            require_invasive=args.require_invasive
        )
        
        row_content = [slide_id, patient_id, full_path]

        if label is None:
            if raw_diags: 
                skipped_amb += 1
                ambiguous_rows.append(row_content + ["AMBIGUOUS", slide_id1])
                ambiguous_details.append({
                    "slide_id": slide_id,
                    "slide_id1": slide_id1,
                    "patient_id": patient_id,
                    "full_path": full_path,
                    "morph_codes": morph_codes,
                    "primary_diagnoses": primary_diag_texts,
                    "could_be_ilc": any("lobular" in str(pd).lower() for pd in primary_diag_texts if pd and "mixed" not in str(pd).lower() and "ductal" not in str(pd).lower())
                })
            else: 
                skipped_noinfo += 1
            continue

        if label == "IDC": kept_idc += 1
        elif label == "ILC": kept_ilc += 1
        else: continue

        # MODIFIED: Use the new slide_id (long) and slide_id1 (short) variables
        rows.append(row_content + [label, slide_id1])

    recovered_ilc = 0
    if args.recover_ilc and ambiguous_details:
        if not args.quiet:
            print(f"\nAttempting to recover pure ILC cases from {len(ambiguous_details)} ambiguous cases...")
        
        for detail in ambiguous_details:
            primary_texts = detail["primary_diagnoses"]
            morph_codes = detail["morph_codes"]
            is_pure_ilc = False
            
            for text in primary_texts:
                text_lower = str(text).lower()
                if ("lobular" in text_lower and 
                    "mixed" not in text_lower and 
                    "ductal" not in text_lower and
                    "and" not in text_lower):
                    
                    if "8523" in morph_codes or "8520" in morph_codes:
                        is_pure_ilc = True
                        break
                    elif not any(c in ["8500", "8521", "8522", "8524"] for c in morph_codes):
                        is_pure_ilc = True
                        break
            
            if is_pure_ilc:
                # MODIFIED: Use the correct IDs for recovered rows
                rows.append([detail["slide_id"], detail["patient_id"], detail["full_path"], "ILC", detail["slide_id1"]])
                kept_ilc += 1
                recovered_ilc += 1
                skipped_amb -= 1
                
                if not args.quiet and recovered_ilc <= 10:
                    print(f"  Recovered ILC: {detail['slide_id']} - {primary_texts[0] if primary_texts else 'N/A'}")
        
        if not args.quiet:
            print(f"Successfully recovered {recovered_ilc} ILC cases from ambiguous set")
            print(f"New totals: {kept_idc} IDC, {kept_ilc} ILC")

    if args.match_paper_counts:
        if not args.quiet:
            print(f"\nBefore matching paper counts: {kept_idc} IDC, {kept_ilc} ILC")
        rows = balance_to_paper_counts(rows, target_idc=749, target_ilc=203, seed=args.split_seed)
        kept_idc = sum(1 for r in rows if r[3] == "IDC")
        kept_ilc = sum(1 for r in rows if r[3] == "ILC")
        if not args.quiet:
            print(f"After matching paper counts: {kept_idc} IDC, {kept_ilc} ILC")

    header = ["slide_id", "patient_id", "full_path", "label", "slide_id1"]

    if args.create_splits:
        train, val, test = create_splits(rows, seed=args.split_seed)
        base_name = os.path.splitext(args.out)[0]
        
        train_path = f"{base_name}_train.csv"
        with open(train_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(train)
        
        val_path = f"{base_name}_val.csv"
        with open(val_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(val)
        
        test_path = f"{base_name}_test.csv"
        with open(test_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(test)
        
        if not args.quiet:
            print(f"\nSplit statistics:")
            print(f"Train: {len(train)} ({sum(1 for r in train if r[3]=='IDC')} IDC, {sum(1 for r in train if r[3]=='ILC')} ILC)")
            print(f"Val: {len(val)} ({sum(1 for r in val if r[3]=='IDC')} IDC, {sum(1 for r in val if r[3]=='ILC')} ILC)")
            print(f"Test: {len(test)} ({sum(1 for r in test if r[3]=='IDC')} IDC, {sum(1 for r in test if r[3]=='ILC')} ILC)")
            print(f"Wrote splits: {train_path}, {val_path}, {test_path}")

    outpath = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    if ambiguous_details:
        ambig_path = f"{os.path.splitext(args.out)[0]}_ambiguous.csv"
        ambig_detail_path = f"{os.path.splitext(args.out)[0]}_ambiguous_detailed.csv"
        
        with open(ambig_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(ambiguous_rows)
        
        # MODIFIED: Update the detailed ambiguous export to use the correct new IDs
        with open(ambig_detail_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slide_id", "slide_id1", "patient_id", "morph_codes", "primary_diagnoses", "could_be_ilc"])
            for detail in ambiguous_details:
                w.writerow([
                    detail["slide_id"],
                    detail["slide_id1"],
                    detail["patient_id"],
                    ";".join(detail["morph_codes"]),
                    ";".join(detail["primary_diagnoses"]),
                    detail["could_be_ilc"]
                ])
        
        if not args.quiet:
            print(f"Wrote ambiguous cases to: {ambig_path}")
            print(f"Wrote detailed ambiguous info to: {ambig_detail_path}")
            
            potential_ilc = [d for d in ambiguous_details if d["could_be_ilc"]]
            if potential_ilc:
                print(f"\nFound {len(potential_ilc)} potential pure ILC cases in ambiguous set:")
                for p in potential_ilc[:10]:
                    print(f"  {p['slide_id']}: {'; '.join(p['primary_diagnoses'][:2])}")
                    
                print(f"\nTo review and potentially add these ILC cases:")
                print(f"1. Check {ambig_detail_path}")
                print(f"2. Look for 'could_be_ilc' = True")
                print(f"3. Manually verify if they are pure ILC")
                print(f"4. Or re-run with --recover-ilc flag to auto-recover them")
            
            ambig_code_dist = defaultdict(int)
            for detail in ambiguous_details:
                for code in detail["morph_codes"]:
                    ambig_code_dist[code] += 1
            
            if ambig_code_dist:
                print(f"\nMorphology codes in ambiguous cases:")
                for code, count in sorted(ambig_code_dist.items()):
                    code_name = ""
                    if code == "8522": code_name = " (mixed)"
                    elif code == "8523": code_name = " (ambiguous, sometimes pure ILC)"
                    elif code == "8524": code_name = " (mixed)"
                    elif code == "8520": code_name = " (ILC)"
                    elif code == "8500": code_name = " (IDC)"
                    print(f"  {code}{code_name}: {count} cases")

    if not args.quiet:
        print(f"\nWrote: {outpath}")
        print(f"Kept (IDC): {kept_idc}")
        print(f"Kept (ILC): {kept_ilc}")
        print(f"Total kept: {kept_idc + kept_ilc}")
        print(f"Skipped (ambiguous): {skipped_amb}")
        print(f"Skipped (no diagnosis info): {skipped_noinfo}")
        print(f"Slides not found in /cases by slide_id: {len(not_found)}")
        if not_found[:10]:
            print("Examples (first 10):")
            for s in not_found[:10]: 
                print("  -", s)
        print(f"\nOptions used:")
        print(f"  primary-only={args.primary_only}")
        print(f"  one-slide-per-patient={args.one_slide_per_patient}")
        print(f"  ilc-include-mixed={args.ilc_include_mixed}")
        print(f"  morphology-only={args.morphology_only}")
        print(f"  require-invasive={args.require_invasive}")
        print(f"  slide-id-regex={args.slide_id_regex}")
        print(f"  match-paper-counts={args.match_paper_counts}")
        print(f"  create-splits={args.create_splits}")

        if args.dump_code_hist and code_hist:
            print("\nICD-O-3 morphology code histogram (unique codes per slide, after filters):")
            for code in sorted(code_hist):
                print(f"  {code}: {code_hist[code]}")
        
        if not args.match_paper_counts and (kept_idc != 749 or kept_ilc != 203):
            print("\n" + "="*60)
            print("RECOMMENDATION: To match M3amba paper (749 IDC / 203 ILC):")
            print("Add --match-paper-counts flag to balance the dataset")
            print("="*60)

if __name__ == "__main__":
    main()