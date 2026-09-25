#!/usr/bin/env python3
"""
PromptShield Benchmark Runner
==============================
Evaluates the CURRENT PromptShield PII/entity detection pipeline
against a manually-annotated ground-truth dataset.

IMPORTANT: This script does NOT modify any production code.
It only calls the existing SensitiveDataDetector.detect() method
as-is and measures its output.

Usage
-----
    python evaluation/promptshield_benchmark/run_benchmark.py

All output files are written to:
    evaluation/promptshield_benchmark/

Files produced
--------------
    predictions.json
    metrics.json
    per_entity_metrics.csv
    latency_results.csv
    component_latency.csv
    detector_contribution.csv
    coverage.csv
    error_analysis.md
"""

import csv
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is on sys.path so `backend` imports work
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_DIR = SCRIPT_DIR

# ---------------------------------------------------------------------------
# Silence noisy loggers during benchmarking (keeps output readable)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
for noisy in ("presidio_analyzer", "gliner", "transformers", "torch"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Import the CURRENT PromptShield detector — no modifications
# ---------------------------------------------------------------------------
try:
    from backend.detector import SensitiveDataDetector
    from backend.models import EntityType
except ImportError as e:
    print(f"[FATAL] Cannot import PromptShield backend: {e}")
    print("  Make sure you run this script from the project root or that")
    print("  the project root is on PYTHONPATH.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Label mapping (sourced from label_mapping.json — reproduced here for
# self-contained execution without requiring JSON read at import time)
# ---------------------------------------------------------------------------
GROUND_TRUTH_LABEL_MAP: Dict[str, Optional[str]] = {
    "EMAIL":               "EMAIL",
    "PHONE":               "PHONE",
    "PHONE_NUMBER":        "PHONE",
    "CREDIT_CARD":         "CREDIT_CARD",
    "API_KEY":             "API_KEY",
    "ACCESS_TOKEN":        "ACCESS_TOKEN",
    "PASSWORD":            "PASSWORD",
    "BANK_ACCOUNT":        "BANK_ACCOUNT",
    "IP_ADDRESS":          "IP_ADDRESS",
    "ORDER_ID":            "ORDER_ID",
    "USER_ID":             "USER_ID",
    "USERNAME":            "USER_ID",
    "CUSTOMER_ID":         "CUSTOMER_ID",
    "TICKET_ID":           "TICKET_ID",
    "PERSON":              "PERSON",
    "ORGANIZATION":        "ORGANIZATION",
    "LOCATION":            "LOCATION",
    "ADDRESS":             "ADDRESS",
    "DATE":                "DATE",
    # Unsupported — PromptShield has no matching EntityType
    "PASSPORT_NUMBER":     None,
    "NATIONAL_ID":         None,
    "MAC_ADDRESS":         None,
    "UUID":                None,
    "IFSC":                None,
    "NATIONALITY":         None,
    "MEDICAL_INFORMATION": None,
    "MEDICATION":          None,
    "EMPLOYEE_ID":         None,
}


def normalize_gt_label(raw_label: str) -> Optional[str]:
    """Return canonical EntityType string for a GT label, or None if unsupported."""
    return GROUND_TRUTH_LABEL_MAP.get(raw_label.upper(), None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_python_packages() -> Dict[str, str]:
    """Return a dict of installed package versions relevant to benchmarking."""
    packages = {}
    for pkg in ["gliner", "presidio-analyzer", "spacy", "torch", "transformers"]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", pkg],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    packages[pkg] = line.split(":", 1)[1].strip()
                    break
        except Exception:
            packages[pkg] = "unknown"
    return packages


# ---------------------------------------------------------------------------
# Strict entity-level matching
# ---------------------------------------------------------------------------

def is_match(pred: Dict, gt: Dict) -> bool:
    """
    Return True if prediction strictly matches a ground-truth entity:
      - start offset must match exactly
      - end offset must match exactly
      - canonical label must match exactly
    """
    return (
        pred["start"] == gt["start"]
        and pred["end"] == gt["end"]
        and pred["label"] == gt["label"]
    )


def compute_tp_fp_fn(
    predictions: List[Dict],
    ground_truths: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Compute strict TP / FP / FN for a single prompt.

    Parameters
    ----------
    predictions  : list of {text, start, end, label, confidence, source}
    ground_truths: list of {text, start, end, label}  — canonical labels only

    Returns
    -------
    tps, fps, fns  — each a list of entity dicts
    """
    gt_matched = [False] * len(ground_truths)
    pred_matched = [False] * len(predictions)

    # Find all TP matches
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truths):
            if not gt_matched[gi] and is_match(pred, gt):
                gt_matched[gi] = True
                pred_matched[pi] = True
                break

    tps = [predictions[i] for i, m in enumerate(pred_matched) if m]
    fps = [predictions[i] for i, m in enumerate(pred_matched) if not m]
    fns = [ground_truths[i] for i, m in enumerate(gt_matched) if not m]

    return tps, fps, fns


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark():
    print("=" * 70)
    print("  PROMPTSHIELD PERFORMANCE BASELINE BENCHMARK")
    print("=" * 70)

    gt_path = BENCHMARK_DIR / "ground_truth.json"
    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        sys.exit(1)

    raw_gt = load_ground_truth(gt_path)
    print(f"\n[INFO] Loaded {len(raw_gt)} prompts from ground truth.")

    # ------------------------------------------------------------------
    # Initialise detector — measure model loading time separately
    # ------------------------------------------------------------------
    print("\n[INFO] Initialising SensitiveDataDetector (loading all models)...")
    init_start = time.perf_counter()
    detector = SensitiveDataDetector(use_spacy=True, use_presidio=True, use_gliner=True)
    init_end = time.perf_counter()
    init_time_ms = (init_end - init_start) * 1000.0
    print(f"[INFO] Model initialisation time: {init_time_ms:.1f} ms")

    status = detector.get_detector_status()
    print(f"\n[INFO] Detector status:")
    for comp, info in status.items():
        print(f"         {comp}: {info}")

    # ------------------------------------------------------------------
    # Warm-up runs (excluded from latency measurement)
    # ------------------------------------------------------------------
    WARMUP_RUNS = 5
    warmup_prompts = [item["text"] for item in raw_gt[:WARMUP_RUNS]]
    print(f"\n[INFO] Running {WARMUP_RUNS} warm-up passes...")
    for wp in warmup_prompts:
        _ = detector.detect(wp)
    print("[INFO] Warm-up complete.")

    # ------------------------------------------------------------------
    # Prepare component-level timers (non-invasive wrapper approach)
    # We time each sub-detector individually by calling them directly
    # on a copy of the pipeline. This does NOT change production behavior.
    # ------------------------------------------------------------------
    from backend.regex_detector import RegexDetector
    from backend.presidio_detector import PresidioDetector
    from backend.gliner_detector import GlinerNERDetector
    from backend.spacy_detector import SpacyNERDetector
    from backend.entity_fusion import EntityFusionEngine

    regex_det   = RegexDetector()
    presidio_det = PresidioDetector()
    gliner_det  = GlinerNERDetector()
    spacy_det   = SpacyNERDetector()
    fusion_eng  = EntityFusionEngine()

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    all_predictions: List[Dict] = []
    all_latencies_ms: List[float] = []
    component_latencies: Dict[str, List[float]] = {
        "regex": [], "presidio": [], "gliner": [], "spacy": [], "fusion": []
    }

    # Prepare normalised GT for scoring
    def prepare_gt_entities(raw_entities: List[Dict]) -> List[Dict]:
        """Filter + normalise GT entities to scoreable canonical form."""
        result = []
        for e in raw_entities:
            canon = normalize_gt_label(e["label"])
            if canon is None:
                continue  # unsupported type — skip from scoring
            result.append({
                "text":  e["text"],
                "start": e["start"],
                "end":   e["end"],
                "label": canon,
            })
        return result

    print("\n[INFO] Running evaluation over all prompts...\n")

    all_tp: List[Dict] = []
    all_fp: List[Dict] = []
    all_fn: List[Dict] = []

    per_prompt_results = []

    for item in raw_gt:
        prompt_id   = item["id"]
        prompt_text = item["text"]
        gt_entities_raw = item.get("entities", [])

        # Normalise GT labels
        gt_entities = prepare_gt_entities(gt_entities_raw)

        # ------ Full pipeline latency ------
        t_total_start = time.perf_counter()
        detected = detector.detect(prompt_text)
        t_total_end = time.perf_counter()
        total_latency_ms = (t_total_end - t_total_start) * 1000.0
        all_latencies_ms.append(total_latency_ms)

        # ------ Component-level latency (parallel instrumentation) ------
        t_r0 = time.perf_counter()
        r_regex = regex_det.detect(prompt_text)
        t_r1 = time.perf_counter()
        component_latencies["regex"].append((t_r1 - t_r0) * 1000.0)

        if presidio_det.is_available():
            t_p0 = time.perf_counter()
            _ = presidio_det.detect(prompt_text)
            t_p1 = time.perf_counter()
            component_latencies["presidio"].append((t_p1 - t_p0) * 1000.0)
        else:
            component_latencies["presidio"].append(0.0)

        if gliner_det.is_available():
            t_g0 = time.perf_counter()
            _ = gliner_det.detect(prompt_text)
            t_g1 = time.perf_counter()
            component_latencies["gliner"].append((t_g1 - t_g0) * 1000.0)
        else:
            component_latencies["gliner"].append(0.0)

        if spacy_det.is_available():
            t_s0 = time.perf_counter()
            _ = spacy_det.detect(prompt_text)
            t_s1 = time.perf_counter()
            component_latencies["spacy"].append((t_s1 - t_s0) * 1000.0)
        else:
            component_latencies["spacy"].append(0.0)

        # Fusion latency (on regex results as a proxy for fusion time)
        t_f0 = time.perf_counter()
        _ = fusion_eng.fuse(r_regex, prompt_text)
        t_f1 = time.perf_counter()
        component_latencies["fusion"].append((t_f1 - t_f0) * 1000.0)

        # ------ Convert detected entities to dicts ------
        pred_entities = []
        for ent in detected:
            pred_entities.append({
                "text":       ent.text,
                "start":      ent.start,
                "end":        ent.end,
                "label":      ent.entity_type.value,
                "confidence": round(ent.confidence, 4),
                "source":     ent.source,
                "detector":   ent.detector,
            })

        # ------ TP / FP / FN ------
        tps, fps, fns = compute_tp_fp_fn(pred_entities, gt_entities)
        all_tp.extend([{**e, "prompt_id": prompt_id} for e in tps])
        all_fp.extend([{**e, "prompt_id": prompt_id, "prompt_text": prompt_text,
                         "expected_entities": gt_entities} for e in fps])
        all_fn.extend([{**e, "prompt_id": prompt_id, "prompt_text": prompt_text,
                         "predicted_entities": pred_entities} for e in fns])

        # ------ Save prediction record ------
        all_predictions.append({
            "prompt_id":       prompt_id,
            "text":            prompt_text,
            "detected_entities": pred_entities,
            "latency_ms":      round(total_latency_ms, 4),
        })

        per_prompt_results.append({
            "id": prompt_id,
            "text": prompt_text,
            "gt": gt_entities,
            "pred": pred_entities,
            "tps": tps,
            "fps": fps,
            "fns": fns,
            "latency_ms": total_latency_ms,
        })

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------
    total_tp = len(all_tp)
    total_fp = len(all_fp)
    total_fn = len(all_fn)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1  = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    total_gt_entities = sum(len(r["gt"]) for r in per_prompt_results)
    total_pred_entities = sum(len(r["pred"]) for r in per_prompt_results)

    # ------------------------------------------------------------------
    # Per-entity metrics
    # ------------------------------------------------------------------
    entity_tp   = defaultdict(int)
    entity_fp   = defaultdict(int)
    entity_fn   = defaultdict(int)
    entity_gt   = defaultdict(int)
    entity_pred = defaultdict(int)

    for r in per_prompt_results:
        for e in r["gt"]:
            entity_gt[e["label"]] += 1
        for e in r["pred"]:
            entity_pred[e["label"]] += 1
        for e in r["tps"]:
            entity_tp[e["label"]] += 1
        for e in r["fps"]:
            entity_fp[e["label"]] += 1
        for e in r["fns"]:
            entity_fn[e["label"]] += 1

    all_entity_types = sorted(
        set(list(entity_gt.keys()) + list(entity_fp.keys()))
    )

    per_entity_rows = []
    per_entity_f1s = []
    for etype in all_entity_types:
        tp = entity_tp.get(etype, 0)
        fp = entity_fp.get(etype, 0)
        fn = entity_fn.get(etype, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_entity_f1s.append(f1)
        per_entity_rows.append({
            "entity_type": etype,
            "tp": tp, "fp": fp, "fn": fn,
            "gt_count": entity_gt.get(etype, 0),
            "pred_count": entity_pred.get(etype, 0),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
        })

    macro_f1 = statistics.mean(per_entity_f1s) if per_entity_f1s else 0.0

    # ------------------------------------------------------------------
    # Latency statistics
    # ------------------------------------------------------------------
    def latency_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        srt = sorted(values)
        n = len(srt)
        return {
            "mean":   round(statistics.mean(srt), 4),
            "median": round(statistics.median(srt), 4),
            "min":    round(srt[0], 4),
            "max":    round(srt[-1], 4),
            "std":    round(statistics.stdev(srt) if n > 1 else 0.0, 4),
            "p95":    round(srt[min(int(math.ceil(0.95 * n)) - 1, n - 1)], 4),
            "p99":    round(srt[min(int(math.ceil(0.99 * n)) - 1, n - 1)], 4),
        }

    lat_stats = latency_stats(all_latencies_ms)
    total_detection_time_s = sum(all_latencies_ms) / 1000.0
    throughput = len(raw_gt) / total_detection_time_s if total_detection_time_s > 0 else 0.0

    comp_stats = {}
    for comp, vals in component_latencies.items():
        comp_stats[comp] = latency_stats(vals)

    # ------------------------------------------------------------------
    # Prompt-length categories
    # ------------------------------------------------------------------
    def classify_length(text: str) -> str:
        n = len(text)
        if n < 100:   return "SHORT"
        if n < 300:   return "MEDIUM"
        return "LONG"

    length_latencies: Dict[str, List[float]] = defaultdict(list)
    for r in per_prompt_results:
        cat = classify_length(r["text"])
        length_latencies[cat].append(r["latency_ms"])

    length_rows = []
    for cat in ["SHORT", "MEDIUM", "LONG"]:
        vals = length_latencies.get(cat, [])
        if not vals:
            continue
        srt = sorted(vals)
        n = len(srt)
        length_rows.append({
            "category": cat,
            "num_prompts": n,
            "mean_latency_ms": round(statistics.mean(srt), 4),
            "p95_latency_ms":  round(srt[min(int(math.ceil(0.95 * n)) - 1, n - 1)], 4),
        })

    # ------------------------------------------------------------------
    # Detector contribution
    # ------------------------------------------------------------------
    contrib_rows = []
    for r in per_prompt_results:
        for ent in r["pred"]:
            contrib_rows.append({
                "prompt_id":   r["id"],
                "entity_text": ent["text"],
                "entity_type": ent["label"],
                "source":      ent["source"],
                "detector":    ent["detector"],
                "confidence":  ent["confidence"],
                "is_tp": any(
                    t["text"] == ent["text"] and t["start"] == ent["start"]
                    and t["end"] == ent["end"] for t in r["tps"]
                ),
            })

    # ------------------------------------------------------------------
    # Coverage analysis
    # ------------------------------------------------------------------
    # Count unsupported GT labels separately
    unsupported_counts: Dict[str, int] = defaultdict(int)
    for item in raw_gt:
        for e in item.get("entities", []):
            canon = normalize_gt_label(e["label"])
            if canon is None:
                unsupported_counts[e["label"]] += 1

    coverage_rows = []
    for etype in all_entity_types:
        gt_count   = entity_gt.get(etype, 0)
        detected   = entity_tp.get(etype, 0)
        missed     = entity_fn.get(etype, 0)
        rec        = detected / gt_count if gt_count > 0 else 0.0
        coverage_rows.append({
            "entity_type": etype,
            "gt_count":    gt_count,
            "detected":    detected,
            "missed":      missed,
            "recall":      round(rec, 4),
        })

    # ------------------------------------------------------------------
    # Write predictions.json
    # ------------------------------------------------------------------
    pred_path = BENCHMARK_DIR / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    print(f"[OUT] Written: {pred_path}")

    # ------------------------------------------------------------------
    # Write metrics.json
    # ------------------------------------------------------------------
    metrics = {
        "dataset_size":           len(raw_gt),
        "ground_truth_entities":  total_gt_entities,
        "predicted_entities":     total_pred_entities,
        "true_positives":         total_tp,
        "false_positives":        total_fp,
        "false_negatives":        total_fn,
        "precision":              round(precision, 4),
        "recall":                 round(recall, 4),
        "micro_f1":               round(micro_f1, 4),
        "macro_f1":               round(macro_f1, 4),
        "mean_latency_ms":        lat_stats.get("mean"),
        "median_latency_ms":      lat_stats.get("median"),
        "min_latency_ms":         lat_stats.get("min"),
        "max_latency_ms":         lat_stats.get("max"),
        "std_latency_ms":         lat_stats.get("std"),
        "p95_latency_ms":         lat_stats.get("p95"),
        "p99_latency_ms":         lat_stats.get("p99"),
        "throughput_prompts_per_second": round(throughput, 4),
        "model_init_time_ms":     round(init_time_ms, 4),
        "warmup_runs":            WARMUP_RUNS,
        "measured_runs":          len(raw_gt),
        "detector_status":        status,
        "component_latency_ms":   comp_stats,
        "length_category_latency": length_rows,
        "unsupported_gt_labels":  dict(unsupported_counts),
    }
    metrics_path = BENCHMARK_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[OUT] Written: {metrics_path}")

    # ------------------------------------------------------------------
    # Write per_entity_metrics.csv
    # ------------------------------------------------------------------
    per_entity_path = BENCHMARK_DIR / "per_entity_metrics.csv"
    with open(per_entity_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "entity_type", "tp", "fp", "fn", "gt_count", "pred_count",
            "precision", "recall", "f1"
        ])
        writer.writeheader()
        writer.writerows(per_entity_rows)
    print(f"[OUT] Written: {per_entity_path}")

    # ------------------------------------------------------------------
    # Write latency_results.csv (per prompt)
    # ------------------------------------------------------------------
    latency_path = BENCHMARK_DIR / "latency_results.csv"
    with open(latency_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "prompt_id", "prompt_length_chars", "length_category", "latency_ms"
        ])
        writer.writeheader()
        for r in per_prompt_results:
            text = r["text"]
            writer.writerow({
                "prompt_id":          r["id"],
                "prompt_length_chars": len(text),
                "length_category":    classify_length(text),
                "latency_ms":         round(r["latency_ms"], 4),
            })
    print(f"[OUT] Written: {latency_path}")

    # ------------------------------------------------------------------
    # Write component_latency.csv
    # ------------------------------------------------------------------
    comp_path = BENCHMARK_DIR / "component_latency.csv"
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "component", "available", "mean_ms", "median_ms", "p95_ms", "note"
        ])
        writer.writeheader()
        comp_notes = {
            "regex":    "Always available — deterministic patterns, no ML",
            "presidio": "Available if presidio-analyzer installed",
            "gliner":   "Available if gliner package + model loaded",
            "spacy":    "Available if en_core_web_sm loaded",
            "fusion":   "Measured on regex-only candidates as proxy for fusion overhead",
        }
        comp_avail = {
            "regex":    True,
            "presidio": status.get("presidio", {}).get("active", False),
            "gliner":   status.get("gliner", {}).get("model_loaded", False),
            "spacy":    status.get("spacy", {}).get("model_loaded", False),
            "fusion":   True,
        }
        for comp in ["regex", "presidio", "gliner", "spacy", "fusion"]:
            cs = comp_stats.get(comp, {})
            writer.writerow({
                "component": comp,
                "available": comp_avail.get(comp, "UNKNOWN"),
                "mean_ms":   cs.get("mean", "N/A"),
                "median_ms": cs.get("median", "N/A"),
                "p95_ms":    cs.get("p95", "N/A"),
                "note":      comp_notes.get(comp, ""),
            })
    print(f"[OUT] Written: {comp_path}")

    # ------------------------------------------------------------------
    # Write detector_contribution.csv
    # ------------------------------------------------------------------
    contrib_path = BENCHMARK_DIR / "detector_contribution.csv"
    with open(contrib_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "prompt_id", "entity_text", "entity_type",
            "source", "detector", "confidence", "is_tp"
        ])
        writer.writeheader()
        writer.writerows(contrib_rows)
    print(f"[OUT] Written: {contrib_path}")

    # ------------------------------------------------------------------
    # Write coverage.csv
    # ------------------------------------------------------------------
    coverage_path = BENCHMARK_DIR / "coverage.csv"
    with open(coverage_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "entity_type", "gt_count", "detected", "missed", "recall"
        ])
        writer.writeheader()
        writer.writerows(coverage_rows)
        # Also write unsupported label rows
        for lbl, cnt in sorted(unsupported_counts.items()):
            writer.writerow({
                "entity_type": f"{lbl} [UNSUPPORTED]",
                "gt_count":    cnt,
                "detected":    "N/A",
                "missed":      "N/A",
                "recall":      "N/A",
            })
    print(f"[OUT] Written: {coverage_path}")

    # ------------------------------------------------------------------
    # Write error_analysis.md
    # ------------------------------------------------------------------
    error_path = BENCHMARK_DIR / "error_analysis.md"
    with open(error_path, "w", encoding="utf-8") as f:
        f.write("# PromptShield Baseline — Error Analysis\n\n")
        f.write(f"**Pipeline:** CURRENT PromptShield Detection Pipeline\n")
        f.write(f"**Dataset:** {len(raw_gt)} prompts — `ground_truth.json`\n\n")
        f.write("---\n\n")

        # False Positives
        f.write("## False Positives\n\n")
        f.write(f"Total FPs: **{total_fp}**\n\n")
        if not all_fp:
            f.write("No false positives detected.\n\n")
        else:
            f.write("| Prompt ID | Predicted Text | Predicted Type | Source | Error Classification |\n")
            f.write("|-----------|---------------|----------------|--------|---------------------|\n")
            for fp in all_fp:
                pid = fp["prompt_id"]
                ptext = fp["text"][:40].replace("|", "/")
                ptype = fp["label"]
                src   = fp.get("source", "SOURCE UNCERTAIN")
                # Basic error classification
                gt_ents = fp.get("expected_entities", [])
                classification = _classify_fp(fp, gt_ents)
                f.write(f"| {pid} | `{ptext}` | {ptype} | {src} | {classification} |\n")

        f.write("\n---\n\n")

        # False Negatives
        f.write("## False Negatives\n\n")
        f.write(f"Total FNs: **{total_fn}**\n\n")
        if not all_fn:
            f.write("No false negatives detected.\n\n")
        else:
            f.write("| Prompt ID | Expected Text | Expected Type | System Predicted Instead | Error Classification |\n")
            f.write("|-----------|--------------|----------------|--------------------------|---------------------|\n")
            for fn in all_fn:
                pid = fn["prompt_id"]
                etext = fn["text"][:40].replace("|", "/")
                etype = fn["label"]
                preds = fn.get("predicted_entities", [])
                predicted_instead = _find_partial_match(fn, preds)
                classification = _classify_fn(fn, preds)
                f.write(f"| {pid} | `{etext}` | {etype} | {predicted_instead} | {classification} |\n")

        f.write("\n---\n\n")

        # Summary
        f.write("## Error Summary by Type\n\n")
        fp_by_type: Dict[str, int] = defaultdict(int)
        fn_by_type: Dict[str, int] = defaultdict(int)
        for fp in all_fp:
            fp_by_type[fp["label"]] += 1
        for fn in all_fn:
            fn_by_type[fn["label"]] += 1

        f.write("### Most Over-Detected Types (FP)\n\n")
        f.write("| Entity Type | FP Count |\n|-------------|----------|\n")
        for etype, cnt in sorted(fp_by_type.items(), key=lambda x: -x[1]):
            f.write(f"| {etype} | {cnt} |\n")

        f.write("\n### Most Frequently Missed Types (FN)\n\n")
        f.write("| Entity Type | FN Count |\n|-------------|----------|\n")
        for etype, cnt in sorted(fn_by_type.items(), key=lambda x: -x[1]):
            f.write(f"| {etype} | {cnt} |\n")

        f.write("\n---\n\n")
        f.write("## Unsupported Ground-Truth Entity Types\n\n")
        f.write("The following GT entity types have no matching PromptShield EntityType.\n")
        f.write("They are excluded from TP/FP/FN scoring.\n\n")
        f.write("| GT Label | Count |\n|----------|-------|\n")
        for lbl, cnt in sorted(unsupported_counts.items()):
            f.write(f"| {lbl} | {cnt} |\n")

    print(f"[OUT] Written: {error_path}")

    # ------------------------------------------------------------------
    # Print final report to console
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CURRENT PROMPTSHIELD PERFORMANCE BASELINE")
    print("=" * 70)

    print(f"\nModel configuration:")
    print(f"  GLiNER model:   urchade/gliner_small-v2.1")
    print(f"  GLiNER labels:  person, organization, location, password, username")
    print(f"  GLiNER threshold: 0.50")
    print(f"  Presidio:       {'ACTIVE' if status['presidio']['active'] else 'INACTIVE'}")
    print(f"  spaCy model:    en_core_web_sm (fallback: en_core_web_md / heuristic)")
    print(f"  spaCy:          {'ACTIVE' if status['spacy']['active'] else 'INACTIVE'}")
    print(f"  Regex:          ALWAYS ACTIVE")
    print(f"  Presidio confidence threshold: 0.40 (scores < 0.40 dropped)")
    print(f"  Fusion:         EntityFusionEngine (strict-span dedup + overlap resolution)")

    print(f"\nDataset:")
    print(f"  Number of prompts:            {len(raw_gt)}")
    print(f"  Number of GT entities:        {total_gt_entities}")
    print(f"  Number of predicted entities: {total_pred_entities}")
    print(f"  Unsupported GT entity types:  {dict(unsupported_counts)}")

    print(f"\nPERFORMANCE:")
    print(f"  Precision:  {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"  Recall:     {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"  Micro F1:   {micro_f1:.4f}  ({micro_f1 * 100:.2f}%)")
    print(f"  Macro F1:   {macro_f1:.4f}  ({macro_f1 * 100:.2f}%)")
    print(f"  TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")

    print(f"\nLATENCY (detection only — model loading excluded):")
    print(f"  Mean latency:    {lat_stats['mean']:.2f} ms")
    print(f"  Median latency:  {lat_stats['median']:.2f} ms")
    print(f"  Min latency:     {lat_stats['min']:.2f} ms")
    print(f"  Max latency:     {lat_stats['max']:.2f} ms")
    print(f"  Std dev:         {lat_stats['std']:.2f} ms")
    print(f"  P95 latency:     {lat_stats['p95']:.2f} ms")
    print(f"  P99 latency:     {lat_stats['p99']:.2f} ms")
    print(f"  Model init time: {init_time_ms:.1f} ms  (reported separately)")

    print(f"\nTHROUGHPUT:")
    print(f"  {throughput:.2f} prompts/second  (warm runtime)")

    print(f"\nPER-ENTITY RESULTS:")
    print(f"  {'Entity Type':<18} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*64}")
    for row in per_entity_rows:
        print(
            f"  {row['entity_type']:<18} {row['tp']:>4} {row['fp']:>4} {row['fn']:>4}"
            f" {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}"
        )

    print(f"\nCOMPONENT LATENCY (per prompt, mean | median | P95):")
    for comp in ["regex", "presidio", "gliner", "spacy", "fusion"]:
        cs = comp_stats.get(comp, {})
        avail = comp_avail.get(comp, False)
        if not avail and comp not in ("regex", "fusion"):
            print(f"  {comp:<12}: NOT AVAILABLE / INACTIVE")
        else:
            print(f"  {comp:<12}: {cs.get('mean', 0):.2f} ms | "
                  f"{cs.get('median', 0):.2f} ms | {cs.get('p95', 0):.2f} ms")

    print(f"\nPROMPT LENGTH LATENCY:")
    for lr in length_rows:
        print(f"  {lr['category']:<8} ({lr['num_prompts']:>3} prompts): "
              f"mean={lr['mean_latency_ms']:.2f} ms  P95={lr['p95_latency_ms']:.2f} ms")

    print(f"\nERROR SUMMARY:")
    top_fp_types = sorted(fp_by_type.items(), key=lambda x: -x[1])[:5]
    top_fn_types = sorted(fn_by_type.items(), key=lambda x: -x[1])[:5]
    print(f"  Top false positive types: {top_fp_types}")
    print(f"  Top false negative types: {top_fn_types}")

    print(f"\n{'='*70}")
    print(f"All output files written to: {BENCHMARK_DIR}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Error classification helpers
# ---------------------------------------------------------------------------

def _classify_fp(fp_ent: Dict, gt_entities: List[Dict]) -> str:
    """Classify a FP entity."""
    text  = fp_ent["text"]
    label = fp_ent["label"]
    start = fp_ent["start"]
    end   = fp_ent["end"]

    # Check for partial overlap with a GT entity
    for gt in gt_entities:
        gt_s, gt_e = gt["start"], gt["end"]
        overlap_start = max(start, gt_s)
        overlap_end   = min(end, gt_e)
        if overlap_start < overlap_end:
            if gt["label"] != label:
                return "wrong entity type"
            if start != gt_s or end != gt_e:
                return "partial span / incorrect boundary"

    # Check for same text but different type
    for gt in gt_entities:
        if gt["text"] == text and gt["label"] != label:
            return "wrong entity type"

    return "false positive (hallucinated entity)"


def _classify_fn(fn_ent: Dict, predictions: List[Dict]) -> str:
    """Classify a FN entity."""
    text  = fn_ent["text"]
    label = fn_ent["label"]
    start = fn_ent["start"]
    end   = fn_ent["end"]

    for pred in predictions:
        p_s, p_e = pred["start"], pred["end"]
        # Exact span, wrong label
        if p_s == start and p_e == end and pred["label"] != label:
            return "wrong entity type (span correct)"
        # Partial overlap
        if max(p_s, start) < min(p_e, end):
            return "partial span / incorrect boundary"
        # Same text, wrong label
        if pred["text"] == text and pred["label"] != label:
            return "wrong entity type (text match)"

    return "missed entity (not detected at all)"


def _find_partial_match(fn_ent: Dict, predictions: List[Dict]) -> str:
    """Describe what the system predicted instead (if anything overlaps)."""
    start = fn_ent["start"]
    end   = fn_ent["end"]
    text  = fn_ent["text"]

    candidates = []
    for pred in predictions:
        p_s, p_e = pred["start"], pred["end"]
        overlap = max(0, min(p_e, end) - max(p_s, start))
        if overlap > 0 or pred["text"] == text:
            candidates.append(f"`{pred['text']}` ({pred['label']})")

    if candidates:
        return "; ".join(candidates[:3])
    return "nothing"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_benchmark()
