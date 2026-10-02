"""Evaluation metrics: MetricTracker, METEOR, ROUGE, CIDEr, EM, VSI-Bench."""

import json
import os
import re
import string

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score


def _ensure_nltk_data():
    """Fetch the NLTK data METEOR needs before any generation runs, instead of
    failing in the metrics pass after a full benchmark."""
    for res, name in (("tokenizers/punkt_tab", "punkt_tab"), ("corpora/wordnet", "wordnet")):
        found = False
        for candidate in (res, res + ".zip"):
            try:
                nltk.data.find(candidate)
                found = True
                break
            except LookupError:
                pass
        if not found:
            if not nltk.download(name, quiet=True):
                raise RuntimeError(f"NLTK data '{name}' is missing and could not be downloaded. "
                                   f"Run: python -m nltk.downloader {name}")


_ensure_nltk_data()
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer

from .bbox import (
    compute_3d_iou,
    compute_3d_obb_iou,
    compute_multi3drefer_f1,
    pano_bbox_to_world_bbox,
    parse_3d_bbox,
    parse_3d_obb,
    parse_multi_3d_bbox,
    parse_pano_3d_bbox,
)


# Datasets where each sample's answer is a single 3D bounding box. Used by
# MetricTracker to decide whether to compute IoU/Acc grounding metrics.
GROUNDING_DATASETS = frozenset({
    "scanrefer", "scanrefer_train", "scanrefer_val",
    "nr3d_train", "sr3d_train",
    "multi3drefer", "multi3drefer_train",
})

# Subset of GROUNDING_DATASETS that emit 9-DoF OBBs (cx,cy,cz,dx,dy,dz,rx,ry,rz)
# instead of 6-DoF AABBs. MetricTracker dispatches to OBB-aware parse + IoU
# when the dataset name is in here, otherwise rotation is silently dropped
# and IoU collapses to AABB-of-corners (incorrect for non-axis-aligned boxes).
OBB_GROUNDING_DATASETS = frozenset()


def _is_grounding_dataset(name):
    return name in GROUNDING_DATASETS


def _is_obb_grounding_dataset(name):
    return name in OBB_GROUNDING_DATASETS


_COT_DIST_RE = re.compile(r"dist\s*=\s*(-?\d+\.?\d*)")


def _parse_regression_value(text: str) -> float:
    """Parse a regression value from plain float or CoT format (``dist=X.X``)."""
    text = text.strip()
    # Try plain float first (fast path).
    try:
        return float(text)
    except ValueError:
        pass
    # CoT format: extract the last ``dist=...`` value.
    m = _COT_DIST_RE.findall(text)
    if m:
        return float(m[-1])
    raise ValueError(f"Cannot parse regression value from: {text!r}")


class MetricTracker:
    def __init__(self, benchmarking=True, exp_name="test", dataset_name=None,
                 pano_grounding_format=False):
        self.benchmarking = benchmarking
        self.exp_name = exp_name
        self.dataset_name = dataset_name  # "vsi_bench", "sp_bench", "sqa3d", or None
        # When True, parse grounding predictions and GT as pano-format JSON
        # `[{"bbox_3d": [u, v, depth, sx, sy, sz], "label": "..."}]` and
        # unproject back to world-centered meters before computing IoU /
        # center distance. Keeps metrics numerically comparable to the
        # non-pano baseline (compute_3d_iou is called on the unprojected box).
        self.pano_grounding_format = bool(pano_grounding_format)
        self.metrics_acc = {
            "METEOR": [], "ROUGE1": [], "ROUGE2": [],
            "ROUGEL": [], "EM@1": [], "EM@R1": []
        }
        # Regression metrics (populated only when both pred and gt parse as floats)
        self._reg_preds = []
        self._reg_gts = []
        self.val_losses = []  # teacher-forced CE loss per batch (optional)
        # Grounding metrics (populated when both pred and gt parse as 3D bboxes)
        self._grounding_ious = []
        self._grounding_center_dists = []
        self._grounding_total = 0
        # Multi-object grounding metrics (Multi3DRefer: F1@IoU)
        self._multi_grounding_f1_25 = []
        self._multi_grounding_f1_50 = []
        self._multi_grounding_total = 0
        self.gts_cider = {}
        self.preds_cider = {}
        self.results_log = []
        self.per_type_scores = {}  # {question_type: [score, ...]}

        if self.benchmarking:
            os.makedirs(self.exp_name, exist_ok=True)
            # Create a dedicated folder for step-by-step metrics
            self.inter_dir = os.path.join(self.exp_name, "intermediate_metrics")
            os.makedirs(self.inter_dir, exist_ok=True)

    def update(self, count, question, prediction, ground_truths, scene_id=None, question_type=None, all_predictions=None, n_source_images=None):
        # Normalize everything upfront for fairness across all metrics
        norm_prediction = normalize(prediction)
        norm_ground_truths = [normalize(gt) for gt in ground_truths]

        # Calculate generic NLP scores
        scores = run_evaluation_metrics(norm_prediction, norm_ground_truths)
        self._accumulate_text_scores(scores)
        self._accumulate_regression(prediction, ground_truths, question_type)
        this_iou = self._accumulate_grounding(prediction, ground_truths)
        self._accumulate_multi_grounding(prediction, ground_truths)
        self._accumulate_cider(count, norm_prediction, norm_ground_truths)

        official_score = self._compute_official_score(
            prediction, ground_truths, question_type, scores, this_iou
        )
        self._track_per_type(question_type, official_score, scores)
        self._append_log(
            scene_id, question, question_type, prediction, ground_truths,
            scores, n_source_images, all_predictions,
        )
        return scores["METEOR"]

    # --- update() helpers (one concern each; behaviour is identical to the
    # former single method, split for readability and per-concern testing) ---

    def _accumulate_text_scores(self, scores):
        """Append the generic NLP scores (METEOR/ROUGE/EM/EM@R1)."""
        self.metrics_acc["METEOR"].append(scores["METEOR"])
        self.metrics_acc["ROUGE1"].append(scores["rouge1"])
        self.metrics_acc["ROUGE2"].append(scores["rouge2"])
        self.metrics_acc["ROUGEL"].append(scores["rougeL"])
        self.metrics_acc["EM@1"].append(scores["EM"])
        self.metrics_acc["EM@R1"].append(scores["EM@R1"])

    def _accumulate_regression(self, prediction, ground_truths, question_type):
        """Accumulate (pred, gt) float pairs for the regression stats.

        Excludes counting tasks -- they parse as floats but aren't distances,
        so distance-tolerance metrics (within_1cm etc.) are meaningless for them.
        Matches the whole counting family (legacy VSI ``object_counting`` plus the
        curriculum ``*_counting_real`` / ``_counting_parity_`` / ``_counting_mod3_``
        variants, which arrive as ``curriculum_<task>``), not just the exact legacy
        string. Supports CoT format like "p1=(...), dist=1.7" via ``dist=``.
        """
        _is_counting = "counting" in str(question_type) if question_type is not None else False
        if _is_counting:
            return
        try:
            pred_val = _parse_regression_value(prediction)
            gt_val = _parse_regression_value(ground_truths[0])
            self._reg_preds.append(pred_val)
            self._reg_gts.append(gt_val)
        except (ValueError, IndexError):
            pass

    def _accumulate_grounding(self, prediction, ground_truths):
        """Single-box 3D grounding IoU / center-distance. Returns this sample's
        IoU (or None if not a grounding dataset / not parseable).

        All bboxes are bare metric float tuples in the scene-centered
        axis-aligned frame; IoU is computed in metric meters.
        """
        if not _is_grounding_dataset(self.dataset_name):
            return None
        self._grounding_total += 1
        _is_obb = _is_obb_grounding_dataset(self.dataset_name)
        if self.pano_grounding_format:
            pred_pano = parse_pano_3d_bbox(prediction)
            gt_pano = parse_pano_3d_bbox(ground_truths[0]) if ground_truths else None
            pred_box = pano_bbox_to_world_bbox(pred_pano) if pred_pano is not None else None
            gt_box = pano_bbox_to_world_bbox(gt_pano) if gt_pano is not None else None
        elif _is_obb:
            pred_box = parse_3d_obb(prediction)
            gt_box = parse_3d_obb(ground_truths[0]) if ground_truths else None
        else:
            pred_box = parse_3d_bbox(prediction)
            gt_box = parse_3d_bbox(ground_truths[0]) if ground_truths else None
        this_iou = None
        if pred_box is not None and gt_box is not None:
            this_iou = compute_3d_obb_iou(pred_box, gt_box) if _is_obb else compute_3d_iou(pred_box, gt_box)
            self._grounding_ious.append(this_iou)
            dist = ((pred_box[0]-gt_box[0])**2 + (pred_box[1]-gt_box[1])**2 + (pred_box[2]-gt_box[2])**2) ** 0.5
            self._grounding_center_dists.append(dist)
        return this_iou

    def _accumulate_multi_grounding(self, prediction, ground_truths):
        """Multi-object grounding F1@0.25 / F1@0.50 (Multi3DRefer)."""
        _multi_grounding_datasets = {"multi3drefer", "multi3drefer_train", "multi3drefer_val"}
        if self.dataset_name not in _multi_grounding_datasets:
            return
        self._multi_grounding_total += 1
        pred_boxes = parse_multi_3d_bbox(prediction)
        gt_boxes = parse_multi_3d_bbox(ground_truths[0]) if ground_truths else []
        f1_25, _, _ = compute_multi3drefer_f1(pred_boxes, gt_boxes, 0.25)
        f1_50, _, _ = compute_multi3drefer_f1(pred_boxes, gt_boxes, 0.50)
        self._multi_grounding_f1_25.append(f1_25)
        self._multi_grounding_f1_50.append(f1_50)

    def _accumulate_cider(self, count, norm_prediction, norm_ground_truths):
        """Stash normalized refs/preds for CIDEr. Uses count as a unique key
        (CIDEr requires exactly one prediction per key)."""
        cider_key = str(count)
        self.gts_cider[cider_key] = norm_ground_truths
        self.preds_cider[cider_key] = [norm_prediction]

    def _compute_official_score(self, prediction, ground_truths, question_type, scores, this_iou):
        """Dataset-specific official score (also stashed into ``scores`` for the
        VSI/probe cases, matching the original behaviour). Returns None when no
        dataset-specific score applies."""
        if self.dataset_name in ("vsi_bench", "sp_bench") and question_type:
            official_score = vsibench_score(prediction, ground_truths[0], question_type)
            scores["official"] = official_score
            return official_score
        if question_type and str(question_type).startswith("curriculum_") and ground_truths:
            # Probe tasks: numeric answers scored with MRA (partial credit for
            # near-misses), classification answers fall back to EM@R1. EM@1=0
            # on "0.78" vs "0.72" was masking real progress on regression probes.
            official_score = curriculum_score(prediction, ground_truths[0], question_type)
            scores["official"] = official_score
            return official_score
        if self.dataset_name == "sqa3d" and question_type:
            return scores["EM@R1"]
        if self.dataset_name == "multi3drefer" and question_type:
            # Use F1@0.25 as the per-type tracking score
            return self._multi_grounding_f1_25[-1] if self._multi_grounding_f1_25 else 0.0
        if _is_grounding_dataset(self.dataset_name) and question_type:
            # Single-box grounding datasets (ScanRefer / Nr3D / Sr3D): use the
            # IoU for *this* sample thresholded at 0.25 as the per-type tracking
            # score, so the periodic per-type breakdown is meaningful (instead of
            # always falling back to EM=0 on free-form bbox text). Unparseable
            # predictions count as misses (score 0).
            return 1.0 if (this_iou is not None and this_iou >= 0.25) else 0.0
        return None

    def _track_per_type(self, question_type, official_score, scores):
        """Append the per-question-type tracking score (official, or EM@1 fallback)."""
        if not question_type:
            return
        if question_type not in self.per_type_scores:
            self.per_type_scores[question_type] = []
        self.per_type_scores[question_type].append(
            official_score if official_score is not None else scores["EM"]
        )

    def _append_log(self, scene_id, question, question_type, prediction,
                    ground_truths, scores, n_source_images, all_predictions):
        """Append the per-sample results-log entry."""
        log_entry = {
            "scene_id": scene_id,
            "question": question,
            "question_type": question_type,
            "prediction": prediction,
            "ground_truth": ground_truths,
            "metrics": scores
        }
        if n_source_images is not None:
            log_entry["n_source_images"] = int(n_source_images)
        if all_predictions is not None:
            log_entry["all_predictions"] = all_predictions
        self.results_log.append(log_entry)

    def update_val_loss(self, loss_value: float):
        """Append a teacher-forced validation loss value (one entry per batch)."""
        self.val_losses.append(loss_value)

    def _compute_vsibench_overall(self):
        """Compute VSI-Bench official overall score.

        Fine-grained subtypes (VSI-Bench direction easy/medium/hard, and ReVSI's
        direction forward/backward, rel-distance closest/farthest, counting and
        room-size single/multiple) are first averaged into their coarse family,
        then all family scores are macro-averaged. For pure VSI-Bench runs only
        the direction subtypes are present, so this matches the original
        easy/medium/hard merge exactly.
        """
        type_means = {}
        for qtype, scores_list in self.per_type_scores.items():
            type_means[qtype] = sum(scores_list) / len(scores_list) if scores_list else 0.0

        task_scores = {}
        parent_means = {}  # family -> list of subtype means (averaged equally)
        for qtype, mean_val in type_means.items():
            parent = _SUBTYPE_TO_PARENT.get(qtype)
            if parent is not None:
                parent_means.setdefault(parent, []).append(mean_val)
            else:
                task_scores[qtype] = mean_val
        for parent, means in parent_means.items():
            task_scores[parent] = sum(means) / len(means)

        return sum(task_scores.values()) / len(task_scores) if task_scores else 0.0

    def all_questions_average(self):
        """Micro-average official per-sample scores across all questions.

        This differs from official_overall for datasets like VSI-Bench where
        official_overall is a macro average over task categories.
        """
        if not self.per_type_scores:
            return None
        total_count = sum(len(v) for v in self.per_type_scores.values())
        if total_count == 0:
            return None
        total_score = sum(sum(v) for v in self.per_type_scores.values())
        return total_score / total_count

    def save(self, step=None):
        """Unified save function: handles periodic steps and final save."""
        if not self.benchmarking:
            return

        stats = compute_final_scores(self.metrics_acc, self.gts_cider, self.preds_cider)
        reg = self.regression_stats()
        if reg:
            stats.update(reg)
        grnd = self.grounding_stats()
        if grnd:
            stats.update(grnd)
        mgrnd = self.multi_grounding_stats()
        if mgrnd:
            stats.update(mgrnd)
        bcs = self.binary_classification_stats()
        if bcs:
            stats.update(bcs)
        if self.val_losses:
            stats["val_loss"] = sum(self.val_losses) / len(self.val_losses)

        # Per-question-type breakdown (ordered by paper display order)
        if self.per_type_scores:
            type_breakdown = {}
            for display_name, key, scores_list in self._ordered_type_items():
                type_breakdown[display_name] = {
                    "key": key,
                    "mean": sum(scores_list) / len(scores_list) if scores_list else 0.0,
                    "count": len(scores_list),
                }
            stats["per_question_type"] = type_breakdown

            if self.dataset_name in ("vsi_bench",):
                stats["official_overall"] = self._compute_vsibench_overall()
            elif self.dataset_name in ("sp_bench", "sqa3d"):
                means = [v["mean"] for v in type_breakdown.values()]
                stats["official_overall"] = sum(means) / len(means) if means else 0.0
            elif self.dataset_name == "multi3drefer":
                mgs = self.multi_grounding_stats()
                stats["official_overall"] = mgs.get("m3dref_F1@0.25", 0.0)

            all_q_avg = self.all_questions_average()
            if all_q_avg is not None:
                stats["all_questions_average"] = all_q_avg
                stats["all_questions_count"] = sum(len(v) for v in self.per_type_scores.values())

        if step is not None:
            # Periodic Intermediate Save
            met_file = os.path.join(self.inter_dir, f"metrics_step_{step}.json")
            res_file = os.path.join(self.exp_name, "qa_results_current.json")

            with open(met_file, "w") as f:
                json.dump(stats, f, indent=4)
            with open(res_file, "w") as f:
                json.dump(self.results_log, f, indent=4)
            print(f"--- Intermediate save at step {step} ---")
        else:
            # Final Save
            met_file = os.path.join(self.exp_name, "final_metrics.json")
            res_file = os.path.join(self.exp_name, "qa_results_final.json")

            with open(met_file, "w") as f:
                json.dump(stats, f, indent=4)
            with open(res_file, "w") as f:
                json.dump(self.results_log, f, indent=4)
            print(f"--- Final Results Saved to '{self.exp_name}' ---")

    def _ordered_type_items(self):
        """Yield (display_name, internal_key, scores_list) in paper display order."""
        if self.dataset_name in ("vsi_bench", "sp_bench"):
            # Merge fine-grained subtypes into their coarse family on the fly so
            # the table shows the same families as the VSI-Bench paper. For pure
            # VSI-Bench runs only the direction subtypes are present, so this
            # reduces to the original easy/medium/hard merge.
            merged = {}
            for k, v in self.per_type_scores.items():
                parent = _SUBTYPE_TO_PARENT.get(k, k)
                merged.setdefault(parent, []).extend(v)
            order = VSIBENCH_DISPLAY_ORDER
            names = VSIBENCH_DISPLAY_NAMES
            for key in order:
                if key in merged:
                    yield names.get(key, key), key, merged[key]
            # Emit any types not in the standard order
            for key, scores in sorted(merged.items()):
                if key not in order:
                    yield names.get(key, key), key, scores
        elif self.dataset_name == "sqa3d":
            order = SQA3D_DISPLAY_ORDER
            for key in order:
                if key in self.per_type_scores:
                    yield key, key, self.per_type_scores[key]
            for key, scores in sorted(self.per_type_scores.items()):
                if key not in order:
                    yield key, key, scores
        else:
            for key, scores in sorted(self.per_type_scores.items()):
                yield key, key, scores

    def binary_classification_stats(self):
        """For binary yes/no tasks, compute GT distribution and majority baseline."""
        gt_labels = []
        for entry in self.results_log:
            gts = entry.get("ground_truth", [])
            if gts:
                gt = normalize(gts[0])
                if gt in ("yes", "no"):
                    gt_labels.append(gt)
        if not gt_labels:
            return None
        yes_count = sum(1 for l in gt_labels if l == "yes")
        no_count = len(gt_labels) - yes_count
        majority_frac = max(yes_count, no_count) / len(gt_labels)
        return {
            "binary_n": len(gt_labels),
            "binary_yes_count": yes_count,
            "binary_no_count": no_count,
            "binary_yes_frac": yes_count / len(gt_labels),
            "binary_majority_baseline": majority_frac,
        }

    def multi_grounding_stats(self):
        """Compute Multi3DRefer metrics (F1@0.25, F1@0.5) if available."""
        if not self._multi_grounding_f1_25:
            return {}
        f1_25 = np.array(self._multi_grounding_f1_25)
        f1_50 = np.array(self._multi_grounding_f1_50)
        return {
            "m3dref_F1@0.25": float(f1_25.mean()),
            "m3dref_F1@0.50": float(f1_50.mean()),
            "m3dref_n": len(f1_25),
            "m3dref_total": self._multi_grounding_total,
        }

    def grounding_stats(self):
        """Compute 3D grounding metrics (Acc@0.25, Acc@0.5, mean IoU) if available."""
        if not self._grounding_ious:
            return {}
        ious = np.array(self._grounding_ious)
        center_dists = np.array(self._grounding_center_dists) if self._grounding_center_dists else np.array([])
        stats = {
            "grnd_Acc@0.25": float((ious >= 0.25).mean()),
            "grnd_Acc@0.5": float((ious >= 0.5).mean()),
            "grnd_Acc@0.1": float((ious >= 0.1).mean()),
            "grnd_Acc@0.05": float((ious >= 0.05).mean()),
            "grnd_mean_IoU": float(ious.mean()),
            "grnd_median_IoU": float(np.median(ious)),
            "grnd_parse_rate": len(self._grounding_ious) / max(self._grounding_total, 1),
            "grnd_n": len(self._grounding_ious),
            "grnd_total": self._grounding_total,
        }
        if len(center_dists) > 0:
            stats["grnd_center_dist_mean"] = float(center_dists.mean())
            stats["grnd_center_dist_median"] = float(np.median(center_dists))
            stats["grnd_center_within_0.5m"] = float((center_dists <= 0.5).mean())
            stats["grnd_center_within_1m"] = float((center_dists <= 1.0).mean())
            stats["grnd_center_within_2m"] = float((center_dists <= 2.0).mean())
            if center_dists.mean() > 50:
                print(
                    f"\n  WARNING: Mean center distance is {center_dists.mean():.0f}m -- "
                    f"physically impossible for indoor scenes. Likely a coordinate "
                    f"normalization bug (box tokens stripped during decode, or "
                    f"pred/GT in different coordinate spaces)."
                )
        return stats

    def regression_stats(self):
        """Compute regression metrics (MAE, within-tolerance, correlation) if available."""
        if len(self._reg_preds) < 2:
            return {}
        preds = np.array(self._reg_preds)
        gts = np.array(self._reg_gts)
        abs_err = np.abs(preds - gts)  # regression answers are already in meters
        stats = {
            "reg_n": len(preds),
            "reg_MAE": float(abs_err.mean()),
            "reg_MedAE": float(np.median(abs_err)),
            "reg_within_1cm": float((abs_err <= 0.01).mean()),
            "reg_within_5cm": float((abs_err <= 0.05).mean()),
            "reg_within_10cm": float((abs_err <= 0.10).mean()),
            "reg_within_0.5m": float((abs_err <= 0.5).mean()),
            "reg_within_1.0m": float((abs_err <= 1.0).mean()),
            "reg_within_2.0m": float((abs_err <= 2.0).mean()),
            "reg_pred_mean": float(preds.mean()),
            "reg_pred_std": float(preds.std()),
            "reg_gt_mean": float(gts.mean()),
        }
        if preds.std() > 0 and gts.std() > 0:
            stats["reg_pearson_r"] = float(np.corrcoef(preds, gts)[0, 1])
        else:
            stats["reg_pearson_r"] = 0.0
        return stats

    def print_summary(self, step):
        """Calculates and prints running averages for all metrics."""
        print("\n" + "="*45)
        print(f"RUNNING STATS - STEP {step}")
        for metric, values in self.metrics_acc.items():
            avg = sum(values) / len(values) if values else 0
            print(f"{metric.ljust(8)}: {avg:.4f}")
        reg = self.regression_stats()
        if reg:
            print("--- Regression (meters) ---")
            print(f"  MAE:    {reg['reg_MAE']:.3f}m  (MedAE: {reg['reg_MedAE']:.3f}m)")
            print(f"  <=1cm:   {reg['reg_within_1cm']:.1%}  <=5cm: {reg['reg_within_5cm']:.1%}  <=10cm: {reg['reg_within_10cm']:.1%}")
            print(f"  <=0.5m:  {reg['reg_within_0.5m']:.1%}  <=1.0m: {reg['reg_within_1.0m']:.1%}  <=2.0m: {reg['reg_within_2.0m']:.1%}")
            print(f"  Pearson r: {reg['reg_pearson_r']:.4f}")
            print(f"  Pred u={reg['reg_pred_mean']:.2f} s={reg['reg_pred_std']:.2f}  |  GT u={reg['reg_gt_mean']:.2f}")
            print(f"  Parseable: {reg['reg_n']}/{len(self.metrics_acc['EM@1'])}")
        if self.val_losses:
            avg_vl = sum(self.val_losses) / len(self.val_losses)
            print(f"{'Val Loss'.ljust(8)}: {avg_vl:.4f}")
        grnd = self.grounding_stats()
        if grnd:
            print(f"--- 3D Grounding ---")
            print(f"  Acc@0.25: {grnd['grnd_Acc@0.25']:.1%}  Acc@0.5: {grnd['grnd_Acc@0.5']:.1%}")
            print(f"  Acc@0.1:  {grnd['grnd_Acc@0.1']:.1%}  Acc@0.05: {grnd['grnd_Acc@0.05']:.1%}")
            print(f"  Mean IoU: {grnd['grnd_mean_IoU']:.4f}  Median IoU: {grnd['grnd_median_IoU']:.4f}")
            if "grnd_center_dist_mean" in grnd:
                print(f"  Center dist: {grnd['grnd_center_dist_mean']:.2f}m (med: {grnd['grnd_center_dist_median']:.2f}m)")
                print(f"  Within 0.5m: {grnd['grnd_center_within_0.5m']:.1%}  1m: {grnd['grnd_center_within_1m']:.1%}  2m: {grnd['grnd_center_within_2m']:.1%}")
            print(f"  Parsed: {grnd['grnd_n']}/{grnd['grnd_total']} ({grnd['grnd_parse_rate']:.1%})")
        mgrnd = self.multi_grounding_stats()
        if mgrnd:
            print(f"--- Multi3DRefer Grounding ---")
            print(f"  F1@0.25: {mgrnd['m3dref_F1@0.25']:.4f}  F1@0.50: {mgrnd['m3dref_F1@0.50']:.4f}")
            print(f"  Samples: {mgrnd['m3dref_n']}/{mgrnd['m3dref_total']}")
        bcs = self.binary_classification_stats()
        if bcs:
            print(f"--- Binary Classification ---")
            print(f"  GT: {bcs['binary_yes_frac']:.1%} yes, {1-bcs['binary_yes_frac']:.1%} no (n={bcs['binary_n']})")
            print(f"  Majority baseline: {bcs['binary_majority_baseline']:.1%}")
        if self.per_type_scores:
            print("--- Per-Type Breakdown ---")
            for display_name, _, scores_list in self._ordered_type_items():
                avg = sum(scores_list) / len(scores_list) if scores_list else 0
                print(f"  {display_name.ljust(12)}: {avg:.4f} (n={len(scores_list)})")
        print("="*45 + "\n")

def normalize(text):
    if isinstance(text, list): text = " ".join(text)
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def max_em(prediction, ground_truths):
    normalized_pred = normalize(prediction)
    for gt in ground_truths:
        if normalized_pred == normalize(gt): return 1
    return 0

def refined_em(prediction, ground_truths):
    """EM@R1 (Refined Exact Match) as defined by LEO (Huang et al., ICML 2024).

    Returns 1 if the normalised prediction exactly matches any GT *or* is a
    substring of any GT (or vice-versa, after whitespace removal).
    """
    # ``normalize`` strips "a"/"an"/"the" as articles, which collapses a bare
    # MCQ letter prediction of "A" to the empty string. Without the non-empty
    # guards below, "" is a substring of every GT and every single-"A"
    # prediction was scored as correct regardless of the target letter.
    if not ground_truths:
        return 0.0
    normalized_pred = normalize(prediction)
    pred_nospace = "".join(normalized_pred.split())
    for gt in ground_truths:
        norm_gt = normalize(gt)
        gt_nospace = "".join(norm_gt.split())
        if pred_nospace and pred_nospace == gt_nospace:
            return 1.0
        if pred_nospace and gt_nospace and (
            pred_nospace in gt_nospace or gt_nospace in pred_nospace
        ):
            return 1.0
    return 0.0

# --- VSI-Bench / SP-Bench Official Metrics ---

VSIBENCH_MCA_TYPES = {
    "object_rel_direction_easy", "object_rel_direction_medium",
    "object_rel_direction_hard", "object_rel_direction",
    "object_rel_distance", "route_planning", "obj_appearance_order",
    # ReVSI multiple-choice subtypes (carry options, letter ground truth).
    "object_rel_direction_forward_easy", "object_rel_direction_forward_hard",
    "object_rel_direction_backward_easy", "object_rel_direction_backward_hard",
    "object_rel_distance_closest", "object_rel_distance_farthest",
}
VSIBENCH_NA_TYPES = {
    "object_abs_distance", "object_counting",
    "object_size_estimation", "room_size_estimation",
    # ReVSI numeric subtypes (no options, numeric ground truth -> MRA).
    "object_counting_single", "object_counting_multiple",
    "room_size_estimation_single", "room_size_estimation_multiple",
}

# Fine-grained subtypes merged into a coarse family for the macro-averaged
# overall and the per-type display table. Averaging subtype means equally
# matches VSI-Bench's easy/medium/hard convention. This is a no-op for any
# family whose subtypes are absent from a run (so pure VSI-Bench runs, which
# only have the direction easy/medium/hard subtypes, are unaffected).
VSIBENCH_SUBTYPE_GROUPS = {
    "object_rel_direction": [
        "object_rel_direction_easy", "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_rel_direction_forward_easy", "object_rel_direction_forward_hard",
        "object_rel_direction_backward_easy", "object_rel_direction_backward_hard",
    ],
    "object_rel_distance": [
        "object_rel_distance_closest", "object_rel_distance_farthest",
    ],
    "object_counting": [
        "object_counting_single", "object_counting_multiple",
    ],
    "room_size_estimation": [
        "room_size_estimation_single", "room_size_estimation_multiple",
    ],
}
_SUBTYPE_TO_PARENT = {
    sub: parent
    for parent, subs in VSIBENCH_SUBTYPE_GROUPS.items()
    for sub in subs
}

# Display order and short names matching the VSI-Bench paper results table.
# After aggregation the 3 direction subtypes are merged into "object_rel_direction".
VSIBENCH_DISPLAY_ORDER = [
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction",
    "route_planning",
    "obj_appearance_order",
]
VSIBENCH_DISPLAY_NAMES = {
    "object_counting":          "Obj. Cnt.",
    "object_abs_distance":      "Abs. Dist.",
    "object_size_estimation":   "Obj. Size",
    "room_size_estimation":     "Room Size",
    "object_rel_distance":      "Rel. Dist.",
    "object_rel_direction":     "Rel. Dir.",
    "route_planning":           "Route Plan",
    "obj_appearance_order":     "Appr. Order",
}

# SQA3D display order (SpaceMind paper convention)
SQA3D_DISPLAY_ORDER = ["What", "Is", "How", "Can", "Which", "Others"]

def vsibench_fuzzy_match(pred_str):
    """Extract first token, strip trailing period -- official VSI-Bench preprocessing."""
    return str(pred_str).strip().split(' ')[0].rstrip('.').strip().lower()

def vsibench_exact_match(pred, target):
    """Case-insensitive exact match after fuzzy extraction."""
    return 1.0 if vsibench_fuzzy_match(pred) == vsibench_fuzzy_match(target) else 0.0

def vsibench_mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    """MRA: fraction of thresholds where relative error is within tolerance."""
    try:
        pred_val = float(vsibench_fuzzy_match(pred))
        target_val = float(target)
    except (ValueError, TypeError):
        return 0.0
    if target_val == 0:
        return 1.0 if pred_val == 0 else 0.0
    num_pts = int((end - start) / interval + 2)
    thresholds = np.linspace(start, end, num_pts)
    relative_error = abs(pred_val - target_val) / abs(target_val)
    accuracy = relative_error <= (1 - thresholds)
    return float(accuracy.mean())

def vsibench_score(pred, target, question_type):
    """Compute the official VSI-Bench score for a single sample."""
    if question_type in VSIBENCH_MCA_TYPES:
        return vsibench_exact_match(pred, target)
    elif question_type in VSIBENCH_NA_TYPES:
        return vsibench_mean_relative_accuracy(pred, target)
    else:
        return vsibench_exact_match(pred, target)


def curriculum_mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    """VSI-Bench MRA adapted for curriculum_* tasks. Uses ``_parse_regression_value``
    so CoT-style answers (``p1=..., p2=..., dist=1.7``) score correctly."""
    try:
        pred_val = _parse_regression_value(str(pred))
        target_val = _parse_regression_value(str(target))
    except (ValueError, TypeError):
        return 0.0
    if target_val == 0:
        return 1.0 if abs(pred_val) < 1e-6 else 0.0
    num_pts = int((end - start) / interval + 2)
    thresholds = np.linspace(start, end, num_pts)
    relative_error = abs(pred_val - target_val) / abs(target_val)
    accuracy = relative_error <= (1 - thresholds)
    return float(accuracy.mean())


def _score_turn_sequence(pred, target):
    """Per-turn accuracy for free-form maze navigation answers.

    Splits both strings on commas, strips and lowercases each token, then
    scores position-by-position over the GT length. Missing prediction tokens
    (pred shorter than GT) count as wrong. Extra tokens are ignored.
    Returns a float in [0, 1].
    """
    gt = [t.strip().lower() for t in str(target).split(",") if t.strip()]
    pr = [t.strip().lower() for t in str(pred).split(",") if t.strip()]
    if not gt:
        return 0.0
    return sum(i < len(pr) and pr[i] == gt[i] for i in range(len(gt))) / len(gt)


def curriculum_score(pred, target, question_type):
    """Score a curriculum_* task. Numeric-answer probes (distances, areas, volumes,
    counts, angles, depth) are scored with MRA so predictions close to GT get
    partial credit instead of the all-or-nothing EM=0. Classification probes
    (yes/no, MCQ letters, first/second/both/neither) use VSI-Bench's fuzzy
    exact match (first token, strip trailing period, lowercase, strict
    equality) -- same protocol as VSI-Bench's MCA tasks, which avoids the
    article-stripping pitfall where a bare "A" prediction normalizes to the
    empty string and matches every GT under EM@R1's substring fallback.
    Multi-box grounding probes emit a JSON list of bbox_3d objects; the
    fuzzy first-token fallback would match every parseable prediction
    trivially, so those are scored with Multi3DRefer F1@0.25 instead.
    Maze navigation probes emit free-form turn sequences ("Turn Left, Turn
    Right, ..."); first-token fuzzy match collapses all of these to "turn",
    so they are scored with per-turn position accuracy instead."""
    if question_type and (
        "multi_box_grounding" in str(question_type)
        or "object_class_grounding" in str(question_type)
    ):
        pred_boxes = parse_multi_3d_bbox(str(pred))
        gt_boxes = parse_multi_3d_bbox(str(target))
        f1, _, _ = compute_multi3drefer_f1(pred_boxes, gt_boxes, 0.25)
        return f1
    if str(target).strip().lower().startswith("turn "):
        return _score_turn_sequence(pred, target)
    try:
        _parse_regression_value(str(target))
        return curriculum_mean_relative_accuracy(pred, target)
    except (ValueError, TypeError):
        return vsibench_exact_match(pred, target)

def calculate_max_rouge(references, candidate, metrics=['rouge1', 'rouge2', 'rougeL']):
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    max_scores = {metric: 0.0 for metric in metrics}
    cand_str = " ".join(candidate) if isinstance(candidate, list) else candidate
    for ref in references:
        ref_str = " ".join(ref) if isinstance(ref, list) else ref
        current_scores = scorer.score(target=ref_str, prediction=cand_str)
        for metric in metrics:
            max_scores[metric] = max(max_scores[metric], current_scores[metric].fmeasure)
    return max_scores

def run_evaluation_metrics(prediction, ground_truths):
    """Calculates METEOR, ROUGE, EM@1, and EM@R1 for a single prediction."""
    m_score = meteor_score([word_tokenize(g) for g in ground_truths], word_tokenize(prediction))
    r_scores = calculate_max_rouge(ground_truths, prediction)
    em_score = max_em(prediction, ground_truths)
    emr1_score = refined_em(prediction, ground_truths)

    return {
        "METEOR": m_score,
        "rouge1": r_scores["rouge1"],
        "rouge2": r_scores["rouge2"],
        "rougeL": r_scores["rougeL"],
        "EM": em_score,
        "EM@R1": emr1_score,
    }

def compute_final_scores(metrics_acc, gts_cider, preds_cider):
    """Aggregates all metrics including CIDEr."""
    c_score, _ = Cider().compute_score(gts_cider, preds_cider)
    final_metrics = {m: (sum(v)/len(v) if v else 0) for m, v in metrics_acc.items()}
    final_metrics["CIDEr"] = c_score
    return final_metrics
