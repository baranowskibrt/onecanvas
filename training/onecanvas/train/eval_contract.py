"""Execution-time enforcement that a training run evaluates itself.

WHY THIS EXISTS. Evaluation was configuration, so it was optional in practice:
``--gen_eval_steps 0``, an unset ``--gen_eval_dataset``, or a launcher variable
nobody exported produced a run that trained for days, wrote a healthy loss
curve, and never scored anything. Writing the rule down in the run guide did
not stop it, because nothing in the execution path ever checked. This module is
that check. It runs inside the trainer, on every launch and on every resume,
and it refuses to enter the optimization loop when the run cannot produce a
readable score.

WHAT IS ENFORCED, in order.

1.  BEFORE OPTIMIZATION. At least one evaluator is attached, is appropriate for
    what the run trains, declares a positive cadence and a non-empty
    population, and its configuration is persisted beside the checkpoints
    (``evaluation_contract.json`` plus the ``evaluation`` block of
    ``resolved_config.json``). An omitted launcher variable is a refusal, never
    a silent skip. The same check runs on resume.
2.  A BASELINE. A run scores itself once before its first optimizer step, so
    every curve has a zero point. On resume the baseline is required only when
    the applicable one is missing from the ledger; a recorded baseline is
    reused rather than recomputed.
3.  SCHEDULED VALIDATION, at the declared cadence, and a FINAL evaluation when
    the last step carries no matching recorded result. A matching completed
    result is reused, never recomputed.
4.  FAILURE IS TERMINAL. An infrastructure or scoring failure (OOM, a dataset
    that resolves to nothing, a scorer that raises, an evaluation that scores
    zero items) preserves whatever resumable state exists and then stops the
    run with an explicit evaluation failure. It must never degrade into
    indefinite unevaluated training. A model that answers wrongly, a sample
    that parses to a wrong number, a family that scores 0.0 -- these are
    evaluation RESULTS and never failures.
5.  COMPLETION IS EARNED. ``training_complete.json`` is written only after the
    metrics and the population accounting of the final evaluation are durably
    on disk.

THE EXCEPTION IS EXPLICIT AND OWNER-AUTHORIZED. Two environment variables,
both required, both recorded in the contract file and printed at launch:

    export ONECANVAS_NO_EVALUATION_AUTHORIZED_BY="owner"
    export ONECANVAS_NO_EVALUATION_REASON="one line saying why this run cannot score"

Setting one without the other is a refusal, so a half-configured escape hatch
cannot read as an authorization.

THE BASELINE IS SEPARATELY SKIPPABLE, because it is the only evaluation that
spends wall clock BEFORE the first optimizer step:

    export ONECANVAS_NO_BASELINE_AUTHORIZED_BY="owner"

The scheduled curve is untouched -- the run still evaluates itself on cadence
and still refuses a missing evaluator -- so this is not the no-evaluation
exception above wearing a smaller name. What it gives up is the curve's zero
point, so a later "it improved from X" must cite the warm start's own recorded
score rather than one measured here.

THE SEAM FOR EXTERNAL EVALUATORS. A run whose task the built-in generation eval
cannot express (a closed tool loop, a multi-turn episode) attaches its own
evaluator through ``ONECANVAS_TRAINER_PLUGIN``. Such an evaluator satisfies
this contract by answering one method on its callback::

    def evaluation_contract_spec(self):
        from onecanvas.train.eval_contract import EvaluatorSpec
        return EvaluatorSpec(
            name="toolmark_closed_loop",
            cadence_steps=self.every,          # must be > 0
            populations={"real_families": self.episodes * len(self.families)},
            final_population=self.episodes * len(self.families),
            baseline=True,
            scoring={"metric": "all_questions_average"},
            headline={"metric": "all_questions_average",
                      "unit": "fraction", "better": "higher"},
            details={"families": sorted(self.families)},
        )

or, when the evaluator is not itself a ``TrainerCallback``, by calling
``register_evaluator(spec)`` before ``trainer.train()``. Declaring the spec is
not a promise the launcher makes; it is read off the evaluator object the run
actually built, so it cannot drift from what runs.

THE LEDGER IS WHERE RESULTS ARE READ. Every evaluator's results reach
``evaluation_ledger.json`` through ``record_evaluation``, and the runs board and
run report draw every run's curve from it, so an evaluator needs no second
publication step to be visible. ``headline`` names the one recorded metric that
curve plots (``metric``), its ``unit`` (``fraction`` for 0-1, ``percent``,
``m`` or any other unit name) and which direction is ``better``, with an
optional plain-language ``label``. ``details["metric_labels"]`` may name the
other recorded keys the same way. A spec without a headline is refused before
training, and a result missing its headline value stops the run like any other
scoring failure.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

CONTRACT_FILE = "evaluation_contract.json"
LEDGER_FILE = "evaluation_ledger.json"
COMPLETION_FILE = "training_complete.json"

BUILTIN_EVALUATOR = "gen_eval"

PHASE_BASELINE = "baseline"


def _baseline_waived():
    """Owner authorization to skip the curve's zero point, or "".

    Separate from the no-evaluation exception: the scheduled curve still runs
    and a missing evaluator is still a refusal. This waives ONLY the measurement
    taken before the first optimizer step, which on a warm start is a score the
    source checkpoint already has.
    """
    return os.environ.get("ONECANVAS_NO_BASELINE_AUTHORIZED_BY", "").strip()

PHASE_SCHEDULED = "scheduled"
PHASE_FINAL = "final"


class EvaluationContractError(RuntimeError):
    """A precondition of the evaluation contract is not met. Raised before the
    first optimizer step, so nothing has been spent when it fires."""


class EvaluationExecutionError(RuntimeError):
    """An attached evaluator could not produce a result. Terminal: the run stops
    rather than continuing to train unevaluated."""


def _canonical(value):
    """JSON-safe, order-stable form of an arbitrary declaration value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        items = [_canonical(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, set) else items
    return str(value)


# ---------------------------------------------------------------------------
# Checkpoint identity: WHICH WEIGHTS were scored
# ---------------------------------------------------------------------------
def checkpoint_identity(output_dir, *, step: int, lineage: Dict[str, Any]) -> Dict[str, Any]:
    """Identify the model state a result was measured on, BY CONTENT.

    The identity is the content digest that ``checkpoint_guard`` computed and
    cached in the checkpoint's SAVE_COMPLETE sentinel when the checkpoint was
    finalized: the adapter, its config, and the learned spatial/depth modules,
    bound to the immutable source-model provenance stored in the checkpoint.

    A byte SIZE was the previous key and it is not an identity. Every LoRA of a
    given rank over the same target modules has the same byte count, so two
    different trainings of the same recipe into the same directory -- exactly
    the case where reusing one's evaluation for the other is wrong -- were
    indistinguishable. The size also cannot see a changed depth embedding at
    all.

    NO COMPLETE CHECKPOINT, NO IDENTITY. When the step carries no finalized
    checkpoint the identity is marked ``complete: False`` and carries no
    content, and ``find_recorded_evaluation`` refuses to reuse such a record.
    An unidentifiable set of weights cannot back a reused number.
    """
    from onecanvas.train.checkpoint_guard import read_content_identity

    ckpt = pathlib.Path(output_dir) / f"checkpoint-{int(step)}"
    content = read_content_identity(ckpt) if ckpt.exists() else None
    ident: Dict[str, Any] = {
        "step": int(step),
        "lineage": _canonical(lineage),
        "checkpoint_dir": ckpt.name,
        "complete": bool(content) and bool(content.get("strong")),
        "content": content,
    }
    blob = json.dumps(
        {"step": ident["step"], "lineage": ident["lineage"],
         "content": (content or {}).get("digest")},
        sort_keys=True, separators=(",", ":"),
    )
    ident["id"] = hashlib.sha1(blob.encode()).hexdigest()[:16]
    return ident


# ---------------------------------------------------------------------------
# Evaluator declaration
# ---------------------------------------------------------------------------
@dataclass
class EvaluatorSpec:
    """What an attached evaluator will actually do, read off the evaluator.

    ``inputs`` and ``scoring`` are what make a recorded result REUSABLE rather
    than merely present. Matching on population size alone says "somebody
    scored 1500 items at this step", which is true of a 320x240 draw and a
    640x480 draw, of an allowlisted pool and the whole benchmark, of the strict
    per-type metric and the micro average. Reusing across any of those is the
    silent-substitution failure wearing an evaluation costume. So an evaluator
    declares WHAT IT READS (``inputs``: resolution, frame count, split/offset,
    scene filter, orientation convention, augmentation, family list) and HOW IT
    SCORES (``scoring``: the selection metric, the answer format, the parser,
    the episode count), and reuse matches the hash of all of it together with
    the checkpoint identity.

    Leave a key out and reuse gets LOOSER, never safer. Put in everything whose
    change would change the number.
    """

    name: str
    cadence_steps: int
    populations: Dict[str, int]
    final_population: Optional[int] = None
    baseline: bool = True
    inputs: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    headline: Dict[str, Any] = field(default_factory=dict)

    def identity(self, *, is_final: bool = False) -> str:
        """A stable hash of evaluator + population + inputs + scoring.

        ``details`` and ``headline`` are deliberately NOT hashed: they say how
        to print and plot the result, and hashing them would invalidate reuse
        whenever the presentation changed.
        """
        payload = {
            "evaluator": self.name,
            "populations": {str(k): int(v) for k, v in sorted(self.populations.items())},
            "final_population": (
                int(self.final_population) if self.final_population is not None else None
            ),
            "is_final": bool(is_final),
            "inputs": _canonical(self.inputs),
            "scoring": _canonical(self.scoring),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def problems(self) -> List[str]:
        out: List[str] = []
        if not self.name:
            out.append("evaluator has no name")
        if int(self.cadence_steps) <= 0:
            out.append(
                f"cadence is {self.cadence_steps}, which disables scheduled "
                "validation; the contract requires a positive cadence"
            )
        if not self.populations:
            out.append("no evaluation population is declared")
        for key, size in self.populations.items():
            if key is None or key == "":
                out.append(
                    "an evaluation dataset is unnamed, so the run would score "
                    "whatever the TRAINING mix happens to be; name it "
                    "explicitly (--gen_eval_dataset)"
                )
            if int(size) <= 0:
                out.append(f"population for {key!r} is {size}, which scores nothing")
        if not self.scoring:
            out.append(
                "no scoring configuration is declared, so a recorded result "
                "could be reused across a change of metric or answer format"
            )
        return out

    def headline_problem(self) -> Optional[str]:
        """Why the declared headline cannot be plotted, or None.

        Kept out of ``problems()`` on purpose: an evaluator with problems is
        set aside as degraded when another one works, and an evaluator must
        never be switched off because its plotting declaration is missing.
        """
        head = self.headline or {}
        if head.get("metric") and head.get("unit") \
                and head.get("better") in ("higher", "lower"):
            return None
        return (
            f"headline is {self.headline!r}; declare "
            "headline={'metric': <recorded metric key>, 'unit': "
            "'fraction'|'percent'|<unit>, 'better': 'higher'|'lower'} so the "
            "runs board can draw this evaluator's curve from the ledger"
        )

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


_REGISTERED: List[EvaluatorSpec] = []


def register_evaluator(spec: EvaluatorSpec) -> None:
    """Declare an evaluator that is not itself a TrainerCallback."""
    if not isinstance(spec, EvaluatorSpec):
        raise TypeError(f"register_evaluator wants an EvaluatorSpec, got {type(spec)!r}")
    _REGISTERED.append(spec)


def collect_specs(callbacks) -> List[EvaluatorSpec]:
    """Every evaluator attached to this run, read off the objects themselves."""
    specs: List[EvaluatorSpec] = list(_REGISTERED)
    for cb in callbacks or []:
        getter = getattr(cb, "evaluation_contract_spec", None)
        if not callable(getter):
            continue
        spec = getter()
        if spec is None:
            continue
        if not isinstance(spec, EvaluatorSpec):
            raise EvaluationContractError(
                f"{type(cb).__name__}.evaluation_contract_spec() returned "
                f"{type(spec).__name__}, not an EvaluatorSpec."
            )
        specs.append(spec)
    return specs


# ---------------------------------------------------------------------------
# The owner-authorized exception
# ---------------------------------------------------------------------------
def no_evaluation_exception() -> Optional[Dict[str, str]]:
    """The explicit, owner-authorized no-evaluation exception, or None.

    Both variables are required. Half of the pair is a refusal rather than an
    authorization, so a stray export cannot turn evaluation off by accident.
    """
    who = os.environ.get("ONECANVAS_NO_EVALUATION_AUTHORIZED_BY", "").strip()
    why = os.environ.get("ONECANVAS_NO_EVALUATION_REASON", "").strip()
    if not who and not why:
        return None
    if not who or not why:
        raise EvaluationContractError(
            "The no-evaluation exception needs BOTH "
            "ONECANVAS_NO_EVALUATION_AUTHORIZED_BY and "
            "ONECANVAS_NO_EVALUATION_REASON. Got "
            f"AUTHORIZED_BY={who!r} REASON={why!r}. A partially configured "
            "escape hatch is not an authorization; set both or attach an "
            "evaluator."
        )
    return {"authorized_by": who, "reason": why}


# ---------------------------------------------------------------------------
# Durable records
# ---------------------------------------------------------------------------
def _atomic_write_json(path: pathlib.Path, payload) -> None:
    """Write JSON so a reader never sees a half-written file, and fsync both the
    file and its directory so the record survives the node dying next."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _read_json(path: pathlib.Path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return default


def read_ledger(output_dir) -> List[Dict[str, Any]]:
    records = _read_json(pathlib.Path(output_dir) / LEDGER_FILE, [])
    return records if isinstance(records, list) else []


def record_evaluation(
    output_dir,
    *,
    step: int,
    phase: str,
    evaluator: str,
    populations: Dict[str, Any],
    metrics: Dict[str, Any],
    identity: Optional[str] = None,
    checkpoint: Optional[Dict[str, Any]] = None,
    spec: Optional[EvaluatorSpec] = None,
) -> Dict[str, Any]:
    """Append one completed evaluation to the run's durable ledger.

    ``populations`` is the accounting, per dataset: how many items were asked
    for, how many the dataset resolved, how many were actually scored, how many
    were dropped. A score with no population behind it is not a result.

    ``identity`` and ``checkpoint`` are what later reuse matches on. The
    evaluator's declared inputs and scoring are written out beside them so a
    reader can see WHY two records differ instead of only that two hashes do.

    The spec's ``headline`` travels with the record, and its value must be a
    finite number in ``metrics``: a result the board cannot plot is a scoring
    failure, not a quiet gap in the curve.
    """
    head = (spec.headline if spec is not None else None) or {}
    if head:
        value = metrics.get(head.get("metric"))
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or value != value or value in (float("inf"), float("-inf")):
            fail_evaluation(
                output_dir, step=step, phase=phase, evaluator=evaluator,
                cause=EvaluationExecutionError(
                    f"declared headline metric {head.get('metric')!r} is "
                    f"{value!r} in the recorded metrics, not a finite number"),
            )
    path = pathlib.Path(output_dir) / LEDGER_FILE
    records = read_ledger(output_dir)
    record = {
        "step": int(step),
        "phase": phase,
        "evaluator": evaluator,
        "identity": identity,
        "checkpoint": checkpoint or {},
        "populations": populations,
        "metrics": metrics,
        "recorded_at": time.time(),
    }
    if spec is not None:
        record["inputs"] = _canonical(spec.inputs)
        record["scoring"] = _canonical(spec.scoring)
        record["headline"] = _canonical(spec.headline)
        if (spec.details or {}).get("metric_labels"):
            # Plain names for the recorded keys, so an evaluator that has no
            # contract entry (one run outside the trainer) is still readable.
            record["metric_labels"] = _canonical(spec.details["metric_labels"])
    if checkpoint is not None and not checkpoint.get("complete"):
        print(
            f"[eval-contract] step {step}: this {phase} result was measured "
            "with NO complete checkpoint identified at that step "
            f"({checkpoint.get('checkpoint_dir')}), so the weights behind it "
            "cannot be named and the result is not reusable while that stays "
            "true. Check whether the save was skipped (low disk) or refused as "
            "torn.",
            flush=True,
        )
    records.append(record)
    _atomic_write_json(path, records)
    return record


def _record_checkpoint(output_dir, record: Dict[str, Any]) -> Dict[str, Any]:
    """The record's checkpoint block, re-resolved from disk when it named none.

    The block a record carries is not independent evidence: ``record_evaluation``
    got it from ``checkpoint_identity(output_dir, step, lineage)``, read off the
    directory at record time. Re-deriving it now from the same three inputs is
    THE SAME evidence, not a weaker one, so a result whose checkpoint could not
    be identified when it was written becomes attributable as soon as
    ``checkpoint_guard`` can read that checkpoint -- which is what a pre-identity
    SAVE_COMPLETE sentinel changed. Without this, a run that resumed from such a
    checkpoint re-measured its baseline on every single requeue and could never
    reuse the hour it had just spent.

    It refuses unless the re-resolved identity is complete AND names the same
    checkpoint directory the record named. A record that named no directory, or
    a step whose checkpoint is now absent or torn, keeps its original block and
    stays unusable.
    """
    ckpt = record.get("checkpoint") or {}
    if ckpt.get("complete") or not ckpt.get("checkpoint_dir"):
        return ckpt
    try:
        step = int(record.get("step"))
    except (TypeError, ValueError):
        return ckpt
    fresh = checkpoint_identity(output_dir, step=step,
                                lineage=ckpt.get("lineage") or {})
    if not fresh.get("complete") or fresh["checkpoint_dir"] != ckpt["checkpoint_dir"]:
        return ckpt
    print(
        f"[eval-contract] step {step}: the {record.get('phase')} result for "
        f"{record.get('evaluator')} named no weights when it was recorded, and "
        f"{fresh['checkpoint_dir']} is readable now and identifies them as "
        f"{fresh['id']}, so the result can be matched instead of re-measured.",
        flush=True,
    )
    return fresh


def find_recorded_evaluation(
    output_dir,
    *,
    step: Optional[int] = None,
    phase: Optional[str] = None,
    evaluator: Optional[str] = None,
    identity: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """A completed result matching this request, so it is reused not redone.

    A MATCH IS THE SAME MEASUREMENT ON THE SAME WEIGHTS, not merely the same
    step and sample count. ``identity`` is the evaluator's hash of population,
    declared inputs and scoring configuration; ``checkpoint_id`` is the hash of
    the lineage and the step. Both must agree, and a record written before this
    tightening (no ``identity`` field) never matches, so an old ledger cannot
    cause a stale reuse -- it causes one extra honest evaluation.

    EVERY REUSE NAMES THE WEIGHTS, INCLUDING A BASELINE'S. Callers that leave
    ``checkpoint_id`` unset do so because the baseline's weights are the run's
    starting point rather than the asking step's, NOT because a baseline may be
    reused off a record that identifies no weights at all. So the completeness
    requirement lives here, on every match, and ``checkpoint_id`` only adds the
    "and it is THIS step's weights" clause on top of it.
    """
    for record in reversed(read_ledger(output_dir)):
        if step is not None and int(record.get("step", -1)) != int(step):
            continue
        if phase is not None and record.get("phase") != phase:
            continue
        if evaluator is not None and record.get("evaluator") != evaluator:
            continue
        if identity is not None and record.get("identity") != identity:
            continue
        ckpt = _record_checkpoint(output_dir, record)
        if checkpoint_id is not None and ckpt.get("id") != checkpoint_id:
            continue
        # A result measured on weights that were never identified by content is
        # not reusable: nothing in the record says WHICH weights produced it.
        if not ckpt.get("complete"):
            continue
        if not record.get("metrics"):
            continue
        return record
    return None


# ---------------------------------------------------------------------------
# The lifecycle, driven for any declared evaluator
# ---------------------------------------------------------------------------
def wants_checkpoint(step: int, *, cadence: int, max_steps: int = 0,
                     stop_step: int = 0) -> bool:
    """Does this step need a complete checkpoint for the contract's sake?

    Every scheduled evaluation boundary does, so that a terminal evaluation
    failure leaves the step resumable and so that the result can name the
    weights it was measured on. SAVE CADENCE AND EVALUATION CADENCE ARE
    DIFFERENT KNOBS -- RUN_SAVE_STEPS is resume granularity, RUN_GEN_EVAL_STEPS
    is how often a curve gets a point -- and with save_steps 125 against a
    3000-step eval cadence they coincide only by arithmetic accident. The final
    step needs one too: it is the step the final evaluation scores, and
    max_steps is rarely a multiple of save_steps.
    """
    if step <= 0:
        return False
    if cadence > 0 and step % cadence == 0:
        return True
    if max_steps and step >= int(max_steps):
        return True
    if stop_step and step >= int(stop_step):
        return True
    return False


# ---------------------------------------------------------------------------
# Cross-rank coordination, on a process group of its own
# ---------------------------------------------------------------------------
# WHY A SEPARATE GROUP AND WHY IT IS NOT ENOUGH ON ITS OWN.
#
# An evaluation failure has to be agreed BEFORE any rank enters a collective
# the failing rank will never reach. Once rank 0 is blocked inside a plugin's
# own all_gather_object, a join_failure issued later by rank 2 is a DIFFERENT
# collective: it cannot rescue rank 0, it can only mismatch with it. So the
# agreement point belongs immediately before the plugin's collective, which is
# what phase_join() is for, and the post-hoc join in the lifecycle driver is
# only an outer net for failures raised outside that window.
#
# The group is gloo and is created eagerly on every rank at contract
# enforcement time, before the first optimizer step. Lazy creation would be a
# collective in itself, and a rank that failed before reaching it would hang
# the others on the creation rather than on the join. Gloo, because CPU-side
# object exchange must still work when the failure was CUDA-side.
#
# THE EXPLICIT LIMIT. A rank that has DIED -- segfault, OOM-killer, node loss --
# cannot join anything, and no in-process protocol can make it. The join is
# therefore bounded by ONECANVAS_EVAL_JOIN_TIMEOUT_S (default 900) so the
# survivors fail with an evaluation error naming the missing ranks instead of
# sitting until ddp_timeout. Recovering from a dead process is torchrun's and
# SLURM's job (--requeue plus auto-resume from the last complete checkpoint),
# not this contract's.
_COORD_GROUP = None
_COORD_GROUP_TRIED = False


def join_timeout_seconds() -> int:
    try:
        return max(30, int(os.environ.get("ONECANVAS_EVAL_JOIN_TIMEOUT_S", "900")))
    except ValueError:
        return 900


def init_coordination_group(verbose: bool = False):
    """Create the contract's own process group. Collective: ALL ranks call it,
    at the same point, before training starts."""
    global _COORD_GROUP, _COORD_GROUP_TRIED
    if _COORD_GROUP_TRIED:
        return _COORD_GROUP
    _COORD_GROUP_TRIED = True
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return None
    if not (dist.is_available() and dist.is_initialized()):
        return None
    if dist.get_world_size() <= 1:
        return None
    try:
        import datetime
        _COORD_GROUP = dist.new_group(
            backend="gloo",
            timeout=datetime.timedelta(seconds=join_timeout_seconds()),
        )
        if verbose:
            print(
                "[eval-contract] coordination group ready (gloo, timeout "
                f"{join_timeout_seconds()}s) for cross-rank failure agreement",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001 - fall back to the default group
        _COORD_GROUP = None
        if verbose:
            print(
                f"[eval-contract] could not create a private coordination "
                f"group ({e}); failure agreement will use the default process "
                "group, which shares its ordering with training collectives",
                flush=True,
            )
    return _COORD_GROUP


def phase_join(local_error, *, step: int, phase: str, evaluator: str,
               output_dir=None, where: str = "before a plugin collective"):
    """Agree on failure BEFORE entering a collective a failed rank cannot reach.

    Call this at the end of the per-rank, collective-free part of an
    evaluation, immediately before the evaluator's own all_gather / all_reduce.
    Every rank passes the exception it caught locally, or None. If ANY rank
    failed, EVERY rank raises here, so none of them enters the collective that
    the failed rank will never reach.

        local_error = None
        try:
            per_rank_result = ...            # no collectives in here
        except Exception as e:
            local_error = e
        eval_contract.phase_join(local_error, step=step, phase=phase,
                                 evaluator=name, output_dir=out)
        dist.all_gather_object(shards, per_rank_result)   # safe now

    Returns None when every rank is healthy.
    """
    joined = join_failure(local_error)
    if not joined:
        return None
    if output_dir is not None:
        fail_evaluation(
            output_dir, step=step, phase=phase, evaluator=evaluator,
            cause=(local_error or RuntimeError(joined)),
        )
    raise EvaluationExecutionError(
        f"evaluation failed {where} at step {step} ({phase}, {evaluator}): "
        f"{joined}"
    )


def is_driven_evaluator(obj) -> bool:
    """An evaluator the contract can drive through the whole lifecycle.

    Two methods: ``evaluation_contract_spec()`` (what it measures) and
    ``run_evaluation(model, step, phase)`` returning ``{"metrics": {...},
    "populations": {...}}``. Everything else -- the baseline, reuse across a
    resume, the final evaluation, the ledger write, the distributed failure
    join -- belongs to the contract, so a plugin cannot implement three of the
    five phases and look complete. Registration is not participation.
    """
    return callable(getattr(obj, "evaluation_contract_spec", None)) and callable(
        getattr(obj, "run_evaluation", None)
    )


try:  # transformers is a hard dependency of the training package
    from transformers import TrainerCallback as _TrainerCallback
except Exception:  # pragma: no cover - keeps the module importable standalone
    class _TrainerCallback:  # type: ignore[no-redef]
        pass


class EvaluationLifecycleCallback(_TrainerCallback):
    """Drives every declared external evaluator through the full lifecycle.

    WHY THIS EXISTS rather than one method per plugin. The first version of this
    contract asked an external evaluator only to declare itself, and the tool-
    loop plugin then had a step-0 point that a requeue skipped, no final
    evaluation, no reuse check against what was already scored, and no ledger
    entry, while passing the start-up check and reading as integrated. Each of
    those is the same omission the built-in path had, reimplemented one layer
    out. They live here once instead.

    ORDERING. The scheduled evaluation runs in ``on_save``, after the step's
    checkpoint is finalized, for the reason the built-in does: an evaluation
    that dies must leave the step resumable. ``on_step_end`` forces the save at
    a cadence step so the two cannot drift apart.
    """

    def __init__(self, driven, output_dir, lineage: Dict[str, Any]):
        self.driven = list(driven)
        self.output_dir = str(output_dir)
        self.lineage = dict(lineage or {})

    # -- TrainerCallback protocol (duck-typed; transformers calls by name) --
    def _rank(self) -> int:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    def _checkpoint(self, step: int) -> Dict[str, Any]:
        return checkpoint_identity(self.output_dir, step=step, lineage=self.lineage)

    def _needed(self, ev, step: int, phase: str, *, is_final: bool) -> bool:
        """Rank 0 decides, every rank obeys. A recorded result for the SAME
        measurement on the SAME weights is reused; anything else is redone."""
        decision = True
        if self._rank() == 0:
            spec = ev.evaluation_contract_spec()
            if phase == PHASE_BASELINE and _baseline_waived():
                print("[eval-contract] baseline waived by "
                      f"ONECANVAS_NO_BASELINE_AUTHORIZED_BY={_baseline_waived()}; "
                      "the scheduled curve is unaffected")
                decision = False
            else:
                if phase == PHASE_BASELINE:
                    # The baseline is the CURVE'S ZERO POINT, measured on the
                    # weights the run started from. A resume at step 5000 must not
                    # redo it against step-5000 weights -- that would be a second
                    # scheduled point wearing the baseline's name. So it matches on
                    # the measurement alone, anywhere in the ledger, and runs at the
                    # resume step only when this run never recorded one.
                    found = find_recorded_evaluation(
                        self.output_dir, phase=PHASE_BASELINE, evaluator=spec.name,
                        identity=spec.identity(is_final=False),
                    )
                else:
                    found = find_recorded_evaluation(
                        self.output_dir, step=step, evaluator=spec.name,
                        identity=spec.identity(is_final=is_final),
                        checkpoint_id=self._checkpoint(step)["id"],
                    )
                decision = found is None
                if not decision:
                    print(
                        f"[eval-contract] {spec.name}: reusing recorded {phase} "
                        f"result from step {found.get('step')} "
                        f"(identity {found.get('identity')})",
                        flush=True,
                    )
        return broadcast_decision(decision)

    def _run(self, ev, model, step: int, phase: str, *, is_final: bool = False):
        spec = ev.evaluation_contract_spec()
        local_error = None
        result = None
        try:
            result = ev.run_evaluation(model=model, step=step, phase=phase)
        except Exception as e:  # noqa: BLE001 - re-raised on every rank below
            local_error = e
        joined = join_failure(local_error)
        if joined:
            fail_evaluation(
                self.output_dir, step=step, phase=phase, evaluator=spec.name,
                cause=(local_error or RuntimeError(joined)),
            )
        if self._rank() != 0:
            return
        metrics = (result or {}).get("metrics") or {}
        populations = (result or {}).get("populations") or {}
        if not metrics or not populations:
            fail_evaluation(
                self.output_dir, step=step, phase=phase, evaluator=spec.name,
                cause=EvaluationExecutionError(
                    f"run_evaluation returned metrics={bool(metrics)} "
                    f"populations={bool(populations)}; a result needs both, "
                    "because a score with no measured population behind it is "
                    "not a result."
                ),
            )
        record_evaluation(
            self.output_dir, step=step, phase=phase, evaluator=spec.name,
            populations=populations, metrics=metrics,
            identity=spec.identity(is_final=is_final),
            checkpoint=self._checkpoint(step), spec=spec,
        )

    def on_train_begin(self, args, state, control, model=None, **kw):
        step = int(state.global_step)
        for ev in self.driven:
            spec = ev.evaluation_contract_spec()
            if spec.problems():
                continue
            # On a fresh run this is step 0. On a resume it is the resume step,
            # and it runs only when no baseline for this measurement and these
            # weights is already recorded -- which is what makes a requeued run
            # that never got its zero point still get one.
            if spec.baseline and self._needed(ev, step, PHASE_BASELINE,
                                              is_final=False):
                self._run(ev, model, step, PHASE_BASELINE)
            # A RESUME ON AN UNSCORED SCHEDULED BOUNDARY SCORES IT HERE. The
            # boundary's checkpoint is saved before its evaluation runs, so a
            # failed or preempted evaluation resumes from exactly this step,
            # and on_save does not fire for it again. Without this the point
            # was silently dropped: toolloop_grounding_interface_v1 stopped on
            # an evaluator defect at step 3000 (job 2966167) with its
            # checkpoint complete, and the next point would have been 4000.
            cadence = int(spec.cadence_steps)
            if (step > 0 and cadence > 0 and step % cadence == 0
                    and self._needed(ev, step, PHASE_SCHEDULED, is_final=False)):
                self._run(ev, model, step, PHASE_SCHEDULED)
        return control

    def on_step_end(self, args, state, control, **kw):
        # REQUEST THE CHECKPOINT, at every boundary this evaluator will score
        # and at the final step, whatever save_steps happens to be.
        step = int(state.global_step)
        for ev in self.driven:
            spec = ev.evaluation_contract_spec()
            if wants_checkpoint(
                step,
                cadence=int(spec.cadence_steps),
                max_steps=int(getattr(state, "max_steps", 0) or 0),
                stop_step=int(getattr(args, "continuation_stop_step", 0) or 0),
            ):
                control.should_save = True
        return control

    def on_save(self, args, state, control, model=None, **kw):
        step = int(state.global_step)
        for ev in self.driven:
            spec = ev.evaluation_contract_spec()
            if spec.cadence_steps <= 0 or step <= 0 or step % spec.cadence_steps:
                continue
            if self._needed(ev, step, PHASE_SCHEDULED, is_final=False):
                self._run(ev, model, step, PHASE_SCHEDULED)
        return control

    def on_train_end(self, args, state, control, model=None, **kw):
        step = int(state.global_step)
        for ev in self.driven:
            spec = ev.evaluation_contract_spec()
            if spec.problems():
                continue
            is_final = spec.final_population is not None
            if self._needed(ev, step, PHASE_FINAL, is_final=is_final):
                self._run(ev, model, step, PHASE_FINAL, is_final=is_final)
        return control


# ---------------------------------------------------------------------------
# The precondition check
# ---------------------------------------------------------------------------
def enforce_before_optimization(
    output_dir,
    callbacks,
    *,
    resuming: bool,
    is_rank_zero: bool = True,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Refuse to start optimizing unless this run can produce a readable score.

    Returns the persisted contract record. Raises EvaluationContractError when
    no attached evaluator satisfies the contract and no owner-authorized
    exception is set.
    """
    # Collective, and the FIRST thing: every rank reaches this point before the
    # first optimizer step, which is the only moment the group can be created
    # without a failed rank being able to hang its peers on the creation.
    init_coordination_group(verbose=is_rank_zero)

    exception = no_evaluation_exception()
    specs = collect_specs(callbacks)
    driven_names = {
        cb.evaluation_contract_spec().name
        for cb in (callbacks or [])
        if is_driven_evaluator(cb) and cb.evaluation_contract_spec() is not None
    }
    driven_names |= {
        d.evaluation_contract_spec().name
        for cb in (callbacks or [])
        for d in getattr(cb, "driven", []) or []
    }

    contract: Dict[str, Any] = {
        "enforced": exception is None,
        "resuming": bool(resuming),
        "recorded_at": time.time(),
        "evaluators": [
            dict(s.as_dict(), identity=s.identity(), driven=(s.name in driven_names))
            for s in specs
        ],
        "context": dict(extra_context or {}),
    }

    if exception is not None:
        contract["exception"] = exception
        if is_rank_zero:
            _persist_contract(output_dir, contract)
            print(
                "\n" + "!" * 70 + "\n"
                "  TRAINING WITHOUT EVALUATION, on an explicit authorization.\n"
                f"  authorized_by: {exception['authorized_by']}\n"
                f"  reason:        {exception['reason']}\n"
                "  This run will produce no score and no curve. The\n"
                "  authorization is recorded in " + CONTRACT_FILE + ".\n"
                + "!" * 70 + "\n",
                flush=True,
            )
        return contract

    if not specs:
        raise EvaluationContractError(
            "NO EVALUATOR IS ATTACHED, so this run would train to completion "
            "without ever scoring itself.\n"
            "  Fix one of:\n"
            "  - set --gen_eval_steps > 0 and --gen_eval_dataset "
            "<dataset[:n],...> (launcher: RUN_GEN_EVAL_STEPS, "
            "RUN_GEN_EVAL_DATASET),\n"
            "  - or attach a task-appropriate evaluator through "
            "ONECANVAS_TRAINER_PLUGIN whose callback answers "
            "evaluation_contract_spec() (see onecanvas/train/eval_contract.py),\n"
            "  - or, only with the owner's authorization, export BOTH "
            "ONECANVAS_NO_EVALUATION_AUTHORIZED_BY and "
            "ONECANVAS_NO_EVALUATION_REASON."
        )

    problems: Dict[str, List[str]] = {}
    for spec in specs:
        found = spec.problems()
        if found:
            problems[spec.name or type(spec).__name__] = found
    if len(problems) == len(specs):
        lines = [
            "EVERY ATTACHED EVALUATOR IS UNUSABLE, so this run would train "
            "without producing a readable score."
        ]
        for name, found in problems.items():
            lines.append(f"  {name}:")
            lines.extend(f"    - {p}" for p in found)
        lines.append(
            "  An omitted launcher variable does not disable evaluation; fix "
            "the configuration above, or set the owner-authorized exception."
        )
        raise EvaluationContractError("\n".join(lines))
    if problems:
        contract["degraded_evaluators"] = problems

    # A RESULT NOBODY CAN FIND is how five runs in a row read as "no evaluation"
    # on the board while scoring themselves on schedule. Every evaluator that
    # will run declares what its curve plots, or the run does not start.
    unplottable = {s.name: s.headline_problem() for s in specs
                   if (s.name or "") not in problems and s.headline_problem()}
    if unplottable:
        raise EvaluationContractError(
            "AN ATTACHED EVALUATOR DECLARES NO HEADLINE, so its results would "
            "reach the ledger with nothing saying which number the runs board "
            "should draw.\n"
            + "\n".join(f"  {name}: {why}" for name, why in unplottable.items()))

    # A CURVE WITH NO ZERO POINT is the case the baseline exists for, and
    # ``baseline=False`` is the one way an evaluator could still produce one.
    # It is allowed per evaluator (a secondary transfer curve does not need its
    # own zero) but not for the run: something must score the starting weights.
    usable = [s for s in specs if (s.name or "") not in problems]
    if not any(s.baseline for s in usable) and not _baseline_waived():
        raise EvaluationContractError(
            "NO ATTACHED EVALUATOR PROVIDES A BASELINE, so this run's curves "
            "would have no zero point and nothing would check that the "
            "evaluation population resolves before compute is spent. "
            f"Evaluators declaring baseline=False: {[s.name for s in usable]}."
        )

    if is_rank_zero:
        _persist_contract(output_dir, contract)
        for spec in specs:
            if (spec.name or "") in problems:
                continue
            print(
                f"[eval-contract] evaluator {spec.name!r}: every "
                f"{spec.cadence_steps} steps over "
                f"{ {k: int(v) for k, v in spec.populations.items()} }"
                f"{'' if spec.final_population is None else f', final {spec.final_population}'}"
                f"{'' if spec.baseline else ', NO baseline'}",
                flush=True,
            )
        for spec in specs:
            if (spec.name or "") in problems or spec.name == BUILTIN_EVALUATOR:
                continue
            if spec.name in driven_names:
                continue
            print(
                f"[eval-contract] evaluator {spec.name!r} is SELF-DRIVEN: it "
                "declares itself but is not driven by the lifecycle callback, "
                "so it owns its own baseline, resume reuse, final evaluation "
                "and ledger writes. Implementing run_evaluation(model, step, "
                "phase) instead hands all four to the contract.",
                flush=True,
            )
    return contract


def _persist_contract(output_dir, contract: Dict[str, Any]) -> None:
    """Write the contract beside the checkpoints and into the resolved config,
    so the run's own settings record what it promised to measure."""
    out = pathlib.Path(output_dir)
    _atomic_write_json(out / CONTRACT_FILE, contract)
    config_path = out / "resolved_config.json"
    config = _read_json(config_path, None)
    if isinstance(config, dict):
        config.setdefault("evaluation", {})["contract"] = contract
        _atomic_write_json(config_path, config)


# ---------------------------------------------------------------------------
# Terminal failure
# ---------------------------------------------------------------------------
def fail_evaluation(output_dir, *, step: int, phase: str, evaluator: str, cause) -> None:
    """Stop the run with an explicit evaluation failure.

    Called after whatever resumable state exists has been written (the periodic
    path finalizes its checkpoint BEFORE scoring, precisely so this path can
    stop without losing the step). The failure is recorded in the ledger so a
    later read sees why the run ended, then re-raised.
    """
    detail = f"{type(cause).__name__}: {cause}"
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0:
        try:
            path = pathlib.Path(output_dir) / LEDGER_FILE
            records = read_ledger(output_dir)
            records.append({
                "step": int(step),
                "phase": phase,
                "evaluator": evaluator,
                "failed": detail,
                "recorded_at": time.time(),
            })
            _atomic_write_json(path, records)
        except Exception:  # the ledger write must not mask the real failure
            pass
    try:
        from onecanvas.train.checkpoint_guard import find_last_complete_checkpoint
        last = find_last_complete_checkpoint(str(output_dir))
    except Exception:
        last = None
    raise EvaluationExecutionError(
        f"EVALUATION FAILED at step {step} ({phase}, evaluator {evaluator!r}): "
        f"{detail}\n"
        "  This is an infrastructure or scoring failure, not a model result, "
        "so the run stops here rather than training on unevaluated.\n"
        f"  Last complete checkpoint: {last or 'none'}\n"
        "  Diagnose the evaluator, then resume; the recorded results for "
        "earlier steps are reused."
    ) from cause


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------
def record_completion(output_dir, *, step: int, is_rank_zero: bool = True) -> Dict[str, Any]:
    """Mark the run successfully complete, but only once the final evaluation's
    metrics and population accounting are durably on disk.

    A run that trained every step and could not score the last one is NOT
    complete, and saying otherwise is how an unevaluated run reads as a result.
    """
    contract = _read_json(pathlib.Path(output_dir) / CONTRACT_FILE, {}) or {}
    ledger = read_ledger(output_dir)

    if contract.get("exception"):
        payload = {
            "step": int(step),
            "completed_at": time.time(),
            "evaluated": False,
            "no_evaluation_exception": contract["exception"],
        }
        if is_rank_zero:
            _atomic_write_json(pathlib.Path(output_dir) / COMPLETION_FILE, payload)
        return payload

    scored = [r for r in ledger if r.get("metrics")]
    if not scored:
        raise EvaluationExecutionError(
            "Training reached the end of its schedule with NO recorded "
            "evaluation result, so it cannot be recorded as complete. The "
            f"ledger at {pathlib.Path(output_dir) / LEDGER_FILE} is empty. An "
            "external evaluator must call record_evaluation() for every result "
            "it produces."
        )
    final = max(scored, key=lambda r: int(r.get("step", 0)))
    if int(final.get("step", -1)) != int(step):
        raise EvaluationExecutionError(
            f"Training ended at step {step} but the newest recorded evaluation "
            f"is at step {final.get('step')}. The final evaluation did not "
            "complete, so this run is not recorded as complete."
        )
    if not final.get("populations"):
        raise EvaluationExecutionError(
            f"The final evaluation at step {step} recorded metrics with no "
            "population accounting, so the score has no measured sample behind "
            "it. Not recording completion."
        )

    payload = {
        "step": int(step),
        "completed_at": time.time(),
        "evaluated": True,
        "baseline_step": next(
            (int(r["step"]) for r in scored if r.get("phase") == PHASE_BASELINE), None
        ),
        "final_evaluation": final,
        "evaluations_recorded": len(scored),
    }
    if is_rank_zero:
        _atomic_write_json(pathlib.Path(output_dir) / COMPLETION_FILE, payload)
        print(
            f"[eval-contract] run complete at step {step}: "
            f"{len(scored)} recorded evaluations, final "
            f"{ {k: v for k, v in list(final.get('metrics', {}).items())[:4]} }",
            flush=True,
        )
    return payload


# ---------------------------------------------------------------------------
# Distributed helpers: every rank must take the SAME branch
# ---------------------------------------------------------------------------
def broadcast_decision(value: bool) -> bool:
    """Rank 0's decision, taken by every rank.

    Reuse and skip decisions read files on shared storage, which is not
    guaranteed to be coherent across nodes at the same instant; a rank that
    decides differently deadlocks the next collective.
    """
    try:
        import torch
    except ImportError:  # no distributed context to reconcile
        return bool(value)
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return bool(value)
    if torch.distributed.get_world_size() <= 1:
        return bool(value)
    box = [bool(value)]
    torch.distributed.broadcast_object_list(box, src=0, group=_COORD_GROUP)
    return bool(box[0])


def join_failure(local_error) -> Optional[str]:
    """Share a per-rank evaluation failure so every rank stops together.

    Returns a description when ANY rank failed, else None. Without this, one
    rank raises, the others walk into the next collective, and the run dies
    later as an unexplained NCCL timeout instead of an evaluation failure.

    Runs on the contract's own gloo group, so contract coordination never
    interleaves with training or plugin collectives on the default group, and
    a CUDA-side failure can still be exchanged. ORDERING STILL MATTERS: this
    rescues nobody who is already blocked inside another collective, which is
    why phase_join() exists and is called BEFORE the evaluator's own.

    The join is bounded (see join_timeout_seconds). A rank that has died cannot
    join at all; the survivors then raise a timeout naming the wait rather than
    hanging until ddp_timeout.
    """
    local = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
    try:
        import torch
    except ImportError:  # no peers to join with
        return local
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local
    world = torch.distributed.get_world_size()
    if world <= 1:
        return local
    gathered = [None] * world
    try:
        torch.distributed.all_gather_object(gathered, local, group=_COORD_GROUP)
    except Exception as e:  # noqa: BLE001 - a peer that cannot join is terminal
        raise EvaluationExecutionError(
            "cross-rank evaluation failure agreement did not complete within "
            f"{join_timeout_seconds()}s ({type(e).__name__}: {e}). At least one "
            "rank did not reach the join, which usually means the process is "
            "gone (OOM-killer, segfault, node loss) rather than merely slow. "
            "That is unrecoverable in-process: let the job die and resume from "
            "the last complete checkpoint. This rank's own state was "
            f"{local or 'healthy'}."
        ) from e
    failed = [(i, g) for i, g in enumerate(gathered) if g]
    if not failed:
        return None
    return "; ".join(f"rank {i}: {g}" for i, g in failed)
