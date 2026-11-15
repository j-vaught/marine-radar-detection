"""Compare selected training code at two Git commits without touching experiment data.

This checks only the specified Ultralytics training prefix, selected helpers,
module-level statements, and named supporting files. It does not establish
whole-library, runtime, package, driver, or hardware equivalence.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from .audit_pairs import TRAIN_FILES

MODULE = "src/revision/ultralytics_models.py"
HELPERS = ("coco_yolo_lines", "stage_split", "_model_spec")
SUPPORT = tuple(
    sorted(
        (
            set(TRAIN_FILES["yolo"])
            | set(TRAIN_FILES["faster_rcnn"])
            | {"src/revision/storage.py", "src/faster_rcnn/dataset.py"}
        )
        - {MODULE}
    )
)
GUARD = ast.parse("""if job.get("evaluation_from"):
    from .yolo_repair import evaluate_from_parent
    return evaluate_from_parent(job)
""").body[0]
TRAIN_CALL = ast.parse("model.train(**args)").body[0]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ast_digest(node: ast.AST) -> str:
    return digest(ast.dump(node, include_attributes=False).encode())


def parts(source: str, *, remove_guard: bool) -> dict[str, str]:
    """Return AST digests; strip only the exact authorized repair guard."""
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if any(
        sum(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body) != 1
        for name in (*HELPERS, "run_job")
    ):
        raise ValueError("Each selected function must exist exactly once")
    run = copy.deepcopy(functions["run_job"])
    matches = [i for i, node in enumerate(run.body) if ast.dump(node) == ast.dump(TRAIN_CALL)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one top-level model.train(**args) statement")
    run.body = run.body[: matches[0] + 1]
    guards = [i for i, node in enumerate(run.body) if ast.dump(node) == ast.dump(GUARD)]
    if remove_guard:
        if len(guards) != 1:
            raise ValueError("Expected exactly one unchanged evaluation_from repair guard")
        del run.body[guards[0]]
    elif guards:
        raise ValueError("Baseline unexpectedly contains the repair guard")
    module_scope = ast.Module(
        body=[
            node
            for node in tree.body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ],
        type_ignores=[],
    )
    return {
        "module_scope": ast_digest(module_scope),
        **{name: ast_digest(functions[name]) for name in HELPERS},
        "run_job_through_model_train": ast_digest(run),
    }


def git_read(repo: Path, revision: str, file: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{file}"], check=True, capture_output=True
    ).stdout


def compare(repo: Path, baseline: str, candidate: str) -> dict:
    resolved = [
        subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", f"{rev}^{{commit}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for rev in (baseline, candidate)
    ]
    left, right = (
        parts(git_read(repo, rev, MODULE).decode(), remove_guard=flag)
        for rev, flag in zip(resolved, (False, True), strict=True)
    )
    checks = {
        name: {
            "baseline_sha256": left[name],
            "candidate_sha256": right[name],
            "equal": left[name] == right[name],
        }
        for name in left
    }
    files = {}
    for file in SUPPORT:
        blobs = []
        for rev in resolved:
            probe = subprocess.run(
                ["git", "-C", str(repo), "cat-file", "-e", f"{rev}:{file}"],
                capture_output=True,
                check=False,
            )
            blobs.append(git_read(repo, rev, file) if probe.returncode == 0 else None)
        a, b = blobs
        files[file] = {
            "baseline_sha256": digest(a) if a is not None else None,
            "candidate_sha256": digest(b) if b is not None else None,
            "baseline_present": a is not None,
            "candidate_present": b is not None,
            "equal": a == b,
            "interpretation": "absent in both commits; no file bytes compared"
            if a is None and b is None
            else "file bytes compared",
        }
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--no-ext-diff", *resolved, "--", MODULE, *SUPPORT],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "scoped_git_diff_sha256": digest(diff),
        "baseline_commit": resolved[0],
        "candidate_commit": resolved[1],
        "status": "pass"
        if all(x["equal"] for x in [*checks.values(), *files.values()])
        else "mismatch",
        "ast_checks": checks,
        "supporting_files": files,
        "guard_removed": "exact evaluation_from early return only from candidate run_job",
        "embedded_mask_owners": ["stage_split AST", "src/revision/torch_models.py bytes"],
        "scope": "Selected syntax-tree regions and named files only; excludes post-training evaluation, external libraries, packages, driver, hardware, and runtime behavior. No target scores accessed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.repo, args.baseline, args.candidate)
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
