"""Command line interface."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from rich.console import Console

from . import __version__
from .config import apply_cli_overrides, load_config
from .diagnostics import Severity
from .engine import check_paths
from .reporters import FORMATS, render_github, render_json, render_sarif, render_terminal
from .reporters.vram import render_json as render_vram_json
from .reporters.vram import render_terminal as render_vram_terminal
from .rules import RULES, all_rules
from .vram.types import OptimizerKind, PrecisionMode, Sharding

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2

RETIRED_RULES = {
    "TG009": (
        "TG009 is deliberately not implemented: in-place operations on tensors "
        "needed for backward already raise a precise error from PyTorch itself, "
        "naming the tensor and its version counter, so a pre-flight check adds "
        "nothing useful."
    ),
}

SUBCOMMANDS = {"check", "explain", "rules", "estimate", "gpus"}


def _split_codes(values: Optional[Sequence[str]]) -> List[str]:
    """Accept both ``--ignore TG001,TG002`` and ``--ignore TG001 --ignore TG002``."""
    out: List[str] = []
    for value in values or []:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torch-preflight",
        description="Static analyzer for PyTorch training code.",
    )
    parser.add_argument("--version", action="version", version=f"torch-preflight {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    check = subparsers.add_parser("check", help="analyze files for PyTorch anti-patterns")
    check.add_argument("paths", nargs="*", default=["."], help="files or directories")
    check.add_argument(
        "-f", "--format", choices=FORMATS, default="terminal", help="output format"
    )
    check.add_argument("--fix", action="store_true", help="apply autofixes in place")
    check.add_argument(
        "--diff", action="store_true", help="show what --fix would change, without writing"
    )
    check.add_argument("--select", action="append", help="only run these rules")
    check.add_argument("--ignore", action="append", help="skip these rules")
    check.add_argument("--exclude", action="append", help="path patterns to skip")
    check.add_argument(
        "--fail-on",
        choices=[s.value for s in Severity],
        help="minimum severity that makes the run fail (default: error)",
    )
    check.add_argument("-j", "--jobs", type=int,
                       help="worker processes (default: CPU count; 1 disables parallelism)")
    check.add_argument("--target-gpu",
                       help="target GPU for the TG010 projected-OOM gate")
    check.add_argument("--config", type=Path, help="path to a config file")
    check.add_argument("--no-color", action="store_true", help="disable coloured output")
    check.add_argument("-q", "--quiet", action="store_true", help="only print the summary")

    explain = subparsers.add_parser("explain", help="show the full write-up for a rule")
    explain.add_argument("code", help="rule code, e.g. TG001")

    subparsers.add_parser("rules", help="list all available rules")

    est = subparsers.add_parser(
        "estimate", help="project peak VRAM for a training script before you launch it"
    )
    est.add_argument("path", nargs="?", help="training script to read the config from")
    est.add_argument("--gpu", help="target GPU or cloud instance (see `torch-preflight gpus`)")
    est.add_argument("--gpu-memory", help="explicit capacity for unlisted hardware, e.g. 48GiB")
    est.add_argument("--model",
                     help="architecture name (llama-2-7b) or entry point (pkg.mod:factory)")
    est.add_argument("--model-args", action="append", metavar="KEY=VALUE",
                     help="constructor arguments for a --model entry point")
    est.add_argument("--params", help="parameter count when the model is unknown, e.g. 7B")
    est.add_argument("--online", action="store_true",
                     help="allow Hugging Face hub lookup for unknown architectures")
    est.add_argument("--batch-size", type=int, help="override per-device micro-batch")
    est.add_argument("--seq-len", type=int, help="override sequence length")
    est.add_argument("--image-size", type=int, help="override image resolution")
    est.add_argument("--dtype", choices=[p.value for p in PrecisionMode],
                     help="override precision mode")
    est.add_argument("--optimizer", choices=[o.value for o in OptimizerKind],
                     help="override optimizer")
    est.add_argument("--world-size", type=int, help="number of ranks")
    est.add_argument("--sharding", choices=[s.value for s in Sharding],
                     help="override sharding strategy")
    est.add_argument("--checkpointing", action="store_true",
                     help="assume gradient checkpointing is on")
    est.add_argument("--flash", action="store_true", help="assume flash attention / SDPA")
    est.add_argument("--inference", action="store_true", help="inference only, no backward")
    est.add_argument("--generate", action="store_true",
                     help="autoregressive decoding, which holds a KV cache")
    est.add_argument("--max-context", type=int,
                     help="total tokens the KV cache must hold (prompt + generated)")
    est.add_argument("-f", "--format", choices=("terminal", "json"), default="terminal")

    gpus = subparsers.add_parser("gpus", help="list known GPUs and cloud instances")
    gpus.add_argument("--instances", action="store_true", help="show cloud instances too")
    return parser


def _run_check(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in (args.paths or ["."])]
    for path in paths:
        if not path.exists():
            print(f"torch-preflight: path does not exist: {path}", file=sys.stderr)
            return EXIT_ERROR

    try:
        cfg = load_config(paths[0].resolve(), args.config)
        cfg = apply_cli_overrides(
            cfg,
            select=_split_codes(args.select),
            ignore=_split_codes(args.ignore),
            exclude=args.exclude,
            fail_on=args.fail_on,
            target_gpu=args.target_gpu,
        )
    except (OSError, ValueError) as exc:
        print(f"torch-preflight: {exc}", file=sys.stderr)
        return EXIT_ERROR

    unknown = (cfg.select or set()) | cfg.ignore
    for code in sorted(unknown - set(RULES)):
        print(f"torch-preflight: warning: unknown rule code {code}", file=sys.stderr)

    want_fix = args.fix or args.diff
    result = check_paths(
        paths, cfg, fix=want_fix, write=args.fix and not args.diff, jobs=args.jobs
    )

    if args.diff:
        _print_diff(result)
    elif args.format == "terminal":
        console = Console(no_color=args.no_color, soft_wrap=True)
        render_terminal(result, console, show_source=not args.quiet)
    elif args.format == "json":
        print(render_json(result))
    elif args.format == "github":
        output = render_github(result)
        if output:
            print(output)
    elif args.format == "sarif":
        print(render_sarif(result))

    return EXIT_FINDINGS if result.should_fail(cfg.fail_on) else EXIT_OK


def _print_diff(result) -> None:
    for file in result.files:
        if file.new_source is None:
            continue
        original = Path(file.path).read_text(encoding="utf-8").splitlines(keepends=True)
        updated = file.new_source.splitlines(keepends=True)
        # Show paths relative to the cwd so the output reads like `git diff`.
        try:
            label = Path(file.path).resolve().relative_to(Path.cwd()).as_posix()
        except ValueError:
            label = Path(file.path).as_posix().lstrip("/")
        sys.stdout.writelines(
            difflib.unified_diff(
                original, updated, fromfile=f"a/{label}", tofile=f"b/{label}"
            )
        )


def _run_explain(code: str) -> int:
    normalized_code = code.upper()
    rule = RULES.get(normalized_code)

    if rule is None:
        retired_reason = RETIRED_RULES.get(normalized_code)
        if retired_reason:
            console = Console()
            console.print(f"torch-preflight: {retired_reason}")
            return EXIT_OK

        # Prefer prefix matches first. This handles inputs like TG01,
        # where the user likely started typing a real rule code.
        prefix_matches = [
            rule_code for rule_code in RULES if rule_code.startswith(normalized_code)
        ]

        if prefix_matches:
            suggestion = prefix_matches[0]
        else:
            # Ask difflib for several candidates, then prefer the
            # lowest-numbered rule when scores tie. This avoids cases
            # like TG02 -> TG012 instead of TG002.
            suggestions = difflib.get_close_matches(normalized_code, RULES, n=3)

            if suggestions:
                suggestion = min(
                    suggestions,
                    key=lambda rule_code: int(rule_code[2:]) if rule_code[2:].isdigit() else float('inf'),
                )
            else:
                suggestion = None

        print(f"torch-preflight: unknown rule {code}", file=sys.stderr)

        if suggestion:
            print(
                f"help: did you mean {suggestion}? "
                f"Run `torch-preflight rules` to see all {len(RULES)}.",
                file=sys.stderr,
            )
        else:
            print(
                f"help: run `torch-preflight rules` to see all {len(RULES)}.",
                file=sys.stderr,
            )

        return EXIT_ERROR

    console = Console()
    console.print(f"[bold]{rule.code}[/bold]  {rule.summary}")
    console.print(
        f"[dim]severity:[/dim] {rule.severity.value}   "
        f"[dim]category:[/dim] {rule.category.value}   "
        f"[dim]name:[/dim] {rule.name}"
    )
    console.print()
    console.print(rule.explanation)
    return EXIT_OK


def _run_rules() -> int:
    console = Console()
    for rule in all_rules():
        console.print(
            f"[bold]{rule.code}[/bold]  [dim]{rule.severity.value:<7}[/dim] "
            f"{rule.summary}  [dim]({rule.category.value})[/dim]"
        )
    return EXIT_OK


def _parse_params(text: str) -> Optional[int]:
    """Parse ``7B`` / ``124M`` / ``6738415616`` into a parameter count."""
    value = text.strip().lower().replace(",", "").replace("_", "")
    for suffix, scale in (("b", 1e9), ("m", 1e6), ("k", 1e3)):
        if value.endswith(suffix):
            try:
                return int(float(value[:-1]) * scale)
            except ValueError:
                return None
    try:
        return int(value)
    except ValueError:
        return None


def _run_estimate(args: argparse.Namespace) -> int:
    from .vram import estimate_config, estimate_script, hardware
    from .vram.providers import static
    from .vram.types import RunConfig

    overrides = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "image_size": args.image_size,
        "precision": PrecisionMode(args.dtype) if args.dtype else None,
        "optimizer": OptimizerKind(args.optimizer) if args.optimizer else None,
        "world_size": args.world_size,
        "sharding": Sharding(args.sharding) if args.sharding else None,
        "gradient_checkpointing": True if args.checkpointing else None,
        "flash_attention": True if args.flash else None,
        "inference_only": True if (args.inference or args.generate) else None,
        "generation": True if args.generate else None,
        "max_context": args.max_context,
    }

    from .vram.providers.meta import EntryPointError, parse_model_args

    try:
        model_args = parse_model_args(args.model_args)
    except EntryPointError as exc:
        print(f"torch-preflight: {exc}", file=sys.stderr)
        return EXIT_ERROR

    gpu_memory = None
    if args.gpu_memory:
        gpu_memory = hardware.parse_memory(args.gpu_memory)
        if gpu_memory is None:
            print(f"torch-preflight: could not parse --gpu-memory {args.gpu_memory!r}",
                  file=sys.stderr)
            return EXIT_ERROR

    if not args.path and not (args.model or args.params):
        print("torch-preflight: give a script to read, or --model/--params", file=sys.stderr)
        return EXIT_ERROR

    try:
        if args.params:
            count = _parse_params(args.params)
            if count is None:
                print(f"torch-preflight: could not parse --params {args.params!r}",
                      file=sys.stderr)
                return EXIT_ERROR
            profile = static.from_param_count(count, name=args.model or "custom")
            config = RunConfig()
            for key, value in overrides.items():
                if value is not None:
                    config = config.replace(**{key: value})
                    config.sources[key] = "command line"
            report = estimate_config(
                profile, config, gpu=args.gpu, gpu_memory=gpu_memory
            )
        elif args.path:
            path = Path(args.path)
            if not path.is_file():
                print(f"torch-preflight: no such file: {path}", file=sys.stderr)
                return EXIT_ERROR
            report = estimate_script(
                str(path),
                gpu=args.gpu,
                gpu_memory=gpu_memory,
                model=args.model,
                online=args.online,
                overrides=overrides,
                model_args=model_args,
            )
        else:
            config = RunConfig()
            for key, value in overrides.items():
                if value is not None:
                    config = config.replace(**{key: value})
                    config.sources[key] = "command line"
            from .vram.providers import resolve_profile

            profile = resolve_profile(
                args.model, allow_network=args.online, config=config,
                model_args=model_args,
            )
            report = estimate_config(
                profile, config, gpu=args.gpu, gpu_memory=gpu_memory
            )
    except (OSError, ValueError) as exc:
        print(f"torch-preflight: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.format == "json":
        print(render_vram_json(report))
    else:
        render_vram_terminal(report, Console())

    return EXIT_FINDINGS if report.band.is_failure else EXIT_OK


def _run_gpus(args: argparse.Namespace) -> int:
    from .vram import hardware

    console = Console()
    console.print("[bold]GPUs[/bold]")
    for gpu in hardware.known_gpus():
        aliases = f"  [dim]({', '.join(gpu.aliases)})[/dim]" if gpu.aliases else ""
        console.print(
            f"  [bold]{gpu.key:<14}[/bold] {gpu.usable_gib:6.1f} GiB usable   "
            f"[dim]{gpu.name}[/dim]{aliases}"
        )

    if args.instances:
        console.print()
        console.print("[bold]Cloud instances[/bold]")
        for instance in hardware.known_instances():
            console.print(
                f"  [bold]{instance.key:<32}[/bold] {instance.count}x "
                f"{instance.gpu.name}  [dim]({instance.provider})[/dim]"
            )
    else:
        console.print()
        console.print("[dim]--instances also lists cloud instance types.[/dim]")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # ``torch-preflight ./src`` is shorthand for ``torch-preflight check ./src``.
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "check")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "explain":
        return _run_explain(args.code)
    if args.command == "rules":
        return _run_rules()
    if args.command == "check":
        return _run_check(args)
    if args.command == "estimate":
        return _run_estimate(args)
    if args.command == "gpus":
        return _run_gpus(args)

    parser.print_help()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
