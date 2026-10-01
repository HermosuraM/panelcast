"""Command line entry point: `panelcast run all`, `panelcast run silver resolve`, `panelcast fetch-reference`."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from panelcast.config import Settings
from panelcast.context import Context
from panelcast.paths import set_project_root

log = logging.getLogger("panelcast")


def _scaled(scale: float | None) -> dict:
    if not scale:
        return {}
    base = Settings().sim["panel"]["n_panelists"]
    return {"panel": {"n_panelists": max(200, int(base * scale))}}


def main(argv: list[str] | None = None) -> int:
    from panelcast.stages import STAGES, run_stages

    parser = argparse.ArgumentParser(prog="panelcast", description=__doc__)
    parser.add_argument("--root", help="project root (folder with conf/ and data/)")
    parser.add_argument("--env", choices=["local", "databricks"], help="override platform detection")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run pipeline stages")
    run.add_argument("stages", nargs="+", choices=[*STAGES, "all"])
    run.add_argument("--scale", type=float, help="multiply the simulated panel size (e.g. 0.1 for a quick run)")

    fetch = sub.add_parser("fetch-reference", help="download SEC EDGAR revenue + Wikipedia pageviews")
    fetch.add_argument("--refresh", action="store_true", help="re-download cached EDGAR JSON")
    fetch.add_argument("--skip-wiki", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("py4j", "urllib3", "httpx", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.root:
        set_project_root(args.root)

    if args.command == "fetch-reference":
        from panelcast.reference.build import fetch_reference

        fetch_reference(Settings(env=args.env), refresh=args.refresh, skip_wiki=args.skip_wiki)
        return 0

    settings = Settings(sim_overrides=_scaled(args.scale), env=args.env)
    ctx = Context(settings)
    t0 = time.time()
    run_stages(ctx, args.stages)
    log.info("done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
