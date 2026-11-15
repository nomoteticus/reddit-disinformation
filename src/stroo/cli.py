"""Command line interface: ``stroo <command>``.

    stroo collect              daily scrape of new submissions and comments
    stroo detect               find informal flags in the comment files
    stroo aggregate 2020       join everything and build the dashboard tables
    stroo dashboard            run the dashboard locally

Each command imports its dependencies only when it runs, so the dashboard can
be used without the scraping or embedding libraries installed.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stroo", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("collect", help="scrape new submissions and comments (needs Pushshift)")

    detect = commands.add_parser("detect", help="match flag patterns in the comment files")
    detect.add_argument("--processes", type=int, default=1, help="spaCy worker processes")

    aggregate = commands.add_parser("aggregate", help="build the UNITED and dashboard tables")
    aggregate.add_argument("year", type=int, help="calendar year to process, e.g. 2020")

    dashboard = commands.add_parser("dashboard", help="run the dashboard locally")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8050)
    dashboard.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "collect":
        from .collect import daily

        daily.run()
    elif args.command == "detect":
        from .flags import detect

        detect.run(n_process=args.processes)
    elif args.command == "aggregate":
        from . import aggregate

        aggregate.run(args.year)
    elif args.command == "dashboard":
        from .dashboard import app

        app.main(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
