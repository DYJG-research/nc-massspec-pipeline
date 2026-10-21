#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_rows(path):
    text = Path(path).read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    if text.startswith("["):
        return json.loads(text)

    return [
        json.loads(x)
        for x in text.splitlines()
        if x.strip()
    ]


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--predictions",
        required=True
    )

    ap.add_argument(
        "--massspecgym",
        required=True
    )

    ap.add_argument(
        "--output",
        required=True
    )

    args = ap.parse_args()

    mass_rows = load_rows(
        args.massspecgym
    )

    truth = {
        str(x["identifier"]):
            str(
                x["molecule"]["smiles"]
            )
        for x in mass_rows
    }

    pred_rows = load_rows(
        args.predictions
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    missing = []

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in pred_rows:

            sid = str(
                row["identifier"]
            )

            if sid not in truth:
                missing.append(sid)
                continue

            out = {
                **row,
                "true_smiles":
                    truth[sid],
            }

            f.write(
                json.dumps(
                    out,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        "prediction rows:",
        len(pred_rows)
    )

    print(
        "missing truth:",
        len(missing)
    )

    if missing:
        raise RuntimeError(
            f"Missing truth for "
            f"{len(missing)} samples"
        )


if __name__ == "__main__":
    main()
