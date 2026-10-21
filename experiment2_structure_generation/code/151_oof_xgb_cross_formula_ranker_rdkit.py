#!/usr/bin/env python3

"""
XGB cross-formula ranker with RDKit chemistry features.

Input:
    rank_features_rdkit.jsonl

Compared with baseline:
    rank_features.jsonl

Added:
    molecular topology descriptors
    physicochemical descriptors

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

import xgboost as xgb



def read_jsonl(path):

    rows=[]

    with open(
        path,
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows



def build_dataframe(
    feature_rows,
    label_rows,
):

    if len(feature_rows)!=len(label_rows):

        raise RuntimeError(
            "Feature/label mismatch"
        )


    records=[]

    feature_names=None


    for frow,lrow in zip(
        feature_rows,
        label_rows
    ):

        if (
            frow["sample_id"]
            !=
            lrow["sample_id"]
            or
            frow["candidate_id"]
            !=
            lrow["candidate_id"]
        ):
            raise RuntimeError(
                "alignment mismatch"
            )


        feats=frow["features"]


        if feature_names is None:

            feature_names=list(
                feats.keys()
            )


        rec={

            "sample_id":
                frow["sample_id"],

            "candidate_id":
                frow["candidate_id"],

            "smiles":
                frow["smiles"],

            "formula":
                frow.get("formula"),

            "formula_rank":
                frow.get(
                    "formula_rank"
                ),

            "branch_route_rank":
                frow.get(
                    "branch_route_rank"
                ),

            "label":
                int(
                    lrow["label"]
                ),

            "split_group":
                lrow["split_group"],

        }


        for name in feature_names:

            rec[name]=float(
                feats[name]
            )


        records.append(rec)



    df=pd.DataFrame(records)


    return (
        df,
        feature_names
    )



def prepare_rank_training(
    df,
    feature_names,
):

    ordered=df.sort_values(
        [
            "sample_id",
            "candidate_id"
        ],
        kind="mergesort"
    ).reset_index(drop=True)


    qid,_=pd.factorize(
        ordered["sample_id"],
        sort=True
    )


    X=ordered[
        feature_names
    ].to_numpy(
        dtype=np.float32
    )


    y=ordered[
        "label"
    ].to_numpy(
        dtype=np.float32
    )


    return (
        ordered,
        X,
        y,
        qid.astype(np.int32)
    )



def fit_ranker(
    X,
    y,
    qid,
    args,
):

    model=xgb.XGBRanker(

        objective="rank:pairwise",

        eval_metric="ndcg@10",

        tree_method="hist",

        n_estimators=args.n_estimators,

        learning_rate=args.learning_rate,

        max_depth=args.max_depth,

        min_child_weight=args.min_child_weight,

        subsample=args.subsample,

        colsample_bytree=args.colsample_bytree,

        reg_lambda=args.reg_lambda,

        reg_alpha=args.reg_alpha,

        random_state=args.seed,

        n_jobs=args.n_jobs,

    )


    try:

        model.fit(
            X,
            y,
            qid=qid,
            verbose=False
        )


    except TypeError:

        group=np.bincount(qid)

        model.fit(
            X,
            y,
            group=group,
            verbose=False
        )


    return model



def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--features",
        default=
        "work/predicted_formula_multihyp_v1/"
        "cross_formula/xgb_oof_v1/"
        "rank_features_rdkit.jsonl"
    )


    parser.add_argument(
        "--labels",
        default=
        "work/predicted_formula_multihyp_v1/"
        "cross_formula/xgb_oof_v1/"
        "rank_labels.jsonl"
    )


    parser.add_argument(
        "--samples",
        default=
        "work/predicted_formula_multihyp_v1/"
        "cross_formula/xgb_oof_v1/"
        "sample_manifest.jsonl"
    )


    parser.add_argument(
        "--output-dir",
        default=
        "work/predicted_formula_multihyp_v1/"
        "cross_formula/xgb_oof_v1/"
        "rdkit_ranker_v1"
    )


    parser.add_argument(
        "--n-splits",
        type=int,
        default=5
    )


    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500
    )


    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05
    )


    parser.add_argument(
        "--max-depth",
        type=int,
        default=6
    )


    parser.add_argument(
        "--min-child-weight",
        type=float,
        default=1
    )


    parser.add_argument(
        "--subsample",
        type=float,
        default=0.85
    )


    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.85
    )


    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=2.0
    )


    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0
    )


    parser.add_argument(
        "--n-jobs",
        type=int,
        default=16
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=20260821
    )


    args=parser.parse_args()


    outdir=Path(
        args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )


    feature_rows=read_jsonl(
        args.features
    )

    label_rows=read_jsonl(
        args.labels
    )

    sample_rows=read_jsonl(
        args.samples
    )


    df,feature_names=build_dataframe(
        feature_rows,
        label_rows
    )


    print(
        "Number of features:",
        len(feature_names)
    )

    print(
        "RDKit features enabled:",
        any(
            x.startswith("mol_")
            or
            x in [
                "ring_count",
                "tpsa",
                "fraction_csp3"
            ]
            for x in feature_names
        )
    )


    sample_df=pd.DataFrame(
        sample_rows
    )


    splitter=GroupKFold(
        n_splits=args.n_splits
    )


    sample_ids=sample_df[
        "sample_id"
    ].to_numpy()


    groups=sample_df[
        "split_group"
    ].to_numpy()


    oof=np.zeros(
        len(df)
    )


    sample_fold={}


    for fold,(tr,va) in enumerate(
        splitter.split(
            np.zeros(
                len(sample_df)
            ),
            groups=groups
        ),
        1
    ):


        train_samples=set(
            sample_ids[tr]
        )

        val_samples=set(
            sample_ids[va]
        )


        for sid in val_samples:
            sample_fold[sid]=fold



        train_df=df[
            df.sample_id.isin(
                train_samples
            )
        ]


        val_df=df[
            df.sample_id.isin(
                val_samples
            )
        ]


        train_order,X,y,qid=prepare_rank_training(
            train_df,
            feature_names
        )


        model=fit_ranker(
            X,
            y,
            qid,
            args
        )


        Xv=val_df[
            feature_names
        ].to_numpy(
            dtype=np.float32
        )


        oof[
            val_df.index
        ]=model.predict(Xv)


        print(
            f"Fold {fold} finished"
        )



    df["oof_score"]=oof


    output=outdir/"oof_candidate_scores.jsonl"


    with output.open(
        "w",
        encoding="utf-8"
    ) as f:

        for r in df.itertuples():

            f.write(
                json.dumps(
                    {
                        "sample_id":r.sample_id,
                        "candidate_id":r.candidate_id,
                        "smiles":r.smiles,
                        "oof_score":float(r.oof_score),
                        "label":int(r.label)
                    },
                    ensure_ascii=False
                )
                +"\n"
            )


    print()
    print("="*80)
    print("DONE")
    print("Output:",output)
    print("Feature count:",len(feature_names))



if __name__=="__main__":
    main()