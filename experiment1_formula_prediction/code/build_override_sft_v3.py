#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Override SFT v3 dataset

Goal:
Train LLM as XGBoost Top1 verifier.

Decision:
KEEP:
    trust XGB Rank1

SWAP:
    replace Rank1 with Rank2


Changes from v2:
1. Balance SWAP / KEEP
2. Reduce replay interference
3. Keep strict no-gold-leakage prompt
"""


import json
import random
import argparse
from pathlib import Path



SYSTEM_PROMPT = r"""你是一名LC-MS/MS中性分子式候选纠错专家。

前置XGBoost已经对候选进行了排序。
你的任务不是重新生成候选，而是判断当前Rank1是否需要被Rank2覆盖。

请综合考虑：
1. precursor质量误差
2. 元素组成合理性
3. DBE合理性
4. MS/MS碎片支持
5. 候选整体化学合理性

如果Rank2在多个独立证据上综合优于Rank1，可以选择SWAP。

如果两个候选证据接近，或者Rank2存在明显化学冲突，应选择KEEP。

不得查询数据库。
不得生成新的候选。
不得输出推理过程。

最终只输出严格JSON：

{"decision":"KEEP"}

或

{"decision":"SWAP"}
"""


def load_jsonl(path):

    data=[]

    with open(path,encoding="utf-8") as f:

        for line in f:

            if line.strip():

                data.append(
                    json.loads(line)
                )

    return data



def get_decision(item):

    try:

        target=json.loads(
            item["messages"][-1]["content"]
        )

        return target.get("decision")

    except:

        return None



def replace_system(item):

    item=json.loads(
        json.dumps(item,ensure_ascii=False)
    )

    item["messages"][0]["content"]=SYSTEM_PROMPT

    return item



def build():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        default="./outputs/override_sft_v2_train_mixed.jsonl"
    )

    parser.add_argument(
        "--dev-input",
        default="./outputs/override_sft_v2_dev_mixed.jsonl"
    )

    parser.add_argument(
        "--replay-jsonl",
        default="./outputs/formula_nothink_sft_v4_train10k.jsonl"
    )

    parser.add_argument(
        "--swap-num",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--keep-num",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--replay-train",
        type=int,
        default=300
    )

    parser.add_argument(
        "--replay-dev",
        type=int,
        default=50
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260821
    )

    parser.add_argument(
        "--output-prefix",
        default="./outputs/override_sft_v3"
    )


    args=parser.parse_args()


    random.seed(args.seed)



    # =========================
    # load override data
    # =========================

    train_data=load_jsonl(args.input)


    swap=[]
    keep=[]


    for x in train_data:

        d=get_decision(x)

        if d=="SWAP":

            swap.append(x)

        elif d=="KEEP":

            keep.append(x)



    print("original:")
    print("SWAP:",len(swap))
    print("KEEP:",len(keep))



    # =========================
    # SWAP sampling
    # =========================


    if len(swap)>=args.swap_num:

        swap_sel=random.sample(
            swap,
            args.swap_num
        )

    else:

        swap_sel=swap.copy()

        print(
            "[WARN] SWAP shortage:",
            len(swap_sel)
        )



    # =========================
    # KEEP sampling
    # =========================


    if len(keep)>=args.keep_num:

        keep_sel=random.sample(
            keep,
            args.keep_num
        )

    else:

        keep_sel=keep.copy()



    override_train=(
        swap_sel+
        keep_sel
    )


    random.shuffle(
        override_train
    )



    # =========================
    # replay
    # =========================


    replay_all=load_jsonl(
        args.replay_jsonl
    )


    replay_train=random.sample(
        replay_all,
        min(
            args.replay_train,
            len(replay_all)
        )
    )



    # replace system prompt
    final_train=[]


    for x in override_train:

        final_train.append(
            replace_system(x)
        )


    final_train.extend(
        replay_train
    )


    random.shuffle(
        final_train
    )



    # =========================
    # dev
    # =========================


    dev_data=load_jsonl(
        args.dev_input
    )


    swap_dev=[
        x for x in dev_data
        if get_decision(x)=="SWAP"
    ]

    keep_dev=[
        x for x in dev_data
        if get_decision(x)=="KEEP"
    ]



    dev_override=[]


    dev_override.extend(
        swap_dev[:len(swap_dev)//2]
    )


    dev_override.extend(
        keep_dev[:len(keep_dev)//2]
    )


    replay_dev=random.sample(
        replay_all,
        min(
            args.replay_dev,
            len(replay_all)
        )
    )


    final_dev=[]


    for x in dev_override:

        final_dev.append(
            replace_system(x)
        )


    final_dev.extend(
        replay_dev
    )


    random.shuffle(
        final_dev
    )



    # =========================
    # save
    # =========================


    out_prefix=Path(
        args.output_prefix
    )


    out_prefix.parent.mkdir(
        exist_ok=True
    )



    train_path=str(
        out_prefix
    )+"_train_mixed.jsonl"


    dev_path=str(
        out_prefix
    )+"_dev_mixed.jsonl"


    summary_path=str(
        out_prefix
    )+"_summary.json"



    with open(
        train_path,
        "w",
        encoding="utf-8"
    ) as f:

        for x in final_train:

            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False
                )+"\n"
            )



    with open(
        dev_path,
        "w",
        encoding="utf-8"
    ) as f:

        for x in final_dev:

            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False
                )+"\n"
            )



    summary={

        "source":
            args.input,

        "selection":{

            "swap_selected":
                len(swap_sel),

            "keep_selected":
                len(keep_sel),

            "replay_train":
                len(replay_train),

            "replay_dev":
                len(replay_dev)
        },

        "train_total":
            len(final_train),

        "dev_total":
            len(final_dev),

        "paths":{

            "train":
                train_path,

            "dev":
                dev_path
        }

    }



    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )



    print("\nDONE")

    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ))



if __name__=="__main__":

    build()