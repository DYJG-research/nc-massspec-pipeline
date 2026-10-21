#!/usr/bin/env python3

import json
import argparse
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter


API_URL = "http://127.0.0.1:8000/v1/chat/completions"


SYSTEM_PROMPT = """
你是一名LC-MS/MS中性分子式候选纠错专家。

前置XGBoost已经对候选进行了全局排序。
你的任务不是重新做全局排序，而是判断是否有足够证据用XGB Rank2覆盖XGB Rank1。

要求：

1. 只能基于给定 precursor、MS/MS 和候选化学特征判断。
2. XGB Rank1 和 Rank2 的身份以及margin都是推理阶段可获得的信息，不是真实标签。
3. 如果证据不足、两个候选都可疑或无法明确支持 Rank2，优先 KEEP。
4. 只有当 Rank2 的综合化学与谱图证据明确优于 Rank1 时才 SWAP。
5. 不得查询数据库，不得生成新候选，不输出推理过程。
6. 最终只输出严格JSON：

{"decision":"KEEP"}

或

{"decision":"SWAP"}
""".strip()



def build_user_prompt(x):

    peaks = []

    for mz, inten in x["prompt_peaks"]:
        peaks.append(
            f"{mz:.4f}:{inten:.1f}"
        )

    r1 = x["shortlist"][0]
    r2 = x["shortlist"][1]


    prompt = f"""
Precursor m/z: {x['precursor_mz']:.6f}

Polarity: {x.get('polarity','unknown')}

MS2 peaks (m/z:relative_intensity, max=100):

{", ".join(peaks)}


XGB Rank1

Formula: {r1['formula']}
BestAdduct: {r1.get('best_adduct')}
ppm: {r1.get('ppm_error')}
DBE: {r1.get('dbe')}
Support: {r1.get('support')}


XGB Rank2

Formula: {r2['formula']}
BestAdduct: {r2.get('best_adduct')}
ppm: {r2.get('ppm_error')}
DBE: {r2.get('dbe')}
Support: {r2.get('support')}


请判断是否应该用 Rank2 覆盖 Rank1。

证据不足时 KEEP。

只输出严格JSON。
"""

    return prompt.strip()



def call_one(item):

    payload = {

        # swift deploy 返回的模型名
        # 如果不知道，可以先 curl /v1/models 查看
        "model": "Qwen3-14B",

        "messages": [

            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },

            {
                "role":"user",
                "content":build_user_prompt(item)
            }

        ],

        "temperature":0,

        "max_tokens":32,

        # 关键：关闭Qwen3 thinking
        "chat_template_kwargs":{
            "enable_thinking":False
        }

    }


    r=requests.post(
        API_URL,
        json=payload,
        timeout=180
    )


    result = r.json()


    if "choices" not in result:

        print("===== API ERROR =====")
        print(result)

        return {
            "identifier": item["identifier"],
            "decision": "KEEP",
            "raw": json.dumps(
                result,
                ensure_ascii=False
            )
        }


    text = result["choices"][0]["message"]["content"]


    decision="KEEP"


    # 只解析最终JSON
    try:

        obj=json.loads(text)

        decision=obj.get(
            "decision",
            "KEEP"
        )

    except:

        if "SWAP" in text:
            decision="SWAP"



    return {

        "identifier":
            item["identifier"],

        "decision":
            decision,

        "raw":
            text

    }



def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=32
    )


    args=parser.parse_args()


    data=[]


    with open(
        args.input,
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                data.append(
                    json.loads(line)
                )


    print(
        "Samples:",
        len(data)
    )


    fout=open(
        args.output,
        "w",
        encoding="utf-8"
    )


    counter=Counter()


    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:


        futures=[

            pool.submit(
                call_one,
                x
            )

            for x in data

        ]


        for future in tqdm(
            as_completed(futures),
            total=len(futures)
        ):

            result=future.result()

            counter[
                result["decision"]
            ] += 1


            fout.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )+"\n"
            )

            fout.flush()



    fout.close()


    print("====================")
    print(counter)
    print("====================")



if __name__=="__main__":
    main()