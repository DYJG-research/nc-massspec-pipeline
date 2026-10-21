import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--full", required=True)
parser.add_argument("--pred", required=True)
parser.add_argument("--output", required=True)

args = parser.parse_args()


# load LLM decisions

decisions={}

with open(args.pred) as f:
    for line in f:
        x=json.loads(line)

        decisions[x["identifier"]] = x["decision"]


count={
    "KEEP":0,
    "SWAP":0,
    "XGB_ONLY":0
}


out=[]


with open(args.full) as f:

    for line in f:

        x=json.loads(line)

        ident=x["identifier"]


        # original top1
        original=x["shortlist"][0]


        if ident in decisions:

            if decisions[ident]=="SWAP":

                new=x["shortlist"][1]

                x["final_formula"]=new["formula"]
                x["final_adduct"]=new["best_adduct"]

                count["SWAP"]+=1


            else:

                x["final_formula"]=original["formula"]
                x["final_adduct"]=original["best_adduct"]

                count["KEEP"]+=1


        else:

            x["final_formula"]=original["formula"]
            x["final_adduct"]=original["best_adduct"]

            count["XGB_ONLY"]+=1


        out.append(x)



with open(args.output,"w") as f:

    for x in out:
        f.write(json.dumps(x,ensure_ascii=False)+"\n")


print(count)
print("total",len(out))