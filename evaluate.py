
import argparse
import csv
import json
import time
from pathlib import Path

from docchat.agents import run_pipeline
from docchat.llm import LLMError, OllamaClient
from docchat.loader import build_chunks
from docchat.retriever import Embedder, HybridRetriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    files = [(Path(p).name, Path(p).read_bytes()) for p in args.docs]
    chunks, warnings = build_chunks(files)
    for w in warnings:
        print("warning:", w)
    if not chunks:
        raise SystemExit("No readable text in the documents.")
    try:
        embedder = Embedder()
    except Exception as exc:
        print("embedding model unavailable, using keyword-only search:", exc)
        embedder = None
    retriever = HybridRetriever(chunks, embedder)
    client = OllamaClient(model=args.model)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    rows, correct = [], 0
    for item in questions:
        start = time.time()
        try:
            result = list(run_pipeline(item["question"], retriever, client,
                                       k=args.k, verify=not args.no_verify))[-1].payload
        except LLMError as exc:
            raise SystemExit(f"Model error: {exc}")
        answer = result.answer.lower()
        if item.get("out_of_scope"):
            ok = result.relevance == "NO_MATCH"
        else:
            ok = (all(s.lower() in answer for s in item.get("must_contain", []))
                  and not any(s.lower() in answer for s in item.get("must_not_contain", [])))
        correct += ok
        verdict = (result.verification or {}).get("verdict", "off")
        rows.append({"question": item["question"], "correct": ok, "verdict": verdict,
                     "revisions": result.revisions, "seconds": round(time.time() - start, 1),
                     "answer": result.answer.replace("\n", " ")[:300]})
        print(f"{'PASS' if ok else 'FAIL'}  [{verdict:>10}]  {item['question']}")

    print(f"\nAccuracy: {correct}/{len(rows)} = {100 * correct / len(rows):.0f}%  "
          f"(model={args.model}, fact-check={'off' if args.no_verify else 'on'}, "
          f"search={retriever.mode})")
    with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Details saved to eval_results.csv")


if __name__ == "__main__":
    main()
