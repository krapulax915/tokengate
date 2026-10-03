"""Live smoke test against a REAL model through a running DietGate (your own API key).

Checks the things the mock cannot: the real wire format, token accounting for both
non-stream and stream requests, cost > 0, and measured gateway overhead. It makes
a handful of tiny requests (default 2 non-stream + 2 stream), so the cost is a few cents at most.

  1) put the real model in config/models.yaml (see config/models.live.example.yaml),
     with prices from the provider's pricing page
  2) export the provider key (OPENAI_API_KEY / ANTHROPIC_API_KEY) and start the gateway
  3) python scripts/live_smoke.py --model <model-id> --api-key <gateway key> --admin-key <admin key>
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid

import httpx

PROMPT = "Reply with exactly one word: pong"  # a unique nonce is appended per request (no cache hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True, help="model id exactly as listed in config/models.yaml")
    ap.add_argument("--api-key", required=True, help="DietGate data-plane key")
    ap.add_argument("--admin-key", default="", help="needed to read token counts of streamed requests")
    ap.add_argument("--n", type=int, default=2, help="requests per mode")
    ap.add_argument(
        "--max-tokens", type=int, default=1024,
        help="reasoning models (e.g. GLM-5.3-Flash) spend max_tokens on hidden thinking first; "
             "too small a value gives an empty answer",
    )
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"}
    admin = {"X-Admin-Key": args.admin_key} if args.admin_key else {}
    failures: list[str] = []
    ids: list[tuple[str, str]] = []

    with httpx.Client(base_url=args.url, timeout=60.0) as c:
        listed = {m["id"] for m in c.get("/v1/models", headers=headers).json().get("data", [])}
        if args.model not in listed:
            print(f"model {args.model!r} is not routable (missing key, or not in models.yaml). Listed: {sorted(listed)}")
            return 2

        for mode in ("non-stream", "stream"):
            for i in range(args.n):
                body = {
                    "model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
                    "messages": [{"role": "user", "content": f"{PROMPT} [{uuid.uuid4().hex[:8]}]"}],
                    "stream": mode == "stream",
                }
                finish = ""
                if mode == "stream":
                    with c.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
                        text, rid = "", r.headers.get("x-dg-request-id", "")
                        used_model = r.headers.get("x-dg-model", "")
                        for line in r.iter_lines():
                            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                                try:
                                    ch = json.loads(line[6:])
                                except ValueError:
                                    continue
                                text += ((ch.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                        status = r.status_code
                    cost = overhead = "(see /admin/requests)"
                else:
                    r = c.post("/v1/chat/completions", json=body, headers=headers)
                    status, rid = r.status_code, r.headers.get("x-dg-request-id", "")
                    used_model = r.headers.get("x-dg-model", "")
                    choice = (r.json().get("choices") or [{}])[0] if status == 200 else {}
                    text = (choice.get("message") or {}).get("content", "") if status == 200 else r.text[:200]
                    finish = choice.get("finish_reason") or ""
                    cost, overhead = r.headers.get("x-dg-cost-usd"), r.headers.get("x-dg-overhead-ms")
                    if status == 200 and not float(cost or 0) > 0:
                        failures.append(f"{mode} #{i}: cost is 0 - check the model prices in models.yaml")
                print(f"[{mode} #{i}] HTTP {status} model={used_model} text={text.strip()[:40]!r} "
                      f"cost_usd={cost} overhead_ms={overhead}")
                if status == 200 and mode == "non-stream" and used_model != args.model:
                    # a failed upstream call silently falls back to another candidate: not a pass
                    failures.append(f"{mode} #{i}: served by {used_model!r}, not {args.model!r} (upstream failed, fallback used?)")
                if status != 200 or not text.strip():
                    hint = " (finish_reason=length: raise --max-tokens, reasoning models think first)" if finish == "length" else ""
                    failures.append(f"{mode} #{i}: HTTP {status} / empty text{hint}")
                ids.append((mode, rid))

        if admin:
            rows = {r["id"]: r for r in c.get("/admin/requests?limit=100", headers=admin).json().get("requests", [])}
            for mode, rid in ids:
                row = rows.get(rid)
                if not row:
                    continue
                print(f"  {mode:10s} {rid[:12]} model={row.get('chosen_model')} prompt_tokens={row.get('prompt_tokens')} "
                      f"completion_tokens={row.get('completion_tokens')} cost_usd={row.get('cost_usd')} "
                      f"overhead_ms={row.get('overhead_ms')} cache_hit={row.get('cache_hit')}")
                if row.get("chosen_model") != args.model:
                    failures.append(f"{mode} {rid[:12]}: served by {row.get('chosen_model')!r}, not {args.model!r} (fallback used?)")
                if row.get("cache_hit"):
                    failures.append(f"{mode} {rid[:12]}: unexpected cache hit (cost 0, no upstream call)")
                if not float(row.get("cost_usd") or 0) > 0:
                    failures.append(f"{mode} {rid[:12]}: cost is 0 - check the model prices in models.yaml")
                if not (row.get("prompt_tokens") and row.get("completion_tokens")):
                    failures.append(f"{mode} {rid[:12]}: token counts are zero - usage accounting broken")
        else:
            print("(pass --admin-key to verify token counts of streamed requests)")

    if failures:
        print("\nFAILED:\n - " + "\n - ".join(failures))
        return 1
    print("\nOK: real-provider wire format, usage and cost accounting look sane.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
