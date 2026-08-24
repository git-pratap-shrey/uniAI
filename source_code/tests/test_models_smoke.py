"""
test_models_smoke.py
────────────────────
Live connectivity smoke test for every model role configured in CONFIG.

Tests each role by calling models.py abstractions (NOT provider SDKs directly)
with a minimal "Hi" message and verifying a non-empty, non-error response.

Run:
    python source_code/tests/test_models_smoke.py

Each test prints PASS / FAIL with the first 80 chars of the response so you
can quickly see which providers are reachable.

Roles tested:
  • chat     → ACTIVE_CHAT_MODEL provider/model
  • router   → ROUTER_CONFIG provider/model  (used by keyword_map + RAG router)
  • embed    → EMBEDDING_CONFIG provider/model
"""

import os
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from source_code.config import CONFIG
import source_code.models as models

# ── Helpers ───────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

HI_PROMPT = "Hi! Just reply with: Hello, I'm working."

results: list[dict] = []


def run_test(label: str, fn, *args, **kwargs) -> bool:
    """Run a single smoke check, print result, return True on pass."""
    print(f"\n{'─'*60}")
    print(f"{BOLD}TEST:{RESET} {label}")
    t0 = time.time()
    try:
        response = fn(*args, **kwargs)
        elapsed = time.time() - t0

        if not response:
            raise ValueError("Empty response returned")
        if str(response).startswith("⚠"):
            raise ValueError(f"Provider error: {response}")

        preview = str(response)[:80].replace("\n", " ")
        print(f"{GREEN}PASS{RESET} ({elapsed:.1f}s)  → \"{preview}\"")
        results.append({"label": label, "status": "PASS", "elapsed": elapsed})
        return True

    except Exception as e:
        elapsed = time.time() - t0
        print(f"{RED}FAIL{RESET} ({elapsed:.1f}s)  → {e}")
        results.append({"label": label, "status": "FAIL", "error": str(e), "elapsed": elapsed})
        return False


# ── Test definitions ──────────────────────────────────────────────────────────

def test_chat():
    """Primary chat model (ACTIVE_CHAT_MODEL)."""
    provider = CONFIG["providers"]["chat"]
    model    = CONFIG["model"]["model"]
    return run_test(
        f"chat  |  provider={provider}  model={model}",
        models.chat,
        HI_PROMPT,
        provider=provider,
        model=model,
    )


def test_router():
    """Router model (used by generate_keyword_map and RAG routing)."""
    provider = CONFIG["providers"]["router"]
    model    = CONFIG["providers"]["router_model"]
    return run_test(
        f"router  |  provider={provider}  model={model}",
        models.chat,
        HI_PROMPT,
        provider=provider,
        model=model,
    )


def test_embed():
    """Embedding model."""
    provider = CONFIG["providers"]["embedding"]
    model    = CONFIG["providers"]["embedding_model"]

    def _embed():
        vectors = models.embed([HI_PROMPT], provider=provider, model=model)
        if not vectors or not vectors[0]:
            raise ValueError("No embedding vector returned")
        return f"vector dim={len(vectors[0])}  first_3={[round(v, 4) for v in vectors[0][:3]]}"

    return run_test(
        f"embed  |  provider={provider}  model={model}",
        _embed,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'═'*60}")
    print("  uniAI  models.py  LIVE  SMOKE  TEST")
    print(f"{'═'*60}{RESET}")

    tests = [test_chat, test_router, test_embed]
    for t in tests:
        t()

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["status"] == "PASS")
    total  = len(results)

    print(f"\n{'═'*60}")
    print(f"{BOLD}SUMMARY: {passed}/{total} passed{RESET}")
    for r in results:
        status_str = f"{GREEN}PASS{RESET}" if r["status"] == "PASS" else f"{RED}FAIL{RESET}"
        elapsed    = f"({r['elapsed']:.1f}s)"
        label      = r["label"]
        print(f"  {status_str}  {elapsed:<8}  {label}")
        if r["status"] == "FAIL":
            print(f"           {YELLOW}↳ {r.get('error', '')}{RESET}")
    print()

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
