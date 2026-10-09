import os

from app.core.config import settings
from app.services.generation import _generate, _load_model


SMOKE_PROMPT = """Return one JSON object only with this shape:
{"claims":[{"text":"one factual sentence","evidence_ids":["food-smoke-test"]}]}.
Use only the following provided evidence and cite its ID exactly.
If you can answer, return one claim; otherwise return {"claims":[]}.

Question: What protein value does the test record state?

Evidence:
[food-smoke-test]
Nutrition record: Test oats contain 5 g protein per 100 g serving.
"""


def main() -> int:
    settings.llm_offline = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"Configured model: {settings.llm_model_id}")
    print("Offline mode: enabled")
    try:
        _load_model()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Model loading: failed ({error})")
        return 1
    print("Model loading: succeeded")

    try:
        generated = _generate(SMOKE_PROMPT)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Generation: failed ({error})")
        return 1

    if not generated.claims:
        print("Generation: completed, but returned no claims")
        return 1

    cited_test_evidence = sum(
        "food-smoke-test" in claim.evidence_ids
        for claim in generated.claims
    )
    print("Generation: succeeded")
    print(f"Claims returned: {len(generated.claims)}")
    print(f"Claims citing the fixed test evidence: {cited_test_evidence}")
    print("Claim verification: not run by this raw-generation smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
