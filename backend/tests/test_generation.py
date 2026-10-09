import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.core.config import settings
from app.services.constraints import ConstraintResult
from app.services.evidence import Evidence
from app.services.generation import (
    EvidenceVerificationError,
    GeneratedAnswer,
    GeneratedClaim,
    MALFORMED_MODEL_OUTPUT_WARNING,
    ModelOutputError,
    _canonical_numbers,
    _claim_verification,
    _claim_is_supported,
    _load_model,
    _generate,
    _build_prompt,
    generate_verified_answer,
)
from app.services.query_processor import process_query


def _evidence(
    evidence_id: str,
    text: str,
    *,
    kind: str = "document",
    metadata: dict[str, object] | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        text=text,
        source_id=1,
        source_title="Test evidence",
        source_url="https://example.test/source",
        source_type="guideline_snippet",
        authority_score=0.8,
        kind=kind,
        metadata=metadata or {},
    )


def test_claim_verification_requires_cited_evidence_and_matching_numbers() -> None:
    evidence = Evidence(
        evidence_id="food-1",
        text="Food: Tempeh\nServing: 100 g\nProtein: 20.3 g",
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="guideline_snippet",
        authority_score=0.7,
        kind="food",
    )
    evidence_map = {evidence.evidence_id: evidence}

    assert _claim_is_supported(
        GeneratedClaim(
            text="Tempeh contains 20.3 g protein per 100 g.",
            evidence_ids=["food-1"],
        ),
        evidence_map,
    )
    assert not _claim_is_supported(
        GeneratedClaim(
            text="Tempeh contains 99 g protein per 100 g.",
            evidence_ids=["food-1"],
        ),
        evidence_map,
    )
    assert not _claim_is_supported(
        GeneratedClaim(
            text="Tempeh contains 20.3 g protein per 100 g.",
            evidence_ids=["missing"],
        ),
        evidence_map,
    )


def test_claim_verification_binds_nutrition_numbers_to_their_field() -> None:
    evidence = _evidence(
        "food-tempeh",
        "Food: Tempeh\nServing: 100 g\nCalories: 105 kcal\nProtein: 20 g",
        kind="food",
        metadata={"name": "Tempeh"},
    )
    evidence_by_id = {evidence.evidence_id: evidence}

    contradicted = _claim_verification(
        GeneratedClaim(
            text="Tempeh contains 105 g protein per 100 g.",
            evidence_ids=["food-tempeh"],
        ),
        evidence_by_id,
    )
    supported = _claim_verification(
        GeneratedClaim(
            text="Tempeh contains 20 g protein per 100 g.",
            evidence_ids=["food-tempeh"],
        ),
        evidence_by_id,
    )

    assert contradicted.status == "contradicted"
    assert supported.status == "supported"
    assert supported.evidence_ids == ("food-tempeh",)


def test_model_loading_passes_offline_mode_to_tokenizer_and_model(
    monkeypatch, tmp_path
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls.append(("tokenizer", {"model_id": model_id, **kwargs}))
            return object()

    class FakeModel:
        def eval(self) -> None:
            pass

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls.append(("model", {"model_id": model_id, **kwargs}))
            return FakeModel()

    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)
    torch_module.float16 = "float16"
    torch_module.float32 = "float32"
    peft_module = ModuleType("peft")
    peft_module.PeftModel = object
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeTokenizerLoader
    transformers_module.AutoModelForCausalLM = FakeModelLoader
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setattr(settings, "llm_offline", True)
    monkeypatch.setattr(settings, "model_cache_path", tmp_path / "cache")
    monkeypatch.setattr(settings, "llm_adapter_path", tmp_path / "no-adapter")

    _load_model.cache_clear()
    try:
        _load_model()
    finally:
        _load_model.cache_clear()

    assert [kind for kind, _ in calls] == ["tokenizer", "model"]
    assert all(call["local_files_only"] is True for _, call in calls)
    assert all(call["model_id"] == settings.llm_model_id for _, call in calls)
    assert "dtype" in calls[1][1]
    assert "torch_dtype" not in calls[1][1]


def test_generation_passes_attention_mask_without_ignored_sampling_settings(
    monkeypatch,
) -> None:
    import torch

    generation_config = SimpleNamespace(
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
    )
    captured: dict[str, object] = {}

    class FakeTokenizer:
        eos_token_id = 2

        def apply_chat_template(self, _messages, **kwargs):
            assert kwargs["return_dict"] is True
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, _tokens, *, skip_special_tokens):
            assert skip_special_tokens is True
            return '{"claims":[]}'

    class FakeModel:
        def __init__(self) -> None:
            self.generation_config = generation_config

        def parameters(self):
            return iter([SimpleNamespace(device=torch.device("cpu"))])

        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(
        "app.services.generation._load_model",
        lambda: (torch, FakeTokenizer(), FakeModel()),
    )

    generated = _generate("A controlled test prompt.")

    assert generated.claims == []
    assert torch.equal(
        captured["attention_mask"],
        torch.tensor([[1, 1]]),
    )
    assert captured["do_sample"] is False
    used_config = captured["generation_config"]
    assert used_config.temperature is None
    assert used_config.top_p is None
    assert used_config.top_k is None
    assert generation_config.temperature == 0.7


def test_generation_reports_missing_citation_field_without_exposing_model_text(
    monkeypatch,
) -> None:
    import torch

    class FakeTokenizer:
        eos_token_id = 2

        def apply_chat_template(self, _messages, **_kwargs):
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
            }

        def decode(self, _tokens, *, skip_special_tokens):
            return '{"claims":[{"text":"private generated text"}]}'

    class FakeModel:
        generation_config = SimpleNamespace(
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

        def parameters(self):
            return iter([SimpleNamespace(device=torch.device("cpu"))])

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2]])

    monkeypatch.setattr(
        "app.services.generation._load_model",
        lambda: (torch, FakeTokenizer(), FakeModel()),
    )

    with pytest.raises(
        ModelOutputError,
        match=r"claims\.0\.evidence_ids: missing",
    ) as caught:
        _generate("A controlled test prompt.")

    assert "private generated text" not in str(caught.value)


def test_invalid_json_is_reported_without_echoing_model_output(monkeypatch) -> None:
    import torch

    class FakeTokenizer:
        eos_token_id = 2

        def apply_chat_template(self, _messages, **_kwargs):
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
            }

        def decode(self, _tokens, *, skip_special_tokens):
            return "private non-JSON model output"

    class FakeModel:
        generation_config = SimpleNamespace(
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

        def parameters(self):
            return iter([SimpleNamespace(device=torch.device("cpu"))])

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2]])

    monkeypatch.setattr(
        "app.services.generation._load_model",
        lambda: (torch, FakeTokenizer(), FakeModel()),
    )

    with pytest.raises(ModelOutputError, match="did not return valid JSON"):
        _generate("A controlled test prompt.")


def test_prompt_requires_concise_evidence_matched_output() -> None:
    food = _evidence(
        "eval-food-record:73",
        "Food: Ikan Bakar\nCalories: 280 kcal\nProtein: 36.0 g",
        kind="food",
        metadata={"name": "Ikan Bakar"},
    )
    question = "Find foods under 300 calories and with more than 30 g protein."
    prompt = _build_prompt(
        question,
        process_query(question),
        [food],
    )

    assert '"evidence_ids":["eval-food-record:73"]' in prompt
    assert "fewest claims needed" in prompt
    assert "ignore unrelated evidence" in prompt
    assert "only the fields used by those conditions" in prompt
    assert "return exactly {\"claims\":[]}" in prompt
    assert "Requested numeric fields: calories, protein" in prompt


def test_unsupported_generated_nutrition_claim_is_rejected(monkeypatch) -> None:
    evidence = _evidence(
        "food-tempeh",
        "Food: Tempeh\nServing: 100 g\nProtein: 20 g",
        kind="food",
        metadata={"name": "Tempeh"},
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Tempeh contains 200 g protein per 100 g.",
                    evidence_ids=["food-tempeh"],
                )
            ]
        ),
    )

    result = generate_verified_answer(
        "Discuss tempeh generally.",
        process_query("Discuss tempeh generally."),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert "not have enough supported evidence" in result.answer
    assert "200 g" not in result.answer
    assert result.citations == []
    assert result.supported_claims == 0
    assert result.rejected_claims == 1
    assert any(
        "contradicted the retrieved evidence" in item
        for item in result.warnings
    )


def test_irrelevant_guideline_evidence_produces_no_supported_claim(monkeypatch) -> None:
    evidence = _evidence(
        "chunk-diabetes",
        "General notes: For a balanced meal, use quarter plate carbohydrate.",
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Moon dust contains 42 micrograms of vitamin B12.",
                    evidence_ids=[evidence.evidence_id],
                )
            ]
        ),
    )

    result = generate_verified_answer(
        "What is the exact vitamin B12 content of moon dust?",
        process_query("What is the exact vitamin B12 content of moon dust?"),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert "not have enough supported evidence" in result.answer
    assert result.citations == []
    assert result.supported_claims == 0
    assert result.rejected_claims == 1


def test_excessive_unsupported_claims_do_not_reach_final_answer(monkeypatch) -> None:
    evidence = _evidence(
        "food-ikan",
        "Food: Ikan Bakar\nCalories: 280 kcal\nProtein: 36.0 g",
        kind="food",
        metadata={"name": "Ikan Bakar"},
    )
    supported_text = (
        "Ikan Bakar contains 280 kcal calories and 36.0 g protein."
    )
    claims = [
        GeneratedClaim(
            text=supported_text,
            evidence_ids=[evidence.evidence_id],
        ),
        GeneratedClaim(
            text="Ikan Bakar has 620 mg of potassium.",
            evidence_ids=[evidence.evidence_id],
        ),
    ] + [
        GeneratedClaim(
            text=f"Ikan Bakar contains an unrelated unsupported fact {index}.",
            evidence_ids=[evidence.evidence_id],
        )
        for index in range(8)
    ]
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(claims=claims),
    )
    question = "Find foods under 300 calories and with more than 30 g protein."

    result = generate_verified_answer(
        question,
        process_query(question),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert result.supported_claims == 1
    assert result.rejected_claims == 9
    assert result.answer == supported_text
    assert len(result.citations) == 1


def test_malformed_model_output_returns_safe_abstention(
    monkeypatch,
) -> None:
    evidence = _evidence(
        "chunk-guideline",
        "Guidance states one clear fact.",
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: (_ for _ in ()).throw(
            ModelOutputError("The model output was malformed.")
        ),
    )

    result = generate_verified_answer(
        "Explain the guidance.",
        process_query("Explain the guidance."),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert "not have enough supported evidence" in result.answer
    assert result.citations == []
    assert result.supported_claims == 0
    assert result.rejected_claims == 1
    assert MALFORMED_MODEL_OUTPUT_WARNING in result.warnings


def test_food_nutrition_claim_requires_the_matching_food_record() -> None:
    chicken = _evidence(
        "food-chicken",
        "Food: Chicken\nProtein: 20 g",
        kind="food",
        metadata={"name": "Chicken"},
    )

    verification = _claim_verification(
        GeneratedClaim(
            text="Tempeh contains 20 g protein.",
            evidence_ids=["food-chicken"],
        ),
        {chicken.evidence_id: chicken},
    )

    assert verification.status == "insufficient"
    assert verification.evidence_ids == ()


def test_guideline_claim_requires_exact_source_sentence_not_lexical_overlap() -> None:
    guideline = _evidence(
        "chunk-guideline",
        "Malaysian dietary guidelines recommend eating a variety of foods.",
    )
    evidence_by_id = {guideline.evidence_id: guideline}

    paraphrase = _claim_verification(
        GeneratedClaim(
            text="Dietary guidance promotes a diverse selection of foods.",
            evidence_ids=["chunk-guideline"],
        ),
        evidence_by_id,
    )
    exact_sentence = _claim_verification(
        GeneratedClaim(
            text="Malaysian dietary guidelines recommend eating a variety of foods.",
            evidence_ids=["chunk-guideline"],
        ),
        evidence_by_id,
    )

    assert paraphrase.status == "insufficient"
    assert exact_sentence.status == "supported"
    assert exact_sentence.evidence_ids == ("chunk-guideline",)


def test_claim_verification_distinguishes_missing_and_conflicting_values() -> None:
    evidence = _evidence(
        "food-tempeh",
        "Food: Tempeh\nProtein: 20 g",
        kind="food",
        metadata={"name": "Tempeh"},
    )
    missing_value = _claim_verification(
        GeneratedClaim(
            text="Tempeh contains 18 g protein.",
            evidence_ids=["food-tempeh"],
        ),
        {evidence.evidence_id: evidence},
    )
    no_support = _claim_verification(
        GeneratedClaim(
            text="The guideline recommends drinking three litres of water.",
            evidence_ids=["food-tempeh"],
        ),
        {evidence.evidence_id: evidence},
    )
    conflicting_evidence = _evidence(
        "food-tempeh-other",
        "Food: Tempeh\nCalories: 150 kcal",
        kind="food",
        metadata={"name": "Tempeh"},
    )
    conflict = _claim_verification(
        GeneratedClaim(
            text="Tempeh has 105 calories.",
            evidence_ids=["food-tempeh-calories", "food-tempeh-other"],
        ),
        {
            "food-tempeh-calories": _evidence(
                "food-tempeh-calories",
                "Food: Tempeh\nCalories: 105 kcal",
                kind="food",
                metadata={"name": "Tempeh"},
            ),
            conflicting_evidence.evidence_id: conflicting_evidence,
        },
    )

    assert missing_value.status == "contradicted"
    assert no_support.status == "insufficient"
    assert conflict.status == "contradicted"


def test_generated_claim_with_fabricated_citation_is_removed(monkeypatch) -> None:
    evidence = _evidence(
        "chunk-real",
        "Dietary guidelines recommend varied meals.",
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Dietary guidelines recommend varied meals.",
                    evidence_ids=["chunk-fabricated"],
                )
            ]
        ),
    )

    result = generate_verified_answer(
        "Explain dietary guidelines.",
        process_query("Explain dietary guidelines."),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert "not have enough supported evidence" in result.answer
    assert result.citations == []
    assert result.rejected_claims == 1
    assert any("lacked sufficient supporting evidence" in item for item in result.warnings)


def test_multiple_claims_keep_only_their_supporting_citations(monkeypatch) -> None:
    first = _evidence(
        "chunk-variety",
        "Malaysian dietary guidance recommends eating a variety of foods.",
    )
    second = _evidence(
        "chunk-plate",
        "The Malaysian healthy plate guide recommends a balanced meal.",
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Malaysian dietary guidance recommends eating a variety of foods.",
                    evidence_ids=["chunk-variety"],
                ),
                GeneratedClaim(
                    text="The Malaysian healthy plate guide recommends a balanced meal.",
                    evidence_ids=["chunk-plate"],
                ),
            ]
        ),
    )

    result = generate_verified_answer(
        "Explain dietary guidelines.",
        process_query("Explain dietary guidelines."),
        [first, second],
        ConstraintResult(
            allowed=[first, second],
            excluded=[],
            warnings=[],
        ),
    )

    assert result.supported_claims == 2
    assert result.rejected_claims == 0
    assert len(result.citations) == 2
    assert {citation.evidence_id for citation in result.citations} == {
        "chunk-variety",
        "chunk-plate",
    }
    assert all(citation.source_url == "https://example.test/source" for citation in result.citations)


def test_hybrid_numeric_and_guideline_claims_use_separate_evidence(monkeypatch) -> None:
    food = _evidence(
        "food-chicken",
        "Food: Chicken\nServing: 1 piece\nCalories: 220 kcal\nProtein: 20 g",
        kind="food",
        metadata={"name": "Chicken"},
    )
    guideline = _evidence(
        "chunk-guideline",
        "Malaysian dietary guidelines recommend choosing a variety of foods.",
    )
    question = (
        "Find foods below 300 calories and explain the dietary guidelines "
        "for choosing them."
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Chicken contains 220 calories.",
                    evidence_ids=["food-chicken"],
                ),
                GeneratedClaim(
                    text="Malaysian dietary guidelines recommend choosing a variety of foods.",
                    evidence_ids=["chunk-guideline"],
                ),
            ]
        ),
    )

    result = generate_verified_answer(
        question,
        process_query(question),
        [food, guideline],
        ConstraintResult(
            allowed=[food, guideline],
            excluded=[],
            warnings=[],
        ),
    )

    assert result.supported_claims == 2
    assert {citation.evidence_id for citation in result.citations} == {
        "food-chicken",
        "chunk-guideline",
    }
    assert "220 calories" in result.answer
    assert "The cited source states:" in result.answer
    assert "variety of foods" in result.answer


def test_no_generated_claims_returns_insufficient_evidence_without_citations(
    monkeypatch,
) -> None:
    evidence = _evidence(
        "chunk-general",
        "The guideline discusses balanced meals.",
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(claims=[]),
    )

    result = generate_verified_answer(
        "Explain dietary guidelines.",
        process_query("Explain dietary guidelines."),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert "not have enough supported evidence" in result.answer
    assert result.citations == []
    assert result.supported_claims == 0
    assert result.rejected_claims == 0


def test_verifier_failure_propagates_without_unverified_fallback(monkeypatch) -> None:
    evidence = _evidence(
        "chunk-general",
        "Dietary guidance recommends balanced meals.",
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Dietary guidance recommends balanced meals.",
                    evidence_ids=["chunk-general"],
                )
            ]
        ),
    )

    def fail_verification(*_args, **_kwargs):
        raise RuntimeError("verifier failed")

    monkeypatch.setattr(
        "app.services.generation._claim_verification",
        fail_verification,
    )

    with pytest.raises(
        EvidenceVerificationError,
        match="Evidence verification failed",
    ):
        generate_verified_answer(
            "Explain dietary guidance.",
            process_query("Explain dietary guidance."),
            [evidence],
            ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
        )


def test_numeric_normalization_preserves_tens() -> None:
    assert _canonical_numbers("20.0 g and 0.50 g") == {"20", "0.5"}


def test_empty_evidence_returns_safe_refusal_without_loading_model() -> None:
    plan = process_query("How much protein is in tempeh?")

    result = generate_verified_answer(
        "How much protein is in tempeh?",
        plan,
        [],
        ConstraintResult(allowed=[], excluded=[], warnings=[]),
    )

    assert not result.citations
    assert "not have enough verified evidence" in result.answer


def test_structured_food_lookup_returns_requested_recorded_nutrient(
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="food-1",
        text=(
            "Food: Tempeh\nServing: 1 serving (100g)\n"
            "Calories: 190 kcal\nProtein: 18 g\nFibre: 4 g"
        ),
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.3,
        kind="food",
        metadata={"name": "Tempeh"},
    )

    def unexpected_model_call(_prompt: str):
        raise AssertionError("A direct structured lookup should not call the LLM.")

    monkeypatch.setattr("app.services.generation._generate", unexpected_model_call)
    question = "How much protein is in tempeh per 100 g?"
    result = generate_verified_answer(
        question,
        process_query(question),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert "Protein: 18 g" in result.answer
    assert "190 kcal" not in result.answer
    assert [citation.evidence_id for citation in result.citations] == ["food-1"]
    assert result.warnings


def test_numeric_food_search_uses_filtered_evidence_not_single_food_lookup(
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="food-chicken",
        text="Food: Chicken\nServing: 1 piece\nCalories: 220 kcal\nProtein: 20 g",
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.7,
        kind="food",
        metadata={"name": "Chicken"},
    )
    question = "Find foods with less than 300 calories and at least 10g protein"
    plan = process_query(question)
    assert plan.constraints.numeric_constraints

    def generated_from_filtered_context(_prompt: str):
        return GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Chicken has 220 calories and 20 g protein.",
                    evidence_ids=["food-chicken"],
                )
            ]
        )

    monkeypatch.setattr(
        "app.services.generation._generate",
        generated_from_filtered_context,
    )
    result = generate_verified_answer(
        question,
        plan,
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert result.answer == "Chicken has 220 calories and 20 g protein."
    assert [citation.evidence_id for citation in result.citations] == [
        "food-chicken"
    ]


def test_serving_size_lookup_chooses_exact_food_and_skips_model(
    monkeypatch,
) -> None:
    apple = Evidence(
        evidence_id="food-apple",
        text="Food: Apple\nServing: 1 medium apple\nCalories: 95 kcal",
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.7,
        kind="food",
        metadata={"name": "Apple"},
    )
    green_banana = Evidence(
        evidence_id="food-green-banana",
        text="Food: Green Banana\nServing: 1 small banana\nCalories: 90 kcal",
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.7,
        kind="food",
        metadata={"name": "Green Banana"},
    )
    banana = Evidence(
        evidence_id="food-banana",
        text="Food: Banana\nServing: 1 medium banana (118 g)\nCalories: 105 kcal",
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.7,
        kind="food",
        metadata={"name": "Banana"},
    )

    def unexpected_model_call(_prompt: str):
        raise AssertionError("Structured lookup must not call the LLM.")

    monkeypatch.setattr("app.services.generation._generate", unexpected_model_call)
    question = "What is the serving size of banana?"
    result = generate_verified_answer(
        question,
        process_query(question),
        [apple, green_banana, banana],
        ConstraintResult(allowed=[apple, green_banana, banana], excluded=[], warnings=[]),
    )

    assert "Banana: Serving: 1 medium banana (118 g)." in result.answer
    assert [citation.evidence_id for citation in result.citations] == ["food-banana"]


def test_structured_lookup_does_not_fall_back_to_llm_for_unmatched_food(
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="food-apple",
        text="Food: Apple\nServing: 1 medium apple\nSodium: 2 mg",
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.7,
        kind="food",
        metadata={"name": "Apple"},
    )

    def unexpected_model_call(_prompt: str):
        raise AssertionError("Unmatched structured queries must not call the LLM.")

    monkeypatch.setattr("app.services.generation._generate", unexpected_model_call)
    question = "What is the sodium content of fried chicken?"
    result = generate_verified_answer(
        question,
        process_query(question),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert "exact nutrition record" in result.answer
    assert result.citations == []


def test_structured_lookup_returns_the_requested_chicken_sodium_value(
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="food-chicken",
        text=(
            "Food: Fried Chicken\nServing: 1 piece (100 g)\n"
            "Sodium: 620 mg\nProtein: 21 g"
        ),
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.7,
        kind="food",
        metadata={"name": "Fried Chicken"},
    )

    def unexpected_model_call(_prompt: str):
        raise AssertionError("Structured lookup must not call the LLM.")

    monkeypatch.setattr("app.services.generation._generate", unexpected_model_call)
    question = "What is the sodium content of fried chicken?"
    result = generate_verified_answer(
        question,
        process_query(question),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert "Fried Chicken: Sodium: 620 mg" in result.answer
    assert "Protein" not in result.answer
