import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.schemas.chat import ChatCitation
from app.services.constraints import ConstraintResult
from app.services.evidence import Evidence
from app.services.query_processor import QueryPlan
from app.services.retrieval import _food_match_score

os.environ.setdefault("HF_HOME", str(settings.model_cache_path))
os.environ.setdefault("TRANSFORMERS_CACHE", str(settings.model_cache_path))


class GeneratedClaim(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class GeneratedAnswer(BaseModel):
    claims: list[GeneratedClaim] = Field(max_length=12)


class VerifiedAnswer(BaseModel):
    answer: str
    citations: list[ChatCitation]
    warnings: list[str]
    supported_claims: int
    rejected_claims: int


class ModelOutputError(RuntimeError):
    pass


STRUCTURED_FIELDS = (
    ("calories", ("calorie", "calories", "energy", "kcal"), "Calories", "kcal"),
    (
        "carbohydrates",
        ("carbohydrate", "carbohydrates", "carbs"),
        "Carbohydrates",
        "g",
    ),
    ("protein", ("protein",), "Protein", "g"),
    ("fat", ("fat",), "Fat", "g"),
    ("fibre", ("fibre", "fiber"), "Fibre", "g"),
    ("sugar", ("sugar",), "Sugar", "g"),
    ("sodium", ("sodium", "salt"), "Sodium", "mg"),
    ("potassium", ("potassium",), "Potassium", "mg"),
    ("glycemic_index", ("glycemic", "glycaemic"), "Glycemic index", ""),
    ("serving_size", ("serving size", "portion size"), "Serving", ""),
)


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    Path(settings.model_cache_path).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.llm_model_id,
        cache_dir=str(settings.model_cache_path),
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.llm_model_id,
        cache_dir=str(settings.model_cache_path),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    adapter_path = Path(settings.llm_adapter_path)
    if _has_reviewed_adapter(adapter_path):
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return torch, tokenizer, model


def _evidence_prompt(evidence_items: list[Evidence]) -> str:
    return "\n\n".join(
        (
            f"[{item.evidence_id}] "
            f"(source={item.source_title}; type={item.source_type}; "
            f"authority={item.authority_score:.2f})\n{item.text}"
        )
        for item in evidence_items
    )


def _has_reviewed_adapter(adapter_path: Path) -> bool:
    manifest_path = adapter_path / "curated_training_manifest.json"
    if not (adapter_path / "adapter_config.json").is_file() or not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        isinstance(manifest, dict)
        and manifest.get("format_version") == 1
        and isinstance(manifest.get("human_reviewed_examples"), int)
        and manifest["human_reviewed_examples"] >= 4
    )


def _profile_constraints(plan: QueryPlan) -> str:
    details: list[str] = []
    if plan.constraints.diets:
        details.append("Dietary requirements: " + ", ".join(plan.constraints.diets))
    if plan.constraints.allergens:
        details.append(
            "Allergies to avoid: " + ", ".join(plan.constraints.allergens)
        )
    if plan.constraints.activity_level:
        details.append("Activity level: " + plan.constraints.activity_level)
    if not details:
        return "No profile constraints are available."
    return "\n".join(details)


def _build_prompt(
    question: str,
    plan: QueryPlan,
    evidence_items: list[Evidence],
) -> str:
    recommendation_rule = (
        "For food recommendations, only recommend a food when its evidence "
        "explicitly verifies compatibility with every listed diet and allergy. "
        "Do not treat missing ingredient or allergen data as safe."
        if plan.recommendation_requested
        else "Do not turn nutrition facts into personalized medical advice."
    )
    return (
        "Return one JSON object only, with this exact shape: "
        '{"claims":[{"text":"one factual sentence","evidence_ids":["chunk-1"]}]}.\n'
        "You are a careful dietary information assistant. Use only the supplied "
        "evidence for factual claims. Cite each claim using evidence IDs exactly "
        "as shown. If there is not enough evidence, return {\"claims\":[]}.\n"
        "Do not invent nutrition values, citations, diagnoses, or medical claims. "
        f"{recommendation_rule}\n\n"
        f"User constraints:\n{_profile_constraints(plan)}\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{_evidence_prompt(evidence_items)}"
    )


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character == "{":
            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ModelOutputError("The local language model did not return valid JSON.")


def _generate(prompt: str) -> GeneratedAnswer:
    torch, tokenizer, model = _load_model()
    message_list = [
        {
            "role": "system",
            "content": (
                "Follow the requested JSON schema exactly. Return no markdown "
                "or explanatory text outside the JSON object."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            message_list,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(
            f"System: {message_list[0]['content']}\n"
            f"User: {prompt}\nAssistant:",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]

    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=settings.max_answer_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0, input_ids.shape[-1] :]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    try:
        return GeneratedAnswer.model_validate(_extract_json(raw_text))
    except ValidationError as error:
        raise ModelOutputError(
            "The local language model returned a response that did not match "
            "the required claims schema."
        ) from error


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+(?:\.[\w]+)?", text.casefold()))


def _canonical_numbers(text: str) -> set[str]:
    numbers = re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", text)
    return {str(float(number)).rstrip("0").rstrip(".") for number in numbers}


def _claim_is_supported(
    claim: GeneratedClaim,
    evidence_by_id: dict[str, Evidence],
    *,
    require_compatible_food: bool = False,
    diets: tuple[str, ...] = (),
    allergens: tuple[str, ...] = (),
) -> bool:
    cited = [
        evidence_by_id[item_id]
        for item_id in claim.evidence_ids
        if item_id in evidence_by_id
    ]
    if len(cited) != len(claim.evidence_ids) or not cited:
        return False
    if require_compatible_food and not any(
        item.kind == "food"
        and (
            not diets
            or item.metadata.get("dietary_data_verified") is True
        )
        and (
            not allergens
            or item.metadata.get("allergen_data_verified") is True
        )
        for item in cited
    ):
        return False

    claim_tokens = _tokens(claim.text) - {
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
        "is", "it", "of", "on", "or", "per", "the", "to", "with",
    }
    evidence_text = " ".join(item.text for item in cited)
    evidence_tokens = _tokens(evidence_text)
    meaningful_tokens = {token for token in claim_tokens if not token.isdigit()}
    overlap = meaningful_tokens.intersection(evidence_tokens)
    lexical_support = len(overlap) / max(len(meaningful_tokens), 1)
    numeric_support = _canonical_numbers(claim.text).issubset(
        _canonical_numbers(evidence_text)
    )
    return lexical_support >= 0.2 and numeric_support


def _citation_for(evidence: Evidence) -> ChatCitation:
    return ChatCitation(
        evidence_id=evidence.evidence_id,
        source_title=evidence.source_title,
        source_url=evidence.source_url,
        source_type=evidence.source_type,
        authority_score=evidence.authority_score,
        excerpt=evidence.text[:360],
    )


def _structured_lookup_answer(
    question: str,
    plan: QueryPlan,
    evidence_items: list[Evidence],
    constraints: ConstraintResult,
) -> VerifiedAnswer | None:
    if not plan.structured_lookup:
        return None

    question_tokens = set(_tokens(question))
    matching_foods = [
        (score, item)
        for item in evidence_items
        if (score := _food_match_score(question, item)) is not None
    ]
    matching_food = (
        max(matching_foods, key=lambda match: match[0])[1]
        if matching_foods
        else None
    )
    if matching_food is None:
        return None

    requested_fields = [
        field
        for field, terms, _, _ in STRUCTURED_FIELDS
        if (
            bool(re.search(r"\b(?:serving|portion)\s+size\b", question.casefold()))
            if field == "serving_size"
            else question_tokens.intersection(terms)
        )
    ]
    if not requested_fields:
        requested_fields = [field for field, _, _, _ in STRUCTURED_FIELDS]

    serving_match = re.search(
        r"(?im)^Serving:\s*(.+?)\s*$",
        matching_food.text,
    )
    serving = serving_match.group(1) if serving_match else "the listed serving"
    food_name = str(matching_food.metadata.get("name", "This food"))
    claims: list[str] = []
    for field, _, label, unit in STRUCTURED_FIELDS:
        if field not in requested_fields:
            continue
        if field == "serving_size":
            if serving_match:
                claims.append(f"{food_name}: {label}: {serving}.")
            continue
        value_match = re.search(
            rf"(?im)^{re.escape(label)}:\s*"
            rf"([0-9]+(?:\.[0-9]+)?)\s*{re.escape(unit)}\s*$",
            matching_food.text,
        )
        if value_match is None:
            continue
        value = value_match.group(1)
        if field == "glycemic_index":
            claims.append(f"{food_name}: {label}: {value}.")
            continue
        unit_suffix = f" {unit}" if unit else ""
        claims.append(
            f"{food_name}: {label}: {value}{unit_suffix} for {serving}."
        )

    if not claims:
        return None

    warnings = list(constraints.warnings)
    if matching_food.is_estimate:
        warnings.append(
            "Some cited nutrition values come from generated estimates and should "
            "not be treated as authoritative measurements."
        )
    if matching_food.authority_score < 0.5:
        warnings.append(
            "Some cited records have limited source verification; treat the "
            "information as provisional."
        )

    answer = " ".join(claims)
    if matching_food.authority_score < 0.5:
        answer = (
            "This information comes from a limited-verification source: "
            + answer
        )
    return VerifiedAnswer(
        answer=answer,
        citations=[_citation_for(matching_food)],
        warnings=warnings,
        supported_claims=len(claims),
        rejected_claims=0,
    )


def generate_verified_answer(
    question: str,
    plan: QueryPlan,
    evidence_items: list[Evidence],
    constraints: ConstraintResult,
) -> VerifiedAnswer:
    if (
        plan.recommendation_requested
        and (
            plan.constraints.diets
            or plan.constraints.allergens
            or plan.constraints.unknown_dietary_constraint
            or plan.constraints.unknown_allergen_constraint
        )
        and not any(item.kind == "food" for item in evidence_items)
    ):
        warnings = list(constraints.warnings)
        warnings.append(
            "No retrieved food had verified ingredient, allergen, and dietary "
            "compatibility data for the requested constraints."
        )
        return VerifiedAnswer(
            answer=(
                "I can't safely recommend a specific food because the available "
                "records do not verify its ingredients and compatibility with "
                "your dietary constraints. Please check the food label or consult "
                "a qualified dietitian."
            ),
            citations=[],
            warnings=warnings,
            supported_claims=0,
            rejected_claims=0,
        )

    if not evidence_items:
        return VerifiedAnswer(
            answer=(
                "I do not have enough verified evidence to answer that safely. "
                "Please consult a qualified health professional for personal "
                "medical advice."
            ),
            citations=[],
            warnings=constraints.warnings,
            supported_claims=0,
            rejected_claims=0,
        )

    structured_answer = _structured_lookup_answer(
        question,
        plan,
        evidence_items,
        constraints,
    )
    if structured_answer is not None:
        return structured_answer
    if plan.structured_lookup:
        return VerifiedAnswer(
            answer=(
                "I do not have an exact nutrition record for that food and "
                "requested information."
            ),
            citations=[],
            warnings=list(constraints.warnings),
            supported_claims=0,
            rejected_claims=0,
        )

    generated = _generate(_build_prompt(question, plan, evidence_items))
    evidence_by_id = {item.evidence_id: item for item in evidence_items}
    supported = [
        claim
        for claim in generated.claims
        if _claim_is_supported(
            claim,
            evidence_by_id,
            require_compatible_food=(
                plan.recommendation_requested
                and bool(
                    plan.constraints.diets
                    or plan.constraints.allergens
                    or plan.constraints.unknown_dietary_constraint
                    or plan.constraints.unknown_allergen_constraint
                )
            ),
            diets=plan.constraints.diets,
            allergens=plan.constraints.allergens,
        )
    ]
    rejected_count = len(generated.claims) - len(supported)
    citation_ids = list(
        dict.fromkeys(
            evidence_id
            for claim in supported
            for evidence_id in claim.evidence_ids
        )
    )
    cited_evidence = [evidence_by_id[item_id] for item_id in citation_ids]
    warnings = list(constraints.warnings)

    if rejected_count:
        warnings.append(
            f"{rejected_count} generated claim(s) were removed because the "
            "retrieved evidence did not support them."
        )
    if any(item.is_estimate for item in cited_evidence):
        warnings.append(
            "Some cited nutrition values come from generated estimates and should "
            "not be treated as authoritative measurements."
        )
    if any(item.authority_score < 0.5 for item in cited_evidence):
        warnings.append(
            "Some cited records have limited source verification; treat the "
            "information as provisional."
        )
    adapter_path = Path(settings.llm_adapter_path)
    if not _has_reviewed_adapter(adapter_path):
        warnings.append(
            "The base open-weight model is active because no adapter trained "
            "from a human-reviewed dataset is available."
        )

    if not supported:
        answer = (
            "I do not have enough supported evidence to provide a reliable "
            "answer to that question."
        )
        citations: list[ChatCitation] = []
    else:
        answer = " ".join(claim.text.strip() for claim in supported)
        citations = [_citation_for(item) for item in cited_evidence]

    return VerifiedAnswer(
        answer=answer,
        citations=citations,
        warnings=warnings,
        supported_claims=len(supported),
        rejected_claims=rejected_count,
    )
