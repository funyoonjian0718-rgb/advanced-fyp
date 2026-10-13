import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.schemas.chat import ChatCitation
from app.services.constraints import ConstraintResult, _declaration_is_verified
from app.services.evidence import Evidence
from app.services.query_processor import QueryPlan
from app.services.retrieval import _food_match_score

os.environ.setdefault("HF_HOME", str(settings.model_cache_path))


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


MALFORMED_MODEL_OUTPUT_WARNING = (
    "The local model response was malformed; no generated claims were accepted."
)


class EvidenceVerificationError(RuntimeError):
    pass


ClaimStatus = Literal["supported", "contradicted", "insufficient"]


@dataclass(frozen=True)
class ClaimVerification:
    status: ClaimStatus
    evidence_ids: tuple[str, ...] = ()


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


def _canonical_numeric_field(field: str) -> str:
    base = re.sub(r"_(?:g|mg)$", "", field.casefold())
    if base == "fiber":
        base = "fibre"
    for canonical, (aliases, _) in NUMERIC_FACT_FIELDS.items():
        if base in aliases:
            return canonical
    return base


NUMERIC_FACT_FIELDS = {
    "calories": (("calories", "calorie", "kcal", "energy"), "kcal"),
    "carbohydrates": (("carbohydrates", "carbohydrate", "carbs"), "g"),
    "protein": (("protein",), "g"),
    "fat": (("fat",), "g"),
    "fibre": (("fibre", "fiber"), "g"),
    "sugar": (("sugar",), "g"),
    "sodium": (("sodium",), "mg"),
    "potassium": (("potassium",), "mg"),
    "glycemic_index": (
        ("glycemic index", "glycaemic index", "glycemic", "glycaemic", "gi"),
        "",
    ),
}


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as error:
        missing_package = (error.name or "unknown").split(".")[0]
        raise RuntimeError(
            f"Required local model dependency '{missing_package}' is missing. "
            "Install the backend AI dependencies from requirements-ai.txt."
        ) from error

    Path(settings.model_cache_path).mkdir(parents=True, exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            settings.llm_model_id,
            cache_dir=str(settings.model_cache_path),
            local_files_only=settings.llm_offline,
        )
        model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model_id,
            cache_dir=str(settings.model_cache_path),
            local_files_only=settings.llm_offline,
            dtype=(
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
    except OSError as error:
        if settings.llm_offline:
            raise RuntimeError(
                f"Configured model '{settings.llm_model_id}' is not fully "
                "available in the local cache while offline mode is enabled."
            ) from error
        raise RuntimeError(
            f"Could not load configured model '{settings.llm_model_id}'."
        ) from error

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
    evidence_id_example = (
        evidence_items[0].evidence_id if evidence_items else "no-evidence"
    )
    numeric_fields = tuple(
        dict.fromkeys(
            _canonical_numeric_field(constraint.field)
            for constraint in plan.constraints.numeric_constraints
        )
    )
    numeric_scope = (
        "Requested numeric fields: "
        + ", ".join(numeric_fields)
        + ". For food results, include only these fields and their exact "
        "recorded values and units."
        if numeric_fields
        else ""
    )
    recommendation_rule = (
        "For food recommendations, only recommend a food when its evidence "
        "explicitly verifies compatibility with every listed diet and allergy. "
        "Do not treat missing ingredient or allergen data as safe."
        if plan.recommendation_requested
        else "Do not turn nutrition facts into personalized medical advice."
    )
    return (
        "Return one JSON object only, with this exact shape: "
        f'{{"claims":[{{"text":"one factual sentence","evidence_ids":["{evidence_id_example}"]}}]}}.\n'
        "Every claim must contain both keys exactly: text and evidence_ids. "
        "evidence_ids must be a non-empty JSON array containing only IDs copied "
        "from the supplied evidence. Do not omit either key or use another name.\n"
        "Answer only what the user asked, using the fewest claims needed. "
        "Do not add greetings, recommendations, background, or other nutrition "
        "facts. Use only supplied evidence that directly answers the question; "
        "ignore unrelated evidence. If no evidence directly supports the answer, "
        "return exactly {\"claims\":[]} rather than guessing.\n"
        "For document evidence, copy a complete relevant source sentence exactly "
        "as written; do not paraphrase, shorten, or complete a fragment. Attach "
        "the ID printed before that evidence. For food records, use the exact "
        "food name, nutrient field, numeric value, and unit from its record. "
        "For numeric-filter questions, report only foods that meet the stated "
        "conditions and only the fields used by those conditions. Never repeat "
        "the requested condition as if it were a food fact. Each claim must be "
        "independently supported by its cited evidence.\n"
        f"{numeric_scope}\n"
        "Do not invent nutrition values, citations, diagnoses, or medical claims. "
        "Do not infer missing values. "
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
        encoded = tokenizer.apply_chat_template(
            message_list,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        encoded = tokenizer(
            f"System: {message_list[0]['content']}\n"
            f"User: {prompt}\nAssistant:",
            return_tensors="pt",
        )

    model_device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)
    generation_config = deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            max_new_tokens=settings.max_answer_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0, input_ids.shape[-1] :]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    try:
        return GeneratedAnswer.model_validate(_extract_json(raw_text))
    except ValidationError as error:
        invalid_fields = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['type']}"
            for item in error.errors(include_input=False)[:3]
        )
        raise ModelOutputError(
            "The local language model returned a response that did not match "
            f"the required claims schema ({invalid_fields})."
        ) from error


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+(?:\.[\w]+)?", text.casefold()))


def _canonical_numbers(text: str) -> set[str]:
    numbers = re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", text)
    return {_canonical_number(number) for number in numbers}


def _canonical_number(value: str) -> str:
    return str(float(value)).rstrip("0").rstrip(".")


def _numeric_facts(text: str) -> dict[str, set[tuple[str, str | None]]]:
    facts: dict[str, set[tuple[str, str | None]]] = {
        field: set() for field in NUMERIC_FACT_FIELDS
    }
    number = r"(?<![\w.])(?P<value>\d+(?:\.\d+)?)(?![\w.])"
    unit = r"\s*(?P<unit>kcal|mg|g)?"
    for line in text.splitlines():
        for field, (aliases, _) in NUMERIC_FACT_FIELDS.items():
            alias_pattern = "(?:" + "|".join(
                re.escape(alias)
                for alias in sorted(aliases, key=len, reverse=True)
            ) + ")"
            patterns = (
                rf"^\s*{alias_pattern}\s*[:=]\s*{number}{unit}",
                rf"\b{alias_pattern}\b\s+(?:is|are|of|about|approximately|"
                rf"contains|has)\s+{number}{unit}",
                rf"{number}{unit}\s+(?:of\s+)?{alias_pattern}\b",
            )
            for pattern in patterns:
                for match in re.finditer(pattern, line, flags=re.IGNORECASE):
                    parsed_unit = match.group("unit")
                    facts[field].add(
                        (
                            _canonical_number(match.group("value")),
                            parsed_unit.casefold() if parsed_unit else None,
                        )
                    )
    return facts


def _normalized_phrase(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def _food_name(item: Evidence) -> str:
    name = item.metadata.get("name")
    if isinstance(name, str) and name.strip():
        return name
    match = re.search(r"(?im)^\s*Food:\s*(.+?)\s*$", item.text)
    return match.group(1) if match else ""


def _food_is_named_in_claim(item: Evidence, claim_text: str) -> bool:
    name = _normalized_phrase(_food_name(item))
    claim = f" {_normalized_phrase(claim_text)} "
    return bool(name) and f" {name} " in claim


def _exact_document_claim_support(
    claim_text: str,
    cited: list[Evidence],
) -> tuple[str, ...]:
    claim = _normalized_phrase(claim_text)
    if not claim:
        return ()
    supported_ids: list[str] = []
    for item in cited:
        if item.kind != "document":
            continue
        sentences = re.split(r"(?<=[.!?])\s+", item.text.strip())
        if any(_normalized_phrase(sentence) == claim for sentence in sentences):
            supported_ids.append(item.evidence_id)
    return tuple(supported_ids)


def _claim_verification(
    claim: GeneratedClaim,
    evidence_by_id: dict[str, Evidence],
    *,
    require_compatible_food: bool = False,
    diets: tuple[str, ...] = (),
    allergens: tuple[str, ...] = (),
) -> ClaimVerification:
    cited = [
        evidence_by_id[item_id]
        for item_id in claim.evidence_ids
        if item_id in evidence_by_id
    ]
    if len(cited) != len(claim.evidence_ids) or not cited:
        return ClaimVerification("insufficient")
    food_evidence = [item for item in cited if item.kind == "food"]
    if require_compatible_food and not any(
        item.kind == "food"
        and (
            not diets
            or _declaration_is_verified(item, "dietary")
        )
        and (
            not allergens
            or _declaration_is_verified(item, "allergen")
        )
        for item in cited
    ):
        return ClaimVerification("contradicted")

    claim_tokens = _tokens(claim.text) - {
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
        "is", "it", "of", "on", "or", "per", "the", "to", "with",
    }
    meaningful_tokens = {token for token in claim_tokens if not token.isdigit()}
    claim_facts = _numeric_facts(claim.text)
    fact_fields = [
        field for field, values in claim_facts.items() if values
    ]
    supported_ids: set[str] = set()

    if fact_fields:
        if food_evidence:
            matching_foods = [
                item
                for item in food_evidence
                if _food_is_named_in_claim(item, claim.text)
            ]
            if not matching_foods:
                return ClaimVerification("insufficient")

            supporting_foods: list[Evidence] = []
            for item in matching_foods:
                item_facts = _numeric_facts(item.text)
                for field in fact_fields:
                    source_values = item_facts[field]
                    claim_values = claim_facts[field]
                    if not source_values:
                        return ClaimVerification("insufficient")
                    if {
                        value for value, _ in source_values
                    } != {value for value, _ in claim_values}:
                        return ClaimVerification("contradicted")

                    expected_unit = NUMERIC_FACT_FIELDS[field][1]
                    if any(
                        claim_unit is not None and claim_unit != expected_unit
                        for _, claim_unit in claim_values
                    ) or any(
                        source_unit is not None and source_unit != expected_unit
                        for _, source_unit in source_values
                    ):
                        return ClaimVerification("contradicted")
                supporting_foods.append(item)

            supported_ids.update(item.evidence_id for item in supporting_foods)
            evidence_text = " ".join(item.text for item in supporting_foods)
            metric_numbers = {
                value
                for field in fact_fields
                for value, _ in claim_facts[field]
            }
            remaining_numbers = _canonical_numbers(claim.text) - metric_numbers
            if not remaining_numbers.issubset(_canonical_numbers(evidence_text)):
                return ClaimVerification("insufficient")
        else:
            document_support = _exact_document_claim_support(claim.text, cited)
            if not document_support:
                return ClaimVerification("insufficient")
            supported_ids.update(document_support)
    else:
        document_support = _exact_document_claim_support(claim.text, cited)
        if document_support:
            supported_ids.update(document_support)
        elif not food_evidence or not meaningful_tokens:
            return ClaimVerification("insufficient")
        elif not any(
            _food_is_named_in_claim(item, claim.text)
            and _normalized_phrase(claim.text)
            in _normalized_phrase(item.text)
            for item in food_evidence
        ):
            return ClaimVerification("insufficient")
        else:
            supported_ids.update(
                item.evidence_id
                for item in food_evidence
                if _food_is_named_in_claim(item, claim.text)
                and _normalized_phrase(claim.text)
                in _normalized_phrase(item.text)
            )

    if not supported_ids:
        return ClaimVerification("insufficient")
    return ClaimVerification(
        "supported",
        tuple(
            item.evidence_id
            for item in cited
            if item.evidence_id in supported_ids
        ),
    )


def _claim_is_supported(
    claim: GeneratedClaim,
    evidence_by_id: dict[str, Evidence],
    *,
    require_compatible_food: bool = False,
    diets: tuple[str, ...] = (),
    allergens: tuple[str, ...] = (),
) -> bool:
    return _claim_verification(
        claim,
        evidence_by_id,
        require_compatible_food=require_compatible_food,
        diets=diets,
        allergens=allergens,
    ).status == "supported"


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
    if not plan.structured_lookup or plan.constraints.numeric_constraints:
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



MEAL_PLAN_PATTERNS = (
    r"\bmeal\s+plans?\b",
    r"\bplan\s+(?:my\s+)?meals?\b",
    r"\bone[- ]day\s+(?:meal\s+)?plan\b",
    r"\b(?:breakfast|lunch|dinner)\b.{0,100}\b(?:snack|meal\s+plan|balanced\s+day)\b",
)


def is_meal_plan_request(question: str) -> bool:
    """Recognise meal-plan requests that should use structured food evidence."""
    return any(
        re.search(pattern, question, flags=re.IGNORECASE)
        for pattern in MEAL_PLAN_PATTERNS
    )


def _record_value(item: Evidence, label: str, unit: str = "") -> str | None:
    suffix = rf"\s*{re.escape(unit)}" if unit else ""
    match = re.search(
        rf"(?im)^\s*{re.escape(label)}:\s*(.+?){suffix}\s*$",
        item.text,
    )
    return match.group(1).strip() if match else None


def generate_meal_plan_answer(
    question: str,
    evidence_items: list[Evidence],
    constraints: ConstraintResult,
) -> VerifiedAnswer:
    """Compose a transparent sample day from already-filtered food records.

    This path does not ask the LLM to invent a combination or nutrition facts.
    It must receive evidence after enforce_constraints has run.
    """
    foods: list[Evidence] = []
    seen_names: set[str] = set()
    for item in evidence_items:
        if item.kind != "food":
            continue
        name = str(item.metadata.get("name", "")).strip()
        normalized = _normalized_phrase(name)
        if not name or not normalized or normalized in seen_names:
            continue
        seen_names.add(normalized)
        foods.append(item)

    if not foods:
        return VerifiedAnswer(
            answer=(
                "I couldn't build a meal-plan example because the available "
                "retrieved evidence did not include usable food records."
            ),
            citations=[],
            warnings=[
                *constraints.warnings,
                "No food items were available for a source-grounded meal-plan example.",
            ],
            supported_claims=0,
            rejected_claims=0,
        )

    # Choose a repeatable but query-conditioned sample instead of always taking
    # the first four database records. The digest only changes selection order;
    # it never changes a food name, nutrition value, or evidence citation.
    # Repeating the exact same question remains stable, while differently worded
    # requests can produce a different combination from the eligible candidates.
    normalized_question = _normalized_phrase(question)
    foods.sort(
        key=lambda item: hashlib.sha256(
            f"{normalized_question}|{item.evidence_id}".encode("utf-8")
        ).hexdigest()
    )

    # Prefer category variety where category metadata is available, then fill
    # remaining slots from the remaining eligible records.
    slots = ("Breakfast", "Lunch", "Dinner", "Snack")
    selected: list[Evidence] = []
    deferred: list[Evidence] = []
    seen_categories: set[str] = set()
    for item in foods:
        category = str(item.metadata.get("category") or "").strip().casefold()
        if category and category not in seen_categories:
            selected.append(item)
            seen_categories.add(category)
        else:
            deferred.append(item)
        if len(selected) == len(slots):
            break
    if len(selected) < len(slots):
        selected.extend(deferred[: len(slots) - len(selected)])
    rendered: list[str] = [
        "Here is a general sample day arranged from food records in the available "
        "dataset. This is an illustrative example, not a clinically validated or "
        "condition-specific meal plan."
    ]
    for slot, item in zip(slots, selected):
        name = str(item.metadata.get("name", "Food item"))
        serving = _record_value(item, "Serving")
        nutrition: list[str] = []
        for label, unit in (
            ("Calories", "kcal"),
            ("Protein", "g"),
            ("Carbohydrates", "g"),
            ("Fibre", "g"),
        ):
            value = _record_value(item, label, unit)
            if value is not None:
                nutrition.append(f"{label.lower()}: {value} {unit}".strip())
        detail_parts = []
        if serving:
            detail_parts.append(f"serving: {serving}")
        if nutrition:
            detail_parts.append("recorded nutrition: " + ", ".join(nutrition))
        details = f" ({'; '.join(detail_parts)})" if detail_parts else ""
        rendered.append(f"{slot}: {name}{details}.")

    warnings = list(constraints.warnings)
    if len(selected) < len(slots):
        warnings.append(
            f"Only {len(selected)} distinct food record(s) were retrieved, so the "
            "example is incomplete rather than a full four-item day."
        )
    if any(item.is_estimate for item in selected):
        warnings.append(
            "Some cited nutrition values are dataset estimates and should not be "
            "treated as authoritative measurements."
        )
    if any(item.authority_score < 0.5 for item in selected):
        warnings.append(
            "Some cited records have limited source verification; treat the "
            "information as provisional."
        )
    warnings.append(
        "Meal slots are an illustrative arrangement of retrieved records, not "
        "evidence that the foods form a nutritionally complete or medically suitable plan."
    )
    return VerifiedAnswer(
        answer="\n\n".join(rendered),
        citations=[_citation_for(item) for item in selected],
        warnings=warnings,
        supported_claims=len(selected),
        rejected_claims=0,
    )

def _claim_is_in_numeric_scope(
    claim: GeneratedClaim,
    plan: QueryPlan,
    evidence_by_id: dict[str, Evidence],
) -> bool:
    requested_fields = {
        _canonical_numeric_field(constraint.field)
        for constraint in plan.constraints.numeric_constraints
    }
    if not requested_fields:
        return True
    cited_food = any(
        evidence_by_id[evidence_id].kind == "food"
        for evidence_id in claim.evidence_ids
        if evidence_id in evidence_by_id
    )
    if not cited_food:
        return True
    claim_fields = {
        field
        for field, values in _numeric_facts(claim.text).items()
        if values
    }
    return bool(claim_fields) and claim_fields.issubset(requested_fields)


def _structured_comparison_answer(
    question: str,
    plan: QueryPlan,
    evidence_items: list[Evidence],
    constraints: ConstraintResult,
) -> VerifiedAnswer | None:
    """Answer named-food comparisons from recorded fields, not model guesses."""
    if plan.intent != "comparison":
        return None

    matched: list[tuple[tuple[int, int, int], Evidence]] = []
    seen: set[str] = set()
    for item in evidence_items:
        score = _food_match_score(question, item)
        if score is not None and item.evidence_id not in seen:
            seen.add(item.evidence_id)
            matched.append((score, item))
    matched.sort(key=lambda pair: pair[0], reverse=True)
    foods = [item for _, item in matched]
    if len(foods) < 2:
        return None

    requested = []
    q = question.casefold()
    for label, unit, aliases in (
        ("Calories", "kcal", ("calorie", "calories", "kcal")),
        ("Protein", "g", ("protein",)),
        ("Carbohydrates", "g", ("carbohydrate", "carbohydrates", "carbs")),
        ("Fat", "g", ("fat",)),
    ):
        if any(re.search(rf"\b{re.escape(alias)}\b", q) for alias in aliases):
            requested.append((label, unit))
    if not requested:
        requested = [("Calories", "kcal"), ("Protein", "g")]

    values: dict[str, dict[str, float]] = {}
    display: dict[str, dict[str, str]] = {}
    for item in foods:
        name = str(item.metadata.get("name", "Food item"))
        values[name] = {}
        display[name] = {}
        for label, unit in requested:
            raw = _record_value(item, label, unit)
            if raw is not None:
                try:
                    values[name][label] = float(raw)
                    display[name][label] = raw
                except ValueError:
                    continue

    # Only compare the two strongest explicitly named food matches.
    foods = foods[:2]
    names = [str(item.metadata.get("name", "Food item")) for item in foods]
    claims: list[str] = []
    for label, unit in requested:
        if all(label in values.get(name, {}) for name in names):
            claims.append(
                f"{names[0]}: {label.lower()} {display[names[0]][label]} {unit} per recorded serving; "
                f"{names[1]}: {label.lower()} {display[names[1]][label]} {unit} per recorded serving."
            )
    if not claims:
        return None

    if re.search(r"\b(?:which|who)\b.{0,80}\b(?:fewer|less|lower)\b.{0,80}\bcalories?\b", q) or re.search(r"\bwhich\s+has\s+fewer\s+calories?\b", q):
        if all("Calories" in values.get(name, {}) for name in names):
            lower = min(names, key=lambda name: values[name]["Calories"])
            claims.append(f"{lower} has fewer recorded calories per serving.")

    warnings = list(constraints.warnings)
    if any(item.is_estimate for item in foods):
        warnings.append("Some cited nutrition values are dataset estimates and should not be treated as authoritative measurements.")
    if any(item.authority_score < 0.5 for item in foods):
        warnings.append("Some cited records have limited source verification; treat the information as provisional.")
    return VerifiedAnswer(
        answer=" ".join(claims),
        citations=[_citation_for(item) for item in foods],
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
        if constraints.has_known_conflict:
            refusal = (
                "I can't safely recommend a specific food because the retrieved "
                "food evidence conflicts with your dietary or allergen "
                "constraints."
            )
        elif constraints.allergen_status_unknown:
            refusal = (
                "I can't safely recommend a specific food because its allergen "
                "information is missing or unclear. Please check the food label "
                "or consult a qualified dietitian."
            )
        elif constraints.dietary_status_unknown:
            refusal = (
                "I can't safely recommend a specific food because its dietary "
                "compatibility could not be verified from the available records."
            )
        else:
            refusal = (
                "I can't safely recommend a specific food because the available "
                "records do not verify its ingredients and compatibility with "
                "your dietary constraints. Please check the food label or consult "
                "a qualified dietitian."
            )
        return VerifiedAnswer(
            answer=refusal,
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

    comparison_answer = _structured_comparison_answer(
        question, plan, evidence_items, constraints
    )
    if comparison_answer is not None:
        return comparison_answer

    structured_answer = _structured_lookup_answer(
        question,
        plan,
        evidence_items,
        constraints,
    )
    if structured_answer is not None:
        return structured_answer
    if plan.structured_lookup and not plan.constraints.numeric_constraints:
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

    try:
        generated = _generate(_build_prompt(question, plan, evidence_items))
    except ModelOutputError:
        return VerifiedAnswer(
            answer=(
                "I do not have enough supported evidence to provide a reliable "
                "answer to that question."
            ),
            citations=[],
            warnings=[*constraints.warnings, MALFORMED_MODEL_OUTPUT_WARNING],
            supported_claims=0,
            rejected_claims=1,
        )
    evidence_by_id = {item.evidence_id: item for item in evidence_items}
    try:
        verifications = []
        for claim in generated.claims:
            verification = _claim_verification(
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
            if not _claim_is_in_numeric_scope(
                claim,
                plan,
                evidence_by_id,
            ):
                verification = ClaimVerification("insufficient")
            verifications.append((claim, verification))
    except (OSError, RuntimeError, ValueError) as error:
        raise EvidenceVerificationError(
            "Evidence verification failed; no unverified answer was returned."
        ) from error
    supported = [
        (claim, verification)
        for claim, verification in verifications
        if verification.status == "supported"
    ]
    rejected_count = len(generated.claims) - len(supported)
    contradicted_count = sum(
        verification.status == "contradicted"
        for _, verification in verifications
    )
    insufficient_count = sum(
        verification.status == "insufficient"
        for _, verification in verifications
    )
    citation_ids = list(
        dict.fromkeys(
            evidence_id
            for _, verification in supported
            for evidence_id in verification.evidence_ids
        )
    )
    cited_evidence = [evidence_by_id[item_id] for item_id in citation_ids]
    warnings = list(constraints.warnings)

    if contradicted_count:
        warnings.append(
            f"{contradicted_count} generated claim(s) contradicted the retrieved "
            "evidence and were removed."
        )
    if insufficient_count:
        warnings.append(
            f"{insufficient_count} generated claim(s) lacked sufficient "
            "supporting evidence and were removed."
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
        rendered_claims: list[str] = []
        for claim, verification in supported:
            supporting_items = [
                evidence_by_id[item_id]
                for item_id in verification.evidence_ids
            ]
            if supporting_items and all(
                item.kind == "document" for item in supporting_items
            ):
                rendered_claims.append(
                    f'The cited source states: "{claim.text.strip()}"'
                )
            else:
                rendered_claims.append(claim.text.strip())
        answer = " ".join(rendered_claims)
        citations = [_citation_for(item) for item in cited_evidence]

    return VerifiedAnswer(
        answer=answer,
        citations=citations,
        warnings=warnings,
        supported_claims=len(supported),
        rejected_claims=rejected_count,
    )
