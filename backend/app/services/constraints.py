import re
from dataclasses import dataclass
from typing import Literal

from app.services.evidence import Evidence
from app.services.query_processor import ALLERGEN_ALIASES, QueryPlan

DIET_CONFLICT_TERMS = {
    "vegetarian": (
        "chicken",
        "beef",
        "pork",
        "lamb",
        "mutton",
        "fish",
        "prawn",
        "shrimp",
        "crab",
        "seafood",
        "meat",
        "anchovy",
    ),
    "vegan": (
        "chicken",
        "beef",
        "pork",
        "lamb",
        "mutton",
        "fish",
        "prawn",
        "shrimp",
        "crab",
        "seafood",
        "meat",
        "milk",
        "dairy",
        "egg",
        "honey",
        "anchovy",
    ),
    "pescatarian": ("chicken", "beef", "pork", "lamb", "mutton", "meat"),
}


@dataclass
class ConstraintResult:
    allowed: list[Evidence]
    excluded: list[Evidence]
    warnings: list[str]


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term.casefold())}\b", text.casefold()) is not None


def _allergen_conflict(evidence: Evidence, allergen: str) -> bool:
    declared = evidence.metadata.get("allergens")
    if isinstance(declared, list):
        if any(_contains_term(str(item), allergen) for item in declared):
            return True
    text_parts = (
        str(evidence.metadata.get("name", "")),
        str(evidence.metadata.get("aliases", "")),
        str(evidence.metadata.get("ingredients", "")),
        evidence.text,
    )
    aliases = ALLERGEN_ALIASES.get(allergen, (allergen,))
    return any(
        _contains_term(part, alias)
        for part in text_parts
        for alias in aliases
    )


DietStatus = Literal["supported", "conflict", "unknown"]


def _diet_status(evidence: Evidence, diet: str) -> DietStatus:
    labels = evidence.metadata.get("dietary_labels")
    normalized_labels = (
        {
            re.sub(r"[_-]+", " ", str(label).casefold()).strip()
            for label in labels
            if str(label).strip()
        }
        if isinstance(labels, list)
        else set()
    )

    explicit_negative = any(
        label in {
            f"non {diet}",
            f"not {diet}",
            f"not suitable for {diet}",
        }
        or f"not suitable for {diet}" in label
        for label in normalized_labels
    )
    text = " ".join(
        str(evidence.metadata.get(field, ""))
        for field in ("name", "category", "ingredients")
    ).casefold()
    if explicit_negative or any(
        _contains_term(text, term)
        for term in DIET_CONFLICT_TERMS.get(diet, ())
    ):
        return "conflict"

    supported_labels = {diet}
    if diet == "vegetarian":
        supported_labels.add("vegan")
    if any(
        label in supported_labels
        or label in {f"{diet}-friendly", f"{diet} friendly"}
        or label == f"suitable for {diet}"
        for label in normalized_labels
    ):
        return "supported"

    return "unknown"


def enforce_constraints(
    evidence_items: list[Evidence],
    plan: QueryPlan,
) -> ConstraintResult:
    if not plan.recommendation_requested:
        return ConstraintResult(evidence_items, [], [])

    allowed: list[Evidence] = []
    excluded: list[Evidence] = []
    warnings: list[str] = []
    if plan.constraints.unknown_allergen_constraint:
        warnings.append(
            "An allergy or intolerance could not be identified reliably, so no "
            "food recommendation can be verified against it."
        )
    if plan.constraints.unknown_dietary_constraint:
        warnings.append(
            "A dietary restriction could not be identified reliably, so no food "
            "recommendation can be verified against it."
        )
    unknown_allergen_data = False
    unknown_diet_data = False

    for evidence in evidence_items:
        conflict = False
        unknown_status = False
        if (
            evidence.kind == "food"
            and (
                plan.constraints.unknown_allergen_constraint
                or plan.constraints.unknown_dietary_constraint
            )
        ):
            excluded.append(evidence)
            continue
        if evidence.kind == "food":
            for allergen in plan.constraints.allergens:
                if _allergen_conflict(evidence, allergen):
                    conflict = True
                    break
                if evidence.metadata.get("allergen_data_verified") is not True:
                    unknown_allergen_data = True

            if not conflict:
                for diet in plan.constraints.diets:
                    status = _diet_status(evidence, diet)
                    if status == "conflict":
                        conflict = True
                        break
                    if (
                        status == "unknown"
                        or evidence.metadata.get("dietary_data_verified") is not True
                    ):
                        unknown_status = True
                        unknown_diet_data = True

        if conflict:
            excluded.append(evidence)
        elif evidence.kind == "food" and (
            unknown_status
            or (
                plan.constraints.allergens
                and evidence.metadata.get("allergen_data_verified") is not True
            )
        ):
            excluded.append(evidence)
        else:
            allowed.append(evidence)

    if unknown_allergen_data:
        warnings.append(
            "The nutrition dataset does not provide verified allergen declarations "
            "for every food, so foods with unknown allergen status were not recommended."
        )
    if unknown_diet_data:
        warnings.append(
            "The nutrition dataset does not verify the requested dietary status "
            "for every food, so foods with unknown status were not recommended."
        )
    if excluded:
        warnings.append(
            "Some retrieved foods were excluded because they conflict with, or "
            "cannot be verified against, the requested dietary constraints."
        )

    return ConstraintResult(allowed, excluded, warnings)
