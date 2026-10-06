import re
from dataclasses import dataclass

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


def _diet_conflict(evidence: Evidence, diet: str) -> bool:
    labels = evidence.metadata.get("dietary_labels")
    if isinstance(labels, list):
        normalized = {str(label).casefold() for label in labels}
        if diet in normalized or (diet == "vegetarian" and "vegan" in normalized):
            return False
        return True

    text = " ".join(
        (
            str(evidence.metadata.get("name", "")),
            str(evidence.metadata.get("category", "")),
            str(evidence.metadata.get("ingredients", "")),
            evidence.text,
        )
    ).casefold()
    return any(
        _contains_term(text, term)
        for term in DIET_CONFLICT_TERMS.get(diet, ())
    )


def enforce_constraints(
    evidence_items: list[Evidence],
    plan: QueryPlan,
) -> ConstraintResult:
    if not plan.recommendation_requested:
        return ConstraintResult(evidence_items, [], [])

    allowed: list[Evidence] = []
    excluded: list[Evidence] = []
    warnings: list[str] = []
    unknown_allergen_data = False
    unknown_diet_data = False

    for evidence in evidence_items:
        conflict = False
        if evidence.kind == "food":
            for allergen in plan.constraints.allergens:
                if _allergen_conflict(evidence, allergen):
                    conflict = True
                    break
                if evidence.metadata.get("allergen_data_verified") is not True:
                    unknown_allergen_data = True

            if not conflict:
                for diet in plan.constraints.diets:
                    if _diet_conflict(evidence, diet):
                        conflict = True
                        break
                    if evidence.metadata.get("dietary_data_verified") is not True:
                        unknown_diet_data = True

        if conflict:
            excluded.append(evidence)
        elif evidence.kind == "food" and (
            (plan.constraints.allergens and evidence.metadata.get("allergen_data_verified") is not True)
            or (plan.constraints.diets and evidence.metadata.get("dietary_data_verified") is not True)
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
            "The nutrition dataset does not verify halal or kosher status for every "
            "food, so foods with unknown status were not recommended."
        )
    if excluded:
        warnings.append(
            "Some retrieved foods were excluded because they conflict with, or "
            "cannot be verified against, the requested dietary constraints."
        )

    return ConstraintResult(allowed, excluded, warnings)
