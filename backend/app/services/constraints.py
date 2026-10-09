from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from app.services.evidence import Evidence

if TYPE_CHECKING:
    from app.services.query_processor import QueryPlan

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

ALLERGEN_ALIASES = {
    "peanut": ("peanut", "peanuts", "groundnut", "groundnuts"),
    "tree nuts": (
        "tree nut",
        "tree nuts",
        "nuts",
        "almond",
        "almonds",
        "cashew",
        "cashews",
        "walnut",
        "walnuts",
        "pistachio",
        "pistachios",
        "pecan",
        "pecans",
        "hazelnut",
        "hazelnuts",
    ),
    "milk": ("milk", "dairy", "lactose", "casein", "whey"),
    "egg": ("egg", "eggs"),
    "soy": ("soy", "soya", "soybean", "soybeans"),
    "wheat": ("wheat",),
    "gluten": ("gluten", "celiac", "coeliac"),
    "sesame": ("sesame", "sesame seed", "sesame seeds"),
    "fish": ("fish", "salmon", "tuna", "cod", "anchovy", "anchovies"),
    "shellfish": (
        "shellfish",
        "prawn",
        "prawns",
        "shrimp",
        "crab",
        "lobster",
        "crayfish",
        "mollusc",
        "molluscs",
        "mussel",
        "mussels",
        "oyster",
        "oysters",
    ),
}

@dataclass
class ConstraintResult:
    allowed: list[Evidence]
    excluded: list[Evidence]
    warnings: list[str]
    has_known_conflict: bool = False
    allergen_status_unknown: bool = False
    dietary_status_unknown: bool = False


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


def _declaration_is_verified(evidence: Evidence, data_type: str) -> bool:
    return (
        evidence.metadata.get(f"{data_type}_data_reviewed") is True
        and evidence.metadata.get(f"{data_type}_data_verified") is True
    )


@dataclass(frozen=True)
class NumericConstraint:
    field: str
    operator: str
    value: float


@dataclass(frozen=True)
class QueryConstraints:
    diets: tuple[str, ...]
    allergens: tuple[str, ...]
    activity_level: str | None
    numeric_constraints: tuple[NumericConstraint, ...] = ()
    unknown_dietary_constraint: bool = False
    unknown_allergen_constraint: bool = False

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


NUMERIC_FIELD_ALIASES = {
    "calories": "calories",
    "kcal": "calories",
    "protein": "protein_g",
    "carbohydrates": "carbohydrates_g",
    "carbs": "carbohydrates_g",
    "carbohydrate": "carbohydrates_g",
    "fat": "fat_g",
    "fiber": "fiber_g",
    "fibre": "fiber_g",
    "sugar": "sugar_g",
    "sodium": "sodium_mg",
    "potassium": "potassium_mg",
    "glycemic index": "glycemic_index",
    "gi": "glycemic_index",
}


COMPARISON_OPERATORS = (
    (
        r"(?:less than|under|below|fewer than)",
        "<",
    ),
    (
        r"(?:at most|no more than)",
        "<=",
    ),
    (
        r"(?:more than|over|above|greater than)",
        ">",
    ),
    (
        r"(?:at least|no less than)",
        ">=",
    ),
    (
        r"(?:equal to|exactly)",
        "=",
    ),
)


def _normalise_numeric_field(field: str) -> str | None:
    return NUMERIC_FIELD_ALIASES.get(field.casefold().strip())


def extract_numeric_constraints(
    text: str,
) -> tuple[NumericConstraint, ...]:
    """
    Extract numerical nutrition constraints from a natural-language query.

    Supports both:

        "less than 300 calories"
        "calories less than 300"

    Example result:

        calories < 300
        protein_g >= 10
    """
    constraints: list[NumericConstraint] = []
    normalized = text.casefold()

    field_pattern = (
        r"(calories?|kcal|protein|carbohydrates?|carbs?|"
        r"fat|fib(?:er|re)|sugar|sodium|potassium|"
        r"glycemic index|gi)"
    )

    operator_pattern = (
    r"(?:less than|under|below|fewer than|"
    r"at most|no more than|"
    r"more than|over|above|greater than|"
    r"at least|no less than|"
    r"equal to|exactly)"
    )

        # ---------------------------------------------------------
    # Range pattern:
    #
    # "between 200 and 400 calories"
    # "protein between 10g and 30g"
    # ---------------------------------------------------------
    range_pattern = re.compile(
        rf"\b(?:between\s+)?"
        rf"(?P<value_min>\d+(?:\.\d+)?)\s*"
        rf"(?:g|mg|kcal|calories?)?\s+"
        rf"and\s+"
        rf"(?P<value_max>\d+(?:\.\d+)?)\s*"
        rf"(?:g|mg|kcal|calories?)?\s*"
        rf"(?P<field>{field_pattern})\b",
        flags=re.IGNORECASE,
    )

    field_first_range_pattern = re.compile(
        rf"\b(?P<field>{field_pattern})\s+"
        rf"between\s+"
        rf"(?P<value_min>\d+(?:\.\d+)?)\s*"
        rf"(?:g|mg|kcal|calories?)?\s+"
        rf"and\s+"
        rf"(?P<value_max>\d+(?:\.\d+)?)\s*"
        rf"(?:g|mg|kcal|calories?)?\b",
        flags=re.IGNORECASE,
    )

    # Operator-style range:
    # "between 200 and 400 calories"
    for match in range_pattern.finditer(normalized):
        field = _normalise_numeric_field(match.group("field"))

        if field is None:
            continue

        constraints.append(
            NumericConstraint(
                field=field,
                operator=">=",
                value=float(match.group("value_min")),
            )
        )

        constraints.append(
            NumericConstraint(
                field=field,
                operator="<=",
                value=float(match.group("value_max")),
            )
        )

    # Field-first range:
    # "calories between 200 and 400"
    for match in field_first_range_pattern.finditer(normalized):
        field = _normalise_numeric_field(match.group("field"))

        if field is None:
            continue

        constraints.append(
            NumericConstraint(
                field=field,
                operator=">=",
                value=float(match.group("value_min")),
            )
        )

        constraints.append(
            NumericConstraint(
                field=field,
                operator="<=",
                value=float(match.group("value_max")),
            )
        )

    # ---------------------------------------------------------
    # Pattern 1:
    # operator -> value -> field
    #
    # Example:
    # "less than 300 calories"
    # "at least 10g protein"
    # ---------------------------------------------------------
    operator_first_pattern = re.compile(
    rf"\b(?P<operator>{operator_pattern})\s+"
    rf"(?P<value>\d+(?:\.\d+)?)\s*(?:g|mg|kcal|calories?)?\s*"
    rf"(?P<field>{field_pattern})\b",
    flags=re.IGNORECASE,
)

    # ---------------------------------------------------------
    # Pattern 2:
    # field -> operator -> value
    #
    # Example:
    # "calories less than 300"
    # "protein at least 10g"
    # ---------------------------------------------------------
    field_first_pattern = re.compile(
    rf"\b(?P<field>{field_pattern})\s+"
    rf"(?P<operator>{operator_pattern})\s+"
    rf"(?P<value>\d+(?:\.\d+)?)\s*(?:g|mg|kcal|calories?)?\b",
    flags=re.IGNORECASE,
)

    def operator_from_text(operator_text: str) -> str:
        for pattern_text, operator in COMPARISON_OPERATORS:
            if re.fullmatch(
                pattern_text,
                operator_text,
                flags=re.IGNORECASE,
            ):
                return operator

        raise ValueError(f"Unsupported comparison operator: {operator_text}")

    # Operator-first matches
    for match in operator_first_pattern.finditer(normalized):
        operator_text = match.group("operator")
        value_text = match.group("value")
        field_text = match.group("field")

        field = _normalise_numeric_field(field_text)

        if field is None:
            continue

        constraints.append(
            NumericConstraint(
                field=field,
                operator=operator_from_text(operator_text),
                value=float(value_text),
            )
        )

    # Field-first matches
    for match in field_first_pattern.finditer(normalized):
        field_text = match.group("field")
        operator_text = match.group("operator")
        value_text = match.group("value")

        field = _normalise_numeric_field(field_text)

        if field is None:
            continue

        constraints.append(
            NumericConstraint(
                field=field,
                operator=operator_from_text(operator_text),
                value=float(value_text),
            )
        )

    return tuple(constraints)

def enforce_constraints(
    evidence_items: list[Evidence],
    plan: QueryPlan,
) -> ConstraintResult:
    has_food_constraints = bool(
        plan.constraints.diets
        or plan.constraints.allergens
        or plan.constraints.unknown_dietary_constraint
        or plan.constraints.unknown_allergen_constraint
    )
    if not has_food_constraints or (
        not plan.recommendation_requested
        and not plan.constraints.numeric_constraints
    ):
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
    has_known_conflict = False
    allergen_status_unknown = plan.constraints.unknown_allergen_constraint
    dietary_status_unknown = plan.constraints.unknown_dietary_constraint

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
                    has_known_conflict = True
                    break
                if not _declaration_is_verified(evidence, "allergen"):
                    unknown_allergen_data = True
                    allergen_status_unknown = True

            if not conflict:
                for diet in plan.constraints.diets:
                    status = _diet_status(evidence, diet)
                    if status == "conflict":
                        conflict = True
                        has_known_conflict = True
                        break
                    if (
                        status == "unknown"
                        or not _declaration_is_verified(evidence, "dietary")
                    ):
                        unknown_status = True
                        unknown_diet_data = True
                        dietary_status_unknown = True

        if conflict:
            excluded.append(evidence)
        elif evidence.kind == "food" and (
            unknown_status
            or (
                plan.constraints.allergens
                and not _declaration_is_verified(evidence, "allergen")
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
    if has_known_conflict:
        warnings.append(
            "Some retrieved foods conflict with a known dietary or allergen "
            "restriction and were excluded."
        )
    if excluded:
        warnings.append(
            "Some retrieved foods were excluded because they conflict with, or "
            "cannot be verified against, the requested dietary constraints."
        )

    return ConstraintResult(
        allowed,
        excluded,
        warnings,
        has_known_conflict=has_known_conflict,
        allergen_status_unknown=allergen_status_unknown,
        dietary_status_unknown=dietary_status_unknown,
    )
