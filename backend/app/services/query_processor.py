import re
from dataclasses import dataclass
from app.services.constraints import (
    ALLERGEN_ALIASES,
    QueryConstraints,
    extract_numeric_constraints,
)


DIET_ALIASES = {
    "vegetarian": ("vegetarian", "veggie"),
    "vegan": ("vegan", "plant-based", "plant based"),
    "halal": ("halal",),
    "kosher": ("kosher",),
    "pescatarian": ("pescatarian",),
}

UNSUPPORTED_DIET_TERMS = (
    "carnivore",
    "whole 30",
    "whole30",
    "fodmap",
    "low-fat",
    "low fat",
    "low-sodium",
    "low sodium",
    "mediterranean",
    "dash",
    "paleo",
    "keto",
    "ketogenic",
    "low-carb",
    "low carb",
)


EXPANSIONS = {
    "protein": ("protein-rich foods", "dietary protein sources"),
    "sodium": ("salt intake", "sodium content"),
    "calories": ("energy content", "kilocalories per serving"),
    "fibre": ("dietary fiber", "high-fibre foods"),
    "fiber": ("dietary fibre", "high-fibre foods"),
    "guideline": ("official dietary recommendations", "nutrition guidance"),
}

NUTRIENT_TERMS = {
    "calorie",
    "calories",
    "kcal",
    "energy",
    "protein",
    "carbohydrate",
    "carbohydrates",
    "fat",
    "fibre",
    "fiber",
    "sugar",
    "sodium",
    "potassium",
    "glycemic",
}

GUIDELINE_TERMS = {
    "guideline",
    "guidelines",
    "recommendation",
    "recommendations",
    "research",
    "evidence",
    "why",
    "explain",
    "health effects",
}

# Words and phrases that usually indicate the user wants to compare
# two or more specific foods.
COMPARISON_PATTERNS = (
    r"\bcompare\s+"
    r"(?!how\s+can\s+i\b)",

    r"\bcomparison\s+of\s+"
    r"(?:the\s+)?(?:two\s+)?"
    r"(?:foods?|meals?|dishes?|products?)\b",

    r"\b(?:versus|vs\.?)\b",

    r"\bdifference\s+between\b"
    r".{0,120}\b(?:food|foods|meal|meals|dish|dishes|"
    r"nutrition|calories|protein|carbohydrates|carbs|fat|"
    r"fiber|fibre|sugar|sodium|potassium|glycemic|diabetes)\b",

    r"\bwhich\b.{0,100}"
    r"\b(?:more|less|higher|lower|fewer|greater|better)\b"
    r".{0,100}\b(?:than|between|or)\b",

    r"\bwhich\b.{0,120}\bbetween\b.{0,150}\band\b",

    r"\bbetween\b.{0,100}\band\b.{0,100}\bwhich\b",
)

# Words and phrases that indicate numerical or categorical filtering
# over multiple food records.
CONSTRAINT_SEARCH_PATTERNS = (
    r"\b(?:less than|fewer than|under|below|at most|no more than)\b"
    r".{0,40}\b(?:calories?|kcal|protein|carbohydrates?|carbs?|"
    r"fat|fibre|fiber|sugar|sodium|potassium)\b",
    r"\b(?:more than|greater than|over|above|at least|no less than)\b"
    r".{0,40}\b(?:calories?|kcal|protein|carbohydrates?|carbs?|"
    r"fat|fibre|fiber|sugar|sodium|potassium)\b",
    r"\bbetween\b.{0,100}\b(?:calories?|kcal|protein|carbohydrates?|"
    r"carbs?|fat|fibre|fiber|sugar|sodium|potassium)\b",
    r"\b(?:foods?|meals?|options?)\b.{0,80}"
    r"\b(?:under|below|less than|fewer than|over|above|more than|"
    r"greater than|at least)\b",
    r"\bnot marked\b.{0,40}\b(?:limit|moderate|suitable)\b",
    r"\b(?:marked|rated)\b.{0,30}\b(?:limit|moderate|suitable)\b",
)

# Questions that ask the system to discover multiple foods based on
# a nutrition property instead of looking up one named food.
SEMANTIC_SEARCH_PATTERNS = (
    r"\bwhich foods\b",
    r"\bwhat foods\b",
    r"\bwhat are some foods\b",
    r"\bfoods that\b",
    r"\bfoods with\b",
    r"\bgood sources? of\b",
    r"\bsource[s]? of\b",
    r"\bhigh[- ](?:protein|fibre|fiber|calories?|sugar|sodium)\b",
    r"\blow[- ](?:calorie|calories|carb|carbs|sugar|sodium)\b",
    r"\brelatively high\b",
    r"\brelatively low\b",
    r"\bwhich\b.{0,80}\bfoods?\b.{0,50}\b(?:lower|higher)\b"
    r".{0,40}\b(?:calories?|protein|fibre|fiber|sugar|sodium|fat)\b",
)

COMMAND_LINE_PREFIX = re.compile(
    r"^\s*(?:"
    r"uvicorn|python(?:\.exe)?|py(?:\.exe)?|pip(?:\d+(?:\.\d+)?)?|"
    r"npm|npx|node|git|alembic|pytest|docker|powershell|"
    r"set-executionpolicy"
    r")\b",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class QueryPlan:
    intent: str
    queries: tuple[str, ...]
    subquestions: tuple[str, ...]
    constraints: QueryConstraints
    personalized: bool
    recommendation_requested: bool
    structured_lookup: bool
    document_lookup: bool


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _normalise_known_diet_typos(value: str) -> str:
    """Correct a small allowlist of unambiguous common dietary misspellings.

    This is intentionally not fuzzy matching: unknown diet terms should remain
    unknown and be handled by the existing clarification/safety logic.
    """
    replacements = {
        r"\bvegeterian\b": "vegetarian",
        r"\bvegatarian\b": "vegetarian",
        r"\bvegeterianism\b": "vegetarianism",
        r"\bvegeterian\b": "vegetarian",
    }
    result = value
    for pattern, correction in replacements.items():
        result = re.sub(pattern, correction, result, flags=re.IGNORECASE)
    return result


def _extract_diets(text: str) -> tuple[str, ...]:
    normalized = _normalise(text)
    matches: list[str] = []
    for diet, aliases in DIET_ALIASES.items():
        for alias in aliases:
            for occurrence in re.finditer(rf"\b{re.escape(alias)}\b", normalized):
                prefix = normalized[max(0, occurrence.start() - 35) : occurrence.start()]
                if re.search(
                    r"\b(?:not|never|no\s+longer|don't\s+follow|do\s+not\s+follow)"
                    r"\s+(?:a\s+)?$",
                    prefix,
                ):
                    continue
                matches.append(diet)
                break
            if diet in matches:
                break
    return tuple(dict.fromkeys(matches))


def _has_unknown_constraint(
    text: str,
    known_aliases: dict[str, tuple[str, ...]],
    *,
    profile_value: bool,
    allergen_constraint: bool,
) -> bool:
    normalized = _normalise(text)
    if not normalized or normalized in {
        "none",
        "no",
        "n a",
        "na",
        "no preference",
        "no restrictions",
        "omnivore",
        "omnivorous",
        "no known allergies",
        "no known food allergies",
    }:
        return False

    context_pattern = (
        r"\b(?:allerg\w*|intoleran\w*|anaphyl\w*|celiac|coeliac)\b"
        if allergen_constraint
        else r"\b(?:dietary\s+(?:restriction|requirement|preference)|"
        r"keto\w*|paleo|low[\s-]?(?:carb|fat|sodium)|carnivore|"
        r"whole[\s-]?30|fodmap|mediterranean|dash|diabet\w*|"
        r"hypertension|high\s+blood\s+pressure|renal\s+disease|"
        r"kidney\s+disease)\b"
    )
    has_constraint_context = re.search(context_pattern, normalized) is not None
    residue = normalized
    if not profile_value:
        residue = re.sub(
            r"\b(?:not|never)\s+(?:be\s+)?(?:allergic|intolerant)\s+to\s+"
            r"[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2}",
            " ",
            residue,
        )
        residue = re.sub(
            r"\b(?:do\s+not|don't|dont)\s+have\s+(?:a\s+)?"
            r"[a-z][a-z-]*\s+(?:allerg(?:y|ies)|intoleran\w*)\b",
            " ",
            residue,
        )
    for aliases in known_aliases.values():
        for alias in sorted(aliases, key=len, reverse=True):
            residue = re.sub(rf"\b{re.escape(alias)}\b", " ", residue)
    residue = re.sub(
        r"\b(?:i|am|a|an|have|has|the|to|and|or|with|my|eat|follow|"
        r"prefer|preference|diet|dietary|allerg(?:y|ies|ic)|intoleran\w*|"
        r"severe|mild|food|foods|known|none|no|not|do|don't|dont|is|of|for|"
        r"free|based|plant|what|can|should|eat|with|any|that|safe|safely|"
        r"my|your|you|have|recommend|recommendation|recommendations|"
        r"which|this|those|these|me|please|tell|about|if|contain|contains|"
        r"high|low|protein|calories|meal|meals|options|am|are|do|does|"
        r"want|looking|following|follow|for|on|all|allergy|allergies|"
        r"disease)\b",
        " ",
        residue,
    )
    residue = re.sub(r"[^a-z0-9]+", " ", residue).strip()
    if not residue:
        if not has_constraint_context:
            return False
        return not any(
            re.search(rf"\b{re.escape(alias)}\b", normalized)
            for aliases in known_aliases.values()
            for alias in aliases
        )
    return profile_value or has_constraint_context


def _extract_allergens(text: str) -> tuple[str, ...]:
    normalized = _normalise(text)
    matches: list[str] = []
    for allergen, aliases in ALLERGEN_ALIASES.items():
        found = False
        for alias in aliases:
            alias_pattern = re.escape(alias)
            for occurrence in re.finditer(
                rf"\b{alias_pattern}\b",
                normalized,
            ):
                start, end = occurrence.span()
                context = normalized[max(0, start - 70) : min(len(normalized), end + 30)]
                escaped_alias = re.escape(alias)
                negative_patterns = (
                    rf"\b(?:not|never)\s+(?:be\s+)?(?:allergic|intolerant)"
                    rf"\s+(?:to\s+)?{escaped_alias}\b",
                    rf"\b(?:no|without)\s+(?:known\s+)?allerg(?:y|ies)"
                    rf"\s+(?:to\s+)?{escaped_alias}\b",
                    rf"\b(?:no|without)\s+{escaped_alias}\s+"
                    rf"(?:allerg(?:y|ies)|intoleran\w*)\b",
                    rf"\b(?:do\s+not|don't|dont)\s+have\s+(?:a\s+)?"
                    rf"{escaped_alias}\s+(?:allerg(?:y|ies)|intoleran\w*)\b",
                )
                if any(
                    re.search(pattern, context)
                    for pattern in negative_patterns
                ):
                    continue
                found = True
                break
            if found:
                break
        if found and allergen not in matches:
            matches.append(allergen)
    return tuple(matches)


def _matches_any_pattern(
    text: str,
    patterns: tuple[str, ...],
) -> bool:
    """
    Return True when the normalized question matches at least
    one of the supplied intent patterns.
    """
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def _is_comparison_query(text: str) -> bool:
    """
    Detect questions that ask the system to compare
    specific foods, meals, products, or nutrition values.

    Conceptual questions such as "What is the difference
    between a snack and a meal?" are not food-record
    comparisons and should remain general questions.
    """
    # Questions asking how to perform a comparison are
    # general questions, not comparison retrieval requests.
    if re.search(
        r"\bhow\s+(?:can|do|should)\s+i\s+compare\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False

    # Conceptual definition questions using "difference between"
    # should not be routed to food comparison retrieval.
    if re.search(
        r"\bwhat\s+is\s+the\s+difference\s+between\s+"
        r"a\s+snack\s+and\s+a\s+meal\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False

    return _matches_any_pattern(text, COMPARISON_PATTERNS)


def _is_constrained_search_query(text: str) -> bool:
    """
    Detect questions that ask for multiple foods satisfying
    numerical or categorical conditions.
    """
    return _matches_any_pattern(text, CONSTRAINT_SEARCH_PATTERNS)


def _is_semantic_search_query(text: str) -> bool:
    """
    Detect questions that ask the system to discover foods
    based on a nutritional property.
    """
    return _matches_any_pattern(text, SEMANTIC_SEARCH_PATTERNS)


def _expand_query(message: str) -> tuple[str, ...]:
    normalized = _normalise(message)
    queries = [normalized]
    for term, expansions in EXPANSIONS.items():
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            queries.extend(
                f"{normalized} {expansion}"
                for expansion in expansions[:1]
            )
    return tuple(dict.fromkeys(queries))[:4]


def _decompose(message: str) -> tuple[str, ...]:
    parts = re.split(
        r";|\b(?:and|also|without|while|whereas)\b",
        message,
        flags=re.IGNORECASE,
    )
    useful = tuple(
        part.strip(" ,.?")
        for part in parts
        if len(part.split()) >= 3
    )
    return useful[:4] if len(useful) > 1 else ()


def process_query(
    message: str,
    *,
    dietary_preference: str | None = None,
    allergies: str | None = None,
    activity_level: str | None = None,
) -> QueryPlan:
    parsed_message = _normalise_known_diet_typos(message)
    normalized = _normalise(parsed_message)
    command_line = COMMAND_LINE_PREFIX.match(message) is not None
    profile_diets = _extract_diets(dietary_preference or "")
    explicit_diets = _extract_diets(parsed_message)
    diets = tuple(dict.fromkeys((*profile_diets, *explicit_diets)))
    profile_allergens = _extract_allergens(allergies or "")
    explicit_allergens = _extract_allergens(message)
    allergens = tuple(dict.fromkeys((*profile_allergens, *explicit_allergens)))
    unknown_allergen_constraint = _has_unknown_constraint(
        allergies or "",
        ALLERGEN_ALIASES,
        profile_value=True,
        allergen_constraint=True,
    ) or _has_unknown_constraint(
        parsed_message,
        ALLERGEN_ALIASES,
        profile_value=False,
        allergen_constraint=True,
    )
    unknown_dietary_constraint = _has_unknown_constraint(
        dietary_preference or "",
        DIET_ALIASES,
        profile_value=True,
        allergen_constraint=False,
    ) or _has_unknown_constraint(
        parsed_message,
        DIET_ALIASES,
        profile_value=False,
        allergen_constraint=False,
    )

    numeric_constraints = extract_numeric_constraints(parsed_message)

    tokens = {token.strip(".,?!:;") for token in normalized.split()}
    structured_lookup = bool(
        NUTRIENT_TERMS.intersection(tokens)
        or re.search(
            r"\bnutrition\s+(?:information|values|facts)\b",
            normalized,
        )
        or (
            re.search(r"\b(?:serving|portion)\s+size\b", normalized)
            and not re.search(
                r"\bwhat\s+does\s+(?:a\s+)?(?:serving|portion)\s+size\s+mean\b",
                normalized,
            )
        )
    )

    # Classify more specific multi-record questions before the
    # generic structured nutrition lookup.
    comparison_query = _is_comparison_query(normalized)
    constrained_search = _is_constrained_search_query(normalized)
    semantic_search = _is_semantic_search_query(normalized)

    document_lookup = bool(
        GUIDELINE_TERMS.intersection(tokens)
    )
    recommendation_requested = bool(
        re.search(
            r"\b(?:meal\s+plans?|one[- ]day\s+(?:meal\s+)?plan|"
            r"plan\s+(?:my\s+)?meals?|recommendations?\s+for\s+me|what should i eat|"
            r"what can i eat|meal ideas|food options|foods can i eat|"
            r"recommend\s+(?:(?:a\s+few|some|a)\s+)?"
            r"(?:foods?|meals?|lunch|dinner|breakfast|snacks?)|"
            r"suggest\s+(?:(?:a\s+few|some|a)\s+)?"
            r"(?:foods?|meals?|lunch|dinner|breakfast|snacks?)|"
            r"could i consider|(?:what|which)\b.{0,40}\b"
            r"(?:foods?|meals?|lunch|dinner|breakfast|snacks?)\b"
            r".{0,30}\b(?:can|should|could|would)\s+i\s+"
            r"(?:eat|consider|choose|have))\b",
            normalized,
        )
    )

    if command_line:
        intent = "out_of_scope"
    elif recommendation_requested:
        intent = "personalized_recommendation"
    elif structured_lookup and document_lookup:
        intent = "hybrid_question"
    elif document_lookup:
        intent = "document_evidence"
    elif constrained_search:
        intent = "constrained_search"
    elif comparison_query:
        intent = "comparison"
    elif semantic_search:
        intent = "semantic_search"
    elif structured_lookup:
        intent = "structured_lookup"
    else:
        intent = "general_dietary_question"

    personalized = bool(diets or allergens or recommendation_requested)
    queries = _expand_query(parsed_message)
    subquestions = _decompose(parsed_message)

    return QueryPlan(
        intent=intent,
        queries=queries,
        subquestions=subquestions,
        constraints=QueryConstraints(
        diets=diets,
        allergens=allergens,
        activity_level=activity_level,
        numeric_constraints=numeric_constraints,
        unknown_dietary_constraint=unknown_dietary_constraint,
        unknown_allergen_constraint=unknown_allergen_constraint,
        ),
        personalized=personalized,
        recommendation_requested=recommendation_requested,
        structured_lookup=structured_lookup,
        document_lookup=(
            False
            if command_line
            else document_lookup or not structured_lookup
        ),
    )
