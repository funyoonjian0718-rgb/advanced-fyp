import re
from dataclasses import dataclass


DIET_ALIASES = {
    "vegetarian": ("vegetarian", "veggie"),
    "vegan": ("vegan", "plant-based", "plant based"),
    "halal": ("halal",),
    "kosher": ("kosher",),
    "pescatarian": ("pescatarian",),
}

ALLERGEN_ALIASES = {
    "peanut": ("peanut", "peanuts"),
    "tree nuts": ("tree nut", "tree nuts", "almond", "cashew", "walnut"),
    "milk": ("milk", "dairy", "lactose"),
    "egg": ("egg", "eggs"),
    "soy": ("soy", "soya"),
    "wheat": ("wheat",),
    "gluten": ("gluten",),
    "sesame": ("sesame",),
    "fish": ("fish",),
    "shellfish": ("shellfish", "prawn", "shrimp", "crab", "lobster"),
}

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
    "nutrition",
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

COMMAND_LINE_PREFIX = re.compile(
    r"^\s*(?:"
    r"uvicorn|python(?:\.exe)?|py(?:\.exe)?|pip(?:\d+(?:\.\d+)?)?|"
    r"npm|npx|node|git|alembic|pytest|docker|powershell|"
    r"set-executionpolicy"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryConstraints:
    diets: tuple[str, ...]
    allergens: tuple[str, ...]
    activity_level: str | None


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


def _extract_diets(text: str) -> tuple[str, ...]:
    normalized = _normalise(text)
    matches = [
        diet
        for diet, aliases in DIET_ALIASES.items()
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases)
    ]
    return tuple(dict.fromkeys(matches))


def _extract_allergens(text: str) -> tuple[str, ...]:
    normalized = _normalise(text)
    matches: list[str] = []
    for allergen, aliases in ALLERGEN_ALIASES.items():
        found = False
        for alias in aliases:
            alias_pattern = re.escape(alias)
            if not re.search(rf"\b{alias_pattern}\b", normalized):
                continue
            if re.search(
                rf"\b(?:not\s+allergic\s+to|no\s+allerg(?:y|ies)\s+to)\s+"
                rf"{alias_pattern}\b",
                normalized,
            ):
                continue
            found = True
            break
        if found and allergen not in matches:
            matches.append(allergen)
    return tuple(matches)


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
    conversation_summary: str = "",
) -> QueryPlan:
    normalized = _normalise(message)
    command_line = COMMAND_LINE_PREFIX.match(message) is not None
    profile_diets = _extract_diets(dietary_preference or "")
    explicit_diets = _extract_diets(message)
    diets = tuple(dict.fromkeys((*profile_diets, *explicit_diets)))
    profile_allergens = _extract_allergens(allergies or "")
    explicit_allergens = _extract_allergens(message)
    allergens = tuple(dict.fromkeys((*profile_allergens, *explicit_allergens)))

    tokens = {token.strip(".,?!:;") for token in normalized.split()}
    structured_lookup = bool(NUTRIENT_TERMS.intersection(tokens))
    document_lookup = bool(
        GUIDELINE_TERMS.intersection(tokens)
    )
    recommendation_requested = bool(
        re.search(
            r"\b(?:recommendations?\s+for\s+me|what should i eat|"
            r"what can i eat|meal ideas|food options|foods can i eat|"
            r"recommend\s+(?:a\s+|some\s+)?"
            r"(?:foods?|meals?|lunch|dinner|breakfast|snacks?)|"
            r"suggest\s+(?:foods?|meals?|a lunch|a dinner|snacks?)|"
            r"could i consider|what.{0,40}\b(?:foods?|meals?|lunch|dinner|"
            r"breakfast|snacks?)|which.{0,40}\b(?:foods?|meals?|lunch|dinner|"
            r"breakfast|snacks?))\b",
            normalized,
        )
    )

    if command_line:
        intent = "out_of_scope"
    elif recommendation_requested:
        intent = "personalized_recommendation"
    elif structured_lookup and document_lookup:
        intent = "hybrid_question"
    elif structured_lookup and not document_lookup:
        intent = "structured_lookup"
    elif document_lookup:
        intent = "document_evidence"
    else:
        intent = "general_dietary_question"

    personalized = bool(diets or allergens or recommendation_requested)
    query_context = " ".join(part for part in (conversation_summary, message) if part)
    queries = _expand_query(query_context)
    subquestions = _decompose(message)

    return QueryPlan(
        intent=intent,
        queries=queries,
        subquestions=subquestions,
        constraints=QueryConstraints(
            diets=diets,
            allergens=allergens,
            activity_level=activity_level,
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
