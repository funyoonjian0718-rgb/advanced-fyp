import re
from dataclasses import dataclass


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
    unknown_dietary_constraint: bool = False
    unknown_allergen_constraint: bool = False


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

    has_constraint_context = bool(
        re.search(
            r"\b(?:allerg\w*|intoleran\w*|anaphyl\w*|dietary\s+"
            r"(?:restriction|requirement|preference)|keto\w*|paleo|"
            r"low[\s-]?(?:carb|fat|sodium)|carnivore|whole[\s-]?30|"
            r"fodmap|mediterranean|dash|diabet\w*|celiac|coeliac|"
            r"hypertension|high\s+blood\s+pressure|renal\s+disease|"
            r"kidney\s+disease)\b",
            normalized,
        )
    )
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
        r"which|this|those|these|me|please|tell|about|contain|contains|"
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
    normalized = _normalise(message)
    command_line = COMMAND_LINE_PREFIX.match(message) is not None
    profile_diets = _extract_diets(dietary_preference or "")
    explicit_diets = _extract_diets(message)
    diets = tuple(dict.fromkeys((*profile_diets, *explicit_diets)))
    profile_allergens = _extract_allergens(allergies or "")
    explicit_allergens = _extract_allergens(message)
    allergens = tuple(dict.fromkeys((*profile_allergens, *explicit_allergens)))
    unknown_allergen_constraint = _has_unknown_constraint(
        allergies or "",
        ALLERGEN_ALIASES,
        profile_value=True,
    ) or _has_unknown_constraint(
        message,
        ALLERGEN_ALIASES,
        profile_value=False,
    )
    unknown_dietary_constraint = _has_unknown_constraint(
        dietary_preference or "",
        DIET_ALIASES,
        profile_value=True,
    ) or _has_unknown_constraint(
        message,
        DIET_ALIASES,
        profile_value=False,
    )

    tokens = {token.strip(".,?!:;") for token in normalized.split()}
    structured_lookup = bool(
        NUTRIENT_TERMS.intersection(tokens)
        or re.search(
            r"\bnutrition\s+(?:information|values|facts)\b",
            normalized,
        )
        or re.search(r"\b(?:serving|portion)\s+size\b", normalized)
    )
    document_lookup = bool(
        GUIDELINE_TERMS.intersection(tokens)
    )
    recommendation_requested = bool(
        re.search(
            r"\b(?:recommendations?\s+for\s+me|what should i eat|"
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
    elif structured_lookup and not document_lookup:
        intent = "structured_lookup"
    elif document_lookup:
        intent = "document_evidence"
    else:
        intent = "general_dietary_question"

    personalized = bool(diets or allergens or recommendation_requested)
    queries = _expand_query(message)
    subquestions = _decompose(message)

    return QueryPlan(
        intent=intent,
        queries=queries,
        subquestions=subquestions,
        constraints=QueryConstraints(
            diets=diets,
            allergens=allergens,
            activity_level=activity_level,
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
