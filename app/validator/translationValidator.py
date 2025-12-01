import re
import json
import pycountry
from langdetect import detect
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

# Supported languages (ISO 639-1 codes or common names)
LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "pl": "Polish",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "el": "Greek",
    "cs": "Czech",
    "ur": "Urdu",
    "ar": "Arabic",
    "eu": "Basque",
}

# Supported countries (ISO 3166-1 alpha-2 codes)
COUNTRIES = {
    "US": "United States",
    "GB": "United Kingdom",
    "ES": "Spain",
    "MX": "Mexico",
    "FR": "France",
    "CA": "Canada",
    "DE": "Germany",
    "IT": "Italy",
    "PT": "Portugal",
    "BR": "Brazil",
    "RU": "Russia",
    "CN": "China",
    "JP": "Japan",
    "KR": "South Korea",
    "TR": "Turkey",
    "NL": "Netherlands",
    "SE": "Sweden",
    "PL": "Poland",
    "UA": "Ukraine",
    "RO": "Romania",
    "TH": "Thailand",
    "VN": "Vietnam",
    "ID": "Indonesia",
    "GR": "Greece",
    "CZ": "Czech Republic",
    "PK": "Pakistan",
    "SA": "Saudi",
    "AE": "United Arab Emirates",
    "EG": "Egypt",
    "IN": "India",
    "AR": "Argentina"
}

# Supported brand tones
BRAND_TONES = [
    "neutral",
    "formal",
    "informal",
    "premium",
    "luxury",
    "friendly",
    "professional",
    "casual",
    "canadian local French (Quebec French for Canada)",
    "European local French",
    "Egyption local arabic",
    "Saudi local arabic",
    "Spain Spanish (Castilian / Español de España)",
    "Mexican Spanish (Español de México).",
    "Argentinian Spanish(Español Rioplatense)."
]

# Common language-country associations (for regional dialect guidance)
LANGUAGE_COUNTRY_ASSOCIATIONS = {
    "en": ["US", "GB", "CA"],
    "es": ["ES", "MX"],
    "fr": ["FR", "CA"],
    "pt": ["PT", "BR"],
    "zh": ["CN"],
    "ja": ["JP"],
    "ko": ["KR"],
    "ar": ["SA", "AE", "EG"],
    "ur": ["PK"],
    "eu": ["ES"],  # Basque is primarily spoken in Spain
}

# Mapping full language names to short codes
FULL_LANGUAGE_MAP = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "turkish": "tr",
    "dutch": "nl",
    "swedish": "sv",
    "polish": "pl",
    "ukrainian": "uk",
    "romanian": "ro",
    "thai": "th",
    "vietnamese": "vi",
    "indonesian": "id",
    "greek": "el",
    "czech": "cs",
    "urdu": "ur",
    "arabic": "ar",
    "basque": "eu",
}

# Mapping full country names to short codes
FULL_COUNTRY_MAP = {
    "united states": "US",
    "united kingdom": "GB",
    "spain": "ES",
    "mexico": "MX",
    "france": "FR",
    "canada": "CA",
    "germany": "DE",
    "italy": "IT",
    "portugal": "PT",
    "brazil": "BR",
    "russia": "RU",
    "china": "CN",
    "japan": "JP",
    "south korea": "KR",
    "turkey": "TR",
    "netherlands": "NL",
    "sweden": "SE",
    "poland": "PL",
    "ukraine": "UA",
    "romania": "RO",
    "thailand": "TH",
    "vietnam": "VN",
    "indonesia": "ID",
    "greece": "GR",
    "czech republic": "CZ",
    "pakistan": "PK",
    "saudi": "SA",
    "united arab emirates": "AE",
    "egypt": "EG",
    "india": "IN",
    "argentina": "AR",
}


def translationValidator(target_language: str, target_country: str, text: str = None) -> tuple[bool, str]:
    """
    Validate if the target language and country are valid and compatible.
    Optionally validate text language for endpoints like update_translated_string.
    """
    # Normalize language
    target_language = target_language.lower().strip()
    if target_language in FULL_LANGUAGE_MAP:
        target_language = FULL_LANGUAGE_MAP[target_language]
    elif target_language not in LANGUAGES:
        supported_langs = ", ".join(
            sorted(LANGUAGES.keys()) + list(FULL_LANGUAGE_MAP.keys()))
        return False, f"Target language '{target_language}' is not supported. Supported languages: {supported_langs}."

    # Normalize country
    target_country = target_country.upper().strip()
    target_country = target_country.replace("FRANCE", "FR").replace(
        "INDIA", "IN").replace("CANADA", "CA").replace("JAPAN", "JP").replace("UAE", "AE")
    if target_country.lower() in FULL_COUNTRY_MAP:
        target_country = FULL_COUNTRY_MAP[target_country.lower()]
    country_pattern = re.compile(r"^[A-Z]{2}$")
    if not country_pattern.match(target_country) or target_country not in COUNTRIES:
        supported_countries = ", ".join(
            sorted(COUNTRIES.keys()) + list(FULL_COUNTRY_MAP.keys()))
        return False, f"Invalid or unsupported target country '{target_country}'. Supported countries: {supported_countries}."

    # Check language-country combination
    if target_language in LANGUAGE_COUNTRY_ASSOCIATIONS:
        if target_country not in LANGUAGE_COUNTRY_ASSOCIATIONS[target_language]:
            return False, f"Language '{target_language}' is not commonly spoken in '{target_country}'. Commonly associated countries: {', '.join(LANGUAGE_COUNTRY_ASSOCIATIONS[target_language])}."

    return True, "OK"


def validate_json_path(data: dict, path: str) -> tuple[bool, str]:
    """
    Validate if the JSON path is valid and exists in the data.
    """
    try:
        keys = path.split(".")
        ref = data
        for k in keys[:-1]:
            if k.isdigit():
                ref = ref[int(k)]
            else:
                ref = ref[k]
        last_key = keys[-1]
        if last_key.isdigit():
            if not isinstance(ref, list) or int(last_key) >= len(ref):
                return False, f"Invalid path: Index '{last_key}' out of range or not a list."
        else:
            if not isinstance(ref, dict) or last_key not in ref:
                return False, f"Invalid path: Key '{last_key}' does not exist."
        return True, "OK"
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return False, f"Invalid path: {str(e)}"


class SafeJsonParser:
    def parse(self, text: str):
        """
        Safely parse JSON-like text, handling common issues like single quotes and markdown.
        """
        # Step 1: Trim whitespace and markdown fences
        text = text.strip()
        text = re.sub(r"^```(?:json|json5|javascript)?\s*", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

        # Step 2: Remove common prefixes
        text = re.sub(r'^[\s`]*[Oo]utput\s*[:\-]*\s*', '', text)
        text = re.sub(r'^[\s`]*[Rr]esponse\s*[:\-]*\s*', '', text)
        text = text.strip()

        # Step 3: Extract JSON-like content
        match = re.search(r'(\[.*|\{.*)', text, re.S)
        if not match:
            raise ValueError("No valid JSON object or array found in response")
        text = match.group(1).strip()

        # Step 4: Sanitize common issues
        text = (
            text.replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
        )
        text = re.sub(r',(\s*[\]}])', r'\1', text)
        text = re.sub(r'\]\s*\[', '], [', text)

        # Step 5: Fix single quotes for JSON keys
        text = re.sub(r"(\{|\s|,)'([^'{\[\]:,]+)'\s*:", r'\1"\2":', text)
        text = text.replace("\\'", "'")

        # Step 6: Fix missing closing brackets
        open_brackets = text.count("[")
        close_brackets = text.count("]")
        if open_brackets > close_brackets:
            missing = open_brackets - close_brackets
            text += "]" * missing
        open_braces = text.count("{")
        close_braces = text.count("}")
        if open_braces > close_braces:
            missing = open_braces - close_braces
            text += "}" * missing

        # Step 7: Parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            text = text.replace("’", "'").replace("“", '"').replace("”", '"')
            try:
                return json.loads(text)
            except Exception as e2:
                raise ValueError(
                    f"Failed to parse sanitized JSON: {e2}\nRaw: {text[:500]}")
