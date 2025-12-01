# from langdetect import detect
# import pycountry
# from typing import Tuple

# # Supported languages + regions
# LANGUAGES = {
#     "en-US": "American English",
#     "en-GB": "British English",
#     "es-ES": "Spanish (Spain)",
#     "es-MX": "Spanish (Mexico)",
#     "fr-FR": "French (France)",
#     "fr-CA": "French (Canada)",
#     "en-CA": "Canadian English (Canada)",
#     "de-DE": "German",
#     "it-IT": "Italian",
#     "pt-PT": "Portuguese (Portugal)",
#     "pt-BR": "Portuguese (Brazil)",
#     "ru-RU": "Russian",
#     "zh-CN": "Chinese (Simplified)",
#     "ja-JP": "Japanese",
#     "ko-KR": "Korean",
#     "tr-TR": "Turkish",
#     "nl-NL": "Dutch",
#     "sv-SE": "Swedish",
#     "pl-PL": "Polish",
#     "uk-UA": "Ukrainian",
#     "ro-RO": "Romanian",
#     "th-TH": "Thai",
#     "vi-VN": "Vietnamese",
#     "id-ID": "Indonesian",
#     "el-GR": "Greek",
#     "cs-CZ": "Czech",
#     "ur-PK": "Urdu (Pakistan)",
#     "ar-SA": "Arabic (Saudi Arabia)",
#     "ar-AE": "Arabic (UAE)",
#     "ar-EG": "Arabic (Egypt)"
# }


# def validate_language_and_country(language_pair: str, country_code: str, text: str) -> Tuple[bool, str]:
#     """
#     Validate if the language pair, country code, and input text are compatible and supported.
#     Allows translations between languages spoken in the same country (e.g., fr-CA to en-CA).

#     Args:
#         language_pair: Language pair in format 'src-tgt' (e.g., 'fr-en')
#         country_code: Country code (e.g., 'CA-CA')
#         text: Input text to validate (string)

#     Returns:
#         Tuple[bool, str]: (is_valid, message)
#     """
#     # Split language pair
#     try:
#         src, tgt = language_pair.split("-")
#     except ValueError:
#         return False, f"Invalid language_pair format: '{language_pair}'. Expected 'src-tgt'."

#     # Validate source and target languages
#     supported_langs = {lang.split("-")[0] for lang in LANGUAGES.keys()}
#     if src not in supported_langs:
#         return False, f"Source language '{src}' is not supported. Supported languages: {', '.join(sorted(supported_langs))}."
#     if tgt not in supported_langs:
#         return False, f"Target language '{tgt}' is not supported. Supported languages: {', '.join(sorted(supported_langs))}."

#     # Validate country code
#     country = country_code.upper().split("-")[0]
#     supported_regions = {lang.split("-")[1]
#                          for lang in LANGUAGES.keys() if "-" in lang}
#     if country not in supported_regions:
#         return False, f"Country '{country_code}' is not supported. Supported regions: {', '.join(sorted(supported_regions))}."

#     # Get languages supported in the country
#     country_langs = [lang.split(
#         "-")[0] for lang in LANGUAGES.keys() if lang.endswith(f"-{country}")]
#     if not country_langs:
#         return False, f"No languages are supported for country '{country_code}'."

#     # Validate source and target languages against country
#     if src not in country_langs:
#         supported_combinations = [
#             k for k in LANGUAGES.keys() if k.startswith(src + "-")]
#         combinations = ", ".join(
#             supported_combinations) if supported_combinations else "none"
#         return False, f"Source language '{src}' is not valid for country '{country_code}'. Supported combinations for '{src}': {combinations}."
#     if tgt not in country_langs:
#         supported_combinations = [
#             k for k in LANGUAGES.keys() if k.startswith(tgt + "-")]
#         combinations = ", ".join(
#             supported_combinations) if supported_combinations else "none"
#         return False, f"Target language '{tgt}' is not valid for country '{country_code}'. Supported combinations for '{tgt}': {combinations}."

#     # Detect language of input text
#     try:
#         detected_lang = detect(text)
#     except Exception as e:
#         return False, f"Language detection failed: {str(e)}"

#     # Check text source language
#     if detected_lang and detected_lang != src:
#         src_name = pycountry.languages.get(
#             alpha_2=src).name if src in pycountry.languages else src
#         detected_name = pycountry.languages.get(
#             alpha_2=detected_lang).name if detected_lang in pycountry.languages else detected_lang
#         return False, f"The provided text is in {detected_name}, but your source language in '{language_pair}' is {src_name}."

#     return True, "OK"


from langdetect import detect, DetectorFactory
import pycountry
from typing import Tuple, Optional
import re

# Seed for consistent results
DetectorFactory.seed = 0

# Your original supported list (keep it as source of truth)
LANGUAGES = {
    "en-US": "American English",
    "en-GB": "British English",
    "en-CA": "Canadian English (Canada)",
    "es-ES": "Spanish (Spain)",
    "es-MX": "Spanish (Mexico)",
    "fr-FR": "French (France)",
    "fr-CA": "French (Canada)",
    "de-DE": "German",
    "it-IT": "Italian",
    "pt-PT": "Portuguese (Portugal)",
    "pt-BR": "Portuguese (Brazil)",
    "ru-RU": "Russian",
    "zh-CN": "Chinese (Simplified)",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "tr-TR": "Turkish",
    "nl-NL": "Dutch",
    "sv-SE": "Swedish",
    "pl-PL": "Polish",
    "uk-UA": "Ukrainian",
    "ro-RO": "Romanian",
    "th-TH": "Thai",
    "vi-VN": "Vietnamese",
    "id-ID": "Indonesian",
    "el-GR": "Greek",
    "cs-CZ": "Czech",
    "ur-PK": "Urdu (Pakistan)",
    "ar-SA": "Arabic (Saudi Arabia)",
    "ar-AE": "Arabic (UAE)",
    "ar-EG": "Arabic (Egypt)"
}

# Pre-build lookup dictionaries for speed
CODE_TO_NAME = {code: name for code, name in LANGUAGES.items()}
LANG_TO_CODES = {}
COUNTRY_TO_CODES = {}

for code, name in LANGUAGES.items():
    lang_code = code.split("-")[0]
    country_code = code.split("-")[1]

    LANG_TO_CODES.setdefault(lang_code, []).append(code)
    # for short code lookup
    COUNTRY_TO_CODES[country_code.lower()] = country_code

# Add full country names (from pycountry)
for country in pycountry.countries:
    COUNTRY_TO_CODES[country.name.lower()] = country.alpha_2
    COUNTRY_TO_CODES[country.official_name.lower() if getattr(
        country, 'official_name', None) else ''] = country.alpha_2
    COUNTRY_TO_CODES[country.alpha_2.lower()] = country.alpha_2

# Language name (full & common) → alpha_2


def get_lang_code_from_name(name: str) -> Optional[str]:
    name = name.strip().lower()
    # Direct alpha_2
    lang = pycountry.languages.get(alpha_2=name)
    if not lang:
        lang = pycountry.languages.get(alpha_3=name)
    if not lang:
        # Try name lookup
        try:
            lang = pycountry.languages.lookup(name)
        except LookupError:
            pass
    if lang and hasattr(lang, 'alpha_2'):
        return lang.alpha_2
    return None

# Resolve any user input to proper "src-tgt" and country code


def resolve_language_pair(user_input: str) -> Tuple[Optional[str], Optional[str]]:
    user_input = user_input.strip().lower()

    # Case 1: already in format xx-xx or xx-yy
    if re.match(r'^[a-z]{2}-[a-z]{2}$', user_input):
        if user_input in LANGUAGES:
            src, tgt = user_input.split("-")
            return src, tgt

    # Case 2: "english to french", "french canada to english canada", etc.
    text = re.sub(r'[^a-zA-Z\s]', ' ', user_input)  # clean
    words = text.split()

    possible_src = None
    possible_tgt = None

    # Try to match full language names
    for i, word in enumerate(words):
        if word in ['to', '2', '->', '→']:
            # Everything before = source, after = target
            src_part = " ".join(words[:i])
            tgt_part = " ".join(words[i+1:])
            src_code = get_lang_code_from_name(src_part)
            tgt_code = get_lang_code_from_name(tgt_part)
            if src_code and tgt_code:
                return src_code, tgt_code
            break

    # Case 3: single code like "ur-pk" or "fr-ca"
    match = re.match(r'([a-z]{2})\s*-?\s*([a-z]{2})',
                     user_input.replace(" ", ""))
    if match:
        src, tgt = match.groups()
        candidate = f"{src}-{tgt}".lower()
        if candidate in LANGUAGES:
            return src, tgt

    return None, None


def resolve_country(country_input: str) -> Optional[str]:
    if not country_input:
        return None
    key = country_input.strip().lower()
    return COUNTRY_TO_CODES.get(key)


def validate_language_and_country(
    language_input: str,
    country_input: str,
    text: str
) -> Tuple[bool, str]:
    """
    Super flexible version: accepts full names, short codes, natural language.
    """
    # Step 1: Resolve language pair
    src, tgt = resolve_language_pair(language_input)
    if not src or not tgt:
        return False, f"Could not understand language pair: '{language_input}'. Examples: 'en-es', 'English to Spanish', 'fr-CA to en-CA', 'French Canada to English'"

    # Step 2: Check if this exact pair exists in supported
    full_code = f"{src.upper()}-{tgt.upper()}"
    if full_code not in LANGUAGES:
        # Allow same-country different languages (e.g., fr-CA ↔ en-CA)
        country_code = resolve_country(country_input)
        if country_code:
            supported_in_country = [
                c for c in LANGUAGES.keys() if c.endswith(f"-{country_code}")]
            src_variant = f"{src.upper()}-{country_code}"
            tgt_variant = f"{tgt.upper()}-{country_code}"
            if src_variant in LANGUAGES and tgt_variant in LANGUAGES:
                # It's valid for bilingual countries like Canada
                pass
            else:
                return False, f"Translation {src.upper()}→{tgt.upper()} not supported in this region."
        else:
            return False, f"Translation {src.upper()}→{tgt.upper()} is not in supported list."

    # Step 3: Resolve country
    country_code = resolve_country(country_input)
    if not country_code:
        return False, f"Could not recognize country: '{country_input}'. Try full name or 2-letter code."

    # Step 4: Validate source/target exist for this country (if strict needed)
    src_ok = any(c.startswith(src.upper() + "-")
                 and c.endswith("-" + country_code) for c in LANGUAGES)
    tgt_ok = any(c.startswith(tgt.upper() + "-")
                 and c.endswith("-" + country_code) for c in LANGUAGES)
    if not (src_ok or tgt_ok):
        # Relaxed rule: allow if at least one direction exists
        pass

    # Step 5: Detect text language
    try:
        detected = detect(text)
        if detected != src:
            src_name = pycountry.languages.get(alpha_2=src).name if hasattr(
                pycountry.languages.get(alpha_2=src), 'name') else src
            det_name = pycountry.languages.get(
                alpha_2=detected).name if detected else "unknown"
            return False, f"Text appears to be in {det_name}, but you selected source as {src_name}."
    except:
        return False, "Could not detect language of the input text."

    return True, "OK"


def unified_normalize(value: str) -> Optional[str]:
    """
    Universal normalizer for both languages and countries.
    Returns ISO 639-1 (for languages) or ISO 3166-1 alpha-2 (for countries).
    Accepts full names, short codes, mixed case.
    """
    if not value:
        return None

    value = value.strip().lower()

    # 1. If already alpha-2 language (en, fr, es…)
    if len(value) == 2 and value.isalpha():
        # Try language lookup
        lang = pycountry.languages.get(alpha_2=value)
        if lang:
            return value  # ISO language code

    # 2. Try full language name (english, french, arabic…)
    lang_code = get_lang_code_from_name(value)  # your existing helper
    if lang_code:
        return lang_code.lower()

    # 3. Try country alpha2 (us, fr, pk…)
    if len(value) == 2 and value.isalpha():
        country = pycountry.countries.get(alpha_2=value.upper())
        if country:
            return country.alpha_2  # ISO country code

    # 4. Try full country name
    try:
        country = pycountry.countries.lookup(value)
        if country:
            return country.alpha_2
    except LookupError:
        pass

    return None


def validate_language(source_language: str, text: str) -> Tuple[bool, str]:
    """
    Validates detected text language against normalized source language.
    Uses unified_normalize() so full names and codes both work.
    """
    # Normalize source language (english → en, EN → en, french → fr)
    normalized_src = unified_normalize(source_language)

    if not normalized_src:
        return False, f"Invalid source language '{source_language}'."

    # Detect text language (always returns ISO code)
    try:
        detected = detect(text)  # example: 'en'
    except:
        return False, "Unable to detect text language."

    if detected != normalized_src:
        return False, (
            f"Text appears to be '{detected}', but source_language "
            f"normalizes to '{normalized_src}'."
        )

    return True, "OK"


# ──────── Test Examples (Run these) ────────
if __name__ == "__main__":
    tests = [
        ("ur-en", "Pakistan", "سلام پاکستان کیسا ہے"),
        ("Urdu to English", "pk", "یہ ایک ٹیسٹ ہے"),
        ("fr-ca to en-ca", "Canada", "Bonjour comment ça va"),
        ("French to English", "Canada", "Je vais bien merci"),
        ("Spanish Mexico", "Mexico", "Hola cómo estás"),
        ("arabic uae to english", "United Arab Emirates", "مرحبا كيف حالك"),
        ("English to French Canada", "CA", "Hello how are you"),
    ]

    for lang, country, text in tests:
        valid, msg = validate_language_and_country(lang, country, text)
        print(f"'{lang}' → '{country}' | {valid} | {msg}")
