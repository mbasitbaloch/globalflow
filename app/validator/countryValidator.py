from langdetect import detect
import pycountry
from typing import Tuple

# Supported languages + regions
LANGUAGES = {
    "en-US": "American English",
    "en-GB": "British English",
    "es-ES": "Spanish (Spain)",
    "es-MX": "Spanish (Mexico)",
    "fr-FR": "French (France)",
    "fr-CA": "French (Canada)",
    "en-CA": "Canadian English (Canada)",
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


def validate_language_and_country(language_pair: str, country_code: str, text: str) -> Tuple[bool, str]:
    """
    Validate if the language pair, country code, and input text are compatible and supported.
    Allows translations between languages spoken in the same country (e.g., fr-CA to en-CA).

    Args:
        language_pair: Language pair in format 'src-tgt' (e.g., 'fr-en')
        country_code: Country code (e.g., 'CA-CA')
        text: Input text to validate (string)

    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    # Split language pair
    try:
        src, tgt = language_pair.split("-")
    except ValueError:
        return False, f"Invalid language_pair format: '{language_pair}'. Expected 'src-tgt'."

    # Validate source and target languages
    supported_langs = {lang.split("-")[0] for lang in LANGUAGES.keys()}
    if src not in supported_langs:
        return False, f"Source language '{src}' is not supported. Supported languages: {', '.join(sorted(supported_langs))}."
    if tgt not in supported_langs:
        return False, f"Target language '{tgt}' is not supported. Supported languages: {', '.join(sorted(supported_langs))}."

    # Validate country code
    country = country_code.upper().split("-")[0]
    supported_regions = {lang.split("-")[1]
                         for lang in LANGUAGES.keys() if "-" in lang}
    if country not in supported_regions:
        return False, f"Country '{country_code}' is not supported. Supported regions: {', '.join(sorted(supported_regions))}."

    # Get languages supported in the country
    country_langs = [lang.split(
        "-")[0] for lang in LANGUAGES.keys() if lang.endswith(f"-{country}")]
    if not country_langs:
        return False, f"No languages are supported for country '{country_code}'."

    # Validate source and target languages against country
    if src not in country_langs:
        supported_combinations = [
            k for k in LANGUAGES.keys() if k.startswith(src + "-")]
        combinations = ", ".join(
            supported_combinations) if supported_combinations else "none"
        return False, f"Source language '{src}' is not valid for country '{country_code}'. Supported combinations for '{src}': {combinations}."
    if tgt not in country_langs:
        supported_combinations = [
            k for k in LANGUAGES.keys() if k.startswith(tgt + "-")]
        combinations = ", ".join(
            supported_combinations) if supported_combinations else "none"
        return False, f"Target language '{tgt}' is not valid for country '{country_code}'. Supported combinations for '{tgt}': {combinations}."

    # Detect language of input text
    try:
        detected_lang = detect(text)
    except Exception as e:
        return False, f"Language detection failed: {str(e)}"

    # Check text source language
    if detected_lang and detected_lang != src:
        src_name = pycountry.languages.get(
            alpha_2=src).name if src in pycountry.languages else src
        detected_name = pycountry.languages.get(
            alpha_2=detected_lang).name if detected_lang in pycountry.languages else detected_lang
        return False, f"The provided text is in {detected_name}, but your source language in '{language_pair}' is {src_name}."

    return True, "OK"
