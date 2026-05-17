"""CV Contract — formal JSON Schema definition and validator for the CV pipeline output."""

import jsonschema

# Canonical contract — import this in any module that needs to know the shape.
CV_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["nutrition", "allergens", "ingredients"],
    "additionalProperties": True,
    "properties": {
        "nutrition": {
            "type": "object",
            "description": (
                "Extracted nutrient labels keyed by nutrient name "
                "(e.g. 'sodium_mg', 'trans_fat_g'). Values are numeric amounts."
            ),
            "additionalProperties": {"type": "number"},
        },
        "allergens": {
            "type": "array",
            "description": "Allergen names detected in the label region.",
            "items": {"type": "string"},
        },
        "ingredients": {
            "type": "array",
            "description": "Ingredient names in the order they appear on the label.",
            "items": {"type": "string"},
        },
    },
}


def validate_cv_output(payload: dict) -> bool:
    """Return True if *payload* conforms to CV_SCHEMA, False otherwise."""
    try:
        jsonschema.validate(instance=payload, schema=CV_SCHEMA)
        return True
    except (jsonschema.ValidationError, jsonschema.SchemaError):
        return False
    except Exception:
        return False
