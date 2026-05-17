"""Unit tests for cv_contract.validate_cv_output and CV_SCHEMA."""

import json
import os
import unittest

import jsonschema

from cv_contract import CV_SCHEMA, validate_cv_output

_EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "cv_contract_example.json")
with open(_EXAMPLE_PATH) as _f:
    EXAMPLE_PAYLOAD = json.load(_f)


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

class TestSchemaDefinition(unittest.TestCase):
    def test_schema_is_valid_json_schema(self):
        """CV_SCHEMA itself must be a legal JSON Schema document."""
        jsonschema.Draft202012Validator.check_schema(CV_SCHEMA)

    def test_schema_requires_three_fields(self):
        self.assertIn("nutrition", CV_SCHEMA["required"])
        self.assertIn("allergens", CV_SCHEMA["required"])
        self.assertIn("ingredients", CV_SCHEMA["required"])


# ---------------------------------------------------------------------------
# Valid payloads
# ---------------------------------------------------------------------------

class TestValidPayload(unittest.TestCase):
    def test_example_is_valid(self):
        self.assertTrue(validate_cv_output(EXAMPLE_PAYLOAD))

    def test_empty_collections_valid(self):
        payload = {"nutrition": {}, "allergens": [], "ingredients": []}
        self.assertTrue(validate_cv_output(payload))

    def test_integer_nutrition_values_valid(self):
        payload = {
            "nutrition": {"sodium_mg": 470, "calories": 250},
            "allergens": ["milk"],
            "ingredients": ["oats"],
        }
        self.assertTrue(validate_cv_output(payload))

    def test_float_nutrition_values_valid(self):
        payload = {
            "nutrition": {"trans_fat_g": 0.0, "saturated_fat_g": 3.5},
            "allergens": [],
            "ingredients": ["palm oil"],
        }
        self.assertTrue(validate_cv_output(payload))


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------

class TestMissingField(unittest.TestCase):
    def test_missing_nutrition(self):
        payload = {k: v for k, v in EXAMPLE_PAYLOAD.items() if k != "nutrition"}
        self.assertFalse(validate_cv_output(payload))

    def test_missing_allergens(self):
        payload = {k: v for k, v in EXAMPLE_PAYLOAD.items() if k != "allergens"}
        self.assertFalse(validate_cv_output(payload))

    def test_missing_ingredients(self):
        payload = {k: v for k, v in EXAMPLE_PAYLOAD.items() if k != "ingredients"}
        self.assertFalse(validate_cv_output(payload))

    def test_empty_dict(self):
        self.assertFalse(validate_cv_output({}))


# ---------------------------------------------------------------------------
# Extra fields (should remain valid — additionalProperties: True)
# ---------------------------------------------------------------------------

class TestExtraField(unittest.TestCase):
    def test_extra_top_level_key_still_valid(self):
        payload = dict(EXAMPLE_PAYLOAD)
        payload["image_id"] = "frame_042"
        payload["confidence"] = 0.97
        self.assertTrue(validate_cv_output(payload))


# ---------------------------------------------------------------------------
# Malformed values
# ---------------------------------------------------------------------------

class TestMalformedValues(unittest.TestCase):
    def test_nutrition_not_dict(self):
        payload = dict(EXAMPLE_PAYLOAD)
        payload["nutrition"] = ["sodium_mg", 470]
        self.assertFalse(validate_cv_output(payload))

    def test_nutrition_value_is_string(self):
        payload = dict(EXAMPLE_PAYLOAD)
        payload["nutrition"] = {"sodium_mg": "high"}
        self.assertFalse(validate_cv_output(payload))

    def test_allergens_not_list(self):
        payload = dict(EXAMPLE_PAYLOAD)
        payload["allergens"] = "milk, wheat"
        self.assertFalse(validate_cv_output(payload))

    def test_allergens_contains_non_string(self):
        payload = dict(EXAMPLE_PAYLOAD)
        payload["allergens"] = ["milk", 42]
        self.assertFalse(validate_cv_output(payload))

    def test_ingredients_not_list(self):
        payload = dict(EXAMPLE_PAYLOAD)
        payload["ingredients"] = {"item": "oats"}
        self.assertFalse(validate_cv_output(payload))

    def test_ingredients_contains_non_string(self):
        payload = dict(EXAMPLE_PAYLOAD)
        payload["ingredients"] = ["oats", None]
        self.assertFalse(validate_cv_output(payload))

    def test_not_a_dict(self):
        self.assertFalse(validate_cv_output("not a dict"))
        self.assertFalse(validate_cv_output(None))
        self.assertFalse(validate_cv_output([]))


if __name__ == "__main__":
    unittest.main()
