from harness.extract import extract_json


def test_bare_json_parses_direct():
    parsed, method = extract_json('{"patient_name": "Jane Doe"}')

    assert parsed == {"patient_name": "Jane Doe"}
    assert method == "direct"


def test_surrounding_whitespace_still_direct():
    parsed, method = extract_json('\n  {"a": 1}\n  ')

    assert parsed == {"a": 1}
    assert method == "direct"


def test_fenced_with_json_language_tag():
    text = '```json\n{"a": 1}\n```'
    parsed, method = extract_json(text)

    assert parsed == {"a": 1}
    assert method == "fenced"


def test_fenced_without_language_tag():
    text = '```\n{"a": 1}\n```'
    parsed, method = extract_json(text)

    assert parsed == {"a": 1}
    assert method == "fenced"


def test_json_with_leading_prose_falls_back_to_braces():
    text = 'Here is the extracted data:\n\n{"a": 1}'
    parsed, method = extract_json(text)

    assert parsed == {"a": 1}
    assert method == "braces"


def test_json_with_trailing_prose_falls_back_to_braces():
    text = '{"a": 1}\n\nLet me know if you need anything else.'
    parsed, method = extract_json(text)

    assert parsed == {"a": 1}
    assert method == "braces"


def test_nested_braces_inside_a_string_value():
    # A naive brace counter truncates at the '}' inside the string value.
    text = 'Result: {"note": "literal {braces} inside", "a": 1}'
    parsed, method = extract_json(text)

    assert parsed == {"note": "literal {braces} inside", "a": 1}
    assert method == "braces"


def test_escaped_quote_inside_string_value():
    text = 'Result: {"note": "she said \\"hi\\"", "a": 1}'
    parsed, method = extract_json(text)

    assert parsed == {"note": 'she said "hi"', "a": 1}
    assert method == "braces"


def test_nested_object_is_kept_whole():
    text = 'Output: {"outer": {"inner": 1}, "a": 2}'
    parsed, method = extract_json(text)

    assert parsed == {"outer": {"inner": 1}, "a": 2}
    assert method == "braces"


def test_unparseable_input_returns_failed():
    parsed, method = extract_json("I could not find that information.")

    assert parsed is None
    assert method == "failed"


def test_empty_input_returns_failed():
    assert extract_json("") == (None, "failed")
    assert extract_json("   ") == (None, "failed")


def test_valid_json_that_is_not_an_object_returns_failed():
    # A bare list is valid JSON but cannot satisfy the 8-field schema.
    assert extract_json("[1, 2, 3]") == (None, "failed")
    assert extract_json('"just a string"') == (None, "failed")


def test_malformed_json_is_not_repaired():
    # Trailing comma. Repairing it here would inflate the model's score.
    parsed, method = extract_json('{"a": 1,}')

    assert parsed is None
    assert method == "failed"


def test_single_quoted_json_is_not_repaired():
    parsed, method = extract_json("{'a': 1}")

    assert parsed is None
    assert method == "failed"
