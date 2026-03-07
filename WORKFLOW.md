# Workflow

## Definition of done

A task is not complete until:

1. All relevant tests pass locally.
2. The code conforms to the standards in [docs/](docs/) and this file.

There is no CI pipeline yet. Local validation is the only gate.

## Running tests

```bash
# Full test suite
pytest

# Specific test file
pytest tests/test_parse.py

# Specific test by name
pytest -k "test_parse_minimal"
```

## Suggested sequences

### Code changes

1. Run affected tests locally.
2. Fix any failures.
3. Run the full suite.
4. Commit.

### Test changes

1. Run the affected tests.
2. Commit.

### Docs/config only

1. Commit directly (no tests needed).

## Failure handling

- Fix and re-run local test failures before committing.
- Do not commit with known test failures unless the failure is pre-existing and unrelated to your change.
