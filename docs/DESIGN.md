# labelman: System Design Document

## 1. Overview

`labelman` is a dataset labeling tool for image fine-tuning workflows. It:

- Maintains a taxonomy of terms in `labelman.yaml`, the central artifact of the system.
- Uses that taxonomy to label images along multiple axes (categories).
- Supports iterative refinement of the taxonomy over time.
- Provides core workflows: **suggest** (propose terms), **label** (apply terms to images), and **check** (validate the taxonomy).
- Integrates with external models (BLIP, CLIP) through shell scripts described by Boutiques descriptors.

`labelman` is not a media organizer. It exists to produce structured labels from a controlled taxonomy, suitable for training or fine-tuning models.

---

## 2. Goals and Non-Goals

### Goals

- Bulk labeling of image datasets against a user-defined taxonomy.
- Support for dozens of categories per dataset, with varied semantics.
- Iterative refinement: the user edits `labelman.yaml` over time and re-labels.
- Human-reviewable, hand-editable YAML configuration.
- Configurable category semantics (exclusivity, multiplicity).
- Pluggable external-model integrations via Boutiques-described scripts.
- Working out-of-the-box defaults for BLIP (captioning/suggest) and CLIP (zero-shot labeling).

### Non-Goals

- Preserving historical taxonomy versions. Only the current `labelman.yaml` matters.
- Acting as a DAM, media browser, or general-purpose organizer.
- Embedding storage-layer assumptions (database, specific filesystem layout beyond conventions).
- Requiring a particular ML framework or runtime beyond the ability to invoke scripts.
- Real-time or interactive labeling UIs.

---

## 3. Core Concepts

| Concept | Definition |
|---|---|
| **Term list** | The file `labelman.yaml`. The single source of truth for the current taxonomy. User-authored and user-edited. |
| **Category** | A labeling axis. Examples: "lighting", "subject_count", "mood". A dataset may have a dozen or more categories. |
| **Term** | A candidate label within a category. Examples: within "lighting": "natural", "studio", "low-key". |
| **Category mode** | Defines how many terms from a category may be applied to a single image. Controls exclusivity and multiplicity. |
| **Threshold** | A confidence cutoff. A term is applied only if the external model's score meets or exceeds the effective threshold. |
| **Suggest** | A workflow that proposes terms and/or categories for user review. Advisory only; the user decides what enters `labelman.yaml`. |
| **Label** | A workflow that applies the current `labelman.yaml` to images, producing per-image labels constrained by category rules. |
| **Integration script** | A shell-invocable executable that wraps an external model (BLIP, CLIP, or custom). Described by a Boutiques descriptor. |

---

## 4. `labelman.yaml` Format

### Design Principles

- The file is named `labelman.yaml`. This is canonical and not configurable.
- The format is YAML, human-editable, and reviewable in diffs.
- Structure is mostly flat. Nesting is used only where semantics require it (categories contain terms).
- Defaults inherit downward: global -> category -> term.

### Schema

```yaml
# labelman.yaml

defaults:
  threshold: 0.3            # Global default threshold. Required.

global_terms:                 # Optional. Literal labels applied to every image.
  - aircraft
  - mooney m20

integrations:
  blip:
    endpoint: http://localhost:8080/caption   # Use built-in script with this URL
  clip:
    endpoint: http://localhost:8081/classify   # Or use script: /path/to/custom.sh

categories:
  - name: lighting           # Unique category identifier. Required.
    mode: exactly-one         # Category mode. Required. See §5.
    threshold: 0.4            # Optional. Overrides global default for this category.
    terms:
      - term: natural
      - term: studio
      - term: low-key
        threshold: 0.35       # Optional. Overrides category threshold for this term.
      - term: high-key

  - name: mood
    mode: zero-or-more
    terms:
      - term: serene
      - term: dramatic
      - term: playful

  - name: background
    mode: zero-or-one
    terms:
      - term: plain
      - term: environmental
      - term: blurred
```

### Field Reference

#### Top level

| Field | Type | Required | Description |
|---|---|---|---|
| `defaults` | object | Yes | Global default settings. |
| `defaults.threshold` | float (0.0–1.0) | Yes | Global confidence threshold. |
| `global_terms` | list of strings | No | Literal labels applied to every image. Bypass detection and thresholds. |
| `integrations` | object | No | Integration configuration for BLIP/CLIP. |
| `integrations.blip` | object | No | BLIP integration config: `endpoint` (URL) or `script` (path). |
| `integrations.clip` | object | No | CLIP integration config: `endpoint` (URL) or `script` (path). |
| `categories` | list | Yes | List of category definitions. |

#### Category level

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Unique identifier for the category. |
| `mode` | enum | Yes | One of: `exactly-one`, `zero-or-one`, `zero-or-more`. See §5. |
| `threshold` | float (0.0–1.0) | No | Overrides `defaults.threshold` for all terms in this category. |
| `terms` | list | Yes | List of terms in this category. Must contain at least one term. |

#### Term level

| Field | Type | Required | Description |
|---|---|---|---|
| `term` | string | Yes | The label string. Unique within its category. |
| `threshold` | float (0.0–1.0) | No | Overrides the category or global threshold for this specific term. |

### Threshold Inheritance

The effective threshold for a term is resolved by specificity:

1. If the term defines `threshold`, use it.
2. Else if the category defines `threshold`, use it.
3. Else use `defaults.threshold`.

This is the only inheritance chain. There are no other cascading fields.

---

## 5. Category Semantics

Categories are labeling axes. Mutual exclusivity is a property of the category, not of individual terms.

### Category Modes

| Mode | Meaning | Labeling behavior |
|---|---|---|
| `exactly-one` | Every image must receive exactly one label from this category. | The highest-scoring term above threshold is selected. If no term meets the threshold, the highest-scoring term is selected anyway (forced assignment). |
| `zero-or-one` | An image may receive at most one label from this category. | The highest-scoring term above threshold is selected. If no term meets the threshold, no label is assigned. |
| `zero-or-more` | An image may receive any number of labels from this category. | All terms above threshold are assigned. |

### Rules

- Terms within a single category compete or coexist based on mode.
- There is no cross-category exclusivity. Categories are independent axes.
- The labeler must not produce output that violates category mode constraints.

---

## 6. Threshold Model

### Semantics

- A term is applied only if its confidence score (from the external model) meets or exceeds the effective threshold.
- Exception: `exactly-one` mode forces assignment of the best-scoring term even if below threshold.
- Thresholds are plain floats in the range [0.0, 1.0].

### Inheritance

As defined in §4: term-level overrides category-level overrides global default. The most specific value wins.

### Design Tradeoff

Per-term thresholds add complexity. The recommended starting position:

- **Always implement** global and per-category thresholds.
- **Support but discourage** per-term thresholds. They exist in the schema for edge cases (e.g., a term that is systematically under- or over-predicted by the model) but should not be the primary tuning mechanism.

If experience shows per-term thresholds are never used, a future revision may remove them. The schema supports them so that removal is a simplification, not a breaking change.

---

## 7. Tool Scenarios

### Scenario 1: Init

**Command:** `labelman init`

**Behavior:**

1. Creates a workspace directory structure (or initializes in the current directory).
2. Writes a starter `labelman.yaml` with:
   - A `defaults` block with a reasonable threshold (e.g., 0.3).
   - One or two example categories with placeholder terms, commented to guide the user.
3. Creates or references default integration descriptors for BLIP and CLIP (see §8).
4. Writes a minimal configuration file (e.g., `labelman.yaml`) pointing to the dataset path and integration descriptors.

**Outcome:** The user has a working starting point. They can immediately run `suggest` against a dataset or begin editing `labelman.yaml` by hand.

---

### Scenario 2: Suggest

**Command:** `labelman suggest [--mode bootstrap|expand] [--sample N] [--images <path>]`

#### Mode A: Bootstrap (no existing taxonomy)

- Precondition: `labelman.yaml` is empty or minimal.
- The tool selects a sample of images from the dataset (configurable sample size).
- It invokes **BLIP** (or a compatible captioning/VQA model) via the integration script.
- BLIP produces captions or descriptive text for each sampled image.
- `labelman` clusters or extracts candidate terms from the captions.
- It proposes a set of candidate categories and terms, written to stdout or a review file.
- The user reviews the proposals and incorporates what they want into `labelman.yaml`.

#### Mode B: Expand (existing taxonomy)

- Precondition: `labelman.yaml` contains categories and terms.
- The tool reads the current taxonomy.
- For each category (or a user-specified subset), it constructs category-aware prompts or questions. For example, for a category "lighting" with terms ["natural", "studio"], it might ask BLIP: "What type of lighting is in this image?" or use CLIP to score a broader candidate set.
- It proposes additional terms within existing categories, or flags images that don't match any existing term well.
- Proposals are advisory. The user decides what to add to `labelman.yaml`.

#### Key Properties

- Suggest never modifies `labelman.yaml` directly. It produces proposals for human review.
- The external model used for suggest is configured via the integration model (§8). BLIP is the default for bootstrap; CLIP may be used for expand.
- Suggest is expected to be noisy. The value is in accelerating taxonomy development, not in producing a final taxonomy.

---

### Scenario 3: Label

**Command:** `labelman label [--images <path>] [--output <path>]`

**Behavior:**

1. Reads `labelman.yaml` in full.
2. Reads the target image set.
3. For each image, invokes **CLIP** (or a compatible zero-shot classification model) via the integration script, passing the current term set.
4. Receives per-term confidence scores.
5. Applies category rules:
   - For `exactly-one`: selects the highest-scoring term (forced if none meets threshold).
   - For `zero-or-one`: selects the highest-scoring term above threshold, or none.
   - For `zero-or-more`: selects all terms above threshold.
6. Writes the label output (see §9).

**Key Properties:**

- Labeling is deterministic given the same `labelman.yaml`, images, and model outputs.
- Only terms present in the current `labelman.yaml` are considered. Removed terms produce no labels.
- The label workflow uses the current taxonomy as-is. There is no "partial" labeling against a subset of categories (though this could be a future option).

---

### Scenario 4: Check

**Command:** `labelman check [--terms <path>]`

The check scenario validates `labelman.yaml` for structural correctness and semantic consistency. It does not invoke any external models. It operates purely on the term list file.

**Validations performed:**

#### Structural checks (YAML and schema)

- The file is valid YAML.
- The top-level structure has `defaults` and `categories` keys.
- `defaults.threshold` is present and is a float in [0.0, 1.0].
- `categories` is a non-empty list.
- Each category has `name`, `mode`, and `terms` fields.
- Each `mode` value is one of the allowed enums: `exactly-one`, `zero-or-one`, `zero-or-more`.
- Each category's `terms` list is non-empty.
- Each term has a `term` field (string, non-empty).
- Any `threshold` values (category or term level) are floats in [0.0, 1.0].

#### Uniqueness checks

- Category names are unique across the file.
- Term names are unique within each category.

#### Semantic checks

- No category has an empty `terms` list.
- An `exactly-one` category should have at least two terms (a single-term `exactly-one` category is technically valid but likely a mistake; emit a warning, not an error).
- Threshold values are reasonable (e.g., warn if a threshold is 0.0 or 1.0, as these are degenerate).

#### Output

- On success: prints a summary (number of categories, total terms, any warnings) and exits 0.
- On failure: prints all errors with file location context (category name, term name) and exits non-zero.
- Warnings (non-fatal) are printed but do not cause a non-zero exit.

**Key properties:**

- Check is fast and offline. No images, no models, no integration scripts.
- Check should be runnable in CI or as a pre-commit hook.
- Check validates only the current `labelman.yaml`. It does not compare to previous versions.

---

## 8. External Tool Integration Model

### Approach

External models (BLIP, CLIP, etc.) are invoked as CLI tools through shell scripts. Each integration is described by a **Boutiques descriptor** — a standalone JSON file that defines the tool's CLI contract: arguments, inputs, outputs, and invocation template.

### Structure

```
integrations/
  blip.boutiques.json      # Boutiques descriptor for the BLIP integration
  blip.sh                   # Shell script that implements the BLIP CLI
  clip.boutiques.json       # Boutiques descriptor for the CLIP integration
  clip.sh                   # Shell script that implements the CLIP CLI
```

### Default Integrations

`labelman init` sets up references to default BLIP and CLIP integrations. These are provided scripts that:

- **BLIP script**: Accepts image path(s), returns captions or VQA answers as structured text (JSON lines or similar).
- **CLIP script**: Accepts image path(s) and a list of candidate terms, returns per-term confidence scores as structured output.

The Boutiques descriptors for these defaults define exactly what arguments are expected, what input files are required, and what output format is produced.

### Custom Integrations

Users may provide their own scripts instead of using the built-in endpoint-calling scripts. To do so:

1. Write a shell script (or any executable) that conforms to a compatible CLI contract.
2. Set `script: /path/to/your/script.sh` in `labelman.yaml` under the relevant integration.

When `script` is set, it is used directly and any `endpoint` value is ignored.

### Boutiques Role

Per the Boutiques model (see `/src/dandriscoll/BOUTIQUES.md`):

- The **Spec** is the `.boutiques.json` descriptor file, bundled with labelman.
- The **Producer** is the integration script (built-in or custom).
- The **Consumer** is `labelman` itself.

Boutiques descriptors for built-in integrations are accessible via `labelman descriptor blip` and `labelman descriptor clip`. These describe the CLI contract of the built-in scripts.

### Integration Contract (Conceptual)

While the exact Boutiques descriptors are an implementation deliverable, the expected contract shapes are:

**Captioning integration (e.g., BLIP for suggest):**
- Input: one or more image paths.
- Optional input: a prompt or question string.
- Output: structured text with per-image captions or answers.

**Classification integration (e.g., CLIP for label):**
- Input: one or more image paths.
- Input: a list of candidate term strings.
- Output: structured data with per-image, per-term confidence scores.

---

## 9. Label Output Model

The label workflow produces three outputs:

### 1. Sidecar files (`.txt`)

Per-image text files with comma-separated labels in stable-diffusion caption format. Each sidecar has the same stem as its image with a `.txt` extension.

```
single, indoor, calm, tense
```

These are directly consumable by fine-tuning pipelines that expect caption files alongside images.

### 2. CSV (`labels.csv`)

A single CSV file containing all images, all scores (including those suppressed by thresholding or category mode), and assignment flags.

Columns:
- `image` — the image path
- `{category}/{term}_score` — raw confidence score from the model
- `{category}/{term}_assigned` — `1` if selected, `0` if suppressed
- `caption` — the comma-separated label string

This file preserves full score information for threshold tuning and debugging.

### 3. HTML report (`report.html`)

A self-contained HTML report showing:

- **Summary**: image count, category count, term count, global threshold
- **Per-category statistics**: assignment counts and percentages for each term
- **Per-image detail table**: all scores with assigned terms highlighted

### Global Baseline Terms

`labelman.yaml` supports an optional `global_terms` list:

```yaml
global_terms:
  - aircraft
  - mooney m20
```

These labels are applied to every image. They bypass detection, thresholds, and category semantics. They do not need to appear in any category. They appear first in the final label output.

### Manual Label Sidecars

Per-image manual labels can be provided via sidecar files:

```
image_001.jpg              # the image
image_001.labels.txt       # the manual label sidecar
```

Sidecar format: plain text, one label per line, whitespace-trimmed, blank lines ignored.

```
tail number n123ab
red stripe livery
```

Manual labels bypass detection and thresholds. They are authoritative literal labels that do not need to appear in the taxonomy. If no sidecar exists, no manual labels are applied (this is not an error).

### Final Label Assembly

Labels are assembled in this order:

1. **Global baseline terms** (from `global_terms` in `labelman.yaml`)
2. **Manual sidecar labels** (from `{image_stem}.labels.txt`)
3. **Detected labels** (from category rules and thresholds)

The final list is deduplicated (first occurrence wins) to produce a stable, deterministic caption. For example:

```
aircraft, mooney m20, tail number n123ab, red stripe livery, single, outdoor, calm
```

Edge cases:
- A manual label that matches a detected label: appears once (at the manual position).
- A manual label not in the taxonomy: included as-is.
- A missing sidecar: no manual labels, not an error.
- An empty sidecar: treated as no manual labels.
- Global term that matches a detected or manual label: appears once (at the global position).

### Constraints

- Output must respect category mode. An `exactly-one` category always has a value. A `zero-or-one` category has zero or one. A `zero-or-more` category has a list.
- Detected labels come only from the current `labelman.yaml` taxonomy.
- Global and manual labels are outside category enforcement and may contain any string.
- The output must be deterministic given the same inputs (taxonomy, images, model scores, sidecars, global terms).
- The CSV includes all scores regardless of assignment, enabling post-hoc threshold analysis.

---

## 10. Rules and Constraints

1. `labelman.yaml` is the sole source of truth for the current taxonomy. There is no secondary term store.
2. The system must be easy to inspect and edit by hand. No binary formats, no opaque IDs.
3. Categories may number in the dozens or more per dataset. The schema and tooling must not degrade at this scale.
4. The schema should minimize unnecessary metadata. If a field isn't needed, don't require it.
5. History and versioning of prior term sets is out of scope. Use git or equivalent externally.
6. Labeling behavior must be deterministic with respect to the current `labelman.yaml`, thresholds, and model outputs. Same inputs produce same labels.
7. The schema should remain stable and readable as datasets grow. Avoid patterns that become unwieldy at scale.
8. Integration scripts are the boundary between `labelman` and ML models. `labelman` does not import or embed ML frameworks.
9. Suggest never writes to `labelman.yaml`. It produces proposals for human review.
10. The Boutiques descriptor is authoritative for each integration's CLI contract.

---

## 11. Illustrative Examples

### Example `labelman.yaml`

```yaml
defaults:
  threshold: 0.3

global_terms:
  - aircraft
  - mooney m20

integrations:
  blip:
    endpoint: http://localhost:8080/caption
  clip:
    endpoint: http://localhost:8081/classify

categories:
  - name: subject_count
    mode: exactly-one
    terms:
      - term: single
      - term: couple
      - term: group

  - name: setting
    mode: zero-or-one
    threshold: 0.45
    terms:
      - term: indoor
      - term: outdoor
      - term: studio

  - name: mood
    mode: zero-or-more
    terms:
      - term: calm
      - term: energetic
      - term: tense
        threshold: 0.25
      - term: joyful
```

### Threshold Inheritance in This Example

| Term | Effective Threshold | Source |
|---|---|---|
| `subject_count/single` | 0.3 | global default |
| `subject_count/couple` | 0.3 | global default |
| `setting/indoor` | 0.45 | category override |
| `setting/studio` | 0.45 | category override |
| `mood/calm` | 0.3 | global default (category has no override) |
| `mood/tense` | 0.25 | term override |
| `mood/joyful` | 0.3 | global default |

### Labeling Example

Given an image scored by CLIP against the above taxonomy:

| Term | Score |
|---|---|
| `subject_count/single` | 0.82 |
| `subject_count/couple` | 0.10 |
| `subject_count/group` | 0.08 |
| `setting/indoor` | 0.40 |
| `setting/outdoor` | 0.38 |
| `setting/studio` | 0.22 |
| `mood/calm` | 0.55 |
| `mood/energetic` | 0.12 |
| `mood/tense` | 0.27 |
| `mood/joyful` | 0.31 |

**Detected labels:**

- `subject_count`: **single** (highest score, `exactly-one` mode, above threshold)
- `setting`: **none** (no term meets category threshold of 0.45; `zero-or-one` allows empty)
- `mood`: **calm**, **tense**, **joyful** (`zero-or-more`; calm=0.55>=0.3, tense=0.27>=0.25, joyful=0.31>=0.3; energetic=0.12<0.3 excluded)

**Final assembled caption** (assuming a manual sidecar with `tail number n123ab`):

```
aircraft, mooney m20, tail number n123ab, single, calm, tense, joyful
```

---

## 12. Open Questions / Implementation Decisions

1. **Per-term thresholds: necessary?** The schema supports them, but they may not be needed in practice. Monitor whether users actually set per-term thresholds. Consider removing in a future simplification pass if unused.

2. **Suggest prompt strategy.** How should expand-mode construct category-aware prompts for BLIP/VQA? Options include templated questions ("What is the {category} of this image?"), open-ended captions filtered by category, or CLIP-based scoring of a broader candidate list. This is a key UX and quality question.

3. **Label output format.** YAML, JSON, JSON Lines, CSV, or per-image sidecar files? The choice depends on dataset scale and what downstream fine-tuning tools expect. This should be decided during implementation, possibly as a configurable option.

4. **Default vs. custom integration balance.** How much should `labelman init` set up automatically vs. require user configuration? The default BLIP/CLIP scripts need to work with minimal setup, but the system must not make it hard to swap in custom models.

5. **Suggest output format.** How should proposals be presented? Options: printed to stdout, written to a review file (e.g., `suggestions.yaml`), or presented as a diff against `labelman.yaml`. The format should make it easy for the user to cherry-pick proposals.

6. **Batch size and parallelism for integration scripts.** Should `labelman` pass one image at a time or batch? The Boutiques descriptor can express this, but the default scripts need a clear convention.

7. **Configuration file scope.** Beyond `labelman.yaml`, does `labelman` need a separate `labelman.yaml` for dataset path, integration references, and other settings? Or can everything live in `labelman.yaml`? Keeping them separate respects the principle that `labelman.yaml` is purely taxonomy. This separation is recommended but not yet finalized.

8. **Score provenance.** Should label output include raw scores from the model, or only the final assigned labels? Including scores aids debugging and threshold tuning but increases output size.

9. **Handling of images that match no terms.** For `zero-or-one` and `zero-or-more` categories, this is well-defined (no label). But should the system flag these images for review? This is a UX question.

10. **Category ordering.** Does the order of categories in `labelman.yaml` matter? It should not affect labeling semantics, but may affect output ordering. Recommend: order is cosmetic only.

---

## 13. Test Suite

The test suite validates that `labelman` behaves correctly across all scenarios. Tests are organized by scenario and by the component under test. All tests should be runnable without external models (BLIP/CLIP) by using mock integration scripts that return deterministic scores.

### 13.1 `labelman.yaml` Parsing Tests

| Test | Description |
|---|---|
| `parse_minimal` | A valid `labelman.yaml` with one category, one term, and global defaults parses without error. |
| `parse_full` | A valid `labelman.yaml` with multiple categories, all three modes, and threshold overrides at every level parses correctly. |
| `parse_empty_file` | An empty file produces a clear parse error. |
| `parse_missing_defaults` | A file missing the `defaults` block produces an error. |
| `parse_missing_threshold` | A file with `defaults` but no `threshold` produces an error. |
| `parse_missing_categories` | A file with no `categories` key produces an error. |
| `parse_empty_categories` | A file with `categories: []` produces an error. |
| `parse_missing_category_name` | A category without `name` produces an error. |
| `parse_missing_category_mode` | A category without `mode` produces an error. |
| `parse_invalid_mode` | A category with `mode: many` (not a valid enum) produces an error. |
| `parse_missing_terms` | A category without `terms` produces an error. |
| `parse_empty_terms` | A category with `terms: []` produces an error. |
| `parse_missing_term_name` | A term entry without `term` field produces an error. |
| `parse_threshold_out_of_range` | A threshold of 1.5 or -0.1 produces an error (global, category, or term level). |
| `parse_duplicate_category_names` | Two categories with the same name produce an error. |
| `parse_duplicate_term_names` | Two terms with the same name within one category produce an error. |
| `parse_duplicate_terms_across_categories` | The same term name in different categories is allowed (not an error). |

### 13.2 Threshold Inheritance Tests

| Test | Description |
|---|---|
| `threshold_global_only` | When only global threshold is set, all terms inherit it. |
| `threshold_category_override` | A category threshold overrides global for all its terms. |
| `threshold_term_override` | A term threshold overrides its category's threshold. |
| `threshold_term_overrides_global` | A term threshold overrides global when no category threshold is set. |
| `threshold_mixed` | A file with thresholds at all three levels resolves each term to the correct effective threshold. |

### 13.3 Check Scenario Tests

| Test | Description |
|---|---|
| `check_valid_file` | A well-formed `labelman.yaml` passes check with exit code 0. |
| `check_invalid_yaml` | A file with invalid YAML syntax fails with a clear error. |
| `check_schema_violations` | A file with missing required fields fails and reports all violations. |
| `check_duplicate_categories` | Duplicate category names are detected and reported. |
| `check_duplicate_terms` | Duplicate term names within a category are detected and reported. |
| `check_single_term_exactly_one` | An `exactly-one` category with only one term produces a warning (not an error). |
| `check_degenerate_threshold` | A threshold of 0.0 or 1.0 produces a warning. |
| `check_summary_output` | On success, check prints category count, total term count, and any warnings. |
| `check_multiple_errors` | All errors are reported in a single run (not just the first one). |

### 13.4 Label Scenario Tests

These tests use a mock CLIP integration that returns deterministic scores.

| Test | Description |
|---|---|
| `label_exactly_one_above_threshold` | For an `exactly-one` category, the highest-scoring term above threshold is selected. |
| `label_exactly_one_below_threshold` | For an `exactly-one` category where no term meets threshold, the highest-scoring term is still selected (forced). |
| `label_zero_or_one_above_threshold` | For a `zero-or-one` category, the highest-scoring term above threshold is selected. |
| `label_zero_or_one_below_threshold` | For a `zero-or-one` category where no term meets threshold, no label is assigned. |
| `label_zero_or_more_multiple` | For a `zero-or-more` category, all terms above threshold are selected. |
| `label_zero_or_more_none` | For a `zero-or-more` category where no term meets threshold, no labels are assigned. |
| `label_zero_or_more_all` | For a `zero-or-more` category where all terms meet threshold, all are selected. |
| `label_threshold_inheritance` | Labels are applied using the correct effective threshold at each level (global, category, term). |
| `label_deterministic` | Running label twice with identical inputs produces identical output. |
| `label_removed_term` | A term removed from `labelman.yaml` between runs does not appear in new label output. |
| `label_added_term` | A term added to `labelman.yaml` between runs is evaluated and may appear in new label output. |
| `label_multi_category` | An image is labeled across multiple categories independently. Each category's mode is respected. |
| `label_multi_image` | Multiple images are labeled, each producing independent results. |
| `label_output_structure` | The output contains image identifiers and per-category labels consistent with category modes. |

### 13.5 Init Scenario Tests

| Test | Description |
|---|---|
| `init_creates_terms_yaml` | `labelman init` creates a `labelman.yaml` file. |
| `init_terms_yaml_valid` | The generated `labelman.yaml` passes `labelman check`. |
| `init_creates_integration_refs` | Init creates or references default Boutiques descriptors for BLIP and CLIP. |
| `init_idempotent_or_guarded` | Running init in a directory that already has a `labelman.yaml` either refuses (to avoid overwriting) or merges safely. Define and test the chosen behavior. |

### 13.6 Suggest Scenario Tests

These tests use a mock BLIP/CLIP integration.

| Test | Description |
|---|---|
| `suggest_bootstrap_produces_proposals` | Bootstrap mode produces at least one proposed category and term from mock captions. |
| `suggest_bootstrap_does_not_modify_terms` | After running suggest in bootstrap mode, `labelman.yaml` is unchanged. |
| `suggest_expand_uses_existing_categories` | Expand mode reads existing categories and proposes terms within them. |
| `suggest_expand_does_not_modify_terms` | After running suggest in expand mode, `labelman.yaml` is unchanged. |
| `suggest_output_is_structured` | Suggest output is parseable (whatever format is chosen) and contains category and term fields. |

### 13.7 Integration / Boutiques Tests

| Test | Description |
|---|---|
| `integration_descriptor_valid_json` | Each default `.boutiques.json` file is valid JSON. |
| `integration_descriptor_matches_script` | The default integration script's `--descriptor` output matches the standalone descriptor file. |
| `integration_clip_contract` | The CLIP integration script accepts image paths and term list, returns per-term scores. Tested with mock data. |
| `integration_blip_contract` | The BLIP integration script accepts image paths, returns captions. Tested with mock data. |
| `integration_custom_script` | A user-provided custom integration script with its own Boutiques descriptor can be invoked by `labelman`. |
| `integration_missing_descriptor` | Referencing a nonexistent descriptor produces a clear error. |
| `integration_invalid_descriptor` | A malformed Boutiques descriptor produces a clear error at invocation time. |

### 13.8 Edge Case and Error Tests

| Test | Description |
|---|---|
| `error_missing_terms_yaml` | Running `label` or `check` without a `labelman.yaml` produces a clear error. |
| `error_no_images` | Running `label` with no images in the target path produces a clear error or empty output. |
| `error_integration_failure` | If the integration script exits non-zero, `labelman` reports the failure clearly. |
| `error_integration_bad_output` | If the integration script returns malformed output, `labelman` reports a parse error. |
| `error_large_taxonomy` | A `labelman.yaml` with 50+ categories and hundreds of terms parses and checks without issues. |

### Test Infrastructure Notes

- **Mock integrations**: Tests should use mock scripts that return canned scores, not real BLIP/CLIP. This ensures tests are fast, deterministic, and runnable without GPU or model downloads.
- **Fixture files**: Provide a set of valid and invalid `labelman.yaml` fixtures for parsing and check tests.
- **Exit codes**: Test both stdout/stderr content and process exit codes where relevant (especially for `check`).
- **No image dependencies for unit tests**: Label scenario tests can use placeholder image paths since the mock integration ignores actual image content.
