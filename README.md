# labelman

Bulk image labeling tool for fine-tuning datasets. labelman maintains a taxonomy of terms in `labelman.yaml` and uses external models (BLIP for captioning/VQA, CLIP for classification) to apply labels to images.

## Install

```bash
pip install -e .
```

Requires Python 3.10+. The only runtime dependency is PyYAML.

## Quick start

```bash
# 1. Create a starter labelman.yaml
labelman init

# 2. Edit labelman.yaml to define your taxonomy and integration endpoints

# 3. Validate the config
labelman check

# 4. Discover terms from your images (uses BLIP)
labelman suggest --mode bootstrap --images ./images --sample 50

# 5. Label images against your taxonomy (uses CLIP)
labelman label --images ./images

# 6. Review and manually edit labels in the browser
labelman ui --images ./images

# 7. Merge manual edits + detected labels into final output
labelman apply --images ./images
```

## Core concepts

| Concept | Description |
|---|---|
| **Term list** | `labelman.yaml` -- the single source of truth for the taxonomy. |
| **Category** | A labeling axis (e.g. "lighting", "mood"). Each has a mode controlling how many terms apply per image. |
| **Term** | A candidate label within a category (e.g. "natural", "studio" within "lighting"). |
| **Threshold** | Confidence cutoff. Inherits: global -> category -> term (most specific wins). |
| **Global terms** | Labels applied to every image unconditionally. |

### Category modes

| Mode | Behavior |
|---|---|
| `exactly-one` | Always assigns the top-scoring term, even if below threshold. |
| `zero-or-one` | Assigns the top-scoring term only if above threshold. |
| `zero-or-more` | Assigns all terms above threshold. |

## File formats

### labelman.yaml

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
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
      - term: object

  - name: mood
    mode: zero-or-more
    threshold: 0.4
    terms:
      - term: calm
      - term: tense
        threshold: 0.25
```

### Sidecar files

Per-image text files with comma-separated labels:

| File | Example | Purpose | Written by |
|---|---|---|---|
| `img.detected.txt` | `person, calm` | Labels from CLIP detection | `labelman label` |
| `img.labels.txt` | `custom tag, -calm` | Manual overrides (prefix `-` to suppress) | User / `labelman ui` |
| `img.txt` | `aircraft, mooney m20, custom tag, person` | Final merged output | `labelman apply` |

Merge order for final output: global terms, then manual labels, then detected labels. Duplicates are removed (first occurrence wins). Suppressed terms (`-term` in `.labels.txt`) are excluded.

## Commands

### `labelman init [--dir DIR] [--force]`

Create a starter `labelman.yaml` with example categories, integration endpoints, and comments.

### `labelman check [--config PATH]`

Validate `labelman.yaml` for structural and semantic correctness. Fast, offline, no models needed.

### `labelman suggest [--mode bootstrap|expand] [--images DIR] [--sample N|25%] [--dry-run]`

Propose new terms. **bootstrap** mode uses BLIP captioning to discover categories from scratch. **expand** mode uses CLIP to find gaps in the existing taxonomy. Proposals are advisory -- you decide what to add.

### `labelman label [--images DIR] [--config PATH] [--output DIR] [-q]`

Score all images against the taxonomy using CLIP. Writes `.detected.txt` sidecars, `labels.csv`, and `report.html`.

### `labelman apply [--images DIR] [--config PATH] [--output DIR]`

Merge `.labels.txt` (manual) + `.detected.txt` (detected) into final `.txt` sidecars, applying suppression and exclusive-category rules.

### `labelman rename --old TERM --new TERM [--config PATH] [--dry-run]`

Rename a term across `labelman.yaml` and all sidecar files (`.labels.txt`, `.detected.txt`, `.txt`) in the config's directory. Use `--dry-run` to preview changes.

### `labelman ui [--images DIR] [--host HOST] [--port PORT]`

Launch a web interface for browsing images and editing manual labels. Supports keyboard navigation, bulk operations, and taxonomy-aware term buttons. Default port: 7933.

### `labelman descriptor blip|clip`

Print the Boutiques JSON descriptor for a built-in integration script.

### Global options

`--verbose` / `-v` can appear anywhere on the command line to enable debug logging to stderr.

## Integrations

External models are called through shell scripts. Each integration can use either a built-in script with an HTTP endpoint, or a custom script:

```yaml
integrations:
  clip:
    endpoint: http://localhost:8081/classify   # built-in script calls this URL
    # script: /path/to/custom-clip.sh         # custom script (endpoint ignored)
```

Run `labelman descriptor blip` or `labelman descriptor clip` to see the CLI contract.

## Development

```bash
pip install -e ".[dev]"

# Run all tests
pytest

# Run a specific test file
pytest tests/test_label.py

# Run a specific test
pytest -k "test_parse_minimal"
```

See [WORKFLOW.md](WORKFLOW.md) for contribution workflow and [docs/DESIGN.md](docs/DESIGN.md) for the full system design.

## Agentic instructions

Copy the block below into your agent's system prompt or CLAUDE.md to give it the context needed to work with labelman effectively.

~~~markdown
## labelman project context

labelman is a bulk image labeling CLI tool (Python 3.10+, PyYAML). The canonical config is `labelman.yaml`.

### Key commands
- `labelman init` -- scaffold a starter config
- `labelman check` -- validate config (offline, no models)
- `labelman suggest --mode bootstrap|expand --images DIR` -- propose terms (BLIP/CLIP)
- `labelman label --images DIR` -- score images with CLIP, write .detected.txt sidecars
- `labelman apply --images DIR` -- merge manual + detected into final .txt sidecars
- `labelman rename --old TERM --new TERM [--dry-run]` -- rename a term across config and all sidecars
- `labelman ui --images DIR` -- web UI for manual labeling (port 7933)

### File layout
```
labelman.yaml           # taxonomy config (categories, terms, thresholds, integrations)
img.detected.txt        # CLIP-detected labels (comma-separated)
img.labels.txt          # manual labels, prefix - to suppress (comma-separated)
img.txt                 # final merged output (comma-separated)
```

### Code layout
```
src/labelman/
  cli.py          # CLI entrypoint, argparse, subcommand handlers
  schema.py       # YAML parsing, dataclasses (TermList, Category, Term)
  label.py        # labeling engine, sidecar I/O, CSV/HTML reports
  suggest.py      # suggest workflow (bootstrap/expand), BLIP/CLIP invocation
  check.py        # config validation
  rename.py       # term renaming across config + sidecars
  integrations.py # script/endpoint resolution, BLIP/CLIP/LLM runners
  web.py          # web UI server
```

### Running tests
```bash
pytest                          # full suite
pytest tests/test_rename.py     # specific file
pytest -k "test_name"           # specific test
```

### Category modes
- `exactly-one`: always assigns top-scoring term (forced even below threshold)
- `zero-or-one`: top-scoring term if above threshold, else none
- `zero-or-more`: all terms above threshold

### Threshold inheritance
term-level > category-level > global default (most specific wins)

### Sidecar merge order (for apply)
global_terms + manual labels + detected labels, deduplicated, minus suppressions

### Conventions
- No CI pipeline; local test pass is the gate
- Edits to labelman.yaml should preserve comments and formatting
- Integration scripts are described by Boutiques descriptors (JSON)
~~~
