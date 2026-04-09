---
title: "labelman"
summary: "Bulk image labeling for fine-tuning datasets."
shipped: 2026-03-28
tags: [machine-learning, image-labeling, cli, datasets, python]
links:
  - label: "GitHub"
    url: "https://github.com/dandriscoll/labelman"
    primary: true
---

## What is it?

labelman is a CLI tool and web app that applies structured labels to image datasets using BLIP, CLIP, and Qwen VL models. Define a taxonomy in YAML, auto-detect labels across thousands of images, then refine them through a built-in web UI. The result: clean, consistent training data for fine-tuning.

## Key Features

- **Taxonomy-driven labeling** — Define categories, terms, and confidence thresholds in a single YAML file. Three category modes (exactly-one, zero-or-one, zero-or-more) control how labels are assigned.
- **AI-assisted discovery** — Bootstrap new taxonomies from scratch with BLIP captioning, or expand existing ones by finding gaps with CLIP classification.
- **Web UI for manual review** — Browse images, edit labels, and do bulk operations with keyboard shortcuts. Everything merges cleanly with detected labels.
- **Sidecar file workflow** — Detected labels, manual overrides, and final output live in separate text files per image. Merge logic handles deduplication and suppressions automatically.
- **Term renaming** — Rename a term across your config and every sidecar file in one command.

---

[View on ishipped.io](https://ishipped.io/card/dandriscoll/labelman)
