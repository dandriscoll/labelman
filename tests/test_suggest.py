"""Tests for the suggest workflow."""

from unittest.mock import patch

import pytest

from labelman.suggest import (
    CategoryProposal,
    ImageSuggestion,
    Proposal,
    SuggestResult,
    bootstrap,
    expand,
    format_suggest_result,
    suggest_to_caption,
    write_suggest_sidecar,
    _extract_words,
)
from labelman.schema import parse


MINIMAL_CONFIG = """\
defaults:
  threshold: 0.3
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
"""

CONFIG_WITH_QUESTION = """\
defaults:
  threshold: 0.3
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
  - name: lighting
    mode: zero-or-one
    question: "What type of lighting is in this image?"
    terms:
      - term: natural
      - term: studio
"""

CONFIG_QUESTION_NO_TERMS = """\
defaults:
  threshold: 0.3
integrations:
  blip:
    endpoint: http://localhost:8080/caption
  clip:
    endpoint: http://localhost:8081/classify
categories:
  - name: lighting
    mode: zero-or-one
    question: "What type of lighting is in this image?"
"""

BOOTSTRAP_CONFIG = """\
defaults:
  threshold: 0.3
integrations:
  blip:
    endpoint: http://localhost:8080/caption
categories:
  - name: placeholder
    mode: exactly-one
    terms:
      - term: unknown
"""

BOOTSTRAP_WITH_QUESTION = """\
defaults:
  threshold: 0.3
integrations:
  blip:
    endpoint: http://localhost:8080/caption
categories:
  - name: setting
    mode: zero-or-one
    question: "Is this photo taken indoors or outdoors?"
"""


def _mock_blip(responses_by_prompt=None, default_responses=None):
    """Create a mock run_blip that returns different responses based on prompt.

    Args:
        responses_by_prompt: dict mapping prompt string -> list of response dicts
        default_responses: list of response dicts when no prompt or no match
    """
    def mock_run_blip(term_list, image_paths, prompt=None, max_tokens=None):
        if prompt and responses_by_prompt and prompt in responses_by_prompt:
            captions = responses_by_prompt[prompt]
        elif default_responses:
            captions = default_responses
        else:
            captions = [{"image": p, "caption": ""} for p in image_paths]

        results = []
        for i, path in enumerate(image_paths):
            if i < len(captions):
                entry = dict(captions[i])
                entry["image"] = path
            else:
                entry = {"image": path, "caption": ""}
            results.append(entry)
        return results
    return mock_run_blip


def _mock_blip_captions(captions):
    """Simple mock: always return these captions regardless of prompt."""
    return _mock_blip(default_responses=[{"caption": c} for c in captions])


def _mock_clip_scores(scores_per_image):
    """Create a mock run_clip that returns given scores."""
    def mock_run_clip(term_list, image_paths):
        results = []
        for i, path in enumerate(image_paths):
            scores = scores_per_image[i] if i < len(scores_per_image) else {}
            results.append({"image": path, "scores": scores})
        return results
    return mock_run_clip


class TestExtractWords:
    def test_filters_stop_words(self):
        words = _extract_words("a dog is running in the park")
        assert "dog" in words
        assert "running" in words
        assert "park" in words
        assert "the" not in words
        assert "is" not in words

    def test_filters_short_words(self):
        words = _extract_words("an ox by me")
        assert "ox" not in words  # 2 chars

    def test_lowercases(self):
        words = _extract_words("Big Red Dog")
        assert "big" in words
        assert "red" in words
        assert "dog" in words


class TestBootstrap:
    @patch("labelman.suggest.run_blip")
    def test_produces_proposals(self, mock_blip):
        mock_blip.side_effect = _mock_blip_captions([
            "a red car parked on the street",
            "a blue car driving on the highway",
            "a red truck on the road",
            "a green car in the parking lot",
        ])
        term_list = parse(BOOTSTRAP_CONFIG)
        result = bootstrap(term_list, ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"])

        assert result.mode == "bootstrap"
        assert len(result.proposals) >= 1
        # "car" appears in 3 captions, should be proposed
        all_terms = [p.term for cp in result.proposals for p in cp.terms]
        assert "car" in all_terms

    @patch("labelman.suggest.run_blip")
    def test_does_not_modify_terms(self, mock_blip):
        mock_blip.side_effect = _mock_blip_captions(["a cat sitting on a mat"])
        config_text = BOOTSTRAP_CONFIG
        term_list_before = parse(config_text)
        cats_before = [(c.name, [t.term for t in c.terms]) for c in term_list_before.categories]

        term_list = parse(config_text)
        bootstrap(term_list, ["img1.jpg"])

        term_list_after = parse(config_text)
        cats_after = [(c.name, [t.term for t in c.terms]) for c in term_list_after.categories]
        assert cats_before == cats_after

    @patch("labelman.suggest.run_blip")
    def test_respects_sample(self, mock_blip):
        mock_blip.side_effect = _mock_blip_captions(["caption one", "caption two"])
        term_list = parse(BOOTSTRAP_CONFIG)
        bootstrap(term_list, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"], sample=2)
        # Should only pass 2 images to BLIP
        call_args = mock_blip.call_args
        assert len(call_args[0][1]) == 2

    @patch("labelman.suggest.run_blip")
    def test_uses_category_questions(self, mock_blip):
        """Bootstrap uses category questions as BLIP VQA prompts."""
        mock_blip.side_effect = _mock_blip(
            responses_by_prompt={
                "Is this photo taken indoors or outdoors?": [
                    {"answer": "outdoors"},
                    {"answer": "indoors"},
                    {"answer": "outdoors"},
                ],
            },
            default_responses=[
                {"caption": "a building with trees"},
                {"caption": "a room with furniture"},
                {"caption": "a park with trees"},
            ],
        )
        term_list = parse(BOOTSTRAP_WITH_QUESTION)
        result = bootstrap(term_list, ["a.jpg", "b.jpg", "c.jpg"])

        # Should have a proposal for the "setting" category with answers
        setting = [cp for cp in result.proposals if cp.category == "setting"]
        assert len(setting) == 1
        assert setting[0].question == "Is this photo taken indoors or outdoors?"
        assert "outdoors" in setting[0].answers
        assert "indoors" in setting[0].answers

    @patch("labelman.suggest.run_blip")
    def test_bootstrap_question_only_category(self, mock_blip):
        """Bootstrap works with a category that has a question but no terms."""
        mock_blip.side_effect = _mock_blip(
            responses_by_prompt={
                "What type of lighting is in this image?": [
                    {"answer": "natural sunlight"},
                ],
            },
            default_responses=[{"caption": "a photo"}],
        )
        term_list = parse(CONFIG_QUESTION_NO_TERMS)
        result = bootstrap(term_list, ["a.jpg"])

        lighting = [cp for cp in result.proposals if cp.category == "lighting"]
        assert len(lighting) == 1
        assert "natural sunlight" in lighting[0].answers


class TestExpand:
    @patch("labelman.suggest.run_blip")
    @patch("labelman.suggest.run_clip")
    def test_uses_existing_categories(self, mock_clip, mock_blip):
        mock_clip.side_effect = _mock_clip_scores([
            {"person": 0.1, "animal": 0.1},  # weak — gap
            {"person": 0.9, "animal": 0.1},  # strong
        ])
        mock_blip.side_effect = _mock_blip_captions([
            "a landscape with mountains and trees",
        ])
        term_list = parse(MINIMAL_CONFIG)
        result = expand(term_list, ["img1.jpg", "img2.jpg"])

        assert result.mode == "expand"
        # Should have proposals from weak images
        assert len(result.proposals) >= 1

    @patch("labelman.suggest.run_blip")
    @patch("labelman.suggest.run_clip")
    def test_does_not_modify_terms(self, mock_clip, mock_blip):
        mock_clip.side_effect = _mock_clip_scores([
            {"person": 0.9, "animal": 0.1},
        ])
        config_text = MINIMAL_CONFIG
        term_list_before = parse(config_text)
        cats_before = [(c.name, [t.term for t in c.terms]) for c in term_list_before.categories]

        term_list = parse(config_text)
        expand(term_list, ["img1.jpg"])

        term_list_after = parse(config_text)
        cats_after = [(c.name, [t.term for t in c.terms]) for c in term_list_after.categories]
        assert cats_before == cats_after

    @patch("labelman.suggest.run_blip")
    @patch("labelman.suggest.run_clip")
    def test_filters_existing_terms(self, mock_clip, mock_blip):
        mock_clip.side_effect = _mock_clip_scores([
            {"person": 0.1, "animal": 0.1},
        ])
        mock_blip.side_effect = _mock_blip_captions([
            "a person walking with their animal companion",
        ])
        term_list = parse(MINIMAL_CONFIG)
        result = expand(term_list, ["img1.jpg"])

        # "person" and "animal" are existing terms — should not be proposed
        new_terms = [
            p.term for cp in result.proposals
            for p in cp.terms if not p.term.startswith("[unused")
        ]
        assert "person" not in new_terms
        assert "animal" not in new_terms

    @patch("labelman.suggest.run_blip")
    @patch("labelman.suggest.run_clip")
    def test_uses_category_question_as_prompt(self, mock_clip, mock_blip):
        """When a category has a question, expand passes it to BLIP as prompt."""
        mock_clip.side_effect = _mock_clip_scores([
            {"person": 0.9, "animal": 0.1, "natural": 0.5, "studio": 0.2},
        ])
        blip_calls = []
        def tracking_blip(term_list, image_paths, prompt=None, max_tokens=None):
            blip_calls.append({"paths": image_paths, "prompt": prompt})
            return [{"image": p, "caption": "warm ambient lighting"} for p in image_paths]
        mock_blip.side_effect = tracking_blip

        term_list = parse(CONFIG_WITH_QUESTION)
        expand(term_list, ["img1.jpg"])

        # BLIP should have been called with the question as prompt
        prompted = [c for c in blip_calls if c["prompt"] is not None]
        assert len(prompted) == 1
        assert "lighting" in prompted[0]["prompt"]

    @patch("labelman.suggest.run_blip")
    @patch("labelman.suggest.run_clip")
    def test_question_proposals_grouped_by_category(self, mock_clip, mock_blip):
        """Proposals from category questions are filed under that category."""
        mock_clip.side_effect = _mock_clip_scores([
            {"person": 0.9, "animal": 0.1, "natural": 0.5, "studio": 0.2},
        ])
        mock_blip.side_effect = lambda tl, paths, prompt=None, max_tokens=None: [
            {"image": p, "caption": "warm ambient fluorescent lighting"} for p in paths
        ]
        term_list = parse(CONFIG_WITH_QUESTION)
        result = expand(term_list, ["img1.jpg"])

        lighting_proposals = [cp for cp in result.proposals if cp.category == "lighting"]
        if lighting_proposals:
            terms = [p.term for p in lighting_proposals[0].terms]
            # "natural" and "studio" are existing terms, should be filtered
            assert "natural" not in terms
            assert "studio" not in terms

    @patch("labelman.suggest.run_blip")
    @patch("labelman.suggest.run_clip")
    def test_expand_question_only_category(self, mock_clip, mock_blip):
        """Expand works with a category that has a question but no terms."""
        mock_clip.side_effect = _mock_clip_scores([
            {},  # no terms to score
        ])
        mock_blip.side_effect = _mock_blip(
            responses_by_prompt={
                "What type of lighting is in this image?": [
                    {"answer": "natural sunlight"},
                ],
            },
            default_responses=[{"caption": "a photo"}],
        )
        term_list = parse(CONFIG_QUESTION_NO_TERMS)
        result = expand(term_list, ["a.jpg"])

        lighting = [cp for cp in result.proposals if cp.category == "lighting"]
        assert len(lighting) == 1
        assert "natural sunlight" in lighting[0].answers

    @patch("labelman.suggest.run_blip")
    @patch("labelman.suggest.run_clip")
    def test_expand_answers_include_vqa_responses(self, mock_clip, mock_blip):
        """Answers from VQA (using 'answer' key) are captured."""
        mock_clip.side_effect = _mock_clip_scores([{}, {}])
        mock_blip.side_effect = _mock_blip(
            responses_by_prompt={
                "What type of lighting is in this image?": [
                    {"answer": "natural light from a window"},
                    {"answer": "studio flash"},
                ],
            },
            default_responses=[{"caption": "photo"}, {"caption": "photo"}],
        )
        term_list = parse(CONFIG_QUESTION_NO_TERMS)
        result = expand(term_list, ["a.jpg", "b.jpg"])

        lighting = [cp for cp in result.proposals if cp.category == "lighting"]
        assert len(lighting) == 1
        assert "natural light from a window" in lighting[0].answers
        assert "studio flash" in lighting[0].answers


class TestFormatSuggestResult:
    def test_output_is_structured(self):
        result = SuggestResult(
            mode="bootstrap",
            proposals=[
                CategoryProposal(
                    category="colors",
                    terms=[
                        Proposal(term="red", source_count=5),
                        Proposal(term="blue", source_count=3),
                    ],
                ),
            ],
        )
        output = format_suggest_result(result)
        assert "category: colors" in output
        assert "term: red" in output
        assert "term: blue" in output
        assert "bootstrap" in output

    def test_empty_proposals(self):
        result = SuggestResult(mode="bootstrap", proposals=[])
        output = format_suggest_result(result)
        assert "No proposals" in output

    def test_format_includes_question_and_answers(self):
        result = SuggestResult(
            mode="bootstrap",
            proposals=[
                CategoryProposal(
                    category="lighting",
                    question="What type of lighting?",
                    answers=["natural", "studio flash"],
                    terms=[Proposal(term="flash", source_count=1)],
                ),
            ],
        )
        output = format_suggest_result(result)
        assert "question:" in output
        assert "What type of lighting?" in output
        assert "answers:" in output
        assert "natural" in output
        assert "studio flash" in output
        assert "term: flash" in output


CONFIG_WITH_GLOBALS = """\
defaults:
  threshold: 0.3
integrations:
  blip:
    endpoint: http://localhost:8080/caption
  clip:
    endpoint: http://localhost:8081/classify
global_terms:
  - photo
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
"""


class TestImageSuggestions:
    @patch("labelman.suggest.run_blip")
    def test_bootstrap_populates_image_suggestions(self, mock_blip):
        mock_blip.side_effect = _mock_blip_captions([
            "a red car parked on the street",
            "a blue truck driving on the highway",
        ])
        term_list = parse(BOOTSTRAP_CONFIG)
        result = bootstrap(term_list, ["img1.jpg", "img2.jpg"])

        assert len(result.image_suggestions) == 2
        by_image = {s.image: s for s in result.image_suggestions}
        assert "img1.jpg" in by_image
        assert "img2.jpg" in by_image
        # Mined terms should be populated from captions
        assert len(by_image["img1.jpg"].mined_terms) > 0

    @patch("labelman.suggest.run_blip")
    def test_bootstrap_vqa_unmatched_answers_go_to_mined(self, mock_blip):
        """VQA answers that don't match existing terms go to mined_terms, not labels."""
        mock_blip.side_effect = _mock_blip(
            responses_by_prompt={
                "Is this photo taken indoors or outdoors?": [
                    {"answer": "outdoors"},
                    {"answer": "indoors"},
                ],
            },
            default_responses=[
                {"caption": "a building with trees"},
                {"caption": "a room with furniture"},
            ],
        )
        term_list = parse(BOOTSTRAP_WITH_QUESTION)
        result = bootstrap(term_list, ["a.jpg", "b.jpg"])

        by_image = {s.image: s for s in result.image_suggestions}
        # "outdoors"/"indoors" are not in the taxonomy terms, so they go to mined
        assert "outdoors" not in by_image["a.jpg"].labels
        assert "outdoors" in by_image["a.jpg"].mined_terms
        assert "indoors" in by_image["b.jpg"].mined_terms

    @patch("labelman.suggest.run_blip")
    def test_bootstrap_vqa_matching_answers_go_to_labels(self, mock_blip):
        """VQA answers matching existing taxonomy terms go to labels."""
        mock_blip.side_effect = _mock_blip(
            responses_by_prompt={
                "What type of lighting is in this image?": [
                    {"answer": "natural light"},
                    {"answer": "It looks like studio lighting"},
                ],
            },
            default_responses=[
                {"caption": "a photo"},
                {"caption": "a photo"},
            ],
        )
        term_list = parse(CONFIG_WITH_QUESTION)
        result = bootstrap(term_list, ["a.jpg", "b.jpg"])

        by_image = {s.image: s for s in result.image_suggestions}
        # "natural" and "studio" are terms in the lighting category
        assert "natural" in by_image["a.jpg"].labels
        assert "studio" in by_image["b.jpg"].labels

    @patch("labelman.suggest.run_blip")
    @patch("labelman.suggest.run_clip")
    def test_expand_populates_clip_detected_labels(self, mock_clip, mock_blip):
        mock_clip.side_effect = _mock_clip_scores([
            {"person": 0.9, "animal": 0.1},
            {"person": 0.1, "animal": 0.8},
        ])
        mock_blip.side_effect = _mock_blip_captions(["a photo"])
        term_list = parse(MINIMAL_CONFIG)
        result = expand(term_list, ["img1.jpg", "img2.jpg"])

        by_image = {s.image: s for s in result.image_suggestions}
        # exactly-one: highest scoring term selected
        assert "person" in by_image["img1.jpg"].labels
        assert "animal" in by_image["img2.jpg"].labels

    @patch("labelman.suggest.run_blip")
    @patch("labelman.suggest.run_clip")
    def test_expand_weak_images_get_mined_terms(self, mock_clip, mock_blip):
        mock_clip.side_effect = _mock_clip_scores([
            {"person": 0.1, "animal": 0.1},  # weak
        ])
        mock_blip.side_effect = _mock_blip_captions([
            "a landscape with mountains and rivers",
        ])
        term_list = parse(MINIMAL_CONFIG)
        result = expand(term_list, ["img1.jpg"])

        by_image = {s.image: s for s in result.image_suggestions}
        assert len(by_image["img1.jpg"].mined_terms) > 0
        assert "landscape" in by_image["img1.jpg"].mined_terms


class TestSuggestToCaption:
    def test_basic_caption(self):
        term_list = parse(MINIMAL_CONFIG)
        suggestion = ImageSuggestion(
            image="img1.jpg",
            labels=["person", "outdoors"],
        )
        caption = suggest_to_caption(term_list, suggestion)
        assert caption == "person, outdoors"

    def test_includes_global_terms(self):
        term_list = parse(CONFIG_WITH_GLOBALS)
        suggestion = ImageSuggestion(
            image="img1.jpg",
            labels=["person"],
        )
        caption = suggest_to_caption(term_list, suggestion)
        assert caption == "photo, person"

    def test_excludes_mined_by_default(self):
        term_list = parse(MINIMAL_CONFIG)
        suggestion = ImageSuggestion(
            image="img1.jpg",
            labels=["person"],
            mined_terms=["landscape", "mountain"],
        )
        caption = suggest_to_caption(term_list, suggestion)
        assert caption == "person"
        assert "landscape" not in caption

    def test_includes_mined_when_requested(self):
        term_list = parse(MINIMAL_CONFIG)
        suggestion = ImageSuggestion(
            image="img1.jpg",
            labels=["person"],
            mined_terms=["landscape", "mountain"],
        )
        caption = suggest_to_caption(term_list, suggestion, include_mined=True)
        assert caption == "person, landscape, mountain"

    def test_deduplicates(self):
        term_list = parse(CONFIG_WITH_GLOBALS)
        suggestion = ImageSuggestion(
            image="img1.jpg",
            labels=["photo", "person"],
            mined_terms=["person", "extra"],
        )
        caption = suggest_to_caption(term_list, suggestion, include_mined=True)
        assert caption == "photo, person, extra"


class TestWriteSuggestSidecar:
    def test_writes_sidecar_next_to_image(self, tmp_path):
        img = tmp_path / "img1.jpg"
        img.write_text("")
        term_list = parse(MINIMAL_CONFIG)
        suggestion = ImageSuggestion(image=str(img), labels=["person"])

        sidecar = write_suggest_sidecar(term_list, suggestion)
        assert sidecar == tmp_path / "img1.detected.txt"
        assert sidecar.read_text() == "person"

    def test_writes_to_output_dir(self, tmp_path):
        img = tmp_path / "images" / "img1.jpg"
        img.parent.mkdir()
        img.write_text("")
        out_dir = tmp_path / "output"
        term_list = parse(MINIMAL_CONFIG)
        suggestion = ImageSuggestion(image=str(img), labels=["person"])

        sidecar = write_suggest_sidecar(term_list, suggestion, output_dir=out_dir)
        assert sidecar == out_dir / "img1.detected.txt"
        assert sidecar.read_text() == "person"

    def test_include_mined(self, tmp_path):
        img = tmp_path / "img1.jpg"
        img.write_text("")
        term_list = parse(MINIMAL_CONFIG)
        suggestion = ImageSuggestion(
            image=str(img),
            labels=["person"],
            mined_terms=["landscape"],
        )

        sidecar = write_suggest_sidecar(term_list, suggestion, include_mined=True)
        assert sidecar.read_text() == "person, landscape"

    def test_without_mined(self, tmp_path):
        img = tmp_path / "img1.jpg"
        img.write_text("")
        term_list = parse(MINIMAL_CONFIG)
        suggestion = ImageSuggestion(
            image=str(img),
            labels=["person"],
            mined_terms=["landscape"],
        )

        sidecar = write_suggest_sidecar(term_list, suggestion, include_mined=False)
        assert sidecar.read_text() == "person"
