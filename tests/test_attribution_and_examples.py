import random
from datetime import date

import pytest

from poster import compose, examples, gh, guardrail, llm, review
from poster.compose import Draft

TODAY = date(2026, 11, 9)
ARTICLE = [{"title": "Leaving my editor after a decade", "source": "blog.example",
            "snippet": "I have used the same editor for a decade, writing plugins every weekend. Last month I finally "
                       "switched to a terminal workflow because startup times kept getting slower."}]
AUTHOR = {"name": "Dev", "focus": "building AI agents and workflow automation", "daily_tools": "Claude Code, n8n"}
NEWS_TOPIC = {"kind": "news", "title": "Leaving my editor after a decade", "angle": ""}


# --------------------------------------------------------------- borrowed first person
def test_article_story_retold_as_mine_is_caught():
    post = "I used the same editor for a decade, writing plugins every weekend.\n\nLast month I switched to a terminal workflow because startup times kept getting slower."
    hits = guardrail.borrowed_first_person(post, AUTHOR, ARTICLE)
    assert len(hits) == 2


def test_attributed_story_and_own_take_pass():
    post = ("A developer wrote that after a decade with one editor, slow startup times pushed them to a terminal workflow.\n\n"
            "I build AI agents and workflow automation, and I watch for the same slow creep in my own tools.")
    assert guardrail.borrowed_first_person(post, AUTHOR, ARTICLE) == []


def test_check_rejects_and_retry_note_explains():
    r = guardrail.check("Last month I finally switched to a terminal workflow because startup times kept getting slower.",
                        author=AUTHOR, topic=NEWS_TOPIC, news_items=ARTICLE, today=TODAY, max_chars=1300, max_hashtags=3)
    assert not r.ok and r.reason == "borrowed_first_person"
    assert "third person" in guardrail.retry_note(r)


def test_fact_posts_are_not_checked_for_borrowing():
    r = guardrail.check("Last month I finally switched to a terminal workflow because startup times kept getting slower.",
                        author=AUTHOR, topic={"kind": "fact", "title": "x", "angle": ""}, news_items=ARTICLE,
                        today=TODAY, max_chars=1300, max_hashtags=3)
    assert r.ok


# --------------------------------------------------------------- copying from examples
EXAMPLE = "Most teams do not have a data problem. They have a decision problem dressed up as a data problem."


def test_copied_phrase_found_and_rejected():
    post = "Here is my view. Most teams do not have a data problem, they have a decision problem dressed up as one."
    assert guardrail.copied_phrase(post, [EXAMPLE], 8) == "most teams do not have a data problem"
    r = guardrail.check(post, author={}, topic={"kind": "fact", "title": "", "angle": ""}, news_items=[], today=TODAY,
                        max_chars=1300, max_hashtags=3, examples=[EXAMPLE], copy_ngram=8)
    assert r.reason == "copied_from_example"


def test_similar_style_without_copying_passes():
    assert guardrail.copied_phrase("Most bugs are not code problems. They are naming problems.", [EXAMPLE], 8) == ""


def test_examples_go_in_the_message_not_the_sources(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "generate", lambda cfg, p, system, user, t, schema=None: seen.append(user) or
                        ({"post_text": "Short post about naming.", "hook": "", "facts_used": [], "warnings": ["None"]}, "m"))
    cfg = {"llm": {"primary": "groq", "fallback": "gemini", "max_attempts": 1, "temperature": 0.6, "retry_temperature": 0.2},
           "post": {"max_chars": 1300, "max_hashtags": 3}, "timezone": "UTC", "bilingual": {"enabled": False}}
    monkeypatch.setattr(compose, "system_prompt", lambda cfg: "sys")
    d = compose.write(cfg, {}, {"kind": "fact", "title": "Naming", "category": "coding_fact"}, [], examples=["An example with 42 in it."])
    assert "STYLE EXAMPLES" in seen[0] and "An example with 42 in it." in seen[0]
    assert '"news": []' in seen[0]          # the example is not part of the input JSON
    assert d.warnings == []                  # a literal "None" warning is dropped


# --------------------------------------------------------------- saving examples
def test_render_and_body_roundtrip():
    text = examples.render("Hook.\n\nBody.", why="great  hook", source="saved", issue_number=12, saved=date(2026, 11, 9))
    assert text.startswith("---\nsaved: 2026-11-09\nsource: saved\nissue: 12\nwhy: great hook\n---\n")
    assert examples.body_of(text) == "Hook.\n\nBody."
    assert examples.slug("Great hook! (short)") == "great-hook-short"


def test_load_skips_readme_and_samples(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("not an example")
    for i in range(5):
        (tmp_path / f"e{i}.md").write_text(examples.render(f"Post {i} " + "x" * 20, why="w", source="saved", issue_number=i))
    monkeypatch.setattr(examples, "DIR", tmp_path)
    got = examples.load({"examples": {"per_draft": 3, "max_chars": 10}}, rng=random.Random(0))
    assert len(got) == 3 and all(len(t) <= 10 and t.startswith("Post") for t in got)
    assert examples.load({"examples": {"per_draft": 0}}) == []


@pytest.fixture
def board(monkeypatch):
    log = []
    for name in ("comment", "close", "put_file"):
        monkeypatch.setattr(gh, name, lambda *a, _n=name, **k: log.append((_n, a)))
    return log


def test_save_from_issue(board):
    examples.save_from_issue({"number": 12, "title": "Great hook", "body": "A post I liked, long enough to be worth saving here."})
    put = [e for e in board if e[0] == "put_file"][0]
    assert put[1][0].endswith("-great-hook-12.md") and "why: Great hook" in put[1][1]
    assert ("close", (12,)) in board


def test_save_from_issue_needs_text(board):
    examples.save_from_issue({"number": 12, "title": "Great hook", "body": "too short"})
    assert [e[0] for e in board] == ["comment"]


def test_example_command_saves_english_part(board):
    topic = {"kind": "fact", "title": "T", "category": "coding_fact", "angle": ""}
    body = gh.render_draft(topic, [], Draft("The English post.", model="m", post_text_he="הפוסט.", translation_model="m"))
    review.save_example({"number": 9, "body": body, "labels": [{"name": "draft"}, {"name": "published"}], "title": "x"}, "")
    put = [e for e in board if e[0] == "put_file"][0]
    assert "source: own" in put[1][1] and put[1][1].rstrip().endswith("The English post.") and "הפוסט" not in put[1][1]


# --------------------------------------------------------------------------- invented experience
PROFILE = {"name": "N", "headline": "AI automation developer", "focus": "building AI agents and automations"}


def test_invented_experience_is_sent_back():
    for sent in ("When I build customer-facing AI agents that call multiple REST APIs, I constantly hit that ceiling.",
                 "I often need the same raw speed that Fortran provides.",
                 "In my agents, retries matter more than speed."):
        assert guardrail.claimed_experience(sent, PROFILE, {"kind": "onthisday"}) == [sent], sent
    for sent in ("I build AI agents and automations.",                       # the profile says so
                 "I think this still matters for anyone building agents.",   # an opinion
                 "Agents that call several APIs hit the same limit.",        # general
                 "What would you change in your pipelines?"):
        assert guardrail.claimed_experience(sent, PROFILE, {"kind": "fact"}) == [], sent
    backed = {"kind": "news", "angle": "I hit rate limits in my agents every week"}
    assert guardrail.claimed_experience("In my agents, rate limits hit every week.", PROFILE, backed) == []
    for kind in ("project", "queued"):  # the poster's own work
        assert guardrail.claimed_experience("When I build agents, I constantly hit that ceiling.", PROFILE, {"kind": kind}) == []


def test_check_rejects_invented_experience_with_a_retry_note():
    post = "Amdahl's law still sets the limit.\n\nWhen I build agents that call several APIs, I constantly hit that ceiling.\n\nWhere do you hit it?"
    res = guardrail.check(post, author=PROFILE, topic={"kind": "onthisday", "title": "Amdahl's law"}, news_items=[],
                          today=date(2026, 11, 16), max_chars=1300, max_hashtags=3)
    assert not res.ok and res.reason == "claimed_experience"
    assert "general terms" in guardrail.retry_note(res)
