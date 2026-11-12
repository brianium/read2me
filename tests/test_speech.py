from read2me.speech import Segment, segments, speak_code, split_long


def texts(md, **kw):
    return [s.text for s in segments(md, **kw)]


def test_headings_become_sentences_with_level():
    segs = segments("# Proposal\n\n## Why\n\nBecause.")
    assert [(s.text, s.heading) for s in segs] == [("Proposal.", 1), ("Why.", 2), ("Because.", 0)]


def test_frontmatter_is_dropped():
    assert texts("---\ntitle: x\ntags: [a]\n---\n\nHello") == ["Hello."]


def test_inline_code_paths_and_labels():
    assert speak_code(".github/workflows/deploy-docs.yml") == "dot github slash workflows slash deploy docs dot yml"
    assert speak_code("[High]") == "High"
    assert speak_code("claude[bot]") == "claude bot"
    assert speak_code("REVIEW.md") == "Review dot md"
    assert speak_code("severity_counts") == "severity counts"
    assert speak_code("fetchV2UserProfile") == "fetch V2 User Profile"
    assert speak_code("templates/openspec/") == "templates slash openspec"


def test_shell_commands_are_not_read_out():
    assert speak_code("grep -o 'version@[0-9.]*' | head -1") == "a code snippet"


def test_versions_dates_issues_tickets():
    [text] = texts("Released `v1.4.0` on 2026-11-04 for PRs #21–#23, see #24 and API-1234.")
    assert text == "Released version 1 point 4 point 0 on November 4, 2026 for PRs 21 to 23, see number 24 and API 1234."


def test_abbreviations_and_arrows():
    [text] = texts("Use a cache, e.g. Redis → faster.")
    assert text == "Use a cache, for example, Redis to faster."


def test_links_read_their_text_and_bare_urls_their_host():
    assert texts("See [the docs](https://example.com/x).") == ["See the docs."]
    assert texts("See <https://github.com/a/b>.") == ["See a link to github.com."]


def test_ordered_list_numbers_and_tasks():
    assert texts("1. First\n2. Second") == ["1. First.", "2. Second."]
    assert texts("- [x] 1.1 Ship it\n- [ ] 1.2 Test it") == ["Done: 1.1 Ship it.", "1.2 Test it."]


def test_openspec_risk_mitigation():
    [text] = texts("- [Link 404s for outsiders] → Listed as an assumption")
    assert text == "Risk: Link 404s for outsiders. Mitigation: Listed as an assumption."


def test_callouts():
    assert texts("> [!WARNING]\n> Do not push.") == ["Warning: Do not push."]


def test_code_blocks_announce_read_skip():
    md = "Before\n\n```yaml\na: 1\nb: 2\n```\n\nAfter"
    assert texts(md) == ["Before.", "Skipping a 2-line YAML code block.", "After."]
    assert texts(md, code="skip") == ["Before.", "After."]
    assert texts(md, code="read") == ["Before.", "a: 1.", "b: 2.", "After."]
    assert texts("```mermaid\ngraph TD\n```") == ["Skipping a diagram."]


def test_tables_read_row_by_row_with_headers():
    md = "| Topic | Source |\n|---|---|\n| Pin | `AGENTS.md` |\n| Globs | README |"
    assert texts(md) == ["A table with 2 rows.", "Topic: Pin. Source: Agents dot md.", "Topic: Globs. Source: README."]


def test_emoji_dropped():
    assert texts("Done ✅ 🚀") == ["Done."]


def test_split_long_keeps_sentences_whole():
    seg = Segment("One two three. Four five six. Seven eight nine.", pause=0.5, heading=0)
    parts = split_long([seg], max_chars=30)
    assert [p.text for p in parts] == ["One two three. Four five six.", "Seven eight nine."]
    assert parts[-1].pause == 0.5
