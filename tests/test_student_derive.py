"""
Unit tests for the deterministic helpers in src/student_derive.py.

These are pure-Python functions (no Ollama call, no PDF I/O), so they pin
exactly the parsing / mapping / split logic without needing a live model.
Run from the resume_parser/ root:

    python -m pytest tests/ -q
"""
from __future__ import annotations

import pytest

from src.student_derive import (
    map_degree,
    map_branch,
    parse_cgpa,
    parse_graduation_year,
    pick_primary_education,
    split_skills_and_tools,
    flatten_internships,
    _clean_state,
    _extract_specialization,
    _parse_tech_stack,
    _condensation_lost_metrics,
    _attribute_link_to_project,
    _label_to_url,
    resolve_hyperlinks,
)


# ── map_degree ──────────────────────────────────────────────────────────────

class TestMapDegree:
    @pytest.mark.parametrize("raw,expected", [
        ("B.Tech in Computer Science and Engineering", "btech"),
        ("B Tech CSE", "btech"),
        ("BTech, Mechanical Engineering", "btech"),
        ("Bachelor of Technology", "btech"),
        ("M.Tech in CSE", "mtech"),
        ("MTech, Mechanical Engineering", "mtech"),
        ("Master of Technology", "mtech"),
        ("MCA", "mca"),
        ("M.C.A.", "mca"),
        ("BCA", "bca"),
        ("B.C.A.", "bca"),
        ("B C A", "bca"),
        ("Bachelor of Computer Applications", "bca"),
        ("Bachelor of Computer Application", "bca"),
        ("MBA", "mba"),
        ("M.B.A. (Marketing)", "mba"),
        ("Master of Business Administration", "mba"),
        ("B.Sc.", "bsc"),
        ("B Sc Physics", "bsc"),
        ("Bachelor of Science", "bsc"),
        ("B.E. Mechanical", "be"),
        ("Bachelor of Engineering", "be"),
        ("Diploma in Computer Science", "diploma"),
        ("Polytechnic", "diploma"),
        # Post-Graduate Diplomas — Pulse enum has no PGD bucket, map to 'other'
        ("PGDM", "other"),
        ("PGDFM (FORESTRY MANAGEMENT)", "other"),
        ("PGDBA Marketing", "other"),
        ("PGP", "other"),
        # Informal Indian academic phrasings (common in IIT student resumes)
        ("Fourth Year Undergraduate, Mechanical Engineering", "btech"),
        ("Final Year Undergraduate, CSE", "btech"),
        ("4th Year Undergraduate Computer Science", "btech"),
        ("Second Year Undergraduate, Electronics", "btech"),
    ])
    def test_recognized_degrees(self, raw, expected):
        assert map_degree(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("M.Sc. Physics", "other"),
        ("M.E.", "other"),
    ])
    def test_other_recognized_but_not_enum(self, raw, expected):
        assert map_degree(raw) == expected

    @pytest.mark.parametrize("raw", [
        None, "", "Bharatanatyam Diploma in Performing Arts",  # last one DOES contain 'diploma'
    ])
    def test_unrecognized_or_empty(self, raw):
        # None / empty returns None
        if not raw:
            assert map_degree(raw) is None


# ── map_branch ──────────────────────────────────────────────────────────────

class TestMapBranch:
    @pytest.mark.parametrize("raw,expected", [
        # Plain departments
        ("B.Tech in Computer Science and Engineering", "cse"),
        ("BTech, CSE", "cse"),
        ("M.Tech Computer Science", "cse"),
        ("B.Tech in Electronics and Communication", "ece"),
        ("MTech ECE", "ece"),
        ("B.Tech Mechanical Engineering", "mechanical"),
        ("MTech, Mechanical Engineering", "mechanical"),
        ("B.Tech in Civil Engineering", "civil"),
        ("B.Tech Aerospace", "aerospace"),
        ("B.Tech in Chemical", "chemical"),
        ("B.Tech Electrical Engineering", "electrical"),
        ("BTech Mechatronics", "mechatronics"),
        ("BTech in Automobile Engineering", "automotive"),
        ("B.Tech Instrumentation", "instrumentation"),

        # Pure-spec degrees (no compound department keyword)
        ("B.Tech in Data Science", "ai_ml"),
        ("B.Tech in Artificial Intelligence and Machine Learning", "ai_ml"),
        ("B.Tech in Cybersecurity", "cybersecurity"),

        # Compound: department wins over specialization
        ("B.Tech in CSE - Data Science", "cse"),
        ("B.Tech in CSE (AIML)", "cse"),
        ("B.Tech in CSE with focus in Cybersecurity", "cse"),
        ("M.Tech ECE - VLSI Design", "ece"),

        # Informal subtitle phrasings — branch should still extract
        ("Fourth Year Undergraduate, Mechanical Engineering", "mechanical"),
        ("Final Year Undergraduate, CSE", "cse"),
    ])
    def test_known_branches(self, raw, expected):
        assert map_branch(raw) == expected

    @pytest.mark.parametrize("raw", [
        None, "",
        "B.Tech, 2022-2026",       # no branch keyword
        "Bachelor of Music",       # unrelated
    ])
    def test_unknown_or_empty(self, raw):
        assert map_branch(raw) is None


# ── _extract_specialization ────────────────────────────────────────────────

class TestExtractSpecialization:
    @pytest.mark.parametrize("raw,branch,expected", [
        ("B.Tech in CSE - Data Science", "cse", "Data Science"),
        ("B.Tech in Computer Science and Engineering - Data Science", "cse", "Data Science"),
        ("B.Tech Computer Science (AIML)", "cse", "AIML"),
        ("B.Tech in Computer Science (Data Science)", "cse", "Data Science"),
        ("M.Tech ECE - VLSI Design", "ece", "VLSI Design"),
        ("B.Tech CSE with specialization in AI", "cse", "AI"),
        ("B.Tech in CSE with focus in Cybersecurity", "cse", "Cybersecurity"),
    ])
    def test_compound_degrees(self, raw, branch, expected):
        assert _extract_specialization(None, raw, branch) == expected

    @pytest.mark.parametrize("raw,branch", [
        ("B.Tech in Computer Science and Engineering", "cse"),
        ("M.Tech, Mechanical Engineering", "mechanical"),
        ("B.Tech in Data Science", "ai_ml"),
    ])
    def test_no_specialization(self, raw, branch):
        assert _extract_specialization(None, raw, branch) is None

    def test_year_range_not_treated_as_specialization(self):
        # 'CSE, 2022-2026' shouldn't parse 2022-2026 as a specialization
        assert _extract_specialization(None, "B.Tech in CSE, 2022-2026", "cse") is None

    def test_llm_field_overrides_regex(self):
        # If the LLM gave us a specialization explicitly, use it
        result = _extract_specialization(
            llm_field="Data Analytics",
            raw_degree="B.Tech in CSE - Some Other Tail",
            branch_enum="cse",
        )
        assert result == "Data Analytics"

    def test_llm_field_rejected_if_just_branch_echo(self):
        # LLM emits 'CSE' as specialization for a CSE degree — drop
        result = _extract_specialization(
            llm_field="CSE",
            raw_degree="B.Tech in CSE - Real Specialization",
            branch_enum="cse",
        )
        # Should fall through to regex on raw_degree
        assert result == "Real Specialization"


# ── parse_cgpa ──────────────────────────────────────────────────────────────

class TestParseCgpa:
    @pytest.mark.parametrize("text,expected", [
        ("8.5/10", (8.5, "10")),
        ("CGPA: 8.5/10", (8.5, "10")),
        ("8.5 / 10", (8.5, "10")),
        ("9.08/10", (9.08, "10")),

        ("71%", (71.0, "percentage")),
        ("Aggregate: 71%", (71.0, "percentage")),
        ("83.5 %", (83.5, "percentage")),

        ("CGPA: 8.5", (8.5, "10")),
        ("CGPA 7.92", (7.92, "10")),
        ("GPA 3.9", (3.9, "10")),   # interpreted as /10 (will need a 4.0-scale fix if needed)
    ])
    def test_recognized(self, text, expected):
        assert parse_cgpa(text) == expected

    @pytest.mark.parametrize("text", [None, "", "Distinction", "First Class", "Honours"])
    def test_unrecognized(self, text):
        assert parse_cgpa(text) == (None, None)


# ── parse_graduation_year ──────────────────────────────────────────────────

class TestParseGraduationYear:
    def test_single_year(self):
        assert parse_graduation_year("2024") == 2024

    def test_explicit_range(self):
        assert parse_graduation_year("2018 - 2022") == 2022

    def test_month_year_range(self):
        assert parse_graduation_year("Jul 2020 - May 2024") == 2024

    def test_present_projects_with_btech_default(self):
        # Started Nov 2022, BTech is 4 years -> 2026
        assert parse_graduation_year("Nov 2022 - Present", degree_enum="btech") == 2026

    def test_present_projects_with_mtech_default(self):
        # Started 2024, MTech is 2 years -> 2026
        assert parse_graduation_year("2024 - PRESENT", degree_enum="mtech") == 2026

    def test_single_month_year(self):
        assert parse_graduation_year("May 2024") == 2024

    @pytest.mark.parametrize("text", [None, "", "garbage", "TBD"])
    def test_unparseable(self, text):
        # None / empty / non-date all return None
        assert parse_graduation_year(text) is None

    def test_present_tokens_return_today(self):
        # 'ongoing', 'present', 'now', 'current' are all parse_date present-tokens
        # by design — they return today's year. This is the documented behavior.
        from src.student_derive import TODAY_YEAR
        assert parse_graduation_year("ongoing") == TODAY_YEAR
        assert parse_graduation_year("Present") == TODAY_YEAR


# ── split_skills_and_tools ─────────────────────────────────────────────────

class TestSplitSkillsAndTools:
    def test_mutual_exclusion_tools_win(self):
        # 'Cursor AI' in both lists — tools wins, skills drops it
        skills, tools = split_skills_and_tools(
            skills_raw=["PRDs", "Cursor AI", "User Stories"],
            tools_raw=["Figma", "Jira", "Cursor AI"],
        )
        assert "Cursor AI" in tools
        assert "Cursor AI" not in skills
        assert "PRDs" in skills

    def test_allowlist_moves_to_tools(self):
        # 'OpenFOAM' put in skills by LLM, but it's branded software -> tools
        skills, tools = split_skills_and_tools(
            skills_raw=["Matlab", "C++", "OpenFOAM", "CFD"],
            tools_raw=[],
        )
        assert "OpenFOAM" in tools
        assert "Matlab" in tools           # matlab is in the allowlist
        assert "C++" in skills
        assert "CFD" in skills

    def test_dedup_within_each(self):
        skills, tools = split_skills_and_tools(
            skills_raw=["Python", "python", "Python "],
            tools_raw=["Figma", "figma"],
        )
        assert len([s for s in skills if s.lower() == "python"]) == 1
        assert len([t for t in tools if t.lower() == "figma"]) == 1

    def test_empty_inputs(self):
        assert split_skills_and_tools([], []) == ([], [])

    def test_non_string_filtered(self):
        skills, tools = split_skills_and_tools(
            skills_raw=["Python", 123, None, "SQL"],
            tools_raw=["Figma", {}, "Jira"],
        )
        assert skills == ["Python", "SQL"]
        assert tools == ["Figma", "Jira"]


# ── _clean_state ───────────────────────────────────────────────────────────

class TestCleanState:
    @pytest.mark.parametrize("inp,expected", [
        ("UP", "Uttar Pradesh"),
        ("U.P.", "Uttar Pradesh"),
        ("TS", "Telangana"),
        ("Raj", "Rajasthan"),
        ("MH", "Maharashtra"),
        ("KA", "Karnataka"),
        ("TN", "Tamil Nadu"),
        ("AP", "Andhra Pradesh"),
        ("WB", "West Bengal"),
        ("DL", "Delhi"),
    ])
    def test_abbreviation_expansion(self, inp, expected):
        assert _clean_state(inp) == expected

    @pytest.mark.parametrize("inp", ["India", "Bharat", "USA", "UK", "Canada"])
    def test_country_dropped(self, inp):
        assert _clean_state(inp) is None

    def test_full_state_name_passes_through(self):
        assert _clean_state("Karnataka") == "Karnataka"

    def test_whitespace_and_punctuation_stripped(self):
        assert _clean_state("  Uttar Pradesh ,") == "Uttar Pradesh"

    @pytest.mark.parametrize("inp", [None, "", "   "])
    def test_empty(self, inp):
        assert _clean_state(inp) is None


# ── _parse_tech_stack ──────────────────────────────────────────────────────

class TestParseTechStack:
    def test_with_tech_prefix(self):
        items = _parse_tech_stack("Tech: PyTorch, Mistral-7B, PEFT, bitsandbytes, HDF5, SBERT, Whisper")
        assert items == ["PyTorch", "Mistral-7B", "PEFT", "bitsandbytes", "HDF5", "SBERT", "Whisper"]

    def test_with_tools_used_prefix(self):
        items = _parse_tech_stack("Tools used: COMSOL Multiphysics, Live-link for Matlab, MATLAB")
        assert items == ["COMSOL Multiphysics", "Live-link for Matlab", "MATLAB"]

    def test_with_technologies_prefix(self):
        items = _parse_tech_stack("Technologies: Python, R")
        assert items == ["Python", "R"]

    def test_parens_not_split(self):
        # 'PyTorch (nightly/cu128)' is one item, comma inside parens shouldn't split
        items = _parse_tech_stack("Tech: PyTorch (nightly/cu128), Mistral-7B")
        assert items == ["PyTorch (nightly/cu128)", "Mistral-7B"]

    def test_empty(self):
        assert _parse_tech_stack(None) == []
        assert _parse_tech_stack("") == []

    def test_no_prefix(self):
        # Bare comma-separated also works (no Tech: prefix)
        items = _parse_tech_stack("Python, R, SQL")
        assert items == ["Python", "R", "SQL"]


# ── _condensation_lost_metrics ─────────────────────────────────────────────

class TestCondensationLostMetrics:
    def test_no_evidence_or_description(self):
        assert _condensation_lost_metrics(None, "anything") == []
        assert _condensation_lost_metrics("evidence", None) == []

    def test_year_tokens_excluded(self):
        # 2025 / 2026 in evidence shouldn't fire as missing — dates live in
        # separate fields, not in description
        evidence = "Nov 2025 - Jan 2026: Some work happened"
        description = "Some work happened"
        assert _condensation_lost_metrics(evidence, description) == []

    def test_bare_small_numbers_excluded(self):
        # '5' and '20' on their own are too noisy to flag
        evidence = "Did 5 things in 20 ways"
        description = "Did things"
        assert _condensation_lost_metrics(evidence, description) == []

    def test_percentage_preserved(self):
        evidence = "Achieved 58% improvement"
        description = "Achieved 58% improvement"
        assert _condensation_lost_metrics(evidence, description) == []

    def test_percentage_dropped_is_flagged(self):
        evidence = "p<0.005, paired t-test"
        description = "Used t-test"
        lost = _condensation_lost_metrics(evidence, description)
        assert "0.005" in lost


# ── flatten_internships ────────────────────────────────────────────────────

class TestFlattenInternships:
    def test_empty(self):
        assert flatten_internships([]) is None
        assert flatten_internships(None) is None

    def test_single_internship(self):
        text = flatten_internships([{
            "role": "Product Intern",
            "company": "Bosscoder Academy",
            "location": "Noida",
            "start": "Jul 2025",
            "end": "Present",
            "achievements": ["Built dashboards", "Conducted user research"],
        }])
        assert "Product Intern - Bosscoder Academy" in text
        assert "Noida" in text
        assert "Jul 2025" in text
        assert "Present" in text
        assert "- Built dashboards" in text
        assert "- Conducted user research" in text

    def test_multiple_internships_blank_line_separated(self):
        text = flatten_internships([
            {"role": "A", "company": "X", "achievements": ["a1"]},
            {"role": "B", "company": "Y", "achievements": ["b1"]},
        ])
        assert text.count("\n\n") == 1

    def test_skip_entries_with_no_role_or_company(self):
        text = flatten_internships([
            {"role": "", "company": "", "achievements": ["should not appear"]},
            {"role": "Real", "company": "Real Co", "achievements": ["yes"]},
        ])
        assert "should not appear" not in text
        assert "yes" in text


# ── _attribute_link_to_project ─────────────────────────────────────────────

class TestAttributeLinkToProject:
    def test_high_overlap_attributes(self):
        # Project title and link's context_text share most significant tokens
        projects = [{"title": "MedScribe - Clinical Documentation Workstation"}]
        link = {"context_text": "MedScribe - Clinical Documentation Workstation Tech: MedGemma"}
        assert _attribute_link_to_project(link, projects) == 0

    def test_low_overlap_returns_none(self):
        # Only one fluffy word matches — should NOT attribute
        projects = [{"title": "MedScribe - Clinical Documentation Workstation"}]
        link = {"context_text": "Some unrelated paragraph mentioning documentation once"}
        assert _attribute_link_to_project(link, projects) is None

    def test_no_context_returns_none(self):
        projects = [{"title": "Whatever"}]
        link = {"context_text": ""}
        assert _attribute_link_to_project(link, projects) is None

    def test_picks_best_when_multiple_match(self):
        projects = [
            {"title": "Hybrid Dataset Cross Modal Adaptation"},
            {"title": "MedScribe Clinical Documentation Workstation"},
        ]
        # Context strongly matches project 2 only
        link = {"context_text": "MedScribe Clinical Documentation Workstation - SOAP notes"}
        assert _attribute_link_to_project(link, projects) == 1


# ── resolve_hyperlinks ─────────────────────────────────────────────────────

class TestResolveHyperlinks:
    def test_linkedin_only(self):
        hyperlinks = [{
            "url": "https://linkedin.com/in/foo",
            "anchor_text": "LinkedIn",
            "context_text": "John Doe ... LinkedIn ... GitHub",
        }]
        li, pf, projects = resolve_hyperlinks(hyperlinks, [], {})
        assert li == "https://linkedin.com/in/foo"
        assert pf is None

    def test_portfolio_anchor_wins_over_github_fallback(self):
        hyperlinks = [
            {"url": "https://github.com/foo", "anchor_text": "GitHub", "context_text": "header"},
            {"url": "https://foo.github.io/", "anchor_text": "Portfolio", "context_text": "header"},
        ]
        li, pf, _ = resolve_hyperlinks(hyperlinks, [], {})
        assert pf == "https://foo.github.io/"

    def test_github_fallback_when_no_portfolio_anchor(self):
        # Unattributed GitHub becomes the portfolio fallback
        hyperlinks = [
            {"url": "https://github.com/foo", "anchor_text": "GitHub", "context_text": "irrelevant text"},
        ]
        li, pf, _ = resolve_hyperlinks(hyperlinks, [], {})
        assert pf == "https://github.com/foo"

    def test_project_attribution(self):
        projects = [
            {"title": "MedScribe Clinical Documentation Workstation"},
            {"title": "Bawarchi Recipe Generation"},
        ]
        hyperlinks = [
            {
                "url": "https://github.com/me/medscribe",
                "anchor_text": "GitHub",
                "context_text": "MedScribe - Clinical Documentation Workstation Tech: MedGemma",
            },
            {
                "url": "https://huggingface.co/me/bawarchi",
                "anchor_text": "HF",
                "context_text": "Bawarchi Recipe Generation - Tech: Llama",
            },
        ]
        li, pf, attributed = resolve_hyperlinks(hyperlinks, projects, {})
        # Project 0 (MedScribe) should have the github link
        assert any("medscribe" in L["url"] for L in attributed[0]["links"])
        # Project 1 (Bawarchi) should have the HF link
        assert any("bawarchi" in L["url"] for L in attributed[1]["links"])

    def test_empty_hyperlinks_returns_originals(self):
        projects = [{"title": "X"}]
        li, pf, p = resolve_hyperlinks(None, projects, {})
        assert li is None and pf is None
        assert p == projects


# ── pick_primary_education ──────────────────────────────────────────────────

class TestPickPrimaryEducation:
    def test_bca_with_12th_picked_as_primary(self):
        # ashok_kumar_sah's case: BCA from a university + 12th PCM. Before the
        # BCA enum addition both ranked 0 -> pick returned None -> picker null'd
        # degree/branch/cgpa/graduationYear on a candidate who clearly has a
        # university degree.
        entries = [
            {"raw_degree": "BCA", "graduation_year_text": "2016-2019"},
            {"raw_degree": "12th PCM", "graduation_year_text": "2013-2015"},
        ]
        top = pick_primary_education(entries)
        assert top is not None
        assert top["raw_degree"] == "BCA"

    def test_btech_beats_12th(self):
        entries = [
            {"raw_degree": "12th CBSE", "graduation_year_text": "2018"},
            {"raw_degree": "B.Tech CSE", "graduation_year_text": "2022"},
        ]
        top = pick_primary_education(entries)
        assert top["raw_degree"] == "B.Tech CSE"

    def test_mtech_beats_btech(self):
        # PG should outrank UG even if UG is more recent
        entries = [
            {"raw_degree": "B.Tech CSE", "graduation_year_text": "2020"},
            {"raw_degree": "M.Tech CSE", "graduation_year_text": "2022"},
        ]
        top = pick_primary_education(entries)
        assert top["raw_degree"] == "M.Tech CSE"

    def test_only_school_entries_returns_none(self):
        # No real degree -> None so caller knows the candidate has no
        # recognized higher education entry
        entries = [
            {"raw_degree": "12th CBSE", "graduation_year_text": "2018"},
            {"raw_degree": "10th SSC", "graduation_year_text": "2016"},
        ]
        assert pick_primary_education(entries) is None

    def test_empty_returns_none(self):
        assert pick_primary_education([]) is None
        assert pick_primary_education(None) is None


# ── _label_to_url ───────────────────────────────────────────────────────────

class TestLabelToUrl:
    def test_full_linkedin_url_passes(self):
        # ashok's case: identity LLM captured the URL printed in the resume
        # body as linkedin_label. Promote it when PDF annotations missed it.
        url = "https://www.linkedin.com/in/aksahprogrammer/"
        assert _label_to_url(url, "linkedin") == url

    def test_bare_linkedin_domain_gets_scheme(self):
        # Resume prints 'linkedin.com/in/foo' without https:// prefix
        result = _label_to_url("linkedin.com/in/foo", "linkedin")
        assert result == "https://linkedin.com/in/foo"

    def test_github_as_portfolio_passes(self):
        url = "https://github.com/saivighnesh2190"
        assert _label_to_url(url, "portfolio") == url

    def test_github_io_as_portfolio_passes(self):
        # Vighnesh case: github.io is portfolio, not project repo
        url = "https://saivighnesh2190.github.io/portfolio/"
        assert _label_to_url(url, "portfolio") == url

    def test_plain_anchor_text_rejected(self):
        # Bare 'LinkedIn' / 'GitHub' / 'Portfolio' is not a URL — don't promote
        assert _label_to_url("LinkedIn", "linkedin") is None
        assert _label_to_url("GitHub", "portfolio") is None
        assert _label_to_url("Portfolio", "portfolio") is None

    def test_wrong_kind_rejected(self):
        # GitHub URL passed as linkedin kind should not match
        assert _label_to_url("https://github.com/foo", "linkedin") is None
        # LinkedIn URL passed as portfolio should not match
        assert _label_to_url("https://linkedin.com/in/foo", "portfolio") is None

    def test_empty_and_none(self):
        assert _label_to_url(None, "linkedin") is None
        assert _label_to_url("", "linkedin") is None
        assert _label_to_url("   ", "linkedin") is None

    def test_non_string_returns_none(self):
        assert _label_to_url(123, "linkedin") is None  # type: ignore[arg-type]
