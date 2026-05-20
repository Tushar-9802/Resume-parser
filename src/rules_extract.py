"""Rules-only resume extractor v2 — no LLM.

Improvements over v1:
  1. Section detection — degree/branch/year/cgpa scoped to education section;
     skills scoped to skills section.
  2. Geo strict-literal — city only when in explicit candidate address line,
     never inferred from institute/employer.
  3. country no-hardcode — only emit if "India" literally in text.
  4. Name fallback — reject section headings/labels; fallback to email-local-part.
  5. Branch — word-boundary + non-engineering-degree gate (BCA/MCA/MBA → null).
  6. Skills — word boundaries, longest-token-first, scoped to skills section,
     alias dedup, expanded mech-eng dictionary.
  7. Education most-recent picker — prefers row with "Present"/"Pursuing" marker.
  8. CGPA expanded to CPI/SGPA/GPA.
  9. Phone hyphen tolerance.
 10. LinkedIn tracking-param strip widened (licu, jobid, lipi, refId, trk, utm_*).
 11. otherProfileUrls deny-list (drive.google.com, docs.google.com, dropbox.com,
     pastebin, etc.).
 12. OCR stub (pytesseract optional; warns if missing).
"""
from __future__ import annotations
import re, json, sys
from pathlib import Path
from datetime import date
from typing import Any
import fitz  # pymupdf

# Canonical-DB resolver (optional — fall through to None if not installed).
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'canonical_db' / 'src'))
    from resolver import colleges as _colleges_resolver, find_college_mentions  # type: ignore
    from resolver import skills as _skills_resolver  # type: ignore
    _CANONICAL_DB_AVAILABLE = True
except Exception:
    _CANONICAL_DB_AVAILABLE = False
    _colleges_resolver = None
    _skills_resolver = None
    find_college_mentions = None


def _canonicalize_skills(skills_dict: dict[str, list[str]]) -> list[dict]:
    """Flatten the 9-bucket skill dict into a list of {raw, canonical_id, bucket}
    entries — recruiter-portal/JD-matching-friendly shape.

    canonical_id is the ESCO concept URI when the skill resolves; null otherwise.
    Most tech tokens (React.js, MongoDB, Kubernetes) won't resolve in ESCO — that's
    expected; the rules-v2 bucket is preserved either way. JD matching joins on
    canonical_id where present, falls back to (lower-cased) raw on miss."""
    if not _CANONICAL_DB_AVAILABLE or _skills_resolver is None:
        return [
            {'raw': s, 'canonical_id': None, 'bucket': bucket}
            for bucket, items in skills_dict.items() for s in (items or [])
        ]
    resolver = _skills_resolver()
    out: list[dict] = []
    for bucket, items in skills_dict.items():
        for s in (items or []):
            row = resolver.resolve(s, allow_fuzzy=False)  # literal match only — fuzzy too noisy
            out.append({
                'raw': s,
                'canonical_id': (row.get('concept_uri') if row else None),
                'bucket': bucket,
            })
    return out

# Lazy-import OCR (slow startup) only when needed.
_OCR_ENGINE = None
def _get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            _OCR_ENGINE = RapidOCR()
        except ImportError:
            _OCR_ENGINE = False  # mark as unavailable
    return _OCR_ENGINE


def _ocr_pdf(doc: fitz.Document) -> str:
    """Render each page to image and OCR it. Returns concatenated text or '' if OCR unavailable."""
    engine = _get_ocr_engine()
    if not engine:
        return ''
    parts: list[str] = []
    for page in doc:
        # Render at 2x scale for better OCR accuracy.
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes('png')
        try:
            result, _ = engine(img_bytes)
        except Exception:
            continue
        if not result:
            continue
        # result is list of [box, text, confidence] — extract text.
        page_lines = [item[1] for item in result if len(item) >= 2 and item[1]]
        parts.append('\n'.join(page_lines))
    return '\n\n'.join(parts)


TODAY = date(2026, 5, 17)


# ---------- patterns ----------

PHONE_RE = re.compile(r'(?:\+?91[\s\-]?)?[6-9]\d{2}[\s\-]?\d{3}[\s\-]?\d{4}\b')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
CGPA_RE  = re.compile(r'\b(?:CGPA|CPI|SGPA|GPA)\s*[:\-]?\s*([\d.]+)\s*(?:/\s*(10|4|5))?', re.I)
YEAR_RANGE_RE = re.compile(
    r'(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|Present|present|Pursuing|pursuing|Current|current|Now|now)',
)
YEAR_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')

# Degree literals — longer first to avoid partial matches. CASE-INSENSITIVE so "B.Tech" matches.
DEGREE_TOKENS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\bM\.?\s?TECH\b', re.I), 'MTECH'),
    (re.compile(r'\bB\.?\s?TECH\b', re.I), 'BTECH'),
    (re.compile(r'\bBACHELOR\s+OF\s+(?:COMPUTER\s+APPLICATIONS?|COMPUTER\s+APPLICATION)\b', re.I), 'BCA'),
    (re.compile(r'\bMASTER\s+OF\s+COMPUTER\s+APPLICATIONS?\b', re.I), 'MCA'),
    (re.compile(r'\bBACHELOR\s+OF\s+SCIENCE\b', re.I), 'BSC'),
    (re.compile(r'\bBACHELOR\s+OF\s+COMMERCE\b', re.I), 'BCOM'),
    (re.compile(r'\bMASTER\s+OF\s+SCIENCE\b', re.I), 'MSC'),
    (re.compile(r'\bMASTER\s+OF\s+BUSINESS\s+ADMINISTRATION\b', re.I), 'MBA'),
    (re.compile(r'\bMASTER\s+OF\s+ARTS\b', re.I), 'MA'),
    (re.compile(r'\bBACHELOR\s+OF\s+ENGINEERING\b', re.I), 'BE'),
    (re.compile(r'\bPOST\s*GRADUATE\s+DIPLOMA\s+IN\s+MANAGEMENT\b|\bPGDM\b', re.I), 'PGDM'),
    (re.compile(r'\bPOST\s*GRADUATE\s+DIPLOMA\s+IN\s+FINANCIAL\s+MANAGEMENT\b|\bPGDFM\b', re.I), 'PGDFM'),
    (re.compile(r'\bBCA\b'), 'BCA'),
    (re.compile(r'\bMCA\b'), 'MCA'),
    (re.compile(r'\bBSC\b|\bB\.?\s?SC\b'), 'BSC'),
    (re.compile(r'\bMSC\b|\bM\.?\s?SC\b'), 'MSC'),
    (re.compile(r'\bBCOM\b|\bB\.?\s?COM\b'), 'BCOM'),
    (re.compile(r'\bMCOM\b|\bM\.?\s?COM\b'), 'MCOM'),
    (re.compile(r'\bMBA\b'), 'MBA'),
    (re.compile(r'\bPHD\b|\bPH\.?\s?D\b', re.I), 'PHD'),
    (re.compile(r'\bB\.?\s?E\b(?:\.|\b)'), 'BE'),
    (re.compile(r'\bM\.?\s?E\b(?:\.|\b)'), 'ME'),  # last — high false-positive risk
]
NON_ENGINEERING_DEGREES = {'BCA', 'MCA', 'BCOM', 'MCOM', 'MBA', 'MA', 'BSC', 'MSC', 'PGDM', 'PGDFM', 'PHD'}

BRANCH_TOKENS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\bCOMPUTER\s+SCIENCE\s+(?:AND|&)\s+ENGINEERING\b', re.I), 'COMPUTER SCIENCE & ENGINEERING'),
    (re.compile(r'\bCOMPUTER\s+SCIENCE\s+ENGINEERING\b', re.I), 'COMPUTER SCIENCE & ENGINEERING'),
    (re.compile(r'\bCOMPUTER\s+SCIENCE\b', re.I), 'COMPUTER SCIENCE'),
    (re.compile(r'\bARTIFICIAL\s+INTELLIGENCE\s+(?:AND|&)\s+DATA\s+SCIENCE\b', re.I), 'ARTIFICIAL INTELLIGENCE AND DATA SCIENCE'),
    (re.compile(r'\bAI\s*(?:&|AND)\s*DS\b', re.I), 'ARTIFICIAL INTELLIGENCE AND DATA SCIENCE'),
    (re.compile(r'\bDATA\s+SCIENCE\b', re.I), 'DATA SCIENCE'),
    (re.compile(r'\bELECTRONICS?\s+(?:AND|&)\s+COMMUNICATION\s+ENGINEERING\b', re.I), 'ELECTRONICS AND COMMUNICATION ENGINEERING'),
    (re.compile(r'\bELECTRONICS?\s+(?:AND|&)\s+COMMUNICATION\b', re.I), 'ELECTRONICS AND COMMUNICATION'),
    (re.compile(r'\bELECTRICAL\s+(?:AND|&)\s+ELECTRONICS\s+ENGINEERING\b', re.I), 'ELECTRICAL AND ELECTRONICS ENGINEERING'),
    (re.compile(r'\bELECTRICAL\s+(?:AND|&)\s+ELECTRONICS\b', re.I), 'ELECTRICAL AND ELECTRONICS'),
    (re.compile(r'\bELECTRICAL\s+ENGINEERING\b', re.I), 'ELECTRICAL ENGINEERING'),
    (re.compile(r'\bMECHANICAL\s+ENGINEERING\b', re.I), 'MECHANICAL ENGINEERING'),
    (re.compile(r'\bCIVIL\s+ENGINEERING\b', re.I), 'CIVIL ENGINEERING'),
    (re.compile(r'\bMECHATRONICS\b', re.I), 'MECHATRONICS'),
    (re.compile(r'\bBIOMEDICAL\b', re.I), 'BIOMEDICAL'),
    (re.compile(r'\bAEROSPACE\b', re.I), 'AEROSPACE'),
    (re.compile(r'\bPETROLEUM\s+ENGINEERING\b', re.I), 'PETROLEUM ENGINEERING'),
    (re.compile(r'\bINDUSTRIAL\s+ENGINEERING\b', re.I), 'INDUSTRIAL ENGINEERING'),
    (re.compile(r'\bCHEMICAL\s+ENGINEERING\b', re.I), 'CHEMICAL ENGINEERING'),
    (re.compile(r'\bINFORMATION\s+TECHNOLOGY\b', re.I), 'IT'),
    (re.compile(r'(?<!I)\bI\.\s?T\.?(?=\s|$)'), 'IT'),  # I.T. with negative lookbehind to avoid I.I.T
    (re.compile(r'\bCSE\b'), 'CSE'),
    (re.compile(r'\bECE\b'), 'ECE'),
    (re.compile(r'\bEEE\b'), 'EEE'),
]

# Section heading detection — title is on its own line (often), case-insensitive.
SECTION_HEADING_PATTERNS = {
    'education':       re.compile(r'^\s*(EDUCATION|ACADEMIC|EDUCATIONAL\s+QUALIFICATION|EDUCATIONAL\s+BACKGROUND|EDUCATION\s+DETAILS|SCHOLASTIC\s+RECORD|学\s*歴)\s*:?\s*$', re.I),
    'experience':      re.compile(r'^\s*(EXPERIENCE|WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EMPLOYMENT|EMPLOYMENT\s+HISTORY|CAREER|職\s*歴)\s*:?\s*$', re.I),
    'skills':          re.compile(r'^\s*(SKILLS|TECHNICAL\s+SKILLS|TECHNICAL\s+PROFICIENCY|TECHNICAL\s+EXPERTISE|PROGRAMMING\s+SKILLS|KEY\s+SKILLS|CORE\s+COMPETENCIES|SOFTWARE\s+SKILLS|TECHNOLOGIES|TECH\s+STACK|TOOLS|MECHANICAL\s+SOFTWARE\s+SKILLS|SOFTWARE\s+PROFICIENCY|TOOLS\s+AND\s+TECHNOLOGIES)\s*:?\s*$', re.I),
    'projects':        re.compile(r'^\s*(PROJECTS|PROJECT\s+EXPERIENCE|ACADEMIC\s+PROJECTS|KEY\s+PROJECTS)\s*:?\s*$', re.I),
    'certifications':  re.compile(r'^\s*(CERTIFICATIONS|CERTIFICATES|COURSES|TRAININGS|PROFESSIONAL\s+CERTIFICATIONS)\s*:?\s*$', re.I),
    'achievements':    re.compile(r'^\s*(ACHIEVEMENTS|AWARDS|ACCOMPLISHMENTS|HONORS|HONOURS)\s*:?\s*$', re.I),
    'languages':       re.compile(r'^\s*(LANGUAGES|LANGUAGES\s+KNOWN)\s*:?\s*$', re.I),
    'summary':         re.compile(r'^\s*(SUMMARY|PROFESSIONAL\s+SUMMARY|PROFILE|OBJECTIVE|CAREER\s+OBJECTIVE|ABOUT\s+ME)\s*:?\s*$', re.I),
    'interests':       re.compile(r'^\s*(INTERESTS|HOBBIES|EXTRACURRICULAR(?:S)?)\s*:?\s*$', re.I),
    'contact':         re.compile(r'^\s*(CONTACT|PERSONAL\s+DETAILS|PERSONAL\s+INFORMATION)\s*:?\s*$', re.I),
    'positions':       re.compile(r'^\s*(POSITIONS\s+OF\s+RESPONSIBILITY|POSITIONS)\s*:?\s*$', re.I),
}

# Strings the name extractor must reject (section headings + common skill labels + generic words).
NAME_REJECT_TOKENS = {
    'education', 'experience', 'skills', 'projects', 'certifications', 'achievements',
    'languages', 'summary', 'profile', 'objective', 'contact', 'interests', 'hobbies',
    'professional summary', 'scholastic record', 'career objective', 'personal details',
    'ms office', 'ms word', 'ms excel', 'powerpoint', 'tableau',
    'hindi', 'english', 'name', 'address', 'phone', 'email', 'curriculum vitae', 'resume',
    'work experience', 'employment', 'professional experience', 'about me',
    'date of birth', 'gender', 'nationality',
}

# Indian state names (literal-match only).
INDIAN_STATES = {
    'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh', 'goa', 'gujarat',
    'haryana', 'himachal pradesh', 'jharkhand', 'karnataka', 'kerala', 'madhya pradesh',
    'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland', 'odisha', 'punjab',
    'rajasthan', 'sikkim', 'tamil nadu', 'tamilnadu', 'telangana', 'tripura', 'uttar pradesh',
    'uttarakhand', 'west bengal',
    'delhi', 'chandigarh', 'puducherry', 'pondicherry', 'jammu and kashmir', 'ladakh',
    'andaman and nicobar islands', 'dadra and nagar haveli', 'daman and diu', 'lakshadweep',
}
STATE_ABBREVIATIONS = {
    'ap': 'Andhra Pradesh', 'ar': 'Arunachal Pradesh', 'br': 'Bihar', 'ct': 'Chhattisgarh',
    'ga': 'Goa', 'gj': 'Gujarat', 'hr': 'Haryana', 'hp': 'Himachal Pradesh', 'jh': 'Jharkhand',
    'ka': 'Karnataka', 'kl': 'Kerala', 'mp': 'Madhya Pradesh', 'mh': 'Maharashtra',
    'mn': 'Manipur', 'ml': 'Meghalaya', 'mz': 'Mizoram', 'nl': 'Nagaland', 'or': 'Odisha',
    'pb': 'Punjab', 'rj': 'Rajasthan', 'sk': 'Sikkim', 'tn': 'Tamil Nadu', 'tg': 'Telangana',
    'ts': 'Telangana', 'tr': 'Tripura', 'up': 'Uttar Pradesh', 'uk': 'Uttarakhand',
    'wb': 'West Bengal', 'dl': 'Delhi',
    'raj': 'Rajasthan', 'mah': 'Maharashtra',
}

# Skills bucketing — sorted longest-first per bucket at runtime.
SKILL_BUCKETS = {
    'programmingLanguages': {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'go', 'golang', 'rust',
        'kotlin', 'swift', 'r', 'scala', 'ruby', 'php', 'perl', 'matlab', 'dart', 'sql', 'shell',
        'bash', 'apex', 'soql', 'objective-c', 'verilog', 'vhdl',
    },
    'frameworks': {
        'react', 'react.js', 'reactjs', 'vue', 'vue.js', 'angular', 'angularjs', 'svelte',
        'next.js', 'nuxt', 'django', 'flask', 'fastapi', 'spring', 'spring boot', 'express',
        'express.js', 'rails', 'laravel', '.net', 'asp.net', 'streamlit', 'gradio', 'tailwind',
        'bootstrap', 'jquery', 'node.js', 'nodejs', 'lightning web components', 'lwc', 'aura',
        'visualforce',
    },
    'libraries': {
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'scikit-learn', 'sklearn',
        'tensorflow', 'pytorch', 'keras', 'jax', 'transformers', 'huggingface', 'opencv',
        'nltk', 'spacy', 'langchain', 'llamaindex', 'lodash', 'redux', 'rxjs',
        'sentence-transformers', 'sbert', 'ggplot2', 'rstudio',
    },
    'databases': {
        'mysql', 'postgresql', 'postgres', 'sqlite', 'mongodb', 'redis', 'cassandra', 'dynamodb',
        'firestore', 'firebase', 'elasticsearch', 'neo4j', 'oracle', 'sql server', 'mssql',
        'mariadb', 'snowflake', 'bigquery',
    },
    'cloudPlatforms': {
        'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'google cloud platform',
        'heroku', 'vercel', 'netlify', 'digitalocean', 'cloudflare', 'oracle cloud',
    },
    'devopsTools': {
        'docker', 'kubernetes', 'k8s', 'jenkins', 'gitlab ci', 'github actions', 'circleci',
        'ansible', 'terraform', 'puppet', 'chef', 'prometheus', 'grafana', 'kafka', 'airflow',
        'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'peft', 'bitsandbytes',
    },
    'aiTools': {
        'openai', 'gpt', 'gpt-4', 'gpt-4o', 'claude', 'gemini', 'gemma', 'medgemma', 'llama',
        'mistral', 'ollama', 'rag', 'lora', 'qlora', 'genkit', 'langflow', 'voiceflow', 'n8n',
        'make.com', 'relevance ai', 'whisper', 'yt-dlp',
    },
    'softwareApplications': {
        'excel', 'word', 'powerpoint', 'ms office', 'microsoft office', 'photoshop',
        'illustrator', 'figma', 'canva', 'tableau', 'power bi', 'looker', 'autocad', 'fusion 360',
        'solidworks', 'wordpress', 'magento', 'jupyter notebook', 'vscode', 'visual studio code',
        'eclipse', 'ansys', 'ansys fluent', 'ansys workbench', 'siemens nx', 'comsol multiphysics',
        'comsol', 'latex', 'catia', 'creo', 'abaqus', 'openfoam', 'labview', 'sap erp', 'sap',
        'geogebra', 'qgis', 'ls-dyna', 'rstudio', 'festo fluidsim', 'webflow', 'notion',
        'cursor ai', 'google analytics', 'mixpanel', 'ms excel', 'ms word', 'ms powerpoint',
    },
}
# Alias dedup map: which item subsumes which. First-seen wins.
SKILL_ALIASES = {
    'ms excel': 'excel', 'excel': 'excel',
    'ms word': 'word', 'word': 'word',
    'ms powerpoint': 'powerpoint', 'powerpoint': 'powerpoint',
    'microsoft office': 'ms office', 'ms office': 'ms office',
    'visual studio code': 'vscode', 'vscode': 'vscode', 'vs code': 'vscode',
    'reactjs': 'react.js', 'react.js': 'react.js', 'react': 'react.js',
    'nodejs': 'node.js', 'node.js': 'node.js',
    'sklearn': 'scikit-learn', 'scikit-learn': 'scikit-learn',
    'postgres': 'postgresql', 'postgresql': 'postgresql',
    'k8s': 'kubernetes', 'kubernetes': 'kubernetes',
    'amazon web services': 'aws', 'aws': 'aws',
    'google cloud': 'gcp', 'google cloud platform': 'gcp', 'gcp': 'gcp',
    'sbert': 'sentence-transformers', 'sentence-transformers': 'sentence-transformers',
    'comsol multiphysics': 'comsol', 'comsol': 'comsol',
    'ansys workbench': 'ansys', 'ansys fluent': 'ansys', 'ansys': 'ansys',
    'angularjs': 'angular', 'angular': 'angular',
    'vue.js': 'vue', 'vue': 'vue',
    'lwc': 'lightning web components', 'lightning web components': 'lightning web components',
}
DISPLAY = {
    'python': 'Python', 'java': 'Java', 'javascript': 'JavaScript', 'typescript': 'TypeScript',
    'c': 'C', 'c++': 'C++', 'c#': 'C#', 'sql': 'SQL', 'go': 'Go', 'golang': 'Go', 'php': 'PHP',
    'html': 'HTML', 'css': 'CSS', 'react.js': 'React.js', 'node.js': 'Node.js',
    'next.js': 'Next.js', 'aws': 'AWS', 'gcp': 'GCP', 'azure': 'Azure',
    'mysql': 'MySQL', 'postgresql': 'PostgreSQL', 'mongodb': 'MongoDB',
    'tensorflow': 'TensorFlow', 'pytorch': 'PyTorch', 'numpy': 'NumPy', 'pandas': 'Pandas',
    'fastapi': 'FastAPI', 'django': 'Django', 'flask': 'Flask',
    'docker': 'Docker', 'kubernetes': 'Kubernetes', 'jenkins': 'Jenkins',
    'git': 'Git', 'github': 'GitHub', 'gitlab': 'GitLab',
    'excel': 'MS Excel', 'word': 'MS Word', 'powerpoint': 'MS PowerPoint', 'ms office': 'MS Office',
    'tableau': 'Tableau', 'figma': 'Figma', 'canva': 'Canva',
    'rag': 'RAG', 'llm': 'LLM', 'matlab': 'MATLAB',
    'gpt': 'GPT', 'gpt-4': 'GPT-4', 'gpt-4o': 'GPT-4o',
    'sap': 'SAP', 'sap erp': 'SAP ERP', 'ms sql': 'MS SQL',
    'ansys': 'ANSYS', 'siemens nx': 'Siemens NX', 'comsol': 'COMSOL Multiphysics',
    'catia': 'CATIA', 'creo': 'CREO', 'abaqus': 'Abaqus', 'openfoam': 'OpenFOAM',
    'latex': 'LaTeX', 'autocad': 'AutoCAD', 'solidworks': 'SolidWorks',
    'jupyter notebook': 'Jupyter Notebook', 'vscode': 'VS Code', 'r': 'R',
    'lora': 'LoRA', 'qlora': 'QLoRA', 'mistral': 'Mistral', 'llama': 'Llama',
    'gemma': 'Gemma', 'medgemma': 'MedGemma', 'ollama': 'Ollama',
    'sentence-transformers': 'Sentence-Transformers', 'huggingface': 'HuggingFace',
    'whisper': 'Whisper', 'yt-dlp': 'yt-dlp', 'peft': 'PEFT', 'bitsandbytes': 'bitsandbytes',
    'rstudio': 'RStudio', 'ggplot2': 'ggplot2',
}

# Deny-list for otherProfileUrls — file-share / artifact hosts, not profile URLs.
URL_DENY_HOSTS = {
    'drive.google.com', 'docs.google.com', 'dropbox.com', 'onedrive.live.com',
    'pastebin.com', 'gist.github.com', 'youtu.be', 'youtube.com', 'imgur.com',
    'tinyurl.com', 'bit.ly', 'goo.gl',
}
# Tracking params to strip from any URL.
TRACKING_PARAMS = re.compile(
    r'[?&](?:jobid|lipi|licu|refId|trk|sk|sub_confirmation|utm_[a-z_]+)=[^&]*',
    re.I,
)


# ---------- PDF helpers ----------

def open_pdf(pdf_path: Path) -> fitz.Document:
    return fitz.open(pdf_path)


def extract_text_and_links(doc: fitz.Document) -> tuple[str, list[tuple[str, str]]]:
    text_parts: list[str] = []
    links: list[tuple[str, str]] = []
    for page in doc:
        text_parts.append(page.get_text("text"))
        for ln in page.get_links():
            uri = ln.get("uri")
            if not uri:
                continue
            rect = ln.get("from")
            anchor = page.get_textbox(rect).strip().replace("\n", " ") if rect else ""
            links.append((anchor, uri))
    return "\n".join(text_parts), links


# ---------- section detection ----------

def detect_sections(text: str) -> dict[str, tuple[int, int]]:
    """Return {section_name: (start_char_idx, end_char_idx)} for each detected heading.
    A section spans from its heading line to the next heading line (or EOF)."""
    lines = text.splitlines()
    offsets: list[int] = []
    cum = 0
    for line in lines:
        offsets.append(cum)
        cum += len(line) + 1  # +1 for newline
    offsets.append(cum)

    # Find heading lines.
    heading_hits: list[tuple[int, str]] = []  # (line_idx, section_name)
    for i, line in enumerate(lines):
        line_strip = line.strip()
        if not line_strip or len(line_strip) > 60:
            continue
        for name, pat in SECTION_HEADING_PATTERNS.items():
            if pat.match(line_strip):
                heading_hits.append((i, name))
                break

    sections: dict[str, tuple[int, int]] = {}
    for j, (line_i, name) in enumerate(heading_hits):
        start = offsets[line_i + 1] if line_i + 1 < len(offsets) else offsets[-1]
        if j + 1 < len(heading_hits):
            end = offsets[heading_hits[j + 1][0]]
        else:
            end = offsets[-1]
        # If same section appears twice (rare), prefer first occurrence.
        if name not in sections:
            sections[name] = (start, end)
    return sections


def section_text(text: str, sections: dict[str, tuple[int, int]], name: str) -> str:
    span = sections.get(name)
    if not span:
        return ''
    return text[span[0]:span[1]]


# ---------- field extractors ----------

def extract_phone(text: str) -> str | None:
    # Look in first 1000 chars (top-of-resume header) to avoid picking up reference phones.
    head = text[:1500]
    for m in PHONE_RE.finditer(head):
        digits = re.sub(r'\D', '', m.group())
        if len(digits) == 12 and digits.startswith('91'):
            digits = digits[2:]
        if len(digits) == 10 and digits[0] in '6789':
            return f"+91-{digits}"
    # Fallback: scan full text.
    for m in PHONE_RE.finditer(text):
        digits = re.sub(r'\D', '', m.group())
        if len(digits) == 12 and digits.startswith('91'):
            digits = digits[2:]
        if len(digits) == 10 and digits[0] in '6789':
            return f"+91-{digits}"
    return None


def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    if not m:
        return None
    # Some PDFs split emails across lines: "user@cit\nchennai.net". Try to glue.
    # Look at the next line if email ends suspiciously short.
    # For now: just return what we got.
    return m.group()


def name_from_email(email: str | None) -> str | None:
    if not email:
        return None
    local = email.split('@')[0]
    # Strip trailing digits (yadavdivyansh128 → yadavdivyansh).
    local = re.sub(r'\d+$', '', local)
    # Common separators: dot, underscore, dash.
    parts = [p for p in re.split(r'[._\-]+', local) if p]
    if not parts:
        return None
    # Filter out generic terms.
    parts = [p for p in parts if p.lower() not in {'resume', 'cv', 'cognavi', 'official', 'work', 'gmail'}]
    if not parts or all(len(p) <= 2 for p in parts):
        return None
    return ' '.join(p.capitalize() for p in parts)


def _looks_like_name(ln: str) -> bool:
    """Quick predicate: line is a plausible name candidate."""
    if not ln or len(ln) > 50:
        return False
    if any(x in ln.lower() for x in ('curriculum vitae', 'resume', 'profile')):
        return False
    if '@' in ln or 'http' in ln.lower() or re.search(r'\d{4,}', ln):
        return False
    if ln.lower().rstrip(':').strip() in NAME_REJECT_TOKENS:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{1,50}", ln):
        return False
    words = ln.split()
    if not (1 <= len(words) <= 5):
        return False
    if any(w.lower() in NAME_REJECT_TOKENS for w in words):
        return False
    if len(words) == 1 and ln.lower() in {'hindi', 'english', 'tamil', 'telugu', 'kannada',
                                            'malayalam', 'marathi', 'gujarati', 'punjabi',
                                            'bengali', 'urdu'}:
        return False
    return True


def extract_name(text: str, email: str | None) -> str | None:
    """First plausible name in top 15 lines. If found name is one word and the NEXT line
    is also a plausible single name word, glue them together (multi-line name headers).
    Fallback: derive from email-local-part."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines[:15]):
        if not _looks_like_name(ln):
            continue
        # If this is a one-word "name" and the next line is also a single-word plausible name,
        # glue ("Nabneet Sourav" + "Mahapatra"; "Kapil" + "Singh" + "Chauhan").
        accumulated = [ln]
        j = i + 1
        while j < min(len(lines), i + 4):
            nxt = lines[j].strip()
            if not _looks_like_name(nxt):
                break
            # Only glue if both this and next look like name fragments (1-3 words each).
            if len(nxt.split()) > 3:
                break
            accumulated.append(nxt)
            if sum(len(a.split()) for a in accumulated) >= 4:
                break
            j += 1
        # Glue if the first line was 1-2 words AND the email-local-part hints at more.
        if len(ln.split()) <= 2 and email:
            local_lower = email.split('@')[0].lower()
            extra = []
            for a in accumulated[1:]:
                if a.lower() in local_lower or a.lower().replace(' ', '') in local_lower:
                    extra.append(a)
            if extra:
                return ' '.join([ln] + extra)
        return ln
    return name_from_email(email)


INSTITUTE_KEYWORDS = re.compile(
    r'\b(institute|university|college|school|i\.?i\.?t|nit|iim|iisc|iiit|polytechnic|academy|faculty|department)\b',
    re.I,
)


def _header_candidate_line(text: str) -> str | None:
    """Find a line in the top of the resume that looks like a candidate's stated location
    (city + state/country, no institute/employer keyword)."""
    # Scan first 20 non-empty lines.
    head_lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:20]
    for ln in head_lines:
        # Skip if too long (probably a sentence/bullet).
        if len(ln) > 100:
            continue
        # Skip if contains an institute/employer keyword.
        if INSTITUTE_KEYWORDS.search(ln):
            continue
        # Skip lines with email/phone/URL (those are contact lines, not location).
        if '@' in ln or 'http' in ln.lower():
            continue
        ll = ln.lower()
        # Must mention a known Indian city/state/country AND have a comma (city, state form).
        if 'india' not in ll and not any(s in ll for s in INDIAN_STATES):
            continue
        if ',' not in ln:
            continue
        # Reject if it looks like a sentence (has verbs/punctuation past commas).
        if re.search(r'\b(?:and|with|for|to|in)\b', ll):
            continue
        return ln
    return None


def _address_block(text: str) -> str | None:
    """Look for an explicit candidate address block in top of doc.
    Patterns (in priority order):
      1. Explicit 'Address:' line
      2. Line containing 6-digit PIN code in first 1500 chars
      3. Header-line city pattern: 'City, State' or 'City, India' near top, no institute keyword."""
    head = text[:1500]
    m = re.search(r'(?im)^\s*Address\s*[:\-]\s*(.+)$', head)
    if m:
        return m.group(1).strip()
    m = re.search(r'(?m)^([^\n]*\b\d{6}\b[^\n]*)$', head)
    if m:
        return m.group(1).strip()
    # Header-line candidate.
    return _header_candidate_line(text)


def extract_geo(text: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (address, city, state, country). Strict-literal — only emit when explicitly
    visible in candidate address block. Never inferred from institute/employer locations."""
    addr = _address_block(text)
    if not addr:
        return None, None, None, None
    # If address came from header-line heuristic, address itself may be brief — return None for
    # the full-address field but still emit city/state/country.
    addr_l = addr.lower()
    is_full_address = bool(re.search(r'\b\d{6}\b', addr)) or addr.lower().startswith('address')

    # UTs that are also cities — preserve as city, don't set state.
    UT_CITIES = {'delhi', 'chandigarh', 'puducherry', 'pondicherry'}

    state: str | None = None
    for s in INDIAN_STATES:
        if s in UT_CITIES:
            continue  # don't set state for UT-cities; they go in city slot
        if re.search(rf'\b{re.escape(s)}\b', addr_l):
            state = s.title() if s != 'tamilnadu' else 'Tamil Nadu'
            break
    if not state:
        for abbr, full in STATE_ABBREVIATIONS.items():
            if re.search(rf'[,\s]{abbr}\b', addr_l) or addr_l.endswith(' ' + abbr):
                state = full
                break

    country = 'India' if re.search(r'\bindia\b', addr_l) else None

    parts = [p.strip() for p in re.split(r'[,\n]', addr) if p.strip()]
    city: str | None = None
    if parts:
        filtered = []
        for p in parts:
            pl = p.lower()
            if re.fullmatch(r'\d{6}', p) or re.fullmatch(r'-?\s*\d{6}', p):
                continue
            if pl == 'india':
                continue
            if pl in STATE_ABBREVIATIONS:
                continue
            # Drop non-UT state names. Keep UT-cities (Delhi, Chandigarh, etc.) — they're cities.
            if pl in INDIAN_STATES and pl not in UT_CITIES:
                continue
            # Strip leading non-letter chars (Japanese 〒 postal mark, etc.).
            p_clean = re.sub(r'^[^A-Za-z]+', '', p).strip()
            if p_clean:
                filtered.append(p_clean)
        if filtered:
            cand = filtered[0] if not is_full_address else filtered[-1]
            # Strip PIN suffix ("Goa 403004" → "Goa"); strip dash/paren tail.
            cand = re.split(r'[-(]', cand)[0]
            cand = re.sub(r'\s*\d{4,6}\s*$', '', cand).strip()
            # Drop everything after a state token that snuck in.
            cand_norm = cand.title()
            # Reject too short or non-letter starts.
            if not (cand and 3 <= len(cand) <= 40 and re.match(r'[A-Za-z]', cand)):
                pass
            elif cand_norm.lower() in INDIAN_STATES and cand_norm.lower() not in UT_CITIES:
                pass  # state name leaked
            else:
                city = cand_norm
    return (addr if is_full_address else None), city, state, country


def extract_urls(links: list[tuple[str, str]], name: str | None, email: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        'linkedinUrl': None, 'portfolioUrl': None, 'githubUrl': None,
        'leetcodeUrl': None, 'codechefUrl': None, 'codeforcesUrl': None,
        'huggingfaceUrl': None, 'kaggleUrl': None, 'otherProfileUrls': [],
    }
    name_tokens: set[str] = set()
    if name:
        name_tokens |= {w.lower() for w in re.findall(r'[A-Za-z]+', name) if len(w) > 2}
    if email:
        local = email.split('@')[0].lower()
        name_tokens |= {t for t in re.split(r'[._\-+]', local) if len(t) > 2}

    def clean(uri: str) -> str:
        u = TRACKING_PARAMS.sub('', uri)
        # Strip trailing ?, & or empty query.
        u = re.sub(r'[?&]+$', '', u)
        # URL-decode angle brackets that PDF authors sometimes embed as <slug>.
        u = u.replace('%3c', '').replace('%3C', '').replace('%3e', '').replace('%3E', '')
        # Normalize linkedin.com → www.linkedin.com.
        u = re.sub(r'https?://linkedin\.com', 'https://www.linkedin.com', u, flags=re.I)
        return u

    def name_anchored(uri: str) -> bool:
        if not name_tokens:
            return True
        return any(tok in uri.lower() for tok in name_tokens)

    other: list[str] = []
    for anchor, uri in links:
        u = clean(uri)
        ul = u.lower()
        host_match = re.search(r'https?://([^/]+)', ul)
        host = host_match.group(1) if host_match else ''
        if host.startswith('www.'):
            host = host[4:]

        if 'linkedin.com/in/' in ul:
            if out['linkedinUrl'] is None or name_anchored(u):
                out['linkedinUrl'] = u
        elif 'github.io' in ul:
            if out['portfolioUrl'] is None or name_anchored(u):
                out['portfolioUrl'] = u
        elif 'github.com/' in ul:
            path = ul.split('github.com/', 1)[1].split('?')[0].strip('/')
            parts = path.split('/')
            if len(parts) == 1 and parts[0]:
                if out['githubUrl'] is None or name_anchored(u):
                    out['githubUrl'] = u
            # repo URLs skipped — belong in projects
        elif 'leetcode.com' in ul:
            out['leetcodeUrl'] = u
        elif 'codechef.com' in ul:
            out['codechefUrl'] = u
        elif 'codeforces.com' in ul:
            out['codeforcesUrl'] = u
        elif 'huggingface.co' in ul:
            out['huggingfaceUrl'] = u
        elif 'kaggle.com' in ul:
            out['kaggleUrl'] = u
        elif ul.startswith('http') and host and host not in URL_DENY_HOSTS:
            # Skip mailto, deny-listed hosts (drive.google etc.).
            other.append(u)
    out['otherProfileUrls'] = list(dict.fromkeys(other))
    return out


def _education_block(text: str, sections: dict[str, tuple[int, int]]) -> str:
    """Return the text of the education section, or a heuristic guess if no section detected.
    Heuristic: first 2500 chars if no education section header (covers most resume tops)."""
    edu = section_text(text, sections, 'education')
    return edu if edu.strip() else text[:2500]


PRESENT_RE = re.compile(r'\b(?:Present|Pursuing|Current|Now|Ongoing|Appearing|Expected)\b', re.I)


def _most_recent_edu_row(edu_text: str) -> tuple[str, int | None, int | None, bool]:
    """Find the education row with the latest end year (or with a 'Present' marker).
    Splits rows on blank lines or sentinel patterns; falls back to whole edu_text as one row.
    Returns (row_text, enrollment_year, graduation_year, is_pursuing)."""
    rows = [r.strip() for r in re.split(r'\n\s*\n', edu_text) if r.strip()]
    if not rows:
        rows = [edu_text]

    best: tuple[str, int | None, int | None, bool, int] | None = None  # last int = score
    for row in rows:
        years = sorted({int(y) for y in YEAR_RE.findall(row)})
        # Plausibility filter: only "education-plausible" years (1990-2030 range).
        years = [y for y in years if 1990 <= y <= 2030]
        has_present = bool(PRESENT_RE.search(row))
        if not years and not has_present:
            continue
        if has_present:
            start_y = years[0] if years else None
            end_y = None
            is_pursuing = True
            score = 9999  # always prefer Present-bearing rows
        elif len(years) >= 2:
            start_y = years[0]
            end_y = years[-1]
            is_pursuing = end_y > TODAY.year
            score = end_y
        else:
            # Single year — graduation year (most common case for completed degrees).
            start_y = None
            end_y = years[0]
            is_pursuing = end_y > TODAY.year
            score = end_y
        if best is None or score > best[4]:
            best = (row, start_y, end_y, is_pursuing, score)

    if best:
        return best[:4]
    return edu_text, None, None, False


def extract_degree(edu_text: str) -> str | None:
    for pat, norm in DEGREE_TOKENS:
        if pat.search(edu_text):
            return norm
    return None


def extract_branch(edu_text: str, degree: str | None) -> str | None:
    # Drop branch when degree is non-engineering — BCA/MCA/MBA/PGDM/PGDFM/etc. don't have branches.
    if degree in NON_ENGINEERING_DEGREES:
        return None
    for pat, norm in BRANCH_TOKENS:
        if pat.search(edu_text):
            return norm
    return None


def extract_cgpa(edu_text: str, full_text: str) -> tuple[float | None, int | None]:
    """Prefer education section, fall back to whole text. CGPA usually appears once per resume."""
    for scope in (edu_text, full_text):
        m = CGPA_RE.search(scope)
        if not m:
            continue
        try:
            cgpa = float(m.group(1))
            if cgpa > 10:
                continue
            scale = int(m.group(2)) if m.group(2) else (10 if cgpa <= 10 else None)
            return cgpa, scale
        except Exception:
            continue
    return None, None


def extract_skills_section(text: str, sections: dict[str, tuple[int, int]]) -> dict[str, list[str]]:
    """Bucket skills found in the skills section only.
    Falls back to whole-doc-with-word-boundaries if no skills section detected."""
    skills_text = section_text(text, sections, 'skills')
    scope = skills_text if skills_text.strip() else text
    scope_lower = scope.lower()

    out = {b: [] for b in SKILL_BUCKETS}
    seen_canonical: set[str] = set()

    # Process buckets, longest-token-first within each, to avoid C-vs-C++ trap.
    for bucket, terms in SKILL_BUCKETS.items():
        sorted_terms = sorted(terms, key=lambda t: (-len(t), t))
        for term in sorted_terms:
            # Word boundary on each side — but escape regex specials and handle dots/hyphens.
            esc = re.escape(term)
            pat = r'(?<![\w])' + esc + r'(?![\w.+#])'
            if re.search(pat, scope_lower):
                canon = SKILL_ALIASES.get(term, term)
                if canon in seen_canonical:
                    continue
                seen_canonical.add(canon)
                disp = DISPLAY.get(canon) or DISPLAY.get(term) or term.title()
                out[bucket].append(disp)
    return out


# ---------- top-level extract ----------

def extract(pdf_path: Path, *, ocr: bool = True) -> dict[str, Any]:
    doc = open_pdf(pdf_path)
    n_pages = doc.page_count
    text, links = extract_text_and_links(doc)
    used_ocr = False
    text_layer_exists = bool(text.strip())

    if not text_layer_exists and ocr:
        text = _ocr_pdf(doc)
        used_ocr = bool(text.strip())
    doc.close()

    if not text.strip():
        return {
            '_sourceFile': pdf_path.name,
            '_textLayerExists': False,
            '_pdfPageCount': n_pages,
            '_extractionMethod': 'rules_only_v2',
            '_warning': 'No text layer and OCR unavailable or produced no output.',
        }

    sections = detect_sections(text)
    edu_text = _education_block(text, sections)

    email = extract_email(text)
    name = extract_name(text, email)
    addr, city, state, country = extract_geo(text)
    urls = extract_urls(links, name, email)
    degree = extract_degree(edu_text) or extract_degree(text)
    branch = extract_branch(edu_text, degree) or extract_branch(text, degree)
    cgpa, scale = extract_cgpa(edu_text, text)
    _row, enroll, grad, pursuing = _most_recent_edu_row(edu_text)
    skills = extract_skills_section(text, sections)

    # Canonical college resolution (optional). Detects college mentions in text,
    # resolves to canonical row; primary college = first resolved. Also returns
    # institution cities for use by extract_geo (already handled via INSTITUTE_KEYWORDS
    # reject, but explicit resolution gives us a college field and rank/grade enrichment).
    college: str | None = None
    college_city: str | None = None
    college_meta: dict | None = None
    if _CANONICAL_DB_AVAILABLE:
        try:
            matches = find_college_mentions(text)
            if matches:
                primary = matches[0]
                college = primary.get('canonical_name')
                college_city = primary.get('city') or None
                college_meta = {
                    'aishe_id': primary.get('aishe_id') or None,
                    'naac_grade': primary.get('naac_grade') or None,
                    'nirf_rank': int(primary['nirf_rank']) if primary.get('nirf_rank') and str(primary['nirf_rank']).isdigit() else None,
                    'state': primary.get('state') or None,
                    'matchCount': len(matches),
                }
        except Exception:
            pass

    out: dict[str, Any] = {
        '_sourceFile': pdf_path.name,
        'name': name,
        'phone': extract_phone(text),
        'email': email,
        'address': addr,
        'city': city,
        'state': state,
        'country': country,
        **urls,
        'degree': degree,
        'branch': branch,
        'college': college,
        'cgpa': cgpa,
        'cgpaScale': scale,
        'enrollmentYear': enroll,
        'graduationYear': grad,
        'isPursuing': pursuing,
        '_collegeMeta': college_meta,
        '_skillsCanonical': _canonicalize_skills(skills),
        **skills,
        '_pdfPageCount': n_pages,
        '_textLayerExists': text_layer_exists,
        '_extractionMethod': 'rules_only_v2' + ('+ocr' if used_ocr else ''),
        '_sectionsDetected': sorted(sections.keys()),
    }
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: rules_extract_v2.py <pdf_dir> <out_dir>", file=sys.stderr)
        return 2
    pdf_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for pdf in sorted(pdf_dir.glob('*.pdf')):
        try:
            data = extract(pdf)
        except Exception as e:
            data = {'_sourceFile': pdf.name, '_error': f'{type(e).__name__}: {e}'}
        (out_dir / (pdf.stem + '.json')).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        n += 1
        print(f"  {pdf.name}")
    print(f"\nWrote {n} files to {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
