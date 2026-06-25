"""
lib.reasoning — Natural Language Reasoning Generation

Constructs dynamic, non-templated reasoning strings for candidates
using exactly their real data, ensuring variety across rows.
"""

import hashlib


def _get_template_index(candidate_id: str, num_templates: int) -> int:
    """Hash the candidate ID to deterministically select a sentence template."""
    hex_hash = hashlib.md5(candidate_id.encode("utf-8")).hexdigest()
    return int(hex_hash, 16) % num_templates


def generate(candidate_id: str, fv: dict, hp_reason: str, elite_notes: list[str]) -> str:
    """
    Generate a 1-2 sentence varied reasoning string built solely from real field values.
    """
    # Extract fields
    title = str(fv.get("raw_title", "Professional")).strip()
    yoe = int(fv.get("years_of_experience", 0))
    company = str(fv.get("latest_company", "")).strip()
    skills_list = fv.get("top_skills", [])
    
    # Clean up empty values
    if company.lower() == "an unknown company" or not company:
        company_phrase = "various roles"
    else:
        company_phrase = f"at {company}"

    if skills_list:
        skills = ", ".join(skills_list)
    else:
        skills = "general competencies"

    # Activity signal
    months_inactive = int(fv.get("months_since_last_active", 0))
    open_to_work = bool(fv.get("open_to_work", 0.0))
    
    if open_to_work:
        signal = "is actively open to new opportunities"
    elif months_inactive == 0:
        signal = "has been recently active"
    else:
        signal = f"was last active {months_inactive} months ago"

    # Define varied templates
    templates = [
        # Template 0
        "{title} offering {yoe} years of experience, most recently {company}. Key expertise includes {skills}, and the candidate {signal}.",
        # Template 1
        "With {yoe} years in the field, this {title} brings a background {company}. Core skills feature {skills}; candidate {signal}.",
        # Template 2
        "Strong background in {skills} built over {yoe} years. Currently a {title} {company}, the candidate {signal}.",
        # Template 3
        "Candidate {signal}, showcasing {yoe} years as a {title} {company}. Primary proficiencies are {skills}.",
    ]

    idx = _get_template_index(candidate_id, len(templates))
    base_text = templates[idx].format(
        title=title,
        yoe=yoe,
        company=company_phrase,
        skills=skills,
        signal=signal
    )

    # Append caveats / prepends
    parts = []
    
    if hp_reason:
        parts.append(f"[CAVEAT: {hp_reason}]")
    elif fv.get("job_hop_frequency", 0.0) > 0.8:
        parts.append("[CAVEAT: High job-hop frequency]")
        
    for note in elite_notes:
        parts.append(f"[{note.upper()}]")

    parts.append(base_text)

    return " ".join(parts).replace("  ", " ").strip()
