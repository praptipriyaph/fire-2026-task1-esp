"""Reasoning templates for the 7 IPC sections.

Structure mirrors the FIRE 2026 Task 01 §04 published example (3-sentence
form: legal-ingredients statement -> exact-fact sentence -> conclusion citing
the substantive section). The exact_fact appears as its own sentence rather
than a dashed parenthetical, which matches the published 302 example's shape
("The presence of the accused with the weapon... satisfy the elements of
murder under Section 300 IPC, making Section 302 the appropriate penal
provision.") and gives the fact full surface presence for ROUGE/BLEU overlap.

Each template ends with a phrase that cites the substantive section
(141/146, 300, 375, 503) where the offence is defined, distinct from the
penal section being predicted.
"""

TEMPLATES = {
    "147": (
        "Section 147 IPC applies because the facts establish that the accused, "
        "being members of an unlawful assembly within the meaning of Section 141 IPC, "
        "used force or violence in prosecution of the common object of such assembly. "
        "{fact}. "
        "The presence of these facts satisfies the ingredients of rioting under "
        "Section 146 IPC, making Section 147 the appropriate penal provision."
    ),
    "201": (
        "Section 201 IPC applies because the accused, knowing or having reason to "
        "believe that an offence had been committed, caused evidence of the commission "
        "of that offence to disappear with the intention of screening the offender "
        "from legal punishment. "
        "{fact}. "
        "This conduct constitutes causing disappearance of evidence of an offence "
        "committed, satisfying the ingredients of Section 201 IPC."
    ),
    "302": (
        "Section 302 IPC applies because the facts establish intentional causing of "
        "death with the requisite mens rea and overt acts attributable to the accused. "
        "{fact}. "
        "The presence of these facts satisfies the elements of murder under "
        "Section 300 IPC, making Section 302 the appropriate penal provision."
    ),
    "376": (
        "Section 376 IPC applies because the facts establish commission of rape as "
        "defined under Section 375 IPC, with the act done against the will or without "
        "the consent of the prosecutrix. "
        "{fact}. "
        "The presence of these facts satisfies the ingredients of rape under "
        "Section 375 IPC, making Section 376 the appropriate penal provision."
    ),
    "420": (
        "Section 420 IPC applies because the facts establish that the accused, by "
        "deceiving another, dishonestly induced the deceived person to deliver "
        "property or to do or omit to do anything which they would not otherwise have done. "
        "{fact}. "
        "The presence of these facts satisfies the ingredients of cheating and "
        "dishonestly inducing delivery of property, making Section 420 the appropriate "
        "penal provision."
    ),
    "498A": (
        "Section 498A IPC applies because the facts establish that the accused, being "
        "the husband or a relative of the husband of a woman, subjected her to cruelty "
        "within the meaning of the Explanation to Section 498A IPC. "
        "{fact}. "
        "The presence of these facts satisfies the ingredients of cruelty by husband "
        "or relatives of husband under Section 498A IPC."
    ),
    "506": (
        "Section 506 IPC applies because the facts establish that the accused committed "
        "criminal intimidation by threatening another with injury to person, reputation, "
        "or property, with intent to cause alarm or to compel an act. "
        "{fact}. "
        "The presence of these facts satisfies the ingredients of criminal intimidation "
        "under Section 503 IPC, making Section 506 the appropriate penal provision."
    ),
}


def build_trace(section_num: str, exact_fact: str) -> str:
    tpl = TEMPLATES.get(section_num)
    fact = exact_fact.strip().rstrip(".")
    if tpl is None:
        return (
            f"Section {section_num} IPC applies because the facts establish the "
            f"ingredients of the offence. {fact}. The presence of these facts satisfies "
            f"the elements of the offence, making Section {section_num} the appropriate "
            f"penal provision."
        )
    return tpl.format(fact=fact)
