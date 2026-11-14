"""Mapping from raw CFPB product fields to the five CrediTrust product lines,
plus light narrative cleaning.

CFPB has renamed its product taxonomy several times since 2011, so each
CrediTrust product is matched against every historical name. BNPL is not a
CFPB product value at all; it is detected from the company name or from
explicit keywords in the narrative and takes priority over the other labels.
"""
from __future__ import annotations

import re

CREDIT_CARD = "Credit card"
PERSONAL_LOAN = "Personal loan"
BNPL = "Buy Now, Pay Later (BNPL)"
SAVINGS = "Savings account"
MONEY_TRANSFER = "Money transfers"

PRODUCTS = [CREDIT_CARD, PERSONAL_LOAN, BNPL, SAVINGS, MONEY_TRANSFER]

# (CFPB product, allowed sub-products). None means any sub-product, including blank.
_RULES: dict[str, list[tuple[str, set[str] | None]]] = {
    CREDIT_CARD: [
        ("Credit card", None),  # legacy product (2011-2017), no prepaid mixed in
        ("Credit card or prepaid card", {
            "General-purpose credit card or charge card",
            "Store credit card",
        }),
    ],
    PERSONAL_LOAN: [
        ("Consumer Loan", {"Installment loan", "Personal line of credit"}),
        ("Payday loan, title loan, or personal loan",
         {"Installment loan", "Personal line of credit"}),
        ("Payday loan, title loan, personal loan, or advance loan",
         {"Installment loan", "Personal line of credit"}),
    ],
    SAVINGS: [
        ("Checking or savings account", {"Savings account", "CD (Certificate of Deposit)"}),
        ("Bank account or service", {"Savings account", "(CD) Certificate of deposit"}),
    ],
    MONEY_TRANSFER: [
        ("Money transfers", None),  # legacy product
        ("Money transfer, virtual currency, or money service", {
            "Domestic (US) money transfer",
            "International money transfer",
            "Mobile or digital wallet",
        }),
    ],
}

_LOOKUP: dict[str, list[tuple[str, set[str] | None]]] = {}
for _label, _rules in _RULES.items():
    for _prod, _subs in _rules:
        _LOOKUP.setdefault(_prod, []).append((_label, _subs))

BNPL_COMPANIES = {
    "Affirm Holdings, Inc",
    "Klarna AB",
    "Sezzle Inc.",
    "Zip Co US Inc.",
    "Afterpay US Services, LLC",
    "SPLITIT USA INC.",
}

# "Affirm" is deliberately not a narrative keyword: "I affirm that..." is common
# legal boilerplate. Affirm complaints are caught through the company field.
BNPL_PATTERN = re.compile(
    r"buy[\s-]+now,?[\s-]+pay[\s-]+later|\bbnpl\b|\bklarna\b|\bafterpay\b|\bsezzle\b"
    r"|\bpay[\s-]+in[\s-]+4\b|\bquadpay\b",
    re.IGNORECASE,
)


def _s(x) -> str:
    return x if isinstance(x, str) else ""


def is_bnpl(company, narrative) -> bool:
    return _s(company).strip() in BNPL_COMPANIES or bool(BNPL_PATTERN.search(_s(narrative)))


def map_product(product, sub_product, company=None, narrative=None) -> str | None:
    """Return the CrediTrust product label for one complaint, or None if it is
    outside the five product lines."""
    if is_bnpl(company, narrative):
        return BNPL
    for label, subs in _LOOKUP.get(_s(product).strip(), []):
        if subs is None or _s(sub_product).strip() in subs:
            return label
    return None


def map_products_frame(df, product="Product", sub="Sub-product",
                       company="Company", narrative="Consumer complaint narrative"):
    """Vectorised version of map_product for a pandas DataFrame. Returns a Series
    of labels (None where the complaint is out of scope)."""
    import pandas as pd

    labels = pd.Series([None] * len(df), index=df.index, dtype=object)
    prod = df[product].fillna("").str.strip()
    subp = df[sub].fillna("").str.strip()
    # assign in reverse priority so earlier rules win, then BNPL overrides all
    for label in reversed(PRODUCTS):
        for p, subs in _RULES.get(label, []):
            mask = prod.eq(p)
            if subs is not None:
                mask &= subp.isin(subs)
            labels[mask] = label
    bnpl = df[company].fillna("").str.strip().isin(BNPL_COMPANIES)
    bnpl |= df[narrative].fillna("").str.contains(BNPL_PATTERN, regex=True)
    labels[bnpl] = BNPL
    return labels
