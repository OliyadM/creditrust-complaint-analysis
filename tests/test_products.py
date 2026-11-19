import pandas as pd

from src.products import (
    BNPL,
    CREDIT_CARD,
    MONEY_TRANSFER,
    PERSONAL_LOAN,
    SAVINGS,
    clean_narrative,
    map_product,
    map_products_frame,
)


def test_credit_card_legacy_and_current_names():
    assert map_product("Credit card", "") == CREDIT_CARD
    assert map_product("Credit card or prepaid card", "Store credit card") == CREDIT_CARD


def test_prepaid_is_excluded():
    assert map_product("Credit card or prepaid card", "General-purpose prepaid card") is None
    assert map_product("Prepaid card", "Gift card") is None


def test_personal_loan_excludes_payday_and_title():
    assert map_product("Payday loan, title loan, or personal loan", "Installment loan") == PERSONAL_LOAN
    assert map_product("Consumer Loan", "Personal line of credit") == PERSONAL_LOAN
    assert map_product("Payday loan, title loan, or personal loan", "Payday loan") is None
    assert map_product("Consumer Loan", "Vehicle loan") is None


def test_savings_only_savings_subproducts():
    assert map_product("Checking or savings account", "Savings account") == SAVINGS
    assert map_product("Bank account or service", "(CD) Certificate of deposit") == SAVINGS
    assert map_product("Checking or savings account", "Checking account") is None


def test_money_transfers():
    assert map_product("Money transfers", "International money transfer") == MONEY_TRANSFER
    assert map_product("Money transfer, virtual currency, or money service",
                       "Mobile or digital wallet") == MONEY_TRANSFER
    assert map_product("Money transfer, virtual currency, or money service", "Virtual currency") is None


def test_bnpl_by_company_or_keyword_overrides():
    assert map_product("Payday loan, title loan, or personal loan", "Installment loan",
                       company="Affirm Holdings, Inc") == BNPL
    assert map_product("Credit card", "", narrative="I used Buy Now, Pay Later at checkout") == BNPL
    assert map_product("Mortgage", "Other", narrative="klarna kept charging me") == BNPL


def test_affirm_as_a_verb_is_not_bnpl():
    assert map_product("Mortgage", "Other", narrative="I affirm that this is true") is None


def test_out_of_scope():
    assert map_product("Mortgage", "Conventional home mortgage") is None
    assert map_product(None, None) is None


def test_frame_matches_scalar_mapping():
    rows = [
        ("Credit card", "", "Bank", "late fee"),
        ("Credit card or prepaid card", "General-purpose prepaid card", "Bank", "x"),
        ("Consumer Loan", "Installment loan", "Lender", "loan"),
        ("Checking or savings account", "Savings account", "Bank", "withdraw"),
        ("Money transfers", "", "Wire Co", "wire"),
        ("Debt collection", "Other", "Sezzle Inc.", "x"),
        ("Mortgage", "Other", "Bank", "used afterpay"),
        ("Mortgage", "Other", "Bank", None),
    ]
    df = pd.DataFrame(rows, columns=["Product", "Sub-product", "Company", "Consumer complaint narrative"])
    expected = [map_product(*r) for r in rows]
    assert map_products_frame(df).tolist() == expected


def test_clean_narrative():
    raw = "On XX/XX/XXXX I paid {$250.00} to XXXX   XXXX bank.\n\nThey   refused."
    assert clean_narrative(raw) == "On I paid $250.00 to bank. They refused."
    assert clean_narrative(None) == ""
