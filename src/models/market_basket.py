"""
Bonus: market-basket / association-rule analysis ("customers who bought X
also bought Y").

This module is NOT extracted from the notebooks -- neither notebook
implements the bonus market-basket task described in the assignment brief.
It's added here as new, clearly-separated code so the project covers the
full brief (RFM + clustering + churn + the optional association-rule
bonus), using the same cleaned sales data the rest of the pipeline uses.

Uses mlxtend's apriori + association_rules on a one-hot encoded
invoice x product matrix. To keep this tractable on ~800k rows / thousands
of SKUs, it restricts the basket matrix to the top-N best-selling products
by default -- adjust `top_n_products` for a deeper (slower) search.
"""
from typing import Optional

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def build_basket_matrix(df_clean: pd.DataFrame, top_n_products: int = 200, country: Optional[str] = "United Kingdom") -> pd.DataFrame:
    """Build a one-hot encoded (Invoice x Description) basket matrix.

    Parameters
    ----------
    df_clean : cleaned sales transactions.
    top_n_products : restrict to the N best-selling products (by total
        quantity) to keep the frequent-itemset search fast and the rules
        readable. Set to None to use every product (slow on the full
        dataset).
    country : restrict to a single country (the notebooks note the UK
        dominates volume, and mixing currencies/markets muddies "bought
        together" patterns). Set to None to use all countries.
    """
    data = df_clean
    if country is not None:
        data = data[data["Country"] == country]

    if top_n_products is not None:
        top_products = (
            data.groupby("Description")["Quantity"].sum()
            .sort_values(ascending=False)
            .head(top_n_products)
            .index
        )
        data = data[data["Description"].isin(top_products)]

    basket = (
        data.groupby(["Invoice", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    basket_encoded = basket.map(lambda x: True if x > 0 else False)

    # Drop single-item baskets: an association needs at least 2 items.
    basket_encoded = basket_encoded[basket_encoded.sum(axis=1) >= 2]
    return basket_encoded


def mine_association_rules(
    basket_encoded: pd.DataFrame,
    min_support: float = 0.02,
    min_confidence: float = 0.3,
    top_n_rules: int = 50,
) -> pd.DataFrame:
    """Run apriori + association_rules, return the top rules by lift.

    Returns a tidy DataFrame with columns: antecedents, consequents,
    support, confidence, lift.
    """
    frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    rules = rules.sort_values("lift", ascending=False).head(top_n_rules)
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].reset_index(drop=True)
