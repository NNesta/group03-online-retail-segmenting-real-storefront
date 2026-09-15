"""
Raw data loading.

Extracted, unmodified in logic, from the first cells of both
`notebooks/unsupervised_learning.ipynb` and
`notebooks/supervised_learning.ipynb`:

    xls = pd.ExcelFile('../data/online_retail_II.xlsx')
    df_2009_2010 = pd.read_excel(xls, sheet_name='Year 2009-2010')
    df_2010_2011 = pd.read_excel(xls, sheet_name='Year 2010-2011')
    df_2009_2010["SourceSheet"] = "2009-2010"
    df_2010_2011["SourceSheet"] = "2010-2011"
    df_all = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)
"""
from pathlib import Path
from typing import Union

import pandas as pd

from src.config import RAW_EXCEL_PATH


def load_raw_transactions(excel_path: Union[str, Path] = RAW_EXCEL_PATH) -> pd.DataFrame:
    """Load and concatenate both sheets of the Online Retail II workbook.

    Parameters
    ----------
    excel_path : the .xlsx file containing the 'Year 2009-2010' and
        'Year 2010-2011' sheets.

    Returns
    -------
    DataFrame with all raw rows from both sheets plus a `SourceSheet`
    column recording which sheet each row came from.
    """
    xls = pd.ExcelFile(excel_path)

    df_2009_2010 = pd.read_excel(xls, sheet_name="Year 2009-2010")
    df_2010_2011 = pd.read_excel(xls, sheet_name="Year 2010-2011")

    df_2009_2010["SourceSheet"] = "2009-2010"
    df_2010_2011["SourceSheet"] = "2010-2011"

    df_all = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)
    return df_all
