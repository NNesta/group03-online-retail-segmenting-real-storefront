# Raw data

This folder holds the source workbook:

```
online_retail_II.xlsx
```

It ships with the downloadable project archive, but it is **excluded from git**
by `.gitignore` because it's about 45 MB. After cloning the repository you'll
need to put it back here.

**Download:** Online Retail II, UCI Machine Learning Repository —
<https://archive.ics.uci.edu/dataset/502/online+retail+ii>

The file must contain the two sheets UCI ships it with:

- `Year 2009-2010`
- `Year 2010-2011`

Nothing in this folder is ever modified by the pipeline — it's read-only input.
Everything generated lands in `../processed/` and `../../models/`.

If you'd rather commit the workbook anyway, remove the `data/raw/*.xlsx` line
from `.gitignore` and consider using Git LFS for a file this size.
