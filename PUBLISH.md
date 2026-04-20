# Publishing checklist: GitHub + Zenodo

This document lists every step you (the corresponding author) must carry out
to turn this repository into a citable Zenodo deposit. **Every command is
run on your own machine, under your own credentials — no secrets are shared
with any AI agent or third party.**

All placeholders marked `TBD` in these files must be filled in before the
first push. A quick search-and-replace for `TBD` will surface every remaining
one.

---

## Phase 0 — Before you touch anything

- [ ] Confirm that the raw Excel file (`data/raw/Seberang Jaya, ... .xlsx`) is
      **not** tracked by git (`.gitignore` already excludes `data/raw/`).
- [ ] Replace every `TBD` in the following files with real values:
  - [ ] `README.md` (author block in the `Citation` section)
  - [ ] `LICENSE` (copyright line)
  - [ ] `CITATION.cff` (authors, affiliation, ORCID)
  - [ ] `.zenodo.json` (creators)
  - [ ] `pyproject.toml` (`authors` field)
- [ ] Confirm the manuscript title on top of `README.md` matches the final
      submitted title.
- [ ] Run the full test + smoke pipeline locally:

  ```bash
  conda env create -f environment.yml
  conda activate aie
  pip install -e ".[dev]"
  pytest -q
  make smoke
  ```

  All green. No warnings about unhandled NaNs.

---

## Phase 1 — Create the GitHub repository

1. Log into GitHub with the account **`research-AI-studies`**.
2. Create a **new empty repository**:
   - Name: `atmospheric-intelligence-engine`
   - Owner: `research-AI-studies`
   - Visibility: **Public** (required for Zenodo integration).
   - **Do NOT** initialise with README, .gitignore, or license — this repo
     already has them.
3. On your local machine, from this repository's root:

   ```bash
   git init -b main
   git add .
   git commit -m "chore: initial public release v0.1.0"
   git remote add origin git@github.com:research-AI-studies/atmospheric-intelligence-engine.git
   git push -u origin main
   ```

4. On GitHub, open **Settings → Actions → General** and confirm that workflows
   are allowed. Push or re-run CI to produce a green badge.

---

## Phase 2 — Link GitHub and Zenodo

1. Log into <https://zenodo.org> with the account **`research-AI-studies`**
   (sign up if you have not yet; use email `ddaydota@gmail.com`). Link your
   ORCID from the Zenodo profile page.
2. Open <https://zenodo.org/account/settings/github/>.
3. Click **Sync now** so Zenodo refreshes the repository list.
4. Toggle the switch next to **`research-AI-studies/atmospheric-intelligence-engine`**
   to **ON**. This authorises Zenodo to archive future releases automatically.

---

## Phase 3 — Reserve a DOI before submission

RSC requires the DOI to appear in the Data Availability Statement of the
submitted manuscript. Reserve it **before** creating the first GitHub release:

1. Go to Zenodo → **Upload** → **New upload**.
2. Fill in the metadata so it matches `.zenodo.json`:
   - Upload type: **Software**
   - Title: as in `.zenodo.json`
   - Authors: as in `.zenodo.json`
   - Description: as in `.zenodo.json`
   - License: **MIT**
3. Under **Basic information → DOI**, click **Reserve DOI**.
4. Copy the reserved DOI (e.g. `10.5281/zenodo.1234567`). **Do not publish
   yet.**
5. In this repository, replace every remaining `10.5281/zenodo.TBD` with the
   reserved DOI:

   - `README.md` (badge and BibTeX)
   - `CITATION.cff` (`doi` field if present)
6. Commit and push:

   ```bash
   git commit -am "docs: insert reserved Zenodo DOI"
   git push
   ```

7. Add the same DOI to the manuscript's **Data Availability Statement** and
   references list before you submit to the journal.

---

## Phase 4 — Cut the first GitHub release

Once the DOI is inserted and CI is green:

1. On GitHub, open **Releases → Draft a new release**.
2. Tag: `v0.1.0` (must match the version in `pyproject.toml`, `CITATION.cff`,
   and `CHANGELOG.md`).
3. Title: `v0.1.0 — initial submission to Environmental Science: Atmospheres`.
4. Description: paste the `[0.1.0]` section of `CHANGELOG.md`.
5. Click **Publish release**.

Zenodo will detect the release within a few minutes and **complete** the
deposit that was reserved in Phase 3, assigning it the DOI you already
embedded in the code and manuscript.

---

## Phase 5 — Verify

- [ ] The Zenodo record for the reserved DOI now shows state "Published" with
      your release archive attached.
- [ ] The DOI badge at the top of `README.md` resolves to the record.
- [ ] GitHub Actions CI is green on `main`.
- [ ] `pip install git+https://github.com/research-AI-studies/atmospheric-intelligence-engine.git@v0.1.0`
      succeeds in a clean environment.
- [ ] The manuscript Data Availability Statement references the real DOI, not
      `TBD`.

---

## Subsequent releases

Bump `version` in `pyproject.toml`, `CITATION.cff`, and add a section to
`CHANGELOG.md`, then cut a new tag (`v0.2.0`, etc.). Zenodo will mint a new
DOI for each GitHub release and also provide a **concept DOI** that always
points to the latest version.
