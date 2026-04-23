# Publishing checklist: GitHub + Zenodo

This document lists every step you (the corresponding author) must carry out
to turn this repository into a citable Zenodo deposit. **Every command is
run on your own machine, under your own credentials — no secrets are shared
with any AI agent or third party.**

Only DOI-related placeholders remain. They are filled in **after** Zenodo
issues the DOI (see Phase 4 below). A quick search-and-replace for `TBD`
will surface every one.

---

## Phase 0 — Before you touch anything

- [ ] Confirm that the raw Excel file (`data/raw/Seberang Jaya, ... .xlsx`) is
      **not** tracked by git (`.gitignore` already excludes `data/raw/`).
- [x] Author block, affiliations, and emails populated in:
      `README.md`, `LICENSE`, `CITATION.cff`, `.zenodo.json`, `pyproject.toml`.
- [ ] (Optional) Add ORCID iDs to `CITATION.cff` and `.zenodo.json` once
      every co-author has registered one at <https://orcid.org>.
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
   (sign up if you have not yet, using the corresponding author's email).
   Link your ORCID from the Zenodo profile page.
2. Open <https://zenodo.org/account/settings/github/>.
3. Click **Sync now** so Zenodo refreshes the repository list.
4. Toggle the switch next to **`research-AI-studies/atmospheric-intelligence-engine`**
   to **ON**. This authorises Zenodo to archive future releases automatically.

---

## Phase 3 — Cut the first GitHub release

Zenodo's current GitHub integration mints the DOI automatically when it
receives the release webhook. You do not pre-reserve; you publish first and
embed the returned DOI afterwards.

1. On GitHub, open **Releases → Draft a new release**.
2. Tag: `v0.1.0` (matches `pyproject.toml` and `CITATION.cff`).
3. Target: `main`.
4. Title: `v0.1.0 – Initial public release`.
5. Release notes: a short description of the deposit (see the actual release
   for the exact text).
6. Leave **Set as a pre-release** unchecked. Do not attach any binaries.
7. Click **Publish release**.

Within 1–3 minutes Zenodo will archive the release automatically and expose
two DOIs on the record page:

- **Concept DOI** — stable across all future versions (what you cite in the
  manuscript).
- **Version DOI** — points only at this specific release.

---

## Phase 4 — Embed the DOI in the repository

Once Zenodo has published the record:

1. Copy the **Concept DOI** from the Zenodo record (e.g. `10.5281/zenodo.19662860`).
2. Replace every remaining `10.5281/zenodo.TBD` in the repo:

   - `README.md` (DOI badge and BibTeX)
   - `CITATION.cff` (`identifiers` block)
3. Commit and push, then optionally cut a patch release (`v0.1.1`) so the
   DOI is baked into the archived tarball as well:

   ```bash
   git commit -am "docs: insert Zenodo concept DOI"
   git push
   ```

4. Add the same DOI to the manuscript's **Data Availability Statement** and
   references list before submission:

   > The model code, trained parameters, deterministic synthetic sample, and
   > scripts required to reproduce every figure are archived on Zenodo at
   > https://doi.org/10.5281/zenodo.19662860 (concept DOI, resolves to the
   > latest archived version).

---

## Phase 5 — Verify

- [ ] The Zenodo record shows state "Published" with the GitHub archive
      attached.
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
