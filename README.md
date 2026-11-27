# CalGrid power scheduling forecast tool

Backup of the code, the schema reference files and the Tableau workbook.
The EIA key and the download cache are not stored here.

## Contents
- `sim/common.py`: settings, downloads, cache and data cleaning used by both scripts
- `sim/step1_vendor.py`: step 1, downloads demand and weather and builds the purchased forecast
- `sim/step23_current_and_tool.py`: steps 2-3, builds the current process and the tool and writes the two Tableau files
- `sim/requirements.txt`: Python packages
- `09_data/`: schema reference files; step 2-3 checks the new Tableau files against them before saving
- `tableau/CalGrid schedule.twbx`: the Tableau workbook

## Run (VS Code, from this folder)
1. Install the packages once: `pip install -r sim/requirements.txt`
2. Create `.env` in this folder with one line: `EIA_API_KEY=your_key` (see `.env.example`).
   A key is free at https://www.eia.gov/opendata/register.php
3. Run step 1, then steps 2-3:
   ```
   python sim/step1_vendor.py
   python sim/step23_current_and_tool.py
   ```
4. The two new Tableau files are in `sim/results/tableau/`.

The first run downloads everything (about 2 h 53 min); later runs read the cache in `sim/data_cache/`.
