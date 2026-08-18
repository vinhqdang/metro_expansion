# Confirmed data sources (verified 2026-08-18 by direct inspection, not just research)

This supersedes the "likely reusable" / unverified claims in the manuscript's Case Studies and Data section with directly-confirmed findings from cloning and inspecting the actual repositories. Update `manuscript/main.tex` §Case Studies and Data accordingly.

## Xi'an: `github.com/dimichai/tabular-tndp` (+ submodule `github.com/sias-uva/mo-tndp`)

**Confirmed available and usable now.** `envs/mo-tndp/cities/xian/`:
- `config.txt`: 29x29 grid, existing metro lines 1 & 2 as grid-coordinate paths, `excluded_od_segments`.
- `od.txt`: 463,010 nonzero OD pairs as `(idx1, idx2, weight)` triples over the 29x29 = 841-cell grid.
- `average_house_price_gid.txt`, `price_groups_1..10.txt`: per-cell house price / price-quantile groupings, used as the equity/fairness signal in the upstream paper.

**License:** MIT (mo-tndp repo). Free to use with attribution.

**Correction to earlier (unverified) research pass:** the original data-availability research suggested this dataset traces back to Wei et al., KDD 2020. The submodule's own `CITATION.bib` instead attributes it to **Michailidis, Röpke, Roijers, Ghebreab, Santos, "Scalable Multi-Objective Reinforcement Learning with Fairness Guarantees using Lorenz Dominance," arXiv:2411.18195 (2024)**. Cite this as the direct data source; the KDD 2020 lineage is unconfirmed and should not be asserted without reading that paper (still paywalled/unobtained).

**API:** `motndp/city.py`'s `City` class reads these files directly (`grid_to_index`/`index_to_grid` helpers) -- worth reusing or mirroring the format rather than reinventing a grid representation for Xi'an.

## Ho Chi Minh City: `github.com/CityScope/CSL_HCMC`

**Confirmed available and usable now -- the strongest data source of all four cities.** `Data/GIS/` contains real shapefiles (not proxies):
- `Metro/Metro_shapefile/`: MetroRoute.shp + Station.shp (routes and stations).
- `Bus/Bus_shapefile/`: BusRoute.shp + BusStop.shp (existing bus network -- our auxiliary signal source). **Coverage caveat:** the 189 stops in `BusStop.shp` fall entirely within a small downtown/pilot-study area, giving nonzero bus-derived signal for only ~32 of the city's 322 candidate zones (~10%) when run through `src/data/hcmc.py`'s `build_bus_stop_density_feature` -- confirmed by running `scripts/smoke_test_hcmc.py`. Full-city coverage will need the OpenStreetMap fallback or a wider bus dataset.
- `OD_2018/OD_shapefile/`: `Survey_OD.shp`, a real 2018 household travel survey OD dataset.
- `Population/`: district- and ward-level population shapefiles (`population_HCMC/`) plus a separate `population_model_area/` covering the CityScope project's focus area.
- `POI/`: two POI shapefile sets (`POI_model_area/`, `POI_simulation_area/`).
- `Cencus/shapefile_cencus/Ward_Cencuss_Age.shp`: ward-level census with age breakdown.

**License:** no LICENSE file in the repo -- redistribution rights are not explicit. Treat as research/fair-use only: use for analysis and cite the repo, do not vendor/mirror the shapefiles into our own repo (already enforced by `.gitignore`'s `data/*/raw/` and shapefile-extension rules).

## Chengdu and Xi'an network geometry (historical, multi-year): `github.com/GZUPA/subway-traffic-data-set`

**Confirmed available and usable now.** Single `.rar` archive (extract with 7-Zip; no `unrar`/`7z` in the default shell PATH, but `C:\Program Files\7-Zip\7z.exe` is installed on this machine). Contains, per city, **year-by-year** station-point and line shapefiles from ~2010 through ~2019+ for both `chengdu` and `xian` (among 38 Chinese cities total) under `1_subway_point/<city>/` and `2_subway_line/<city>/`.

This is a genuine bonus beyond what the earlier research pass found: because it's multi-year, it opens a validation angle the manuscript doesn't currently mention -- checking whether the policy's predicted highest-priority expansion segments match what these cities *actually built next*, using an earlier year's network as the starting state and a later year's as ground truth. Worth considering as an additional evaluation protocol, not just a data source.

**License:** explicitly stated in the repo's (Chinese-language) README as an open dataset ("本数据集为开源数据集"), requesting only citation of the dataset in any research output (contact: liangweidong@upagz.cn). No ridership/flow data, geometry only.

## Not yet inspected

- `github.com/tsinghua-fib-lab/MetroGNN` (Beijing/Changsha data, candidate 5th/6th city per the open-ended city set) -- not cloned yet.
- Chengdu's own open-data portal (`data.chengdu.gov.cn`) for a possible metro ridership/passenger-flow dataset -- requires manual portal navigation, not yet done.
- Hanoi: no confirmed structured OD/POI dataset found in either research pass; still relying on the gravity-model proxy plan (WorldPop + OSM POI + published ridership calibration).

## Repo hygiene note

All four repos above were cloned into `external/` (gitignored, not committed) purely for inspection. Do not vendor their contents into this repo; write our own loader/preprocessing code in `src/data/` that fetches or reads from these sources at data-pipeline-run time, respecting each source's license terms.
