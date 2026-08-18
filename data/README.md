# Per-city data

Each `<city>/` subdirectory holds that city's raw and processed inputs: rail network topology, planned extensions, OD demand (real or proxy), population/POI density, administrative boundaries, and bus-network auxiliary signal.

Data acquisition status and sources per city are documented in `manuscript/main.tex` (Case Studies and Data section) and were compiled by a research pass on 2026-08-18 -- see git history / conversation record for the full source list (OSM/Overpass, GADM, WorldPop, CityScope/CSL_HCMC, dimichai/tabular-tndp, GZUPA/subway-traffic-data-set, etc.).

Large raw data files and anything under a redistribution-restricting license are gitignored; only small processed/derived files and provenance metadata should be committed.
