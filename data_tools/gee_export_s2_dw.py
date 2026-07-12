"""AUIE Gate A — Earth Engine exports (run in Colab, not locally).

Queues in ONE session:
  1. Seven annual Sentinel-2 dry-season composites (2019-2025), 7 bands, 10 m
  2. Dynamic World 4-class label mosaics for the SAME window/grid (2024 for
     training; the script can queue other years for temporal-transfer checks)

Setup in Colab:
    import ee
    ee.Authenticate()
    ee.Initialize(project="YOUR_GEE_PROJECT_ID")
then paste/run this file.

Exports land in Google Drive folder AUIE_exports. Expect the queue to take
hours — kick everything off, then go build make_chips.py locally.

Class scheme (single source of truth, mirror of config.py):
  0 = background/bare   1 = vegetation   2 = waterbody   3 = built-up
Shrub/scrub -> vegetation (Bengaluru scrub is ecologically vegetation; note
the choice in the paper). Snow/ice -> background (absent in Bengaluru).
"""
import ee

# ---------------------------------------------------------------- AOI
# TODO(Obana): replace with your uploaded BBMP boundary asset for exact
# clipping, e.g. ee.FeatureCollection("projects/YOUR_PROJECT/assets/bbmp").
# The rectangle below covers BBMP with margin; ward-level clipping happens
# later in the analytics layer, so a rectangle is fine for training data.
AOI = ee.Geometry.Rectangle([77.44, 12.80, 77.80, 13.14])

CRS = "EPSG:32643"          # UTM 43N — one grid for every export, every year
SCALE = 10
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
WINDOW = ("-01-01", "-03-31")   # dry season, same window every year — never vary
S2_BANDS = ["B2", "B3", "B4", "B8", "B8A", "B11", "B12"]
EXPORT_NAMES = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
DRIVE_FOLDER = "AUIE_exports"
LABEL_YEARS = [2024]        # add e.g. 2019 later for temporal-transfer checks


def _mask_s2_clouds(img: ee.Image) -> ee.Image:
    """SCL-based mask: drop cloud shadow(3), clouds(8,9), cirrus(10)."""
    scl = img.select("SCL")
    bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(bad.Not())


def s2_composite(year: int) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(AOI)
        .filterDate(f"{year}{WINDOW[0]}", f"{year}{WINDOW[1]}")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(_mask_s2_clouds)
    )
    return (
        col.select(S2_BANDS, EXPORT_NAMES)
        .median()
        .clip(AOI)
        .toUint16()  # reflectance x10000 fits uint16; halves file size
    )


def dw_four_class(year: int) -> ee.Image:
    """Dynamic World mode label over the same window, remapped to 4 classes."""
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(AOI)
        .filterDate(f"{year}{WINDOW[0]}", f"{year}{WINDOW[1]}")
        .select("label")
        .reduce(ee.Reducer.mode())
        .rename("label")
    )
    # DW: 0 water 1 trees 2 grass 3 flooded_veg 4 crops 5 shrub 6 built 7 bare 8 snow
    remapped = dw.remap(
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [2, 1, 1, 1, 1, 1, 3, 0, 0],
    ).rename("class")
    return remapped.clip(AOI).toUint8()


def queue(img: ee.Image, name: str) -> None:
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        region=AOI,
        crs=CRS,
        scale=SCALE,
        maxPixels=1e10,
    )
    task.start()
    print(f"queued: {name}")


if __name__ == "__main__":
    for y in YEARS:
        queue(s2_composite(y), f"s2_bbmp_{y}_janmar")
    for y in LABEL_YEARS:
        queue(dw_four_class(y), f"dw4_bbmp_{y}_janmar")
    print("\nAll tasks queued. Monitor at https://code.earthengine.google.com/tasks")
    print("While waiting: run make_chips.py against last year's local test data,")
    print("or proceed with the gazetteer/resolver refactor.")
