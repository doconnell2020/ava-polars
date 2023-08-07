"""
This script reads in the raw avalanche incident data and reprocesses the coordinate 
information so that it is standardised.

There are 4 categories of 'location_coords' in the source file,
therefore it needs 4 different treatments.

Rules to think about
- We can assume all longtitude should be negative (Western Hemisphere)
- Decimal precision is not very important
- Latitude should not be >90, otherwise swap.
- UTM is only 2 distict types
- 4 types to cover:
    1. LatLon :  Reversed and sign needed
    2. Lat/lng : Correct
    3. Lat/Long Decimal Degrees : Correct
    4. UTM (starts with) : Remove letters, function parse
"""
import time
from geopy.geocoders import Nominatim
import numpy as np
import pandas as pd
import pyproj
import warnings

start = time.time()

warnings.filterwarnings("ignore")


df = pd.read_csv("data/can_avs_raw.csv")

# Create new df with reduced info
new_df = df[["ob_date", "location_coords", "location_coords_type"]].copy()

new_df["location_coords"] = new_df["location_coords"].astype(str)
new_df["location_coords_type"] = new_df["location_coords_type"].astype(str)

# Create idxs for all types of location coordinate type
# These two are in the correct position
lat_lng_idx = new_df["location_coords_type"] == "Lat/lng"
lat_lng_dd_idx = new_df["location_coords_type"] == "Lat/Long Decimal Degrees"

# These rows have their latitude and longitude reversed, so that will have to be caught
# during assignment
lat_lon_idx = new_df["location_coords_type"] == "LatLon"

# The UTM format has more cleaning to go through than the others; removing letters,
# the "assumed" word etc.
utm_idx = new_df["location_coords_type"].str.startswith("UTM")


# Define some processing functions


def split_coordinates(series: pd.Series) -> tuple:
    """
    Splits the series from a point to individual latitude and longitude series.
    """
    coordinates = series.str.strip("[]").str.split(", ")
    latitude = coordinates.apply(lambda x: x[0])
    longitude = coordinates.apply(lambda x: x[1])
    return latitude, longitude


# The UTM type requires more wrangling to parse


def parse_utm(df: pd.DataFrame) -> tuple:
    """
    Processing function for UTM location type.
    """
    coord_specs = df["location_coords_type"].str.strip("(assumed)").str.split()
    zone = coord_specs.apply(lambda x: x[1])
    datum = coord_specs.apply(lambda x: x[2])
    eastings, northings = split_coordinates(df["location_coords"])

    zone = zone.str.extract(r"^(\d+)", expand=False).values
    datum = datum.values
    eastings = eastings.values
    northings = northings.values

    lats, longs = [], []
    for i in range(len(df)):
        if datum[i] == "Unknown":
            lats.append(np.nan)
            longs.append(np.nan)
        else:
            utm_proj = pyproj.Proj(proj="utm", zone=zone[i], datum=datum[i])
            wgs84_proj = pyproj.Proj(proj="latlong", datum="WGS84")
            long, lat = pyproj.transform(
                utm_proj, wgs84_proj, eastings[i], northings[i]
            )
            lats.append(lat)
            longs.append(long)

    return lats, longs


# Function to check if coordinates are within Canada
def is_within_canada(latitude: float, longitude: float) -> bool:
    geolocator = Nominatim(user_agent="my-app")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    if location is None:
        return False
    country = location.raw["address"].get("country")
    return country == "Canada"


# These two are in the correct position
lat_lng_idx = new_df["location_coords_type"] == "Lat/lng"
lat_lng_dd_idx = new_df["location_coords_type"] == "Lat/Long Decimal Degrees"

# These rows have their latitude and longitude reversed, so that will have to be caught
# during assignment
lat_lon_idx = new_df["location_coords_type"] == "LatLon"

# The UTM format has more cleaning to go through than the others; removing letters,
# the "assumed" word etc.
utm_idx = new_df["location_coords_type"].str.startswith("UTM")

lats_1, longs_1 = split_coordinates(new_df["location_coords"][lat_lng_idx])
new_df["latitude"] = lats_1
new_df["longitude"] = longs_1

lats_2, longs_2 = split_coordinates(new_df["location_coords"][lat_lng_dd_idx])
new_df["latitude"][lat_lng_dd_idx] = lats_2
new_df["longitude"][lat_lng_dd_idx] = longs_2

# Recall, the lats and longs in these rows are reversed, hence reverse assignment
longs_3, lats_3 = split_coordinates(new_df["location_coords"][lat_lon_idx])
new_df["latitude"].loc[lat_lon_idx] = lats_3
new_df["longitude"].loc[lat_lon_idx] = longs_3


lats_4, longs_4 = parse_utm(new_df[utm_idx])
new_df["latitude"].loc[utm_idx] = lats_4
new_df["longitude"].loc[utm_idx] = longs_4
# --------------------------------------------------------------------------------------
# Potential optimisation but unable to implement yet. Keep getting Index error or access
# before assignment.

# Use Numpy to conditionally apply functions
# new_df["latitude"], new_df["longitude"] = np.where(
#    df["location_coords_type"] == utm_idx,
#    parse_utm(new_df),
#    split_coordinates(new_df["location_coords_type"]),
# )

## Reversing lat_lon_idx values.

# new_df.loc[lat_lon_idx, ["latitude", "longitude"]] = new_df.loc[
#    lat_lon_idx, ["longitude", "latitude"]
# ].values
#
# --------------------------------------------------------------------------------------


## Latittude cannot be >90. If it is, it is swapped with longitude
new_df["latitude"] = abs(new_df["latitude"].astype(float))


idx = new_df["latitude"] > 90

new_df.loc[idx, ["latitude", "longitude"]] = new_df.loc[
    idx, ["longitude", "latitude"]
].values


# All longitude values should be <0 and all latitude should be >0 based on location.
new_df["longitude"] = -1 * abs(new_df["longitude"].astype(float))
new_df["latitude"] = abs(new_df["latitude"].astype(float))

new_df = new_df.dropna()

new_df["IsWithinCanada"] = new_df.apply(
    lambda row: is_within_canada(row["latitude"], row["longitude"]), axis=1
)

new_df = new_df.loc[new_df["IsWithinCanada"] == True]  # noqa: E712

new_df.drop(columns="IsWithinCanada").dropna().to_csv(
    "./demo/data/can_avs_lat_long_date.csv", index=False
)

time_taken = time.time() - start
print(
    "Time to taken for transform_ava_coords.py to run: {}s.".format(
        round(time_taken, 3)
    )
)
