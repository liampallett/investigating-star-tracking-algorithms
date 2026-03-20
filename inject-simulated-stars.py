"""
Simulate sensor noise by introducing fake stars to replicate sensor noise.
Set a bounding box first, based on satellite Dec and RA. Check for stars visible ±(FOV)/2 deg from this point, this will be our "image".
"""
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord

HIPPARCOS = pd.read_csv("data/hipparcos_vmag6.csv")
HIPPARCOS = HIPPARCOS.dropna(subset=["RAICRS", "DEICRS"])

def get_stars_in_fov(df, ra, dec, fov):
    """
    Finds all visible stars within a FOV of a sensor.
    :param df: Frame of data from the Hipparcos catalogue.
    :param ra: Right ascension of the satellite in degrees.
    :param dec: Declination of the satellite in degrees.
    :param fov: FOV of the sensor. (Default 4 deg as seen in the research paper)
    :return: A Pandas DataFrame object containing only the stars visible in the sensor FOV.
    """
    ra_range = {
        "minBound": ra - (fov / 2),
        "maxBound": ra + (fov / 2)
    }
    dec_range = {
        "minBound": max(-90, dec - fov / 2),
        "maxBound": min(90, dec + fov / 2)
    }

    if ra_range["minBound"] < ra_range["maxBound"]:
        filtered_frame = df[(df["RAICRS"] > ra_range["minBound"]) & (df["RAICRS"] < ra_range["maxBound"]) &
                            (df["DEICRS"] > dec_range["minBound"]) & (df["DEICRS"] < dec_range["maxBound"])]
    else:
        filtered_frame = df[((df["RAICRS"] > ra_range["minBound"]) | (df["RAICRS"] < ra_range["maxBound"])) &
                            (df["DEICRS"] > dec_range["minBound"]) & (df["DEICRS"] < dec_range["maxBound"])]

    observer_coords = SkyCoord(ra=ra, dec=dec, unit="deg")
    star_coords = SkyCoord(ra=filtered_frame["RAICRS"].values, dec=filtered_frame["DEICRS"].values, unit="deg")

    separations = observer_coords.separation(star_coords)
    filtered_frame = filtered_frame[separations.deg <= fov / 2]

    return filtered_frame


def generate_false_stars(ra, dec, max_no_false, fov):
    """
    Generates a certain number of false stars.
    :param ra: Right ascending of the satellite in degrees.
    :param dec: Declination of the satellite in degrees.
    :param max_no_false: Maximum number of false stars to generate. Real number is random between 1 and max_no_false.
    :param fov: FOV of the image sensor in degrees.
    :return: A Pandas DataFrame object containing the false star data.
    """
    no_false = np.random.randint(1, max_no_false)
    ra_range = {
        "minBound": ra - (fov / 2),
        "maxBound": ra + (fov / 2)
    }
    dec_range = {
        "minBound": max(-90, dec - fov / 2),
        "maxBound": min(90, dec + fov / 2)
    }

    false_stars = []

    for i in range(no_false):
        star_hip = -1
        star_ra = np.random.uniform(ra_range["minBound"], ra_range["maxBound"]) % 360
        star_dec = np.random.uniform(dec_range["minBound"], dec_range["maxBound"])
        star_vmag = np.random.normal(5, 0.5)
        star_vmag = np.clip(star_vmag, 1, 6)

        star = {
            "HIP": star_hip,
            "RAICRS": star_ra,
            "DEICRS": star_dec,
            "Vmag": star_vmag
        }

        false_stars.append(star)

    return pd.DataFrame(false_stars)


def run_simulation(repetitions=1000, sensor_fov=4, max_no_false_stars=4):
    """
    Simulates the satellite view a certain number of times for NN training.
    :param repetitions: Number of simulations to run. (Default 1000 as seen in the research paper)
    :param sensor_fov: FOV of the satellite sensor. (Default 4 deg as seen in the research paper)
    :param max_no_false_stars: Maximum number of false stars to generate per simulation. (Default 4 for suitable number to train NN)
    :return: A Pandas DataFrame object containing visible and fake star data for every simulation.
    """
    stars_frame_list = []
    catalogue_coords = SkyCoord(ra=HIPPARCOS["RAICRS"].values, dec=HIPPARCOS["DEICRS"].values, unit="deg")

    for i in range(repetitions):
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        filtered_frame = get_stars_in_fov(HIPPARCOS, ra, dec, sensor_fov)
        filtered_frame["label"] = 1
        false_frame = generate_false_stars(ra, dec, max_no_false_stars, sensor_fov)
        false_frame["label"] = 0

        stars_frame = pd.concat([filtered_frame, false_frame], ignore_index=True)
        star_coords = SkyCoord(ra=stars_frame["RAICRS"].values, dec=stars_frame["DEICRS"].values, unit="deg")
        idx, sep, _ = star_coords.match_to_catalog_sky(catalogue_coords)
        stars_frame["min_star_sep"] = sep.deg
        stars_frame_list.append(stars_frame)

    return pd.concat(stars_frame_list)


training_data = run_simulation(5000, 4, 5)

print(f"Training samples: {len(training_data)}")
print(training_data["label"].value_counts())

test_data = run_simulation(1000, 4, 5)

print(f"\nTest samples: {len(test_data)}")
print(test_data["label"].value_counts())

training_data.to_csv("data/training_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)