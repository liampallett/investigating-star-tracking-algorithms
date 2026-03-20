"""
Query the Hipparcos catalogue via VizieR for star data with magnitudes < 6.0. Save to data/hipparcos_vmag6.csv for manipulation.
"""
from astroquery.vizier import Vizier
from astropy.io import ascii

vizier = Vizier(columns=["HIP", "RAICRS", "DEICRS", "Vmag"], # Only necessary positional elements
                column_filters={"Vmag": "<6.0"}) # Same values as discussed in the paper (Max Visible Magnitude of 6.0)
vizier.ROW_LIMIT = -1 # Set no row limit (ensuring we get all the data)

catalogue_list = vizier.query_constraints(catalog="I/239/hip_main") # Main Hipparcos catalogue (same as research paper)

ascii.write(catalogue_list[0],"data/hipparcos_vmag6.csv", format="csv", overwrite=True)