import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import SkyCoord

galaxies_to_look_for = [
    'J081239.52+483645.3',
    'J085946.92+392305.6',
    'J120122.32+021108.5',
    'J141454.13-020822.9',
    'J210455.31-003522.2'
]

mass_dataset = 'totlgm_dr7_v5_2.fit'
info_dataset = 'gal_info_dr7_v5_2.fit'
with fits.open(info_dataset) as hdul:
    data = hdul[1].data
    ras = np.asarray(data['ra'], dtype=float)   # degrees
    decs = np.asarray(data['dec'], dtype=float) # degrees

    # Remove invalid placeholder or malformed coordinates before SkyCoord.
    valid = (
        np.isfinite(ras)
        & np.isfinite(decs)
        & (ras >= 0.0)
        & (ras < 360.0)
        & (decs >= -90.0)
        & (decs <= 90.0)
    )

    if not np.all(valid):
        n_bad = np.size(valid) - np.count_nonzero(valid)
        print(f"Skipping {n_bad} rows with invalid RA/Dec values.")

    ras = ras[valid]
    decs = decs[valid]

    # Convert to SDSS-style names: JHHMMSS.ss+DDMMSS.s
    coords = SkyCoord(ra=ras * u.deg, dec=decs * u.deg, frame="icrs")
    ra_str = coords.ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
    dec_str = coords.dec.to_string(unit=u.deg, sep='', precision=1, pad=True, alwayssign=True)

    sdss_names = np.char.add('J', np.char.add(ra_str, dec_str))

    # now we can find the indices of the galaxies we care about
    indices = []
    for name in galaxies_to_look_for:
        # Find galaxy with the closest coordinates to the name we're looking for
        target_coord = SkyCoord.from_name(name)
        target_ra = target_coord.ra.deg
        target_dec = target_coord.dec.deg
        # Calculate the angular separation between the target and all galaxies in the dataset
        separations = np.sqrt((ras - target_ra) ** 2 + (decs - target_dec) ** 2)
        closest_index = np.argmin(separations)
        if separations[closest_index] < 1.0 / 3600.0:  # within 1 arcsecond
            indices.append(closest_index)
            print(f"Galaxy {name} found at index {closest_index} with separation {separations[closest_index] * 3600:.2f} arcseconds.")
        else:
            print(f"Galaxy {name} not found in dataset.")

    # Now read the mass dataset and extract the masses for the galaxies we found
    with fits.open(mass_dataset) as mass_hdul:
        mass_data = mass_hdul[1].data
        masses = mass_data['AVG'][indices]  # assuming 'AVG' is the column with the mass values
        for name, mass in zip(galaxies_to_look_for, masses):
            print(f"Galaxy: {name}, Mass: {mass}")