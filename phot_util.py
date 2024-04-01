import pyphot
import numpy as np
import astropy.units as u
import astropy.constants as cs


def extract_info(mags, magerrs, bands):
    """get flux information from magnitudes and errors.

    Args:
        mags (float or list): magnitudes
        magerrs (float or list): errors
        bands (str or list): filter names, e.g. 'PS1_g' or 'SDSS_r', it should match the names in pyphot library.
    
    Returns:
        wave_effs (list): effective wavelengths
        fluxes (list): fluxes
        flux_errs (list): flux errors
        bps (list): bandpasses
    """
    wave_effs = get_effective_wavelength(bands)
    fluxes, flux_errs = mag_to_flux(mags, magerrs, bands)
    bps = get_bandpass(bands)
    return wave_effs, fluxes, flux_errs, bps


def get_effective_wavelength(filter_names):
    lib = pyphot.get_library()
    if isinstance(filter_names, str):
        filter_name = filter_names
        _filter = lib[filter_name]
        wave_eff = _filter.cl.to('AA').value
        return wave_eff
    elif isinstance(filter_names, list):
        wave_effs = []
        for filter_name in filter_names:
            _filter = lib[filter_name]
            wave_eff = _filter.cl.to('AA').value
            wave_effs.append(wave_eff)
        return wave_effs


def get_bandpass(filter_names):
    lib = pyphot.get_library()
    if isinstance(filter_names, str):
        filter_name = filter_names
        _filter = lib[filter_name]
        bp = _filter.width.to('AA').value
        return bp
    elif isinstance(filter_names, list):
        bps = []
        for filter_name in filter_names:
            _filter = lib[filter_name]
            bp = _filter.width.to('AA').value
            bps.append(bp)
        return bps


def mag_to_flux(mags, mag_errs, bands):
    def mag_to_flux_single(mag, mag_err, band):
        if 'PS1_' in band or 'SDSS_' in band or 'GALEX_' in band:
            # Get flux from AB mag
            flux, flux_err = mag_to_flux_AB(mag, mag_err)
            wave_eff = get_effective_wavelength(band)
            # Get effective wavelength for bandpass
            flux_lambda = convert_f_nu_to_f_lambda(flux, wave_eff)
            flux_lambda_err = convert_f_nu_to_f_lambda(flux_err, wave_eff)
        else:
            f0 = get_zero_flux(band)
            flux_lambda = 10 ** (-0.4 * mag) * f0
            flux_lambda_err = abs(-0.4 * flux_lambda * np.log(10) * mag_err)
        return flux_lambda, flux_lambda_err
    if isinstance(mags, float):
        return mag_to_flux_single(mags, mag_errs, bands)
    elif isinstance(mags, list):
        fluxes = []
        flux_errs = []
        for mag, mag_err, band in zip(mags, mag_errs, bands):
            flux, flux_err = mag_to_flux_single(mag, mag_err, band)
            fluxes.append(flux)
            flux_errs.append(flux_err)
        return fluxes, flux_errs


def mag_to_flux_AB(mag, mag_err):
    """Calculate flux in erg s-1 cm-2 Hz-1."""
    flux = 10 ** (-.4 * (mag + 48.6))
    flux_err = abs(-.4 * flux * np.log(10) * mag_err)
    return flux, flux_err


def convert_f_nu_to_f_lambda(f, l):
    """Convert flux from erf s-1 cm-2 Hz-1 to erg s-1 cm-2 AA-1.
        f: flux in erg s-1 cm-2 Hz-1
        l: wavelength in AA
    """
    return f * cs.c.to(u.AA / u.s).value / l ** 2


def get_zero_flux(band):
    lib = pyphot.get_library()
    if isinstance(band, str):
        filt = lib[band]
        f0 = filt.Vega_zero_flux.to('erg/(AA * cm ** 2 * s)').value
        return f0
    elif isinstance(band, list):
        f0s = []
        for b in band:
            filt = lib[b]
            f0 = filt.Vega_zero_flux.to('erg/(AA * cm ** 2 * s)').value
            f0s.append(f0)
        return f0s
