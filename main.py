import sys
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
# from . import phot_util
import phot_util


Vizier.ROW_LIMIT = -1
Vizier.columns = ['all']
Catalogs.ROW_LIMIT = -1
Catalogs.columns = ['all']


class GaiaInfo:
    def __init__(self, ra, dec, radius):
        self.ra = ra
        self.dec = dec
        self.radius = radius
        self.result_table = Gaia.query_object_async(coordinate=SkyCoord(ra=self.ra, dec=self.dec, unit=(u.deg, u.deg)), radius=self.radius)
        self.source_id = self.result_table['SOURCE_ID'][0]
        self.ra = self.result_table['ra'][0]
        self.dec = self.result_table['dec'][0]
        self.parallax = self.result_table['parallax'][0]
        self.parallax_error = self.result_table['parallax_error'][0]
        distance, distance_errlow, distance_errhigh = self.get_distance()
        self.distance = distance
        self.distance_errlow = distance_errlow
        self.distance_errhigh = distance_errhigh
        self.pmra = self.result_table['pmra'][0]
        self.pmdec = self.result_table['pmdec'][0]
        self.phot_g_mean_mag = self.result_table['phot_g_mean_mag'][0]
        self.phot_bp_mean_mag = self.result_table['phot_bp_mean_mag'][0]
        self.phot_rp_mean_mag = self.result_table['phot_rp_mean_mag'][0]
        self.phot_g_mean_mag_err, self.phot_bp_mean_mag_err, self.phot_rp_mean_mag_err = self.get_magerr()
        self.bp_rp = self.result_table['bp_rp'][0]

    def get_distance(self):
        if self.parallax > 0 and self.parallax_error > 0:
            pal_dis = np.random.normal(self.parallax, self.parallax_error)
            disdistance = 1 / pal_dis
            distance = np.median(disdistance)
            distance_errlow = np.percentile(disdistance, 16)
            distance_errhigh = np.percentile(disdistance, 84)
            return distance, distance_errlow, distance_errhigh
        else:
            return np.nan, np.nan, np.nan

    def get_magerr(self):
        phot_g_mean_flux = self.result_table['phot_g_mean_flux'][0]
        phot_g_mean_flux_error = self.result_table['phot_g_mean_flux_error'][0]
        phot_bp_mean_flux = self.result_table['phot_bp_mean_flux'][0]
        phot_bp_mean_flux_error = self.result_table['phot_bp_mean_flux_error'][0]
        phot_rp_mean_flux = self.result_table['phot_rp_mean_flux'][0]
        phot_rp_mean_flux_error = self.result_table['phot_rp_mean_flux_error'][0]
        phot_g_mean_mag_err = np.abs(2.5 / np.log(10) * phot_g_mean_flux_error / phot_g_mean_flux)
        phot_bp_mean_mag_err = np.abs(2.5 / np.log(10) * phot_bp_mean_flux_error / phot_bp_mean_flux)
        phot_rp_mean_mag_err = np.abs(2.5 / np.log(10) * phot_rp_mean_flux_error / phot_rp_mean_flux)
        return phot_g_mean_mag_err, phot_bp_mean_mag_err, phot_rp_mean_mag_err

    def get_gaia_info(self, keyword):
        return self.result_table[keyword][0]

    def __str__(self):
        return f'''
        RA: {self.ra}
        DEC: {self.dec}
        Parallax: {self.parallax}
        Parallax Error: {self.parallax_error}
        Distance: {self.distance} kpc
        Distance Error Low: {self.distance_errlow} kpc
        Distance Error High: {self.distance_errhigh} kpc
        Proper Motion RA: {self.pmra}
        Proper Motion DEC: {self.pmdec}
        G Mean Magnitude: {self.phot_g_mean_mag}
        BP Mean Magnitude: {self.phot_bp_mean_mag}
        RP Mean Magnitude: {self.phot_rp_mean_mag}
        G Mean Magnitude Error: {self.phot_g_mean_mag_err}
        BP Mean Magnitude Error: {self.phot_bp_mean_mag_err}
        RP Mean Magnitude Error: {self.phot_rp_mean_mag_err}
        BP-RP: {self.bp_rp}
        '''
    
    def __repr__(self):
        return self.__str__()


class SED:
    __apass_mags = ['Vmag', 'Bmag', 'g_mag', 'r_mag', 'i_mag']
    __apass_errs = ['e_Vmag', 'e_Bmag', 'e_g_mag', 'e_r_mag', 'e_i_mag']
    __apass_filters = ['GROUND_JOHNSON_V', 'GROUND_JOHNSON_B',
                       'SDSS_g', 'SDSS_r', 'SDSS_i']

    __wise_mags = ['W1mag', 'W2mag']
    __wise_errs = ['e_W1mag', 'e_W2mag']
    __wise_filters = ['WISE_RSR_W1', 'WISE_RSR_W2']

    __ps1_mags = ['gmag', 'rmag', 'imag', 'zmag', 'ymag']
    __ps1_errs = ['e_gmag', 'e_rmag', 'e_imag', 'e_zmag', 'e_ymag']
    __ps1_filters = ['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y']

    __tmass_mags = ['Jmag', 'Hmag', 'Kmag']
    __tmass_errs = ['e_Jmag', 'e_Hmag', 'e_Kmag']
    __tmass_filters = ['2MASS_J', '2MASS_H', '2MASS_Ks']

    __sdss_mags = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
    # __sdss_errs = ['e_gmag', 'e_rmag', 'e_imag']
    __sdss_errs = ['e_umag', 'e_gmag', 'e_rmag', 'e_imag', 'e_zmag']
    # __sdss_filters = ['SDSS_g', 'SDSS_r', 'SDSS_i']
    __sdss_filters = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z']

    __galex_mags = ['FUV', 'NUV']
    __galex_errs = ['e_FUV', 'e_NUV']
    __galex_filters = ['GALEX_FUV', 'GALEX_NUV']

    __tess_mags = ['Tmag']
    __tess_errs = ['e_Tmag']
    __tess_filters = ['TESS']

    __irac_mags = ['_3.6mag', '_4.5mag']
    __irac_errs = ['e_3.6mag', 'e_4.5mag']
    __irac_filters = ['SPITZER_IRAC_36', 'SPITZER_IRAC_45']

    catalogs = {
        'APASS': [
            'II/336/apass9', list(zip(__apass_mags, __apass_errs,
                                      __apass_filters))
        ],
        'Wise': [
            'II/328/allwise', list(zip(__wise_mags, __wise_errs,
                                       __wise_filters))
        ],
        'Pan-STARRS': [
            'II/349/ps1', list(zip(__ps1_mags, __ps1_errs, __ps1_filters))
        ],
        # 'Gaia': [
        #     'I/355/gaiadr3', list(zip(__gaia_mags, __gaia_errs, __gaia_filters))
        # ],
        '2MASS': [
            'II/246/out', list(zip(__tmass_mags, __tmass_errs, __tmass_filters))
        ],
        'SDSS': [
            'V/147/sdss12', list(zip(__sdss_mags, __sdss_errs, __sdss_filters))
        ],
        'GALEX': [
            'II/312/ais', list(zip(__galex_mags, __galex_errs, __galex_filters))
        ],
        'TESS': [
            'TIC', list(zip(__tess_mags, __tess_errs, __tess_filters))
        ],
        'STROMGREN_PAUNZ': [
            'J/A+A/580/A23/catalog', -1
        ],
        'STROMGREN_HAUCK': [
            'II/215/catalog', -1
        ],
        'MERMILLIOD': [
            'II/168/ubvmeans', -1
        ],
    }

    def __init__(self, gaia_info, radius=5 * u.arcsec):
        self.gaia_info = gaia_info
        self.gaia_id = gaia_info.source_id
        self.ra = gaia_info.ra
        self.dec = gaia_info.dec
        self.radius = radius
        self.gaia_query()
        self.vizier_table_names = self.get_vizier_table_names()
        print(self.vizier_table_names)
        self.vizier_tables = self.get_vizier_tables()
        print(self.vizier_tables)
        self.mags = []
        self.mag_errs = []
        self.filters = []
        self.data_sources = []
        self._retrieve_gaia()
        self._retrieve_from('APASS')
        self._retrieve_from('Wise')
        # self._retrieve_from_wise()
        # self._retrieve_from('Pan-STARRS')
        self._retrieve_from_ps1()
        self._retrieve_from('2MASS')
        # self._retrieve_from_tmass()
        self._retrieve_from('SDSS')
        # self._retrieve_from_sdss()
        self._retrieve_from('GALEX')
        # self._retrieve_from_galex()
        self._retrieve_from_tess()
        wave_eff, flux, flux_err, bandpass = phot_util.extract_info(
            self.mags, self.mag_errs, self.filters)
        self.wave_eff = wave_eff
        self.flux = flux
        self.flux_err = flux_err
        self.bandpass = bandpass

    def gaia_query(self):
        """Query Gaia to get different catalog IDs."""
        # cats = ['tycho2', 'panstarrs1', 'sdssdr9',
        #         'allwise', 'tmass', 'apassdr9']
        # names = ['tycho', 'ps', 'sdss', 'allwise', 'tmass', 'apass']
        cats = ['panstarrs1', 'sdssdr9',
                ]
        names = ['ps', 'sdss']
        IDS = {
            'TYCHO2': '',
            'APASS': '',
            '2MASS': '',
            'Pan-STARRS': '',
            'SDSS': '',
            'Wise': '',
            'Gaia': self.gaia_id,
            'SkyMapper': self.gaia_id,
        }
        for c, n in zip(cats, names):
            if c == 'apassdr9':
                cat = 'APASS'
            elif c == 'tmass':
                cat = '2MASS'
                c = 'tmass'
            elif c == 'panstarrs1':
                cat = 'Pan-STARRS'
            elif c == 'sdssdr9':
                cat = 'SDSS'
            elif c == 'allwise':
                cat = 'Wise'
            elif c == 'tycho2':
                cat = 'TYCHO2'
            query = f"""
            SELECT
                {n}.original_ext_source_id
            FROM
                gaiadr2.gaia_source AS gaia
            JOIN
                gaiadr2.{c}_best_neighbour AS {n}
            ON gaia.source_id={n}.source_id
            WHERE
                gaia.source_id={self.gaia_id}
            """
            j = Gaia.launch_job_async(query)
            r = j.get_results()
            if len(r):
                IDS[cat] = r[0][0]
            else:
                IDS[cat] = 'skipped'
                print('Star not found in catalog ' + cat, end='.\n')
        IDS['GALEX'] = ''
        IDS['TESS'] = ''
        IDS['MERMILLIOD'] = ''
        IDS['STROMGREN_PAUNZ'] = ''
        IDS['STROMGREN_HAUCK'] = ''
        self.ids = IDS
        return IDS

    def _retrieve_gaia(self):
        filtnames = ['Gaia_BP', 'Gaia_RP', 'Gaia_G']
        mags = [self.gaia_info.phot_bp_mean_mag, 
                self.gaia_info.phot_rp_mean_mag, 
                self.gaia_info.phot_g_mean_mag]
        mag_errs = [self.gaia_info.phot_bp_mean_mag_err, 
                    self.gaia_info.phot_rp_mean_mag_err, 
                    self.gaia_info.phot_g_mean_mag_err]
        for mag, mag_err, filtername in zip(mags, mag_errs, filtnames):
            self._add_mag(mag, mag_err, filtername)
            self.data_sources.append('Gaia')

    def _retrieve_from(self, data_source):
        table_name = self.catalogs[data_source][0]
        if table_name not in self.vizier_tables.keys():
            print(data_source, table_name, 'Not aviailable')
            return
        cat = self.vizier_tables[table_name]
        if cat is None:
            return
        for magname, mag_errname, filtername in self.catalogs[data_source][1]:
            mag = cat[magname][0]
            mag_err = cat[mag_errname][0]
            if not np.isfinite(mag) or not np.isfinite(mag_err):
                continue
            self._add_mag(mag, mag_err, filtername)
            self.data_sources.append(data_source)

    def _retrieve_from_ps1(self):
        if self.ids['Pan-STARRS'] == 'skipped':
            return
        table_name = self.catalogs['Pan-STARRS'][0]
        cat = self.vizier_tables[table_name]
        if cat is None:
            return
        quality = cat['Qual'][0]
        is_good_quality = (quality & 4) or (quality & 16)
        is_good_quality = is_good_quality and not (quality & 128)
        if not is_good_quality:
            return
        for magname, mag_errname, filtername in self.catalogs['Pan-STARRS'][1]:
            mag = cat[magname][0]
            mag_err = cat[mag_errname][0]
            if not np.isfinite(mag) or not np.isfinite(mag_err):
                continue
            self._add_mag(mag, mag_err, filtername)
            self.data_sources.append('Pan-STARRS')

    def _add_mag(self, mag, mag_err, filtername):
        self.mags.append(mag)
        self.mag_errs.append(mag_err)
        self.filters.append(filtername)

    def get_vizier_tables(self):
        cats = Vizier.query_region(
            SkyCoord(ra=self.ra, dec=self.dec, unit=(u.deg, u.deg), frame='icrs'), radius=self.radius, catalog=self.vizier_table_names)
        return cats

    def get_vizier_table_names(self):
        vizier_table_names = []
        for catalog in self.catalogs:
            vizier_table_name = self.catalogs[catalog][0]
            vizier_table_names.append(vizier_table_name)
        return vizier_table_names

    def get_TIC(self):
        cat = Catalogs.query_region(
            SkyCoord(ra=self.ra, dec=self.dec, unit=(u.deg, u.deg), frame='icrs'), radius=self.radius, catalog='TIC')
        return cat

    def _retrieve_from_tess(self):
        tic = self.get_TIC()
        print(self.catalogs['TESS'][1][0])
        magname, errname, filtername = self.catalogs['TESS'][1][0]
        mag = tic[magname][0]
        mag_err = tic[errname][0]
        if not np.isfinite(mag) or not np.isfinite(mag_err):
            return
        self._add_mag(mag, mag_err, filtername)
        self.data_sources.append('TESS')
        # tic.sort('dstArcSec')

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(111)
        waves = np.array(self.wave_eff)
        fluxes = np.array(self.flux)
        flux_errs = np.array(self.flux_err)
        bandpasses = np.array(self.bandpass)
        filtnames = np.array(self.filters)
        data_sources = np.array(self.data_sources)
        set_data_sources = set(data_sources)
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'X', 'd', 'H', '1', '2', '3', '4', '8']
        # colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black']
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        for ind, data_source in enumerate(set_data_sources):
            mask = data_sources == data_source
            color = colors[ind]
            tmp_waves = waves[mask]
            tmp_fluxes = fluxes[mask]
            tmp_flux_errs = flux_errs[mask]
            tmp_bandpasses = bandpasses[mask]
            tmp_filtnames = filtnames[mask]
            count = 0
            for wave, flux, flux_err, bandpass, filtn in zip(tmp_waves, tmp_fluxes, tmp_flux_errs, tmp_bandpasses, tmp_filtnames):
                ax.errorbar(wave, flux, yerr=flux_err, xerr=bandpass, label=filtn, fmt='o', color=color, marker=markers[count], elinewidth=0.5)
                count += 1
            # marker = markers[ind]
            # ax.errorbar(waves[mask], fluxes[mask], yerr=flux_errs[mask], xerr=bandpasses[mask], label=data_source, fmt='o', marker=marker)
        ax.set_xlabel('Wavelength (Angstrom)')
        ax.set_ylabel('Flux (erg/s/cm2/Angstrom)')
    # def get_flux_from_mag(self):



class XSpec:
    def __init__(self, source_id):
        self.source_id = source_id
        datalink = Gaia.load_data([self.source_id,], retrieval_type='ALL', data_release='Gaia DR3', data_structure='COMBINED')
        # print(datalink)
        keyname = f'XP_SAMPLED-Gaia DR3 {source_id}.xml'
        print('datalink keys =', datalink.keys())
        data_xp = datalink[keyname]
        self.xspec = data_xp
        self.wave = self.xspec[0].array['wavelength'].astype(float).data * 10
        self.flux = self.xspec[0].array['flux'].astype(float).data * 100
        self.flux_err = self.xspec[0].array['flux_error'].astype(float).data * 100
        self.flux_unit = u.erg / u.s / u.cm**2 / u.AA

        # keyname_CONTINUOUS = f'XP_CONTINUOUS-Gaia DR3 {source_id}.xml'
        # data_xp_CONTINUOUS = datalink[keyname_CONTINUOUS]

        # keyname_MSC = f'MCMC_MSC-Gaia DR3 {source_id}.xml'
        # data_xp_MSC = datalink[keyname_MSC]

        # keyname_GSPPHOT = f'MCMC_GSPPHOT-Gaia DR3 {source_id}.xml'
        # data_xp_GSPPHOT = datalink[keyname_GSPPHOT]

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(111)
        ax.errorbar(self.wave, self.flux, yerr=self.flux_err, label='Gaia', color='grey')
        ax.set_xlabel('Wavelength (Angstrom)')
        ax.set_ylabel('Flux (erg/s/cm2/Angstrom)')
        ax.legend()
        return ax


def main():
    try:
        ra = float(sys.argv[1])
        dec = float(sys.argv[2])
    except ValueError as e:
        strra = sys.argv[1]
        strdec = sys.argv[2]
        coo = SkyCoord(strra, strdec, unit=(u.hourangle, u.deg), frame='icrs')
        ra = coo.ra.deg
        dec = coo.dec.deg
    radius = 5 * u.arcsec
    gaia_info = GaiaInfo(ra, dec, radius)
    source_id = gaia_info.source_id
    xspec = XSpec(source_id)
    ax = xspec.plot()
    sed = SED(gaia_info)
    sed.plot(ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()