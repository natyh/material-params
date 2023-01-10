"""
Color Experimental
==================

Contains experimental functions related to RGB processing. These functions are
in a separate module since they are not (and may never be) ready for wider use.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright Naty Hoffman 2023

import numpy as np
import colour
import color_ior

from colour.models import RGB_COLOURSPACES

# Shared color data
cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

# Data for ACEScg (AP1) RGB space
d60_spd = colour.SDS_ILLUMINANTS["D60"]
d60_wp = colour.XYZ_to_xy(colour.sd_to_XYZ(d60_spd))
ap1_wp = RGB_COLOURSPACES["ACEScg"].whitepoint # Not quite D60
ap1_xyz2rgb = RGB_COLOURSPACES["ACEScg"].matrix_XYZ_to_RGB

# Data for sRGB / Rec.709 RGB space
d65_spd = colour.SDS_ILLUMINANTS["D65"]
srgb_wp = RGB_COLOURSPACES["sRGB"].whitepoint # D65
srgb_xyz2rgb = RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB

# Normalization factors to correct for inaccuracies in the color math.
unity_wavelengths = np.arange(360, 781, 20)
unity_vals = np.ones(unity_wavelengths.shape)
unity_spectral_dict = dict(zip(unity_wavelengths, unity_vals))
unity_spd = colour.SpectralDistribution(unity_spectral_dict, name="Unity")
unity_XYZ_ap1 = colour.sd_to_XYZ(unity_spd, cmfs, d60_spd) / 100
# NOTE: Need chromatic adaptation since D60 and AP1 have slightly different white points
unity_RGB_ap1 = colour.XYZ_to_RGB(unity_XYZ_ap1, d60_wp, ap1_wp, ap1_xyz2rgb)
norm_RGB_ap1 = 1.0 / unity_RGB_ap1
unity_XYZ_srgb = colour.sd_to_XYZ(unity_spd, cmfs, d65_spd) / 100
unity_RGB_srgb = colour.XYZ_to_RGB(unity_XYZ_srgb, srgb_wp, srgb_wp, ap1_xyz2rgb)
norm_RGB_srgb = 1.0 / unity_RGB_srgb

def spectral_ior_to_rgb_ap1_ior(wavelength_vals, c_ior_data):
    """
    This function uses a theoretically dubious method to compute RGB complex IOR values from
    spectral values. The IOR values are treated as spectral reflectance values: multiplied with
    the appropriate white point illuminant SPD, convolved with CMFs and then transformed into
    the appropriate RGB space.
    In this case, the RGB space used is ACEScg (AP1).

    Parameters
    ----------
    wavelength_vals: numeric array
        Array of wavelength values.
    c_ior_data: complex number array
        Array of complex index of refraction values at the wavelengths in wavelength_vals.
    """

    assert len(wavelength_vals) == len(c_ior_data), "Vectors are not the same length."
    r_data = np.empty(len(c_ior_data), dtype=np.float64)
    i_data = np.empty(len(c_ior_data), dtype=np.float64)
    for i in range(len(c_ior_data)):
        r_data[i] = np.real(c_ior_data[i])
        i_data[i] = np.imag(c_ior_data[i])
    r_spectral_dict = dict(zip(wavelength_vals, r_data))
    i_spectral_dict = dict(zip(wavelength_vals, i_data))
    r_spd = colour.SpectralDistribution(r_spectral_dict, name="SampleReal")
    i_spd = colour.SpectralDistribution(i_spectral_dict, name="SampleImag")

    r_XYZ = colour.sd_to_XYZ(r_spd, cmfs, d60_spd) / 100
    i_XYZ = colour.sd_to_XYZ(i_spd, cmfs, d60_spd) / 100

    # NOTE: Need chromatic adaptation since D60 and AP1 have slightly different white points
    r_RGB = colour.XYZ_to_RGB(r_XYZ, d60_wp, ap1_wp, ap1_xyz2rgb)
    i_RGB = colour.XYZ_to_RGB(i_XYZ, d60_wp, ap1_wp, ap1_xyz2rgb)

    # Normalize to account for minor inaccuracies in color math
    r_RGB = r_RGB * norm_RGB_ap1
    i_RGB = i_RGB * norm_RGB_ap1

    return r_RGB, i_RGB

def spectral_ior_to_rgb_srgb_ior(wavelength_vals, c_ior_data):
    """
    This function uses a theoretically dubious method to compute RGB complex IOR values from
    spectral values. The IOR values are treated as spectral reflectance values: multiplied with
    the appropriate white point illuminant SPD, convolved with CMFs and then transformed into
    the appropriate RGB space.
    In this case, the RGB space used is sRGB / Rec.709.

    Parameters
    ----------
    wavelength_vals: numeric array
        Array of wavelength values.
    c_ior_data: complex number array
        Array of complex index of refraction values at the wavelengths in wavelength_vals.
    """

    assert len(wavelength_vals) == len(c_ior_data), "Vectors are not the same length."
    r_data = np.empty(len(c_ior_data), dtype=np.float64)
    i_data = np.empty(len(c_ior_data), dtype=np.float64)
    for i in range(len(c_ior_data)):
        r_data[i] = np.real(c_ior_data[i])
        i_data[i] = np.imag(c_ior_data[i])
    r_spectral_dict = dict(zip(wavelength_vals, r_data))
    i_spectral_dict = dict(zip(wavelength_vals, i_data))
    r_spd = colour.SpectralDistribution(r_spectral_dict, name="SampleReal")
    i_spd = colour.SpectralDistribution(i_spectral_dict, name="SampleImag")

    r_XYZ = colour.sd_to_XYZ(r_spd, cmfs, d65_spd) / 100
    i_XYZ = colour.sd_to_XYZ(i_spd, cmfs, d65_spd) / 100
    r_RGB = np.dot(srgb_xyz2rgb, r_XYZ) # When no chromatic adaptation, equivalent to colour.XYZ_to_RGB and much faster
    i_RGB = np.dot(srgb_xyz2rgb, i_XYZ) # When no chromatic adaptation, equivalent to colour.XYZ_to_RGB and much faster

    # Normalize to account for minor inaccuracies in color math
    r_RGB = r_RGB * norm_RGB_srgb
    i_RGB = i_RGB * norm_RGB_srgb

    return r_RGB, i_RGB

def rgb_ap1_f0_to_scalar_eta(rgb_f0):
    """
    This function estimates a scalar eta value for a dielectric given its RGB F0 in the AP1/ACEScg space.

    Parameters
    ----------
    rgb_f0: array-like
        RGB F0 value for the dielectric in the AP1/ACEScg space.
    """
    sqrt_lum_f0 = np.sqrt(color_ior.ap1_luminance(rgb_f0))
    return (1.0 + sqrt_lum_f0) / (1.0 - sqrt_lum_f0)

def rgb_srgb_f0_to_scalar_eta(rgb_f0):
    """
    This function estimates a scalar eta value for a dielectric given its RGB F0 in the sRGB space.

    Parameters
    ----------
    rgb_f0: array-like
        RGB F0 value for the dielectric in the sRGB space.
    """
    sqrt_lum_f0 = np.sqrt(color_ior.srgb_luminance(rgb_f0))
    return (1.0 + sqrt_lum_f0) / (1.0 - sqrt_lum_f0)
