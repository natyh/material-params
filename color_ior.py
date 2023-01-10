"""
Color IOR
============

Contains functions related to RGB IOR processing such as converting spectral IOR
to RGB reflectance values. Depends on the spectral_ior module.
"""
import numpy as np
import spectral_ior
import colour

from colour.models import RGB_COLOURSPACES

# Shared color data
cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

# Data for ACEScg (AP1) RGB space
d60_spd = colour.SDS_ILLUMINANTS["D60"]
d60_wp = colour.XYZ_to_xy(colour.sd_to_XYZ(d60_spd))
ap1_wp = RGB_COLOURSPACES["ACEScg"].whitepoint # Not quite D60
ap1_xyz2rgb = RGB_COLOURSPACES["ACEScg"].matrix_XYZ_to_RGB
ap1_rgb2xyz = RGB_COLOURSPACES["ACEScg"].matrix_RGB_to_XYZ

# Data for sRGB / Rec.709 RGB space
d65_spd = colour.SDS_ILLUMINANTS["D65"]
srgb_wp = RGB_COLOURSPACES["sRGB"].whitepoint # D65
srgb_xyz2rgb = RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB
srgb_rgb2xyz = RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ

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

def lstar(linval):
    CIE_E = 216 / 24389
    CIE_K = 24389 / 27
    if (linval <= CIE_E):
        return CIE_K * linval
    else:
        return 1.16 * linval ** (1.0 / 3.0) - 0.16

def rms_l(vec1, vec2):
    """
    Compute the RMS error in L* space between two scalar vectors (assumed to be scalar color data, e.g.
    single-wavelength samples or RGB channels)

    Parameters
    ----------
    vec1: numeric array
        First array.
    vec2: numeric array
        Second array.
    """
    assert len(vec1) == len(vec2), "Vectors are not the same length."
    sum_sq = 0.0
    for i in range(0, len(vec1)):
        diff = lstar(vec1[i]) - lstar(vec2[i])
        sum_sq = sum_sq + (diff * diff)
    return np.sqrt(sum_sq / len(vec1))

def wrms_l(vec1, vec2, vecw):
    """
    Compute the Weighted RMS error in L* space between two scalar vectors (assumed to be scalar color data, e.g.
    single-wavelength samples or RGB channels)

    Parameters
    ----------
    vec1: numeric array
        First array.
    vec2: numeric array
        Second array.
    vecw: numeric array
        Weights array.
    """
    assert len(vec1) == len(vec2), "Vectors 1 and 2 are not the same length."
    assert len(vec1) == len(vecw), "Vector 1 and the weight vector are not the same length."
    sum_sq = 0.0
    sum_w = 0.0
    for i in range(0, len(vec1)):
        diff = lstar(vec1[i]) - lstar(vec2[i])
        sum_sq = sum_sq + vecw[i] * (diff * diff)
        sum_w = sum_w + vecw[i]
    return np.sqrt(sum_sq / sum_w)

# NOTE - this code assumes colors are in AP1 space, illuminated with the AP1 reference illuminant!!!
def ap1_de2000(rgb1, rgb2):
    # NOTE - we are multiplying by the RGB_to_XYZ matrix instead of using the colour.RGB_to_XYZ() function.
    # In the case where there is no chromatic adaptation, this is equivalent. For AP1 we should actually do
    # color adaptation since the chromaticity coordinates of the D60 illuminant and the AP1 white point are
    # slightly different. For the purposes of computing delta-E error this should not matter as much, and doing
    # it this way is much faster, which is important when processing entire images.
    xyz1 = np.dot(ap1_rgb2xyz, rgb1)
    xyz2 = np.dot(ap1_rgb2xyz, rgb2)
    lab1 = colour.XYZ_to_Lab(xyz1, illuminant=ap1_wp)
    lab2 = colour.XYZ_to_Lab(xyz2, illuminant=ap1_wp)
    return colour.difference.delta_E_CIE2000(lab1, lab2)

# NOTE - this code assumes colors are in AP1 space, illuminated with the AP1 reference illuminant!!!
def rms_ap1_de2000(a_r1, a_g1, a_b1, a_r2, a_g2, a_b2):
    """
    Compute the RMS of the CIE deltaE 2000 error between two arrays of colors, each expressed as separate arrays per color channel.

    Parameters
    ----------
    a_r1: numeric array
        First R channel array.
    a_g1: numeric array
        First G channel array.
    a_b1: numeric array
        First B channel array.
    a_r2: numeric array
        Second R channel array.
    a_g2: numeric array
        Second G channel array.
    a_b2: numeric array
        Second B channel array.
    """
    assert len(a_r1) == len(a_g1) and len(a_g1) == len (a_b1), "First color channel vectors are not the same length."
    assert len(a_r2) == len(a_g2) and len(a_g2) == len (a_b2), "Second color channel vectors are not the same length."
    assert len(a_r1) == len(a_r2), "First and second color vectors are not the same length."
    sum_sq = 0.0
    for i in range(0, len(a_r1)):
        rgb1 = (a_r1[i], a_g1[i], a_b1[i])
        rgb2 = (a_r2[i], a_g2[i], a_b2[i])
        diff = ap1_de2000(rgb1, rgb2)
        sum_sq = sum_sq + (diff * diff)
    return np.sqrt(sum_sq / len(a_r1))

# NOTE - this code assumes colors are in AP1 space, illuminated with the AP1 reference illuminant!!!
def max_ap1_de2000(a_r1, a_g1, a_b1, a_r2, a_g2, a_b2):
    """
    Find the largest CIE deltaE 2000 error between two arrays of colors, each expressed as separate arrays per color channel.
    Returns a tuple containing the largest error and the index in the array where the largest error occurs.
    If multiple array indices have the same largest error the last one will be returned.

    Parameters
    ----------
    a_r1: numeric array
        First R channel array.
    a_g1: numeric array
        First G channel array.
    a_b1: numeric array
        First B channel array.
    a_r2: numeric array
        Second R channel array.
    a_g2: numeric array
        Second G channel array.
    a_b2: numeric array
        Second B channel array.
    """
    assert len(a_r1) == len(a_g1) and len(a_g1) == len (a_b1), "First color channel vectors are not the same length."
    assert len(a_r2) == len(a_g2) and len(a_g2) == len (a_b2), "Second color channel vectors are not the same length."
    assert len(a_r1) == len(a_r2), "First and second color vectors are not the same length."
    max_err = 0.0
    index_max_err = 0
    for i in range(0, len(a_r1)):
        rgb1 = (a_r1[i], a_g1[i], a_b1[i])
        rgb2 = (a_r2[i], a_g2[i], a_b2[i])
        diff = ap1_de2000(rgb1, rgb2)
        if diff > max_err:
            max_err = diff
            index_max_err = i
    return (max_err, index_max_err)

def spectral_ior_to_spd_f0(wavelength_vals, c_ior_data, e = 1.0):
    assert len(wavelength_vals) == len(c_ior_data), "Vectors are not the same length."
    f_data = np.empty(len(c_ior_data), dtype=np.float64)
    for i in range(len(c_ior_data)):
        f_data[i] = spectral_ior.fresnel_reflectance_normal_incidence(c_ior_data[i], e)
    spectral_dict = dict(zip(wavelength_vals, f_data))
    return colour.SpectralDistribution(spectral_dict, name="Sample")

def spectral_ior_to_spd_f(wavelength_vals, c_ior_data, theta, e = 1.0):
    assert len(wavelength_vals) == len(c_ior_data), "Vectors are not the same length."
    f_data = np.empty(len(c_ior_data), dtype=np.float64)
    if e == 1.0:
        for i in range(len(c_ior_data)):
            f_data[i] = spectral_ior.fresnel_reflectance(c_ior_data[i], theta)
    else:
        for i in range(len(c_ior_data)):
            f_data[i] = spectral_ior.fresnel_reflectance_ext_medium(c_ior_data[i], theta, e)

    spectral_dict = dict(zip(wavelength_vals, f_data))
    return colour.SpectralDistribution(spectral_dict, name="Sample")

def spectral_ior_to_rgb_f0_ap1(wavelength_vals, c_ior_data, e = 1.0):
    """
    Compute normal-incidence RGB Fresnel reflectance (in the AP1/ACEScg space) from complex spectral IOR.

    Parameters
    ----------
    wavelength_vals: numeric array
        Array of wavelength values.
    c_ior_data: complex number array
        Array of complex index of refraction values at the wavelengths in wavelength_vals.
    e: real number
        Real-values IOR of external medium (defaults to 1.0).
    """
    F_spd = spectral_ior_to_spd_f0(wavelength_vals, c_ior_data, e)
    XYZ = colour.sd_to_XYZ(F_spd, cmfs, d60_spd) / 100

    # NOTE: Need chromatic adaptation since D60 and AP1 have slightly different white points
    RGB = colour.XYZ_to_RGB(XYZ, d60_wp, ap1_wp, ap1_xyz2rgb)

    # Normalize to account for minor inaccuracies in color math
    RGB = RGB * norm_RGB_ap1

    return RGB

def spectral_ior_to_rgb_f_ap1(wavelength_vals, c_ior_data, theta, e = 1.0):
    """
    Compute RGB Fresnel reflectance (in the AP1/ACEScg space) given complex spectral IOR and incidence angle.

    Parameters
    ----------
    wavelength_vals: numeric array
        Array of wavelength values.
    c_ior_data: complex number array
        Array of complex index of refraction values at the wavelengths in wavelength_vals.
    theta: numeric
        Incidence angle (in degrees).
    e: real number
        Real-values IOR of external medium (defaults to 1.0).
    """
    F_spd = spectral_ior_to_spd_f(wavelength_vals, c_ior_data, theta, e)
    XYZ = colour.sd_to_XYZ(F_spd, cmfs, d60_spd) / 100

    # NOTE: Need chromatic adaptation since D60 and AP1 have slightly different white points
    RGB = colour.XYZ_to_RGB(XYZ, d60_wp, ap1_wp, ap1_xyz2rgb)

    # Normalize to account for minor inaccuracies in color math
    RGB = RGB * norm_RGB_ap1

    return RGB

def spectral_ior_to_rgb_f0_srgb(wavelength_vals, c_ior_data, e = 1.0):
    """
    Compute normal-incidence RGB Fresnel reflectance (in the sRGB/Rec. 709 space) from complex spectral IOR.

    Parameters
    ----------
    wavelength_vals: numeric array
        Array of wavelength values.
    c_ior_data: complex number array
        Array of complex index of refraction values at the wavelengths in wavelength_vals.
    e: real number
        Real-values IOR of external medium (defaults to 1.0).
    """
    F_spd = spectral_ior_to_spd_f0(wavelength_vals, c_ior_data, e)
    XYZ = colour.sd_to_XYZ(F_spd, cmfs, d65_spd) / 100
    RGB = np.dot(srgb_xyz2rgb, XYZ) # When no chromatic adaptation, equivalent to colour.XYZ_to_RGB and much faster

    # Normalize to account for minor inaccuracies in color math
    RGB = RGB * norm_RGB_srgb

    return RGB

def spectral_ior_to_rgb_f_srgb(wavelength_vals, c_ior_data, theta, e = 1.0):
    """
    Compute RGB Fresnel reflectance (in the sRGB/Rec. 709 space) given complex spectral IOR and incidence angle.

    Parameters
    ----------
    wavelength_vals: numeric array
        Array of wavelength values.
    c_ior_data: complex number array
        Array of complex index of refraction values at the wavelengths in wavelength_vals.
    theta: numeric
        Incidence angle (in degrees).
    e: real number
        Real-values IOR of external medium (defaults to 1.0).
    """
    F_spd = spectral_ior_to_spd_f(wavelength_vals, c_ior_data, theta, e)
    XYZ = colour.sd_to_XYZ(F_spd, cmfs, d65_spd) / 100
    RGB = np.dot(srgb_xyz2rgb, XYZ) # When no chromatic adaptation, equivalent to colour.XYZ_to_RGB and much faster

    # Normalize to account for minor inaccuracies in color math
    RGB = RGB * norm_RGB_srgb

    return RGB

def ap1_linear_rgb_to_ap0_linear(rgb_ap1):
    """
    Compute AP0 (ACES2065-1) linear RGB values given AP1 (ACEScg) linear RGB values.

    Parameters
    ----------
    rgb_ap1: array-like
        Linear rgb values in AP1 space.
    """
    rgb_ap0 = colour.RGB_to_RGB(rgb_ap1, colour.models.ACES_CG_COLOURSPACE, colour.models.ACES_2065_1_COLOURSPACE)
    rgb_ap0[0] = max(0.0, min(1.0, rgb_ap0[0]))
    rgb_ap0[1] = max(0.0, min(1.0, rgb_ap0[1]))
    rgb_ap0[2] = max(0.0, min(1.0, rgb_ap0[2]))
    return rgb_ap0

def ap1_linear_rgb_to_srgb_linear(rgb_ap1):
    """
    Compute linear sRGB / Rec. 709 values given AP1 linear RGB values.

    Parameters
    ----------
    rgb_ap1: array-like
        Linear rgb values in AP1 space.
    """
    rgb_srgb = colour.RGB_to_RGB(rgb_ap1, colour.models.ACES_CG_COLOURSPACE, colour.models.sRGB_COLOURSPACE)
    rgb_srgb[0] = max(0.0, min(1.0, rgb_srgb[0]))
    rgb_srgb[1] = max(0.0, min(1.0, rgb_srgb[1]))
    rgb_srgb[2] = max(0.0, min(1.0, rgb_srgb[2]))
    return rgb_srgb

def srgb_encoding_cctf(rgb_lin):
    """
    Compute nonlinear sRGB values given linear sRGB values.

    Parameters
    ----------
    rgb_lin: array-like
        Linear rgb values in sRGB space.
    """
    return colour.sRGB_COLOURSPACE.encoding_cctf(rgb_lin)

def ap1_linear_rgb_to_srgb_nonlinear(rgb_ap1):
    """
    Compute nonlinear sRGB values given AP1 linear RGB values.

    Parameters
    ----------
    rgb_ap1: array-like
        Linear rgb values in the AP1 space.
    """
    rgb_lin = ap1_linear_rgb_to_srgb_linear(rgb_ap1)
    rgb_nonlin = srgb_encoding_cctf(rgb_lin)
    rgb_nonlin[0] = max(0.0, min(1.0, rgb_nonlin[0]))
    rgb_nonlin[1] = max(0.0, min(1.0, rgb_nonlin[1]))
    rgb_nonlin[2] = max(0.0, min(1.0, rgb_nonlin[2]))
    return rgb_nonlin

def ap1_luminance(rgb_in):
    """
    Compute luminance of an RGB triplet in the AP1/ACEScg space.

    Parameters
    ----------
    rgb_in: array-like
        Linear rgb values in the AP1 space.
    """
    return colour.RGB_luminance(rgb_in, colour.ACES_CG_COLOURSPACE.primaries, ap1_wp)

def srgb_luminance(rgb_in):
    """
    Compute luminance of an RGB triplet in the sRGB space.

    Parameters
    ----------
    rgb_in: array-like
        Linear rgb values in the sRGB space.
    """
    return colour.RGB_luminance(rgb_in, colour.sRGB_COLOURSPACE.primaries, srgb_wp)
