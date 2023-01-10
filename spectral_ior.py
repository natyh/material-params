"""
Spectral IOR
============

Contains functions related to spectral IOR processing such as reading common spectral IOR
file formats and computing reflectance from complex IOR.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright Naty Hoffman 2023

import numpy as np

standard_lambda_sampling = np.arange(360, 781)

def fresnel_reflectance_normal_incidence(c, e = 1.0):
    """
    Compute normal-incidence Fresnel reflectance (for a single wavelength) from complex surface IOR and real external medium IOR.

    Parameters
    ----------
    c : complex number
        Complex index of refraction of surface.
    e : real number
        Real index of refraction of external medium (defaults to 1.0).
    """
    etak2 = c.imag * c.imag
    return np.float64( ((c.real-e)*(c.real-e) + etak2) / ((c.real+e)*(c.real+e) + etak2) )

def fresnel_reflectance(c, t):
    """
    Compute Fresnel reflectance (for a single wavelength) from complex IOR and angle of incidence.

    Parameters
    ----------
    c : complex number
        Complex index of refraction.
    t: numeric
        Angle of incidence in degrees.
    """
    # Math from Sebastien Lagarde's blog post on Fresnel.
    sinTheta = np.sin(np.deg2rad(t))
    sinTheta2 = sinTheta * sinTheta
    cosTheta2 = 1 - sinTheta2
    cosTheta = np.sqrt(cosTheta2)
    eta2 = c.real * c.real
    etak2 = c.imag * c.imag
    temp0 = eta2 - etak2 - sinTheta2
    a2plusb2 = np.sqrt(temp0 * temp0 + 4.0 * eta2 * etak2)
    temp1 = a2plusb2 + cosTheta2
    a = np.sqrt(0.5 * (a2plusb2 + temp0))
    temp2 = 2.0 * a * cosTheta
    Rs = (temp1 - temp2) / (temp1 + temp2)
    temp3 = cosTheta2 * a2plusb2 + sinTheta2 * sinTheta2
    temp4 = temp2 * sinTheta2
    Rp = Rs * (temp3 - temp4) / (temp3 + temp4)
    return np.float64(0.5 * (Rp + Rs))

def fresnel_reflectance_ext_medium(c, t, e):
    """
    Compute Fresnel reflectance (for a single wavelength) from complex IOR and angle of incidence.

    Parameters
    ----------
    c : complex number
        Complex index of refraction.
    t: numeric
        Angle of incidence in degrees.
    e : real number
        Real index of refraction of external medium.
    """

    return fresnel_reflectance(c * (1.0/e), t)

def read_rii_format_ior_file(data_idx, iorfilelist):

    # Assuming that data_idx is the index of the start of the numerical data (triples
    # of float numbers).
    wavelength_num = (len(iorfilelist) - data_idx) // 3
    #ERRORCHECK THAT REMAINDER ACTUALLY DIVIDES EXACTLY INTO 3

    # Read 'wavelength_num' wavelengths and pairs of numbers (convert wavelengths from um to nm)
    wavelength_vals = np.empty(wavelength_num, dtype=np.float64)
    c_ior_data = np.empty(wavelength_num, dtype=np.complex64)
    for i in range(0, wavelength_num):
        wavelength_vals[i] = 1000.0 * float(iorfilelist[i*3+data_idx])
        c_ior_data[i] = complex(float(iorfilelist[i*3+(data_idx+1)]), float(iorfilelist[i*3+(data_idx+2)]))

    return wavelength_vals, c_ior_data

def read_sopra_format_ior_file(iorfilelist):

    # Read header
    wavelength_type = int(iorfilelist[0])
    #ERRORCHECK THAT TYPE IS ONE OF ALLOWED VALUES
    wavelength_start = np.float64(iorfilelist[1])
    wavelength_stop = np.float64(iorfilelist[2])
    #ERRORCHECK THAT START IS GREATER THAN 0 AND THAT STEP IS GREATER THAN START (IF NUM IS 1, CAN THEY BE THE SAME?)
    wavelength_num = int(iorfilelist[3])
    #ERRORCHECK THAT NUM IS GREATER THAN 1 (OR CAN IT BE 1?)

    # Read 'wavelength_num' pairs of numbers and store as single-precision complex data
    c_ior_data = np.empty(wavelength_num, dtype=np.complex64)
    for i in range(0, wavelength_num):
        c_ior_data[i] = complex(float(iorfilelist[i*2+4]), float(iorfilelist[i*2+5]))

    # Create array of wavelengths
    wavelength_vals = np.linspace(wavelength_start, wavelength_stop, num=wavelength_num, endpoint=False).astype('float64')

    # Convert wavelengths to nm (if not already in that unit) and ensure increasing order
    if (wavelength_type == 1):
        wavelength_vals = 1239.84187 / wavelength_vals  # Convert from eV to nm
        wavelength_vals = wavelength_vals[::-1]  # Reverse order so nm values are in increasing order
        c_ior_data = c_ior_data[::-1]  # Reverse order of reflectance data to match wavelengths
    elif (wavelength_type == 2):
        wavelength_vals = 1000.0 * wavelength_vals  # Convert from um to nm
    elif (wavelength_type == 3):
        wavelength_vals = 1.e+7 / wavelength_vals  # Convert from 1/cm to nm
        wavelength_vals = wavelength_vals[::-1]  # Reverse order so nm values are in increasing order
        c_ior_data = c_ior_data[::-1]  # Reverse order of reflectance data to match wavelengths

    return wavelength_vals, c_ior_data

def read_ior_file(ior_filename, resample=True):
    """
    Read common spectral IOR file formats and return two arrays: one of wavelengths and one of complex IOR values.

    Parameters
    ----------
    ior_filename : string
        Path of spectral ior file.
    resample : bool, optional (True by default)
        Whether to resample the spectral IOR to a regular 1nm spacing from 360 nm to 780 nm.
    """
    # First put all whitespace-separated strings in a list and then iterate over the list
    with open(ior_filename) as f:
        iorfilelist = []
        for line in f:
            iorfilelist.extend(line.replace(',',' ').split()) # some files use commas, so replace commas with spaces

    # There are two flavors of RII format - in one, the numerical data is preceded by "DATA nk", and in the other, it
    # is preceded by "data: |"
    data_idx = iorfilelist.index("data:")+2 if "data:" in iorfilelist else (iorfilelist.index("DATA")+2 if "DATA" in iorfilelist else -1)

    if (data_idx == -1):
        wavelength_vals, c_ior_data = read_sopra_format_ior_file(iorfilelist)
    else:
        wavelength_vals, c_ior_data = read_rii_format_ior_file(data_idx, iorfilelist)

    # Optionally use linear interpolation to resample the spectral complex ior into a 1nm sampling between 360 and 780nm.
    if resample:
        c_ior_data = np.interp(standard_lambda_sampling, wavelength_vals, c_ior_data).astype('complex64')
        wavelength_vals = np.copy(standard_lambda_sampling)

    return wavelength_vals, c_ior_data

