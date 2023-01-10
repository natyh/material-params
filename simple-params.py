#!/bin/env python3

import numpy as np
import os
from scipy.optimize import minimize
import spectral_ior
import color_ior
import color_exp

# Schlick approximation
def schlick_approx_fresnel(r, t):
    cosTheta = np.cos(np.deg2rad(t))
    cosTheta = max(cosTheta, 0.0)
    omc = 1.0 - cosTheta
    omc2 = omc * omc
    omc5 = omc2 * omc2 * omc
    return r + (1.0 - r) * omc5

# Compute (scalar) Gulbrandsen edge tint given (scalar) F0 and n values
def gulbrandsen_edge_tint(r, n):
    sqrtR = np.sqrt(r)
    gulFactor = (1.0 + sqrtR) / (1.0 - sqrtR) 
    return (gulFactor - n) / (gulFactor - ((1.0 - r)/(1.0 + r)))

# Process a single ior file
def process_ior_file(ior_filename, colorGamutID):

    # Only suppprted colorGamutID values are 0 and 1
    if (colorGamutID != 0) and (colorGamutID != 1):
        sys.exit("Unsuppported color gamut ID!")

    # Read file into arrays. Note that the data is resampled into a regular 1 nm spacing starting
    # at 360 nm (this is important for the point-sampling logic).
    wavelength_vals, c_ior_data = spectral_ior.read_ior_file(ior_filename)

    # Create vector of angles (in degrees for axis labels)
    theta = np.linspace(0.0, 90.0, 100, dtype=np.float64)

    # Compute reflectance for each point on angle axis
    a_r = np.empty(theta.shape[0], dtype=np.float64)
    a_g = np.empty(theta.shape[0], dtype=np.float64)
    a_b = np.empty(theta.shape[0], dtype=np.float64)
    for i in range(0, theta.shape[0]):
        if (colorGamutID == 0): # sRGB
            RGB_lin = color_ior.spectral_ior_to_rgb_f_srgb(wavelength_vals, c_ior_data, theta[i])
        elif (colorGamutID == 1): # AP1
            RGB_lin = color_ior.spectral_ior_to_rgb_f_ap1(wavelength_vals, c_ior_data, theta[i])
        a_r[i] = RGB_lin[0]
        a_g[i] = RGB_lin[1]
        a_b[i] = RGB_lin[2]

    # Compute RGB F0
    if (colorGamutID == 0): # sRGB
        F0_lin = color_ior.spectral_ior_to_rgb_f0_srgb(wavelength_vals, c_ior_data)
    elif (colorGamutID == 1): # AP1
        F0_lin = color_ior.spectral_ior_to_rgb_f0_ap1(wavelength_vals, c_ior_data)
    f0_r = np.float64(F0_lin[0])
    f0_g = np.float64(F0_lin[1])
    f0_b = np.float64(F0_lin[2])

    # Boolean - is this a dielectric? Used for some conditionals later
    bDielectric = f0_r + f0_g + f0_b < 1.0

    # Convolved RGB approximation: convert spectral IOR to RGB if it were stimulus data,
    # including convolving with the XYZ color-matching functions and converting the result
    # into the appropriate RGB space. This is only used as a starting value for fitting.
    if (colorGamutID == 0): # sRGB
        conv_RGB_real, conv_RGB_imag = color_exp.spectral_ior_to_rgb_srgb_ior(wavelength_vals, c_ior_data)
    elif (colorGamutID == 1): # AP1
        conv_RGB_real, conv_RGB_imag = color_exp.spectral_ior_to_rgb_ap1_ior(wavelength_vals, c_ior_data)
    c_r_conv = complex(conv_RGB_real[0], conv_RGB_imag[0])
    c_g_conv = complex(conv_RGB_real[1], conv_RGB_imag[1])
    c_b_conv = complex(conv_RGB_real[2], conv_RGB_imag[2])

    # declare variables and functions shared across fitting methods
    cur_ref = a_r
    a_temp = np.empty(theta.shape[0], dtype=np.float64)

    # Fitting: fit R G B values for n & k

    def fit_err(vec_nk):
        for i in range(0, theta.shape[0]):
            c_nk = complex(vec_nk[0], vec_nk[1])
            a_temp[i] = spectral_ior.fresnel_reflectance(c_nk, theta[i])
        return color_ior.rms_l(cur_ref, a_temp)

    # Fit red n, k using convolved value as initial guess
    cur_ref = a_r
    res = minimize(fit_err, [c_r_conv.real, c_r_conv.imag], method='Nelder-Mead')
    c_r_fit = complex(res.x[0], res.x[1])

    # Fit green n, k using convolved value as initial guess
    cur_ref = a_g
    res = minimize(fit_err, [c_g_conv.real, c_g_conv.imag], method='Nelder-Mead')
    c_g_fit = complex(res.x[0], res.x[1])

    # Fit blue n, k using convolved value as initial guess
    cur_ref = a_b
    res = minimize(fit_err, [c_b_conv.real, c_b_conv.imag], method='Nelder-Mead')
    c_b_fit = complex(res.x[0], res.x[1])

    # Check for negative values (should only happen in rare cases, with small color gamuts like sRGB)
    if c_r_fit.real < 0.0:
        print("WARNING: fitted R value of n is negative - clamping to 0!")
        c_r_fit = complex(0.0, c_r_fit.imag)
    if c_g_fit.real < 0.0:
        print("WARNING: fitted G value of n is negative - clamping to 0!")
        c_g_fit = complex(0.0, c_g_fit.imag)
    if c_b_fit.real < 0.0:
        print("WARNING: fitted B value of n is negative - clamping to 0!")
        c_b_fit = complex(0.0, c_b_fit.imag)
    if c_r_fit.imag < 0.0:
        print("WARNING: fitted R value of k is negative - clamping to 0!")
        c_r_fit = complex(c_r_fit.real, 0.0)
    if c_g_fit.imag < 0.0:
        print("WARNING: fitted G value of k is negative - clamping to 0!")
        c_g_fit = complex(c_g_fit.real, 0.0)
    if c_b_fit.imag < 0.0:
        print("WARNING: fitted B value of k is negative - clamping to 0!")
        c_b_fit = complex(c_b_fit.real, 0.0)

    # Calculate fitted curves
    a_r_fit = np.empty(theta.shape[0], dtype=np.float64)
    a_g_fit = np.empty(theta.shape[0], dtype=np.float64)
    a_b_fit = np.empty(theta.shape[0], dtype=np.float64)
    for i in range(0, theta.shape[0]):
        a_r_fit[i] = spectral_ior.fresnel_reflectance(c_r_fit, theta[i])
        a_g_fit[i] = spectral_ior.fresnel_reflectance(c_g_fit, theta[i])
        a_b_fit[i] = spectral_ior.fresnel_reflectance(c_b_fit, theta[i])

    if bDielectric:
        # Fitting a (dielectrics only): fit scalar value of n (assuming k==0)
        def fit_scalar_dielectric_err(n):
            for i in range(0, theta.shape[0]):
                c_nk = complex(n, 0.0)
                a_temp[i] = spectral_ior.fresnel_reflectance(c_nk, theta[i])
            if (colorGamutID == 0): # sRGB
                return color_ior.rms_srgb_de2000(a_r, a_g, a_b, a_temp, a_temp, a_temp)
            elif (colorGamutID == 1): # AP1
                return color_ior.rms_ap1_de2000(a_r, a_g, a_b, a_temp, a_temp, a_temp)

        # Fit n using green convolved value as initial guess
        res = minimize(fit_scalar_dielectric_err, [c_g_conv.real], method='Nelder-Mead')
        scalar_n_fit = res.x[0]

        # Calculate fitted curve
        a_s_nfit = np.empty(theta.shape[0], dtype=np.float64)
        c_s_nfit = complex(scalar_n_fit, 0.0)
        for i in range(0, theta.shape[0]):
            a_s_nfit[i] = spectral_ior.fresnel_reflectance(c_s_nfit, theta[i])

        # Fitting b (dielectrics only): fit RGB value of n (assuming k==0)
        def fit_dielectric_err(n):
            for i in range(0, theta.shape[0]):
                c_nk = complex(n, 0.0)
                a_temp[i] = spectral_ior.fresnel_reflectance(c_nk, theta[i])
            return color_ior.rms_l(cur_ref, a_temp)

        # Fit red n using convolved value as initial guess
        cur_ref = a_r
        res = minimize(fit_dielectric_err, [c_r_conv.real], method='Nelder-Mead')
        n_r_nfit = res.x[0]

        # Fit green n using convolved value as initial guess
        cur_ref = a_g
        res = minimize(fit_dielectric_err, [c_g_conv.real], method='Nelder-Mead')
        n_g_nfit = res.x[0]

        # Fit blue n using convolved value as initial guess
        cur_ref = a_b
        res = minimize(fit_dielectric_err, [c_b_conv.real], method='Nelder-Mead')
        n_b_nfit = res.x[0]

        # Calculate fitted curves
        a_r_nfit = np.empty(theta.shape[0], dtype=np.float64)
        a_g_nfit = np.empty(theta.shape[0], dtype=np.float64)
        a_b_nfit = np.empty(theta.shape[0], dtype=np.float64)
        c_r_nfit = complex(n_r_nfit, 0.0)
        c_g_nfit = complex(n_g_nfit, 0.0)
        c_b_nfit = complex(n_b_nfit, 0.0)
        for i in range(0, theta.shape[0]):
            a_r_nfit[i] = spectral_ior.fresnel_reflectance(c_r_nfit, theta[i])
            a_g_nfit[i] = spectral_ior.fresnel_reflectance(c_g_nfit, theta[i])
            a_b_nfit[i] = spectral_ior.fresnel_reflectance(c_b_nfit, theta[i])

    # Calculate some values used later.
    gg_r = gulbrandsen_edge_tint(f0_r, c_r_fit.real)
    gg_g = gulbrandsen_edge_tint(f0_g, c_g_fit.real)
    gg_b = gulbrandsen_edge_tint(f0_b, c_b_fit.real)

    # Schlick curves
    a_r_schlick = np.empty(theta.shape[0], dtype=np.float64)
    a_g_schlick = np.empty(theta.shape[0], dtype=np.float64)
    a_b_schlick = np.empty(theta.shape[0], dtype=np.float64)
    for i in range(0, theta.shape[0]):
        a_r_schlick[i] = schlick_approx_fresnel(f0_r, theta[i])
        a_g_schlick[i] = schlick_approx_fresnel(f0_g, theta[i])
        a_b_schlick[i] = schlick_approx_fresnel(f0_b, theta[i])

    name_str = os.path.splitext(os.path.basename(ior_filename))[0]
    print("Results for", name_str, ":")
    if (colorGamutID == 0): # sRGB
        print("(all RGB values are linear in sRGB gamut)")
    elif (colorGamutID == 1): # AP1
        print("(all RGB values are linear in AP1 gamut)")
    print("F0 RGB:", np.float32(f0_r), np.float32(f0_g), np.float32(f0_b))
    print("RGB Gulbrandsen edge tint (from F0 and fitted RGB n):", np.float32(gg_r), np.float32(gg_g), np.float32(gg_b))
    print("Fitted RGB n:", np.float32(c_r_fit.real), np.float32(c_g_fit.real), np.float32(c_b_fit.real))
    print("Fitted RGB k:", np.float32(c_r_fit.imag), np.float32(c_g_fit.imag), np.float32(c_b_fit.imag))
    if bDielectric:
        print("Fitted scalar n (assuming k==0):", np.float32(scalar_n_fit))
        print("Fitted RGB n (assuming k==0):", np.float32(n_r_nfit), np.float32(n_g_nfit), np.float32(n_b_nfit))
    print("")


if __name__ == "__main__":
    import sys

    # List of supported RGB gamut IDs:
    # 0 - sRGB
    # 1 - AP1

    # Change this line to change color gamut (TODO - have this be set on the command line)
    colorGamutID = 1

    for fname in sys.argv[1:]:
        print("processing %", fname)
        process_ior_file(fname, colorGamutID)
