# material-params
Python scripts for computing physical material parameters

The main script is simple-params.py.
The only command-line argument supported is the filename(s) of spectral IOR (index of refraction) files in one of two formats:
RII, and SOPRA. The "[Full database record]" files from refractiveindex.info are supported as a variant of the RII format.

Parameter values are output in the sRGB color space by default.
The other color space supported is AP1 - to get results in the AP1 color space edit the value of "colorGamutID" near the end of simple-params.py.
