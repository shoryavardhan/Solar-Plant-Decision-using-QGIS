# Solar-Plant-Decision-using-QGIS
Solar power plant site selection using a GIS-based approach, studied over different cities of India. 

# Objective
The objective of the project is to show cities in order of their suitability of having a solar power plant. The
project classifies top 25 cities as “most recommended”,last few as “not recommended” and rest as
“normally recommended”.

# Background Study and Findings
For this project we studied various research papers for site selection, solar potential determination, normalization
methods and suitability index method. we used various python libraries and QGIS software. Our dataset is based
on attributes which are as follows: 
# 1.Population – 
Total number of citizens in a particular city.
# 2.PVOUT - 
Photovoltaic Electricity Output is amount of energy, converted by a PV system into electricity
[kWh/kWp] that is expected to be generated according to the geographical conditions of a site and a configuration of
the PV system. Three configurations of a PV system are considered: (i) Small residential; (ii) Medium-size
commercial; and (iii) Ground-mounted large scale. 

# 3.GHI - 
Global Horizontal Irradiation is the total solar energy received on a unit area of horizontal surface. It
includes energy from the sun that is received in a direct beam (the horizontal component of DNI) and DHI.

# 4.DNI - 
Direct Normal Irradiation which is the beam energy component received on a unit area of surface directly facing to the sun at all times. DNI is of particular interest for solar installations that track the sun and for concentrating solar technologies(that can only make use of the direct beam component of irradiation)

# 5.GTI - 
Global Tilted Irradiation is the total energy received on a unit area of a tilted surface .It includes direct beam
and diffused components .A high values of long term GTI average is the most important resource parameter for
project developers.We have used various online platforms both govt and private such as data.gov.in , solargis.com and
globalsolaratlas.info etc.

# AHP process - 
The analytic hierarchy process (AHP) is a structured technique for organizing and analyzing complex
decisions, based on mathematics and psychology. It was developed by Thomas L. Saaty in the 1970s and has been
extensively studied and refined since then.It has particular application in group decision making, and
is used around the world in a wide variety of decision situations, in fields such as government, business, industry,
healthcare, shipbuilding and education.We used this process for weight calculation of our attributes.

# Suitability Index - 
Suitability index is a numerical index which represents the capacity of various attributes to
support a solar power plant in that area. Lower index signifies lower chances of installing power plant in that city. Fuzzification - Fuzzification is to divide the continuous quantity in fuzzy domain into several levels, according to
the requirement, each level can be regarded as a fuzzy variable and corresponds to a fuzzy subset or a
membership function


# How to run?
The QGIS file in the project is the only file you need to run.
You will require QGIS 2.8.1 for the same.
