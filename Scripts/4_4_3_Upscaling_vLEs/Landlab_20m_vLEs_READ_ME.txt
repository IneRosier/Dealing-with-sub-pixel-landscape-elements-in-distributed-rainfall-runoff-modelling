script used in section 4.4.3 of the manuscript to calculate the impact of vLEs at 20m resolution

directories towards input and output data should be adapted

Five different scaling methods for hydro-physical parameters can be used as described in section 4.4.2 of the manuscript. Adapt the parameter 'scaling' in the script accordingly:
	scaling = "none" --> no scaling
	scaling = "FlLength_manning" --> delta scaling of the manning's roughness coefficient
	scaling = "FlLength_Ks" --> delta scaling of the saturated hydraulic conductivity
	scaling =  "gamma" --> alpha scaling of the saturated hydraulic conductivity using the partitioning parameter to determine the pixels in which the scaling is applied
	scaling =  "std" --> alpha scaling of the saturated hydraulic conductivity using the standard deviation of the slope to determine the pixels in which the scaling is applied

Three different weighting methods to determine the hydro-physical parameters of vLE pixels can be used as described in section 4.4.3 of the manuscript. Adapt the parameter 'weight' in the script accordingly:
	weight = "Full" --> priority method to determine hydro-physical parameters of vLE pixels
	weight = "" --> area based weighting to determine hydro-physical parameters of vLE pixels
	Weight = "FA" --> upslope area weighting to determine hydro-physical parameters of vLE pixels