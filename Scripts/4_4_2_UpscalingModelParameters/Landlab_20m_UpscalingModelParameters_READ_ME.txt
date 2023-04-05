OUTPUT TO BE COMPARED WITH OUTPUT SCRIPT SECTION 4.4.1
directories towards input and output data should be adapted

Five different scaling methods for hydro-physical parameters can be used as described in section 4.4.2 of the manuscript. Adapt the parameter 'scaling' in the script accordingly:
	scaling = "none" --> no scaling
	scaling = "FlLength_manning" --> delta scaling of the manning's roughness coefficient
	scaling = "FlLength_Ks" --> delta scaling of the saturated hydraulic conductivity
	scaling =  "gamma" --> alpha scaling of the saturated hydraulic conductivity using the partitioning parameter to determine the pixels in which the scaling is applied
	scaling =  "std" --> alpha scaling of the saturated hydraulic conductivity using the standard deviation of the slope to determine the pixels in which the scaling is applied