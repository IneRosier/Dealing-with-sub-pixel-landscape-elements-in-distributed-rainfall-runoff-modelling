Subfolder level 1
	> 5m
		tif files used to represent vLE configurations. Cell value 0 represents a landscape pixel, cell value 1 represent a vLE pixel
		file name: "Subc1_" + raster resolution + "_FA" + value of upslope area per metre vLE + ".tif"

	> 20m
		Subfolder level 2
			> Weight_Area
				tif files used to scale the manning's coefficient and saturated hydraulic conductivity using area based weights. 
				The cell values in the tif files represent the area proportion of the cell used as weights in Eq. 10.
				Values larger than zero were converted to 1 to use in the vLE priority method.
				file name: "Subc1_" + raster resolution + "_FA" + value of upslope area per metre vLE + ".tif"

			> Weight_UA
				tif files used to scale the manning's coefficient and saturated hydraulic conductivity using upslope area based weights. 
				The cell values in the tif files represent the upslope area proportion of the cell used as weights in Eq. 11.
				file name: "Subc1_" + raster resolution + "weihgt_FA" + value of upslope area per metre vLE + ".tif"