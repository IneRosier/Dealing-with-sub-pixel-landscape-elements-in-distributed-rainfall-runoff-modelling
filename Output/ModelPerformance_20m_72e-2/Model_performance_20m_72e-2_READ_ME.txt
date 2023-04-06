Lanblab output files for model runs at 20m resoluton with no vLEs present in watershed

Subfolder level 1
> noScaling: Landlab output files for saturated hydraulic conductivity equal to 0.72 millimetre per hour and manning's n calculated by area weighted arithmetic mean

> AlphaScaling_PFThreshold: Landlab output files for saturated hydraulic conductivity equal to 0.72 millimetre per hour and saturated hydaulic conductivity scaled by amplification factor alpha (Eq. 4) where the partitioning parameter was lower than  0.75

> AlphaScaling_stdThreshold: Landlab output files for saturated hydraulic conductivity equal to 0.72 millimetre per hour and saturated hydaulic conductivity scaled by amplification factor alpha (Eq. 4) where the standard deviation of the slope was higher than 0.01

> DeltaScaling_Ks: Landlab output files for saturated hydraulic conductivity equal to 0.72 millimetre per hour and saturated hydaulic conductivity scaled by scaling factor delta (Eq. 7)

> DeltaScaling_Manning: Landlab output files for saturated hydraulic conductivity equal to 0.72 millimetre per hour and Manning's roughness coefficient scaled by scaling factor delta (Eq. 7)



Filenames in subfolders: "dataframes_" + date (YYYYMMDD) + ".xlsx"
In each xlsx file there are two sheets:

Sheet 1: "int":
	column 2: "time": model time (in seconds)
	column 3: "timestep": model timestep (in seconds)
	column 4: "discharge_sum": sum of all discharge (in cubic metres per second) at all nodes in watershed
	column 5: "depth_sum" sum of surface water depth (in metres) for all cells in watershed for current timestep
	column 6: "max_h": maximal surface water depth (in metres) in watershed for current timestep
	column 7: "infiltration_depth": cummultative infiltration depth (in metres) in watershed
	column 8: "actual_infiltration": infiltration (in metres) in watershed for current timestep
	column 9: "h_beforeInf_sum": sum of surface water depth (in metres) for all cells in watershed for current timestep before infiltration
	column 10: "h_afterInf_sum": sum of surface water depth (in metres) for all cells in watershed for current timestep after infiltration 
	column 11: "Boundary_discharge_sum": discharge leaving watershed for current time step (in cubic metres)
	column 12: "Q_volume": discharge volume at outlet of watershed for current time step (in cubic metres)
	column 13: "cumm_Q_volume": cummulative discharge volume at outlet of watershed (in cubic metres)
	column 14: "h_volume": surface water depth in watershed for current time step (in cubic metres)
	column 15: "cumm_I_vol": cummulative infiltration volume in watershed (in cubic metres)
	column 16: "P_volume": precipitation volume in watershed for current time step (in cubic metres)
	column 17: "cumm_P_volume": cummulative precipitation volume in watershed (in cubic metres)
	column 18: "h0_vol": surface water depth before model inition in watershed (in cubic metres)
	column 19: "I0_vol": infiltration volume before model initiation in watershed (in cubic metres)
	column 20: "balance": water balance in watershed (in cubic metres)

Sheet 2: "c10171" 
	column 2: "time": model time (in seconds)
	column 3: "timestep": model timestep (in seconds)
	column 4: "depth": surface water depth (in metres) for current timestep in cell with id 10171 (=outlet cell)
	column 5: "depth_before_inf": surface water depth (in metres) for current timestep before infiltration in cell with id 10171 (=outlet cell)
	column 6: "depth_after_inf": surface water depth (in metres) urrent timestep before infiltration in cell with id 10171 (=outlet cell)
	column 7: "infiltration_depth": cummultative infiltration depth (in metres) in cell with id 10171 (=outlet cell)
	column 8: "actual_infiltration": infiltration (in metres) in cell with id 10171 (=outlet cell)
	column 9: "h_beforeInf_10171": surface water depth (in metres) for current timestep before infiltration in cell with id 10171 (=outlet cell)
	column 10: "h_afterInf_10171": surface water depth (in metres) urrent timestep before infiltration in cell with id 10171 (=outlet cell)
	column 11: "discharge_10171": divergence of link-based fluxes at cell with id 10171 (=outlet cell)
	column 12: "surface_water_discharge_10171": discharge rate for current time step in cell with id 10171 (=outlet cell) (in cubic metres per second)
	column 13: "rainfall_intensity": rainfall intensity for current time step for current time step (in cubic metres per second)


