Lanblab output files for model runs at 5m resoluton for saturated hydraulic conductivity of 0.72 millimetre per hour. 


Subfolder level 1
> no_vLEsPresent: Landlab output files for subcatchment with no vLEs present in subwatershed
	Filenames: "dataframes_no_vLE_" + date (YYYYMMDD) + ".xlsx"
> vLEsPresent: Landlab output files for subcatchment with vLEs present in subwatershed
	Filenames: "dataframes_FA" + upslope area (in square metre per metre vLE) + date (YYYYMMDD) + ".xlsx"

In each xlsx file there are two sheets:

Sheet 1: "int":
	column 2: "time": model time (in seconds)
	column 3: "timestep": model timestep (in seconds)
	column 4: "discharge_sum": sum of all discharge (in cubic metres per second) at all nodes in subwatershed
	column 5: "depth_sum" sum of surface water depth (in metres) for all cells in subwatershed for current timestep
	column 6: "max_h": maximal surface water depth (in metres) in subwatershed for current timestep
	column 7: "infiltration_depth": cummultative infiltration depth (in metres) in subwatershed
	column 8: "actual_infiltration": infiltration (in metres) in subwatershed for current timestep
	column 9: "h_beforeInf_sum": sum of surface water depth (in metres) for all cells in subwatershed for current timestep before infiltration
	column 10: "h_afterInf_sum": sum of surface water depth (in metres) for all cells in subwatershed for current timestep after infiltration 
	column 11: "Boundary_discharge_sum": discharge leaving subwatershed for current time step (in cubic metres)
	column 12: "Q_volume": discharge volume at outlet of subwatershed for current time step (in cubic metres)
	column 13: "cumm_Q_volume": cummulative discharge volume at outlet of subwatershed (in cubic metres)
	column 14: "h_volume": surface water depth in subwatershed for current time step (in cubic metres)
	column 15: "cumm_I_vol": cummulative infiltration volume in subwatershed (in cubic metres)
	column 16: "P_volume": precipitation volume in subwatershed for current time step (in cubic metres)
	column 17: "cumm_P_volume": cummulative precipitation volume in subwatershed (in cubic metres)
	column 18: "h0_vol": surface water depth before model inition in subwatershed (in cubic metres)
	column 19: "I0_vol": infiltration volume before model initiation in subwatershed (in cubic metres)
	column 20: "balance": water balance in subwatershed (in cubic metres)

Sheet 2: "c9715" 
	column 2: "time": model time (in seconds)
	column 3: "timestep": model timestep (in seconds)
	column 4: "depth": surface water depth (in metres) for current timestep in cell with id 9715 (=outlet cell)
	column 5: "depth_before_inf": surface water depth (in metres) for current timestep before infiltration in cell with id 9715 (=outlet cell)
	column 6: "depth_after_inf": surface water depth (in metres) urrent timestep before infiltration in cell with id 9715 (=outlet cell)
	column 7: "infiltration_depth": cummultative infiltration depth (in metres) in cell with id 9715 (=outlet cell)
	column 8: "actual_infiltration": infiltration (in metres) in cell with id 9715 (=outlet cell)
	column 9: "h_beforeInf_9715": surface water depth (in metres) for current timestep before infiltration in cell with id 9715 (=outlet cell)
	column 10: "h_afterInf_9715": surface water depth (in metres) urrent timestep before infiltration in cell with id 9715 (=outlet cell)
	column 11: "discharge_9715": divergence of link-based fluxes at cell with id 9715 (=outlet cell)
	column 12: "surface_water_discharge_9715": discharge rate for current time step in cell with id 9715 (=outlet cell) (in cubic metres per second)
	column 13: "rainfall_intensity": rainfall intensity for current time step for current time step (in cubic metres per second)


