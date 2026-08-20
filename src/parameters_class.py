class Parameters:
    def __init__(self, 
                 gg, T0, lr, TD, rho_c, hpc, cp_c, 
                 tt, dt, dtr, start_dtr, end_dtr,
                 U, K, m, n, icflag, islope, Ui, Ki, ee, pixel,
                 hl, hdn, hk, hm, hn, crit_slope,
                 muon, cosmo_thickness, cosmo_topocorr, cosmo_aa, dx_cosmo, t_record):
    
        # thermo model parameters
        self.gg = gg                                # geothermal gradient °C/km
        self.T0 = T0                                # surface temperature at sea level in °C
        self.lr = lr                                # atmospheric lapse rate in °C/km
        self.TD = TD                                # thermal diffusivity in m2/s
        self.rho_c = rho_c                          # crustal density in kg/m3
        self.hpc = hpc*rho_c                        # crustal heat production in W/m3
        self.cp_c = cp_c                            # specific heat capacity of granite in J/kg*K
        
        # time parameters
        self.tt = tt                                # model duration in Myr
        self.dt = dt                                # time step in yr
        self.dtr = dtr                              # laps time record in Myr
        self.start_dtr = start_dtr                  # start of model record
        self.end_dtr = end_dtr                      # end of model record
        
        # river profile model parameters
        self.U = U                                  # uplift in mm/yr
        self.K = K                                  # erodibility in yr-1
        self.m = m                                  # area exponent
        self.n = n                                  # slope exponent 
        self.icflag = icflag                        # slope shape (1:constant slope; 2:constant elevation; 3: steady state elevation)
        self.islope = islope                        # constant slope in m/m
        self.Ui = Ui                                # initial uplift rate in mm/yr (if icflag = 3)
        self.Ki = Ki                                # initial erodibility in yr-1 (if icflag = 3)
        self.ee = ee                                # elevation uncertainty in m
        self.pixel = pixel                          # area unit (0:pixel, 1:m2)
        
        # hillslope model parameters
        self.hl = hl                                # length of hillslopes in m
        self.hdn = hdn                              # number of hillslope node
        self.hk = hk                                # hillslope diffusion rate
        self.hm = hm                                # distance exponent
        self.hn = hn                                # slope exponent
        self.crit_slope = crit_slope                # critical hillslope angle in °
        
        # cosmo model parameters
        self.muon = muon                            # muon model production (1:Braucher et al., 2013; 2:Heisinger et al., 2002ab)
        self.cosmo_thickness = cosmo_thickness      # sample thickness in cm
        self.cosmo_topocorr = cosmo_topocorr        # topographic shielding
        self.cosmo_aa = cosmo_aa                    # atmospheric model
        self.dx_cosmo = dx_cosmo                    # distance between sample locations for which tcn are calculated in m
        self.t_record = t_record                    # time-span to model tcn calculation in years