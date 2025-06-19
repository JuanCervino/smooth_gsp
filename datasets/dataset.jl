using SpeedyWeather
# https://speedyweather.github.io/SpeedyWeather.jl/dev/how_to_run_speedy/

path=pwd()
print(path)

spectral_grid = SpectralGrid(trunc = 6, nlev=1)


time_stepping = Leapfrog(spectral_grid)

initial_conditions = ZonalJet(;latitude = -25.0, umax = -80)

model = ShallowWaterModel(;spectral_grid,time_stepping,initial_conditions)
simulation = initialize!(model)

# Run that simulation.
run!(simulation,output=true,period=Day(20))

