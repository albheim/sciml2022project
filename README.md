# Project for 2022 SciML course at DTU in Copenhagen

We look at the Windkessel models for the pressure/flow relationship in a beating heart 
and try a few different approaches, including universal differential equations, to compare
to these models. We also run symbolic regression on the found functions to see how well they 
can be simplified to interpretable expressions with fewer parameters.

The main scripts to run are the `training_*.jl` scripts, and they are for
* `model_parameters` - Use the same order linear model and simply do an optimization over the parameters.
* `nn_with_model` - Use the parameters for the standard Windkessel model, but add a neural network in the dynamics equations to capture unmodeled dynamics. Only train the neural network and let the other parameters be the standard ones.
* `nn` - Use neural network for the whole dynamics equation and the observation equation. Optimize the networks as well as the initial conditions for the internal states.

At the start of each script a variable `nstate` is set to either 1 or 2 deciding on the order of the model generated.

To run you first need julia, this was tested with julia 1.7.2 but will likely work on newer versions as well. 
Start by cloning the repo and open a terminal in the repo. The environment is instantiated by 
```julia
julia --project -e "import Pkg; Pkg.instantiate()"
```
and then the scripts can be run as
```julia
julia --project training_<experiment>.jl --order n --repeat k
```
where `<experiment>` is replaced to denote one of the three scripts, `n` is set to the order of the approximation (default 1, can also be 2) and `repeat` is set to the number of times to repeat the data (default 1, higher integers will help enforce periodicity).
