using DifferentialEquations
using Flux: Chain, Dense, relu, ADAM
using DiffEqFlux
using Plots

# Define the Logistic Growth Equation parameters
r = 0.1  # growth rate
K = 100.0  # carrying capacity
y0 = 10.0  # initial population

# Define the true ODE function
function logistic!(du, u, p, t)
    du .= r * u * (1 - u / K)
end

# Define the initial condition
u0 = [y0]

# Time span for the solution
tspan = (0.0, 10.0)

# Solve the ODE using the traditional method for comparison
prob = ODEProblem(logistic!, u0, tspan)
sol = solve(prob, Tsit5())

# PINN setup
# Define a neural network structure
dudt_model = Chain(Dense(1, 10, tanh), Dense(10, 1))

# Define the physics-informed loss function
function pinn_loss(p)
    pred_dudt = dudt_model(sol.t, p)
    true_dudt = r .* sol.u .* (1 .- sol.u ./ K)
    return sum(abs2, pred_dudt .- true_dudt)
end

# Training the model using ADAM optimizer
loss_function(p) = pinn_loss(p)
opt = ADAM(0.01)
p, re = Flux.Optimise.optimise(loss_function, Flux.params(dudt_model), opt, maxiters = 500)

# Predicting the population over time using the trained PINN
predicted_population = dudt_model(sol.t, p)

# Plotting the results
plot(sol.t, sol.u, label="True Solution (ODE)", linewidth=2)
plot!(sol.t, predicted_population, label="PINN Prediction", linestyle=:dash, linewidth=2)
xlabel!("Time (t)")
ylabel!("Population (y(t))")
title!("Logistic Growth Equation Solution using PINNs vs ODE")

# Additional insights from the graph
# Plotting the error between the true solution and the PINN prediction
error = abs.(sol.u .- predicted_population)
plot(sol.t, error, label="Error (True - PINN)", linewidth=2, color=:red)
xlabel!("Time (t)")
ylabel!("Error")
title!("Error between True Solution and PINN Prediction")
