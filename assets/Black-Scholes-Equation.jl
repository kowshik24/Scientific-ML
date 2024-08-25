using LinearAlgebra

function black_scholes(S₀, K, r, σ, T, N, M)
    # Initialize parameters
    Smax = 4 * S₀
    Δt = T / N
    Δx = Smax / M
    x = range(0, Smax, length=M+1)
    t = range(0, T, length=N+1)
    
    # Initialize option values
    V = zeros(M+1, N+1)
    
    # Set final condition
    V[:,N+1] = max.(x .- K, 0)
    
    # Solve using implicit finite difference
    for n in N:-1:1
        a = Δt * (r * (x .- Δx/2) ./ Δx)
        b = 1 .+ Δt * (r .+ 0.5 * σ^2 * (x ./ Δx).^2)
        c = -Δt * (r * (x .+ Δx/2) ./ Δx)
        
        A = diagm(-1 => a[2:end], 0 => b, 1 => c[1:end-1])
        V[:,n] = A \ V[:,n+1]
    end
    
    return V[findfirst(x .>= S₀),1]
end

# Example usage
S₀ = 100
K = 100
r = 0.05
σ = 0.2
T = 1
N = 100
M = 100

price = black_scholes(S₀, K, r, σ, T, N, M)
println("Option price: $price")




using Flux: Chain, Dense, relu
using DifferentialEquations
using Zygote
using Plots
# Parameters
S₀ = 100
K = 100
r = 0.05
σ = 0.2
T = 1.0

# Define the Neural Network Model
model = Chain(
    Dense(2, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 1)
)

# Loss function for the PINN
function loss_function(x, y_true)
    # Transpose the input to match the expected input size for the neural network
    x_t = x'
    
    # Predict the option price using the neural network
    y_pred = model(x_t)
    
    # Calculate the residual of the Black-Scholes equation
    dVdt = gradient(() -> model(x_t), x_t[2,:])[1]
    dVdS = gradient(() -> model(x_t), x_t[1,:])[1]
    d2VdS2 = gradient(() -> gradient(() -> model(x_t), x_t[1,:])[1], x_t[1,:])[1]
    residual = dVdt + 0.5 * σ^2 * x_t[1,:]^2 * d2VdS2 + r * x_t[1,:] * dVdS - r * y_pred
    
    # Mean Squared Error for the true values and the residual
    mse_loss = Flux.Losses.mse(y_pred, y_true)
    residual_loss = mean(abs2.(residual))
    
    return mse_loss + residual_loss
end

# Training Data
S_train = collect(range(0, stop=4*S₀, length=100))  # Stock prices
t_train = collect(range(0, stop=T, length=100))  # Time points
x_train = hcat(repeat(S_train, inner=100), repeat(t_train, outer=100))  # Combine S and t
y_train = max.(S_train .- K, 0)  # True option prices at maturity

# Training Loop
epochs = 1000
learning_rate = 0.01
opt = ADAM(learning_rate)

for epoch in 1:epochs
    # Forward pass and calculate loss
    loss = loss_function(x_train, y_train)
    
    # Backpropagation
    Flux.train!(model, [(x_train, y_train)], loss -> loss_function(x_train, y_train), opt)
    
    # Print loss every 100 epochs
    if epoch % 100 == 0
        println("Epoch $epoch, Loss: $loss")
    end
end

# Plot the results
S_test = collect(range(0, stop=4*S₀, length=100))
t_test = T
x_test = hcat(S_test, fill(t_test, 100))
y_pred = model(x_test')
plot(S_test, y_pred, label="Predicted Option Price")
xlabel!("Stock Price")
ylabel!("Option Price")
title!("Black-Scholes Option Pricing using PINN")