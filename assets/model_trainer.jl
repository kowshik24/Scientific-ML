using Flux: Chain, Dense, relu, ADAM
using Zygote
using DifferentialEquations
using Plots
using Statistics  # Import Statistics module for mean function

# Define a simple feedforward neural network
model = Chain(
    Dense(1, 17, relu), 
    Dense(17, 1024, relu), 
    Dense(1024, 1)
)

# Define the Loss Function
function loss(x, y)
    y_pred = model(x)
    mse = mean((y .- y_pred).^2)  # Calculate mean squared error
    return mse
end

# Define the Training Loop
function train!(model, x, y; epochs=1000, lr=0.01)
    ps = Flux.params(model)
    opt = ADAM(lr)
    for epoch in 1:epochs
        Flux.train!(loss, ps, [(x, y)], opt)
        if epoch % 100 == 0
            println("Epoch $epoch, Loss: $(loss(x, y))")
        end
    end
end

# Generate Training Data
function f!(du, u, p, t)
    x, y = u
    du[1] = 1.01*x - x*y
    du[2] = -1.01*y + x*y
end

u0 = [rand(), rand()]
tspan = (0.0, 10.0)
prob = ODEProblem(f!, u0, tspan)
sol = solve(prob, Tsit5())

# Prepare data for training
x = reshape(sol.t, :, 1)  # Time points as inputs
y = reshape(sol[1, :], :, 1)  # First component of the solution as output

# Ensure x and y are matrices with proper dimensions
x = convert(Matrix, x)  # Convert to Matrix if needed
y = convert(Matrix, y)  # Convert to Matrix if needed

# Adjust dimensions if necessary
x = x'  # Transpose x to be (1, N) where N is the number of samples
y = y'  # Transpose y to be (1, N) where N is the number of samples

# Verify the new shapes
println("Adjusted x shape: ", size(x))
println("Adjusted y shape: ", size(y))

# Train the Neural Network
train!(model, x, y)

# Generate Predictions
x_test = range(0, stop=10, length=100)
x_test = reshape(x_test, :, 1)  # Ensure x_test is in the correct shape
x_test = x_test'  # Transpose x_test to be (1, 100)

y_test = model(x_test)

# Plot the Results
plot(sol.t, sol[1, :], label="True Solution", xlabel="Time", ylabel="Solution", title="PDE Solver with Neural Network")
plot!(x_test[1, :], y_test[1, :], label="Predicted Solution", linestyle=:dash)


# Plot the Error
error = abs.(y_test .- sol[1, :])
plot(x_test[1, :], error[1, :], label="Error", xlabel="Time", ylabel="Error", title="Error between True and Predicted Solution")

# Return the model for further analysis
model
