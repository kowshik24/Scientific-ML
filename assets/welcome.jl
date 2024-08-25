println("Hello, World!")


x = 3

println("Tha value of x is $x")


# PDE solver

using DifferentialEquations
using Plots

function f(du, u, p, t)
    x, y = u
    du[1] = 1.01*x - x*y
    du[2] = -1.01*y + x*y
end

u0 = [1.0, 1.0]
tspan = (0.0, 100.0)
prob = ODEProblem(f, u0, tspan)
sol = solve(prob)

plot(sol, vars=(1,2))
