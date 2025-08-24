using ModelingToolkit
using DifferentialEquations
using MethodOfLines

@variables x t
@variables c(..)

∂t = Differential(t)
∂x = Differential(x)

# advection-diffusion constants
R = 1.0
D = 1.0
v = 1.0
cₒ = 1.0

# advection-diffusion equation
eqn = R * ∂t(c(x, t)) ~ D * ∂x(c(x, t))^2 - v * ∂x(c(x, t))

# initial and boundary conditions
bcs = [c(x, 0) ~ cₒ]

# space and time domains
dom = [x ∈ (0, 1), t ∈ (0, 1)]

# define PDE system
@named sys = PDESystem(eqn, bcs, dom, [x, t], [c(x, t)])

# convert the PDE into an ODE problem
prob = discretize(sys, MOLFiniteDifference([x => 100], t))

# solve the problem
sol = solve(prob, Tsit5(), saveat=0.2)