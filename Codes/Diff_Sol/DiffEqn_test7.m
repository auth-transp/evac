%trying to use ODE solvers in DiffEqn_test3%
syms tau vi0 x(t)
tau=0.25
vi0=1.35
Dx = diff(x,t)
eqn = diff(x,t,2)==-1*((1/tau)*Dx)+(vi0/tau)
cond = [x(0)==0, Dx(0)==1.35]
xSol = ode23t(eqn,[0 60],cond)