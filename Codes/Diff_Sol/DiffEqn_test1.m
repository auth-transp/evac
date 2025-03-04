%solution of equation without using any conditions%
syms tau vi0 x(t)
eqn = diff(x,t,2) == -1*((1/tau)*diff(x,t))+(vi0/tau)
tau=0.25
vi0=1.35
xSol(t) = dsolve(eqn)