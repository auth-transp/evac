%solution of equation wuth using any conditions
%at t=0; x=0 & at t=0; v=0
syms tau vi0 x(t)
Dx = diff(x,t)
eqn = diff(x,t,2)==-1*((1/tau)*Dx)+(vi0/tau)
tau=0.25
vi0=1.35
cond = [x(0)==0, Dx(0)==1.35]
t=[0:1:60]
xSol = dsolve(eqn,cond)