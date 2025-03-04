%Solving the same as 'DiffEqn_test2' but by uing ode23t function%
function dxdt=odefcn(x,t,tau,vi0)
dxdt=zeros(2,1)
dxdt(1)=x(2)
dxdt(2)=(1/tau)*(vi0-x(2))
tau=0.25
vi0=1.35
tspan=[0 60]
x0=[0 0]
[t,x]=ode23(@(x,t)odefcn(x,t,tau,vi0),tspan,x0)