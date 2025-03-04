%function defined using ODE45 solver for second order differential by
%converting second order differential to a set of two first order DE
%x''+(x'/tau)=(vi0/tau)
%let x'=x(2),
%then x''=x'(2)
%so x=x(1)
%Final Equations - 1. dx(1)/dt=x(2)
%                  2. dx(2)/dt=(1/tau)*(vi0-x(2))
%x(2) - Velocity
%x(1) - 'x' co-ordinate

function dxdt = odefcn1(t,x,tau,vi0,mi,sis,xs,Vi,Vinf,nis)
dxdt=zeros(2,1);
dxdt(1)=x(2);
dxdt(2)=(1/tau)*(vi0-x(2))+((sis*exp(-(x(1)-xs))*heaviside(x(1)-xs)+(sis*(Vi/Vinf)*(1-heaviside(x(1)-xs)*nis)))/mi);
end