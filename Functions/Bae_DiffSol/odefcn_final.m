% Final function with just one step function for dis%
function dxdt = odefcn_final(t,x,tau,vi0,mi,sis,xs,Vi,Vinf,nis)
dxdt=zeros(2,1);
dxdt(1)=x(2);
dis=(x(1)-xs);
dxdt(2)=(1/tau)*(vi0-x(2))+(((sis*exp(-dis)*step(dis)*nis)+(sis*(Vi/Vinf)*(1-step(dis)))*nis)/mi);
end
function s=step(val)
    if val>=0
        s=1;
    else
        s=0;
    end
end