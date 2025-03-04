%without the function for nis
function dxdt=odefcn7(t,x,vi0,tau,sis,nis,m,xs,Vi,Vinf)
    dxdt=zeros(2,1);
    dxdt(1)=x(2);
    dis=(x(1)-xs);
    dxdt(2)=((1/tau)*(vi0-x(2)))+(((sis*nis)/m)*(exp(-dis))*step(dis))+(((sis*nis)/m)*(Vinf/Vi)*(1-step(dis)));
end
function s=step(val)
    if val<0
        s=1;
    elseif val>=0
        s=0;
    end
end 