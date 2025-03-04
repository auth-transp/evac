%nis loop attempt
function dxdt=odefcn6(t,x,vi00,tau,sis,nis,m,xs,Vi,Vinf)
    dxdt=zeros(2,1);
    dis=(x(1)-xs);
    q=chng(x(2));
    p=nis*q;
    vi0=q*vi00;
    dxdt(1)=x(2);
    dxdt(2)=((1/tau)*(vi0-x(2)))+(((sis*p)/m)*(exp(-dis))*step(dis))+(((sis*p)/m)*(Vi/Vinf)*(1-step(dis)));
    disp([num2str(t) ' & ' num2str(x(1)) ' & ' num2str(x(2)) ' & ' num2str(p) ' &' num2str(Vinf/Vi)]);
end
function s=step(val)
    if val<0
        s=0;
    else
        s=1;
    end
end
function q=chng(vel)
    if vel<0
        q=-1;
    elseif vel == 0
        q=-1;
    else
        q=1;
    end
end
