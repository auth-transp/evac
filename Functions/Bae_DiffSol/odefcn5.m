% force keeps incresing even when the visibility is very low
% trying to solve by inverting the function
function dxdt = odefcn5(t,x,tau,vi00,mi,sis,xs,Vi,Vinf,nis)
dxdt=zeros(2,1);
dxdt(1)=x(2);
n=f_nis(x(2));
vi0=-n*vi00;
dis=(x(1)-xs);
dxdt(2)=(1/tau)*(vi0-x(2))+(((sis*nis)/mi)*(exp(-(dis))*step(dis)))+(((sis*nis)/mi)*(Vinf/Vi)*(1-step(dis)));
disp([num2str(t) ' & ' num2str(x(1)) ' & ' num2str(x(2)) ' & ' num2str(nis) ' &' num2str(Vinf/Vi)]);
end

function s=step(val)
    s=1;
    if val<0
        s=1;
    elseif val>=0
        s=0;
    end
end
function n=f_nis(vel)
    n=-1;
    if vel>0
            n=-1;
    elseif vel<0
            n=1;
    end
end