% introduced a new function 'n' for 'nis' to change its value from -1 to 1 when
% the velocity becomes negative
% made the average velocity change directions when the inst. velocity
% changes direction by making vi0 a function of 'n'
function dxdt = odefcn4(t,x,tau,vi00,mi,sis,xs,Vi,Vinf,nis)
dxdt=zeros(2,1);
dxdt(1)=x(2);
n=f_nis(x(2));
vi0=-n*vi00;
dis=abs(x(1)-xs);
dxdt(2)=(1/tau)*(vi0-x(2))+(((sis*nis)/mi)*((exp(-dis)*step(dis))+((Vi/Vinf)*(1-step(dis)))));
disp([num2str(t) ' & ' num2str(x(1)) ' & ' num2str(x(2)) ' & ' num2str(nis)]);
end

function s=step(val)
    if val>=0
        s=1;
    else
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