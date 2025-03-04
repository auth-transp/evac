function dxdt = odefcn8(t,x,tau,vi00,mi,sis,xs,Vi,Vinf)
n=f_nis(x(2));
vi0=-n*vi00;
dis=round((x(1)-xs),1);

dxdt=zeros(2,1);
dxdt(1)=x(2);
dxdt(2)=(1/tau)*(vi0-x(2))+(((sis*n)/mi)*((exp(-abs(dis))*step(dis))+((Vinf/Vi)*(1-step(dis)))));

term_out=(exp(-abs(dis))*step(dis));
term_in=((Vinf/Vi)*(1-step(dis)));
disp([num2str(t) ' &x ' num2str(x(1)) ' &dis ' num2str(dis) ' &vel ' num2str(x(2)) ' & ' num2str(n) ' &in ' num2str(term_in) ' &out ' num2str(term_out) ]);
end

function s=step(val)
    if val<0
        s=1; %inside
    else
        s=0; %outside
    end
end
function n=f_nis(vel)
    if vel>=0
            n=-1;
    elseif vel<0
            n=1;
    end
end