function dxdt = odefcn9(t,x,tau,vi0,mi,sis,xs,Vi,Vinf,nis)

dis=round((x(1)-xs),1);

dxdt=zeros(2,1);
dxdt(1)=x(2);
dxdt(2)=(1/tau)*(vi0-x(2))+(((sis*nis)/mi)*((exp(-abs(dis))*step(dis))+((Vinf/Vi)*(1-step(dis)))));

term_out=(exp(-abs(dis))*step(dis));
term_in=((Vinf/Vi)*(1-step(dis)));
disp([num2str(t) ' &x ' num2str(x(1)) ' &dis ' num2str(dis) ' &vel ' num2str(x(2)) ' &in ' num2str(term_in) ' &out ' num2str(term_out)]);
end

function s=step(val)
    if val<0
        s=1; %inside
    else
        s=0; %outside
    end
end
