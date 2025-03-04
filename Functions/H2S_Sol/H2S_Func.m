% function for H2S term addition
function dxdt = H2S_Func(t,x,k1,k2,k3,k4,Qm,tau,vi0,mi,n);
dxdt=zeros(2,1);
dxdt(1)=x(2);
dxdt(2)=(1/tau)*(vi0-x(2))+(n/mi)*((k1*chng1(Qm/(4*3.14*x(2))))+(k2*chng2(Qm/(4*3.14*x(2))))+(k3*chng3(Qm/(4*3.14*x(2))))+(k4*chng4(Qm/(4*3.14*x(2)))));
end

function f1=chng1(val)
    if val<1
        f1=0;
    elseif val>=[1;5]
        f1=(val/5);
    else
        f1=1;
    end
end

function f2=chng2(val)
    if val<20
        f2=0;
    elseif val>=[20;50]
        f2=(val/50);
    else
        f2=1;
    end
end

function f3=chng3(val)
    if val<100
        f3=0;
    elseif val>=[100;500]
        f3=(val/500);
    else
        f3=1;
    end
end

function f4=chng4(val);
    if val<500
        f4=0;
    else
        f4=1;
    end
end