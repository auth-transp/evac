%Toxic Load Approach

function dxdt=H2S_Func_TL(t,x,vi0,tau,mi)   
dxdt=zeros(2,1);
dxdt(1)=x(2);

dxdt(2)=((1/tau)*(vi0-x(2))-((1/mi)*50*cTLV4(ctf(x(:,1)),t,1,AgentSetup('H2S'))));
end