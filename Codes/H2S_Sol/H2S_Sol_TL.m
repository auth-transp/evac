%Toxic Load Approach

clear;
tau=1;
vi0=1.35;
mi=80;

[t,x]=ode23(@(t,x) H2S_Func_TL(t,x,vi0,tau,mi),[0 60],[0 1.35]);

for i=1:length(t)
    dxdt = H2S_Func_TL(t,x,vi0,tau,mi);
    fs(i)=mi*dxdt(2);
end

subplot(4,1,1);
plot(t,fs)
title('fs vs t')
subplot(4,1,2);
plot(t,x(:,2))
title('v vs t')
subplot(4,1,3);
plot(t,x(:,1))
title('x vs t')
subplot(4,1,4);
plot(x(:,1),ctf(x));
title('c vs x')

disp(cTLV4(ctf(x(:,1)),t,3,AgentSetup('H2S')))