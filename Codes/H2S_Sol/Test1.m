
% Test case for cTLV5 without calculating TL at every point
% clear;
% x=1:200;
% t=1:75;
% ToxicLoad=cTLV4(ctf(x),t,2,AgentSetup('H2S'))
% plot(x,ctf(x))


%Test case for TLvsF function by manually inputing the Toxic Load
% Fmax=120;
% ToxicLoad=(0:0.01:3);
% F=TLvsF(ToxicLoad,Fmax);
% plot(ToxicLoad,F)
% title('TL vs F');
% xlabel('TL');
% ylabel('F(N)')


% %Test case for ctf
% x=1:60;
% c=ctf(x);
% plot(x,c)

%Test case for Toxic Load

% clear;
% for x=2:102;
% t=1;
% ct=ctf(x);
% TL(x)=cTLV6(ct,t,1,SymptomsSetup('H2S'));
% TL=sum(TL);
% end


% x=1;
% t=0.05;
% TL=cTLV5(ctf(x),t,1,SymptomsSetup('H2S'));
% TL=sum(TL>1)+TL(min(3,sum(TL>1)+1))*(1-(TL(3)>1));


%Test Case for Toxic Force
TL=3;
tau=1;
m=80;
v0=1.35;
vo=2;
F=m*((v0-vo)/tau)
Ft=ToxicForce(TL,m,vo,tau)
