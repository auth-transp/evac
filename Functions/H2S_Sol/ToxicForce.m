% Toxic Force as a function of literature velocities and TL

function F=ToxicForce(TL,m,v,tau)

if TL==0
    F=0;
elseif TL>0 && TL<=1
    v00=2;
    F=m*((v00-v)/tau);
% elseif TL==1
%     v00=2;
%     F=-m*((v00-v)/tau);
elseif TL>1 && TL<=2
    v00=1;
    F=m*((v00-v)/tau);
% elseif TL==2
%     v00=1;
%     F=m*((v00-v)/tau);
elseif TL>2 && TL<3
    v00=0.5;
    F=m*((v00-v)/tau);
elseif TL>=3
    v00=0;
    F=m*((v00-v)/tau);
end