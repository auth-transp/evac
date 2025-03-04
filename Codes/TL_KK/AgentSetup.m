
function [Agent]=AgentSetup(agent)
Atime=[10 30 60 240 480]; % in min
switch agent
    case 'Cl2'
        Arho=[0.5 0.5 0.5 0.5 0.5,
            2.8 2.8 2.0 1.0 0.71,
            50 28 20 10 7.1]; % in ppm
        MW=70.9;
    case 'H2S'
        Arho=[0.755 0.60 0.51 0.36 0.33,
            41 32 27 20 17,
            76 59 50 37 31]; % in ppm
        MW=34;
end


Arho=Arho';%*MW/24.04; %1 atm @ 20oC
Atime=Atime*60; % in secs
taumin=30;
taumax=3600*24;

Brho=zeros(7,3);
alpha=zeros(6,3);
rhomax=zeros(1,3);
rhomin=zeros(1,3);
Btime=zeros(7,3);

%Initialize
for k=1:3
    for b=1:5
        Brho(b+1,k)=Arho(b,k);
        Btime(b+1,k)=Atime(b);
    end
    Btime(0+1,k)=taumin;
    Btime(6+1,k)=taumax;
end
%power low trial values
for k=1:3
    for b=2:5
        if Brho(b-1+1,k)==Brho(b+1,k)
            alpha(b,k)=0;
        else
            alpha(b,k)=log(Atime(b)/Atime(b-1))/log(Brho(b-1+1,k)/Brho(b+1,k));
        end
    end
        alpha(1,k)=alpha(2,k);
        alpha(6,k)=alpha(5,k);
end 
%extrapolate for the edges
for k=1:3
    if alpha(2,k)==0
        rhomax(k)=Brho(1+1,k);
        Brho(0+1,k)=rhomax(k);
    else
        rhomax(k)=Brho(1+1,k)*(Btime(1+1,k)/taumin)^(1/alpha(1,k));
        Brho(0+1,k)=rhomax(k);
    end
    if Brho(4+1,k)==Brho(5+1,k)
        rhomin(k)=Brho(5+1,k);
        Brho(6+1,k)=rhomin(k);
    else
        rhomin(k)=Brho(5+1,k)*(Btime(5+1,k)/taumax)^(1/alpha(5,k));
        Brho(6+1,k)=rhomin(k);
    end
end
%Correct to account for threshold values
for k=1:3
    for b=1:6
        if alpha(b,k)==0
            Btime(b+1,k)=Btime(b-1+1,k);
        end
    end
end
for k=1:3
    for b=2:4
        if alpha(b-1,k)==0 && alpha(b,k)>0
            alpha(b,k)=log(Btime(b+1,k)/Btime(b-1+1,k))/log(Brho(b-1+1,k)/Brho(b+1,k));
        end
    end
end
Agent.Btime=Btime;
Agent.Brho=Brho';
Agent.alpha=alpha;
end