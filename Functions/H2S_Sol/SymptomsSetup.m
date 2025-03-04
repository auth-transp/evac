
function [Agent]=SymptomsSetup(agent)
Atime=[0.17 0.83 1.67 4.17 8.33]; % in min - 10, 50, 100, 250, 500 seconds
switch agent
    case 'Cl2'
        Arho=[0.5 0.5 0.5 0.5 0.5,
            2.8 2.8 2.0 1.0 0.71,
            50 28 20 10 7.1]; % in ppm
        MW=70.9;
    case 'H2S'
        Arho=[4.85 4.23 4.17 4.06 3.82,
            180.79 157.56 155.43 151.37 142.48,
            485.62 423.22 417.49 406.59 382.71]; % in ppm
        MW=34;
end


Arho=Arho';%*MW/24.04; %1 atm @ 20oC
Atime=Atime*60; % in secs
taumin=.1; %arbitrary value
taumax=1e6; %arbitrary value

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