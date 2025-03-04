function [TL]=cTLV4(ct,tt,AEGLk,Agent)
 %This function computes the Toxic Load Rate, and the Toxic Load, using as input the AEGL
%concentrations and times, as well as the instantanoues concentration obtained from
%the dispersion model for a specific location.
a=Agent.alpha;
Btime=Agent.Btime; %[200,600,1800,3600,14400,28800,86400];%time extrapolated to tmin and tmax
% Brho(1,:)=[0,1.063851913,0.851081531,0.723419301,0.510648918,0.468094842,0];%AEGL1 concentrations
% Brho(2,:)=[0,58.15723794,45.39101498,38.29866889,28.36938436,24.11397671,0];%AEGL2 Concentrations
% Brho(3,:)=[0,107.8036606,83.68968386,70.9234609,52.48336106,43.97254576,0];%AEGL3 Concentrations
%Compute cmax (i.e. Brho(1)) and cmin (i.e. Brho(7))corresponding to tmin and tmax, respectively:
%cmax=Brho(1):
Brho=Agent.Brho;
    TL=zeros(1,3);
for k=1:3
    %cmax=Brho(k,2)*((Btime(2)/Btime(1))^(1/a(1)));
    %Brho(k,1)=cmax;
    %cmin=Brho(7):
    %cmin=Brho(k,6)*((Btime(6)/Btime(7))^(1/a(6)));
    %Brho(k,7)=cmin; 
    
    cmin=Brho(k,7);
    cmax=Brho(k,1);
    
    %Determine in which band the instantanous concentration is, then calculate
    %the TL_rate:

    for tstep=2:length(tt)
        if ct(tstep)>cmax
            TL_rate=1/Btime(1);
        elseif ct(tstep)<cmin
            TL_rate=0;
        else 
            for i=2:length(Btime)
                if Brho(k,i-1)<ct(tstep)<Brho(k,i)
                    TL_rate=(1/Btime(i))*((ct(tstep)/Brho(k,i))^(a(i-1)));%Toxic load rate in units of s^-1
                break;
                end
            end   
        end
        dt=tt(tstep)-tt(tstep-1);
        TL(k)=TL(k)+TL_rate*dt;%Toxic Load
    end
end
TL=squeeze(TL(AEGLk)); %temporary extract only one of the AEGLk
end   