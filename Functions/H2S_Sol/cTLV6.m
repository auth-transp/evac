% cTLV5 modified for PanicSim
% Instead of using time steps, it uses the previous toxic load and adds it
% to the newly calculated toxic load, for every step of the ODE
% see also: cTLV4, cTLV5

function [TL]=cTLV6(ct,tt,AEGLk,Agent,NAgent,TLcurrent)
%This function computes the Toxic Load Rate, and the Toxic Load, using as input the AEGL
%concentrations and times, as well as the instantanoues concentration obtained from
%the dispersion model for a specific location.
a=Agent.alpha;
Btime=Agent.Btime;
Brho=Agent.Brho;
TL=zeros(NAgent,3);
%NAgent = size(agents, 1); %get number of agents
    for iAg=1:NAgent
        TL(iAg,:)=TLcurrent(iAg,1:3);
        for k=1:3   
            cmin=Brho(k,7);
            cmax=Brho(k,1);

            %Determine in which band the instantanous concentration is, then calculate
            %the TL_rate:

            %for tstep=2:length(ct)
                if ct(iAg)>cmax
                    TL_rate=1/Btime(1);
                elseif ct(iAg)<cmin
                    TL_rate=0;
                else 
                    for i=2:length(Btime)
                        if Brho(k,i-1)<ct<Brho(k,i)
                            TL_rate=(1/Btime(i))*((ct(iAg)/Brho(k,i))^(a(i-1)));%Toxic load rate in units of s^-1
                        end
                    end   
                end
                %dt=tt(tstep)-tt(tstep-1);
                TL(iAg,k)=TL(iAg,k)+TL_rate*tt;%Toxic Load
            %end
        end
    %TL(iAg,1:4)=sum(TL>1)+TL(min(3,sum(TL>1)+1))*(1-(TL(3)>1)); %One value of Toxic load between 0&3 as per scheme of H2S_SOL_FD_TL
    end
end   