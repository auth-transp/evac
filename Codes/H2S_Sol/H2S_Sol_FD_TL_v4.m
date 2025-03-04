function [TL]=H2S_Sol_FD_TL_v4(Distance,Concentration)
    % return toxic load for contour calculations, otherwise run for plotting of method etc

% add ctf path
    addpath('../../Functions/H2S_Sol/');
    %clear;
    

    % Solution with new force terms analogous to motivational force
    % ft= m(v00-v0)/tau
    % where v00 is desired velocity for symptom
    % and v0 is the agent's desired velocity
    % instead of cTLV5 it employs the cTLV6 for a more representative TL (it
    %   calculates the time step TL and adds it to the previous

    if ~exist('Concentration', 'var')
        Concentration=450; 
    end %optional variables
    if ~exist('Distance', 'var')
        Distance=10000;
    end
    
    tin=0;      %Intial Time
    tfin=600;    %Final Time
    h=1;        %Step size for finite difference (s)

    % a1=0.8;
    % b1=1.2;

    m=80;       %Mass of agent (kg)

    v0=1.35;    %Average velocity of agent (m/s)
    t=(tin-2):h:(tfin);   %Time domain for finite difference (s)
    vo=0;       %Initial velocity of agent (m/s)

    x=zeros(1,length(t));       %Intialization
    xctf=0:1:300;              %for calculation of the concentration
    vel=zeros(1,length(t));     %Initialization
    force=zeros(1,length(t));   %Initialization
    TLsave=zeros(length(t),4);  %Initialization
    ctfsave=zeros(1,length(t));
    ToxAgent=SymptomsSetup('H2S');


    tau=1;          % Randomly generated tau
    symptau=1;      % Randomly generated sympotom tau
    mode='gauss2'; value=Concentration;
    concx=ctfV2(xctf,mode,value);

    for i=4:length(t)
        x(3)=0;                                    %Boundary Condition 1 - x(0)
        x(2)=(-vo*h);                              %Calculated by hand using boundary condition v(0) = x(0)-x(-1)/h
        x(1)=x(2);                                 %Just so x(1) has a value, doesn't affect solution
        %TL=cTLV5(ctf(x),t,1,SymptomsSetup('H2S')); %Toxic Load Function
        TL=cTLV6(ctfV2(x(i-1),mode,value),h,1,ToxAgent,1,TLsave(i-1,1:3)); %Toxic Load Function

        TLsave(i,1)=TL(1);
        TLsave(i,2)=TL(2);
        TLsave(i,3)=TL(3);
    %Toxic Load value extraction loop 
                if TL(1)>1  %Only level 1
                    TL(1)=1;
                end

                if TL(2)>1 % Only level 2
                    TL(2)=1;
                end

                if TL(3)>1 % Only level 3
                    TL(3)=1;
                end
                TL=sum(TL);
                TLsave(i,4)=TL;
    % Symptom desired velocity defined as a function of TL
                    if TL==0    
                       v00=v0;
                    elseif TL>0 && TL<=1
                        v00=1.35*(exp(0.393*TL));
                    elseif TL>1 && TL<=2
                        v00=-1.78*log(TL) + 2.063;
                    elseif TL>2 && TL<3
                        v00=-1.78*log(TL) + 2.063;
                    elseif TL>=3
                        v00=0;
                    end
                    %v00=v0;
    %             a3=0.8;
    %             b3=1.2;
    %             rnum3=(b3-a3).*rand(1,1) + a3; 
    %             
    %             v00=v00*rnum3;                      %Randomized v00

        x(i)=(((x(i-1)*(2*tau+h))-(x(i-2)*tau)+((h^2)*v0))/(tau+h))-(((v0-v00)/symptau)*tau*h^2)/(tau+h);  %Simplified Equation for finite difference
        vel(3)=vo;                              %Intial Velocity of Agent
        vel(i)=((x(i)-x(i-1))/h);               %Velocity of agent as calculated from FD
        force(i)=(m*(v0-vel(i))/tau)-(m*(v0-v00)/symptau); %Mutiplying base equation by m; and using the right side
        if vel(i)==0; break ;end
        ctfsave(i)=ctfV2(x(i),mode,value);
        %     if vel(i)>=0            %if loop for making velocity only positive
    %         vel(i)=vel(i);
    %     else
    %         vel(i)=0;
    %     end
    %     if x(i)>=x(i-1)         %if loop for making agent only go forward
    %         x(i)=x(i);
    %     else
    %         x(i)=x(i-1);
    %     end
          if x(i)>=Distance; break; end
    end


    %Saving
    save('output.mat','t','x','vel','force','ctfsave','TLsave','xctf','concx')
    max(concx)
    %Graphs

    subplot(5,2,1);
    plot(t(3:end),x(3:end));
    title('x vs t');
    xlabel('time(s)');
    ylabel('Dist(m)');

    subplot(5,2,5);
    plot(t(3:end),TLsave(3:end,4));
    title('TL vs t');
    xlabel('time(s)');
    ylabel('TL(-)');

    subplot(5,2,7);
    plot(t(3:end),vel(3:end));
    title('vel vs t');
    xlabel('time(s)');
    ylabel('Vel(m/s)');

    subplot(5,2,9);
    plot(t(3:end),force(3:end));
    title('force vs t');
    xlabel('time(s)');
    ylabel('Force(N)');

    subplot(5,2,3);
    plot(t(3:end),ctfsave(3:end));
    title('Conc vs t');
    xlabel('time(s)');
    ylabel('Conc(ppm)');

    subplot(5,2,4);
    plot(xctf,concx);
    title('Conc vs t');
    xlabel('x(m)');
    ylabel('Conc(ppm)');
end