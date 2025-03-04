%Concentration as a function of 'x' to be used with cTLV6
%Master file for ambient concentration



function ct=ctfV2(x,mode,value)
clear ct;

    if ~exist('value', 'var')
        value=450;
    end
    switch mode;
        case 'flat'
               ct=value;
        case 'gauss1'
            ct=56000*normpdf(x,150,50);
        case 'gauss2'
            ct=5400*normpdf(x,150,3.8);
        case 'zero'
            ct=0;
        case 'linear'
            if  x>=0 && x<20; %Flat line 
                ct=380;
            elseif x>=20 && x<200 %
                ct=(5-380)/(200-20)*(x-20)+380;    
            elseif x<0 % 0ppm for negtive x
                ct=0;
            else           %
                ct=5;
            end
    end

end

% function ct=ctf(x)
% clear ct;
% for i=1 : length(x)
%     if  x(i)>0 && x(i)<=6; %Flat line at 0-6 m
%         ct(i)=0;
%      elseif x(i)>6 && x(i)<12 % slow rise to 50ppm at 6-12 m
%          ct(i)=((25/3)*x(i))-50;
%      elseif x(i)>=12 && x(i)<=30 %slow fall to 20ppm at 12-30
%          ct(i)=(-(10/9)*x(i))+(160/3);
%      elseif x(i)>30 && x(i)<=40 %slow rise to 35ppm at 30-40m
%          ct(i)=((3/2)*x(i))-25;
%      elseif x(i)<=0 % 0ppm for negtive x
%          ct(i)=0;
%      else           %35ppm constant after 40m
%          ct(i)=35;
%      end
% end
% end

% function ct=ctf(x)
% clear ct;
% for i=1 : length(x)
%     if x(i)>0;
%         ct(i)=380;
%     else
%         ct(i)=0;
%     end
% end