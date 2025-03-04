%Concentration as a function of 'y' to be used with cTLV5
%Master file for ambient concentration

function ct=ctfy(y)
clear ct;
for i=1 : length(y)
    if y(i)>6;
        ct(i)=1;
    else
        ct(i)=0;
    end
end