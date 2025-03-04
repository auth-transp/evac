function [ color ] = colorMapTL( TL )
%COLORMAPPRESSURE returns a color related to a pressure value
%   a TL of 0 will result in green
%   a TL of 2 will result in yellow
%   a TL of 3 will result in red
%   pressure values in between take color values in between

forestGreen = [34,139,34]/255;
yellow = [255, 255, 0]/255;
red = [255, 0, 0]/255;

maxTL = 3;
middleTL = 2;
minTL = 0;

if TL < minTL
    color = forestGreen;
elseif TL < middleTL
    percentage = (TL-minTL)/(middleTL-minTL);
    color = percentage*yellow + (1-percentage)*forestGreen;
elseif TL < maxTL
    percentage = (TL-middleTL)/(maxTL-middleTL);
    color = percentage*red + (1-percentage)*yellow;
else
    color = red;
end

end