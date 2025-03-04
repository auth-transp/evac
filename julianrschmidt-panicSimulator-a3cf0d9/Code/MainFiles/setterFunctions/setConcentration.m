function [ settings ] = setConcentration( settings, Concentration )
%SETCONCENTRATION changes wall angle to desired value, also adapts xMax
settings.Concentration = Concentration;
if settings.xMaxCalcBool
    settings = setXMax(settings, calcXMax(settings));
end
end