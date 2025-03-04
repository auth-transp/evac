%Dose response assuming for symmptoms assuming that graph in Guidotti holds for all symptoms
%Response function taken from Excel Sheet
%Plots for concentration limits of symptoms
%1-5 for odor perception and teary eyes - C1
%20-200 for Lung+Eye irritation from minor to severe - C2
%250-500 for Pulmonary Edema - C3
%500+ Knockdown
%Mortality Function => time=231394e-0.009x

%Module 1 - Finding time for a given concentration range

% C1=0.1:0.05:5;    %Concentration limits for odor perception and teary eyes
% C2=20:1:50;       %Concentration limits for Lung+Eye irritation
% C3=250:1:500;     %Concentration limits for Pulmonary Edema

% exp1=(exp(-13.4*C1));
% exp2=(exp(-0.268*C2));
% exp3=(exp(-0.018*C3));

% time1=231394*(exp1); %response for odor perception and teary eyes
% time2=231394*(exp2); %response for Lung+Eye irritation
% time3=231394*(exp3); %response for Pulmonary Edema

% subplot(3,1,1)
% plot(C1,time1)
% title('Odor Perception');
% xlabel('Conc(ppm)');
% ylabel('time(sec)');
% 
% subplot(3,1,2)
% plot(C2,time2)
% title('Lung+Eye Irritation');
% xlabel('Conc(ppm)');
% ylabel('time(sec)');
% 
% subplot(3,1,3)
% plot(C3,time3)
% title('Pulmonary Edema');
% xlabel('Conc(ppm)');
% ylabel('time(sec)');

%Module 2 - Finding concentrtion for a given time

clear;
time1=1200;
C1=(log(time1/231394))/-13.4  %Concentration for Symptom 1

time2=1200;
C2=(log(time2/231394))/-0.268 %Concentration for Symptom 2

time3=1200;
C3=(log(time2/231394))/-0.018 %Concentration for Symptom 3


