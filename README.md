# EVAC_H2S_Matlab

## 1. Structure of files (folders)
There are total of 5 folders
- [Codes](https://github.tamu.edu/SECAReLab/EVAC_H2S_Matlab/blob/master/README.md#3-single-agent-solution-h2s_sol) - This is where the codes for single agent solutions are - 'H2S_Sol'
- Functions - The functions required for the project are stored here including those used in PanicSimulator and 'Single Agent Solution'
- julianrschmidt-panicSimulator-a3cf0d9 - This is the folder that contains the edited code for the [Panic Simualtor](https://github.tamu.edu/SECAReLab/EVAC_H2S_Matlab#2-panic-simulator), renaming this folder may cause issues with the program
- videos - This is the folder where PanicSimulator stores the simulation videos if requested, renaming will cause errors in saving of videos
- Workspaces - This folder contains MATLAB workspaces that contained some results of relevance

## 2. Panic Simulator
### 2.1 How to create and run custom geometries and agent positions?
1. Run Panic Simulator
2. Go to 'Arena Editor' from 'Options'
3. Manually create geometry from tools and agent positions
4. Go to 'File' and 'Save Agents As' in the folder 'presets'
5. Repeat with 'Save Walls As'
6. Go to folder 'presets' and open the array 'defaultSettings'
7. Make sure 'agentPositionStyle' is set to 'filename'
8. Make sure 'wallPositionStyle' is set to 'filename'
9. Copy and paste the path of the saved file from '4.' in 'agentPositionFilename'
10. Copy and paste the path of the saved file from '5.' in 'wallPositionFilename'
11. Overwrite 'defaultSettings.mat' file with changes
12. DO NOT RENAME THE 'defaultSettings.mat' FILE

### 2.2 How to run multiple simulation runs for different concentrations?
1. Go to file 'Code'->'Automate'->'createAutomateObj.m'
2. Change second matrix in line 19 of code
3. Format for change [Initial Concentration, Final Concentration, Step Size]
4. Run PanicSimulator
5. Go to 'Automate' in 'Options'
6. From drop down menu choose 'Concentration'
7. Press 'Ok' and Run

### 2.3 How to terminate the simulation in case all agents do not exit?
This is done by a new variable at which the simulation terminates irrespective of agent position
1. Go to file 'Code'->'Automate'->'automate.m'
2. Go to line 24 of code and in the 'if' loop change the value of 'simulationObj.tSimulation' as per requirment
The simulation will terminate after this time (in seconds)

### 2.4 How to set the concentration if 'Automate' is not being used?
1. Go to folder 'presets' and open the array 'defaultSettings'
2. Change the value of variable 'Concentration' as per requirment
3. Overwrite 'defaultSettings.mat' file with changes
4. DO NOT RENAME THE 'defaultSettings.mat' FILE


=======
## 3. Single Agent Solution (H2S_Sol)
### 3.1 How to run individual test cases by finite difference? (H2S_Sol)
1. Open file Codes->H2S_Sol->H2S_Sol_FD_TL_v3.m
2. Open file Functions->H2S_Sol->ctf.m
3. To change concentration, change value of ct(i) in ctf.m. There are two functions, one is for dynamic concentrations, and the other for constant concentration.
4. Change value of 'tfin' in H2S_Sol_FD_TL_v3.m to change simulation length (in seconds)
5. To change solution step size, change value of 'h' in H2S_Sol_FD_TL_v3.m
6. Run H2S_Sol_FD_TL_v3.m to obtain results in terms of graphs.
>>>>>>> 8c8399458291425a0d588f2a01ce76bd91792999
