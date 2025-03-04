using Agents, Agents.Pathfinding
using Random
using ColorTypes
import ImageMagick
using FileIO: load
using GLMakie 
using InteractiveDynamics
using Images
using DataFrames
using Statistics

#TL_RATE = dTL/dt = (rho/Brho(b,k))**alpha(b,k) / Btime(b,k); rho is the density of the agent, Brho is the AEGL band density

#1 Symptom speed set up
#2 Toxicload rate calculate_dispersion (toxic dispersion model)

@agent Agent ContinuousAgent{2} begin
    toxicload::Float64
end


# Conc Cfx 
#     function Ct(x, mode, value=450)
#         if mode == "flat"
#             Ct = value
#         elseif mode == "gauss1"
#             Ct = 56000 * pdf(Normal(150, 50), x)
#         elseif mode == "gauss2"
#             Ct = 5400 * pdf(Normal(150, 3.8), x)
#         elseif mode == "zero"
#             Ct = 0
#         end
#     end
    

#1 Symptom setup
function symptom_setup(agent_chemical)
    Atime = [0.17, 0.83, 1.67, 4.17, 8.33] #min 
    Arho = zeros(3, 5)

    if agent_chemical == "H2S"
        Arho[1, :] = [4.85, 4.23, 4.17, 4.06, 3.82]
        Arho[2, :] = [180.79, 157.56, 155.43, 151.37, 142.48]
        Arho[3, :] = [485.62, 423.22, 417.49, 406.59, 382.71] #ppm
        MW = 34 #Molecular Weight of h2S
    end

    Arho = Arho';
    Atime = Atime*60; #seconds
    taumin = 0.1;
    taumax = 1e6;
    Brho = zeros(7,3);
    alpha = zeros(6,3);
    rhomax = zeros(1,3);
    Btime = zeros(7,3);

    for k=1:3
        for b=1:5
            Brho[b + 1, k] = Arho[b,k]
            Btime[b + 1] = Atime[b]
        end
        Btime[0+1, k] = taumin;
        Btime[6+1, k] = taumax;
    end

    for k=1:3
        for b=2:5
            if Brho[b-1+1, k]==Brho[b+1, k]
                alpha[b, k] = 0;
            else
                alpha[b, k] = log(Atime[b]/Atime[b-1])/log(Brho[b-1+1, k]);
            end
        end

        alpha[1, k] = alpha[2, k];
        alpha[6, k] = alpha[5, k];
    end

    for k=1:3
        if alpha[2, k]==0
            rhomax[k] = Brho[1+1, k];
            Brho[0+1, k] = rhomax[k];
        else
            rhomax[k] = Brho[1+1, k]*(Btime[1+1, k]/taumin)^(1/alpha[1, k]);
            Brho[0+1, k] = rhomax[k];
        end

        if Brho[4+1, k]==Brho[5+1, k]
            rhomin[k] = Brho[5+1, k];
            Brho[6+1, k] = rhomin[k];
        else
            rhomin[k] = Brho[5+1, k]*(Btime[5+1, k]/taumax)^(1/alpha[5, k]);
            Brho[6+1, k] = rhomin[k];
        end

    end

    for k=1:3
        for b=1:6
            if alpha[b, k]==0
                Btime[b+1, k]=Btime[b-1+1, k];
            end
        end
    end

    for k=1:3
        for b=2:4
            if alpha[b-1, k]==0 && alpha[b, k]>0
                alpha[b, k]=log(Btime[b+1, k]/Btime[b-1+1, k])/log(Brho[b-1+1, k]);
            end
        end
    end

    agent_chemical.Btime=Btime;
    agent_chemical.Brho=Brho';
    agent_chemical.alpha=alpha;
end




#2 Toxicload rate-Toxic dispersion model
function ToxicLoad(Ct, dt, AEGLk, agent_chemical, n_agents, TLcurrent) 
    a = agent_chemical.alpha
    Btime = agent_chemical.Btime;
    Brho = agent_chemical.Brho;
    TL = zeros(nAgent, 3);
    
    for iAg = 1:n_agents
        TL[iAg, :] =TLcurrent(iAg, 1:3);
        for k = 1:3
            Cmin = Brho(k, 7);
            Cmax = Brho(k,1);

            if Ct(iAg)>Cmax
                TL_rate = 1/Btime(1);
            elseif Ct(iAg)<Cmin
                TL_rate = 0;
            else
                for i = 2:length(Btime)
                    if Brho(k, i-1)<Ct<Brho(k,i)
                        TL_rate = (1/Btime(i))*((Ct(iAg)/Brho(k,i))^(a(i-1))); #Toxic Load rate in units of s^-1
                    end
                end
            end
            TL[iAg, k] = TL(iAg, k) + TL_rate*dt;
        end
    end
end

