% this is the combined force solved with all parameters for the smoke term
% in Bae's paper and the first velocity term
fc=((mi/taui)*(vi0-vi))+((a1*exp((r12-d12)/b1)*(lambda1+((1-lambda1)*(1+cos(phi12)/2))))*n12)+((aw*exp((r1w-d1w)/bw)*(lambdaw+((1-lambdaw*(1+cos(phi1w)/2))))*n1w))+(sis*exp(-dis)*heaviside(dis)+(sis*(vi/vinf)*(1-heaviside(dis)*nis)))