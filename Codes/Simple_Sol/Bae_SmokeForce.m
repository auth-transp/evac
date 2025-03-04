% Test1 - For this case all forces were ignored other than the smoke interaction
% force, also the force was not taken as a step function as indicated in
% the paper.
dis=2
sis=125
vinf=30
vi=15
mi=80
taui=0.25
vi0=[1.35;0;0;0;0]
vi=[1;1.2;1.35;1.4;1.5]
nis=[1]
fi=(((mi/taui)*(vi0-vi))+(sis*exp(-dis))+(sis*(vi/vinf))*nis)
%nis was redifened to [1] as it did not work otherwise due to matrix
%multiplication