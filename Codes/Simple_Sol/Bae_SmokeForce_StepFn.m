%test-2 in this test the 'dis' term is written as a step function using the
%heaviside command
Vi=5;
fi=(((mi/taui)*(vi0-vi))+(sis*exp(-dis)*step(dis)+(sis*(Vi/Vinf)*(1-step(dis))))*nis)