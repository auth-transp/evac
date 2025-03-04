% draw TL contours for a range of x and concentration
clear;
xrange=[5,10,20,30,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400];
concrange=[1,5,10,15,20,25,50,60,70,80,90,100,125,150,175,200,225,250,300,350,400,500];
m=size(xrange,2);
n=size(concrange,2);
TLContour=zeros(m,n);
parfor i=1:m
    for j=1:n
       TLContour(i,j)=H2S_Sol_FD_TL_v4(xrange(i),concrange(j));
    end    
end
contour(concrange,xrange,TLContour,[1,2,3]);
