%·Ö¶Îº¯Êý»æÖÆ
x = (0:1:100);
y = zeros(size(x));
for i = 1:length(x)
    if  0<=x(i) && x(i)<=25
        y(i) = 1;
    elseif  25<x(i) && x(i)<=100
        y(i) = 1/(1+((x(i)-25)/5)^2)^2;
    
    end 
 end       

plot(x,y)