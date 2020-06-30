%解方程
clc;
clear;

syms d i theta;

eq1=2*d*sind((i+theta)/2)*cosd(6.5)==500*10^(-6);
eq2=i-theta==13;
eq3=(d*cosd(theta))/750==4*10^(-6);

[d, i, theta] = solve(eq1, eq2, eq3, d, i, theta)  %加上分号运行不会显示结果