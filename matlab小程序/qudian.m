y=0:0.05:3.5;
x=13.265-13.265*((1-(y.^2)/(19.863*19.863)).^0.5);
pathout = 'D:\caomeng\ѧϰ\3learn\MATLAB\test\111.xlsx';
Title = {'x';'y'};
xlswrite(pathout,Title,1,'A1');
xlswrite(pathout,x,1,'B1');
xlswrite(pathout,y,1,'B2')