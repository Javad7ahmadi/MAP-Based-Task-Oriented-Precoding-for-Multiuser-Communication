clc;close all;clear;iiii=0;



iiii=iiii+1;
load('OurStatistics.mat')
hold all;plot(test_acc_curve, 'LineWidth', 2);


iiii=iiii+1;
load('mcr2_statistics.mat')
hold all;plot(test_acc_curve, 'LineWidth', 1);


ylabel('Test Accuracy', 'Interpreter', 'latex')

xlabel('Epoch', 'Interpreter', 'latex')


grid on
box on