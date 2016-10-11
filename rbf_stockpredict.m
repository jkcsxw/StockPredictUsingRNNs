% rbf_stockpredict.m

%% 清除工作空间中的变量和图形
clear,clc
close all

load elm_stock
%% ????
t1 = 1 : 280;
%t1 = t1';

ty = zeros(1,280);

for i=1:280
    ty(1,i) = price(i);
end

%% ????
t = 1:57;
%t = t';

y = zeros(1,57);
for i=1:57
    y(1,i) = price(i+280);
end

%% ???????
net = newrb(t1, ty);

%% ????

testy = sim(net, t);

%% ??
figure('Name', '???');
plot(t, y, 'b-');

hold on;
plot(t, testy, 'r--');
legend('initial data','RBF prediction');
title('RBF prediction');
hold off
grid on