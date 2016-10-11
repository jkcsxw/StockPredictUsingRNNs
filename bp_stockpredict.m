%% 清除工作空间中的变量和图形
clear,clc
close all

%% 1.加载337期上证指数开盘价格
load elm_stock

whos
rng(now)
price=price(:);

train_num=280;
price1=price(1:train_num);  %前280个数据为训练组
price2=price(train_num+1:337);      %后面的数据为测试组

x=price1';
%[x,ps]=mapminmax(x,0,1);   %归一化处理，可自行添加或不加

lag=6;    % 自回归阶数
iinput=x; % x为原始序列（行向量）
n=length(iinput);

%准备输入和输出数据
inputs=zeros(n-lag,lag);
for i=1:n-lag
    inputs(i,:)=iinput(i:i+lag-1)';
end
targets=x(lag+1:end);

%创建网络
P=inputs;
P=P';
T=targets;
net=newff(minmax(P),[10,1],{'logsig','purelin'},'trainlm');
inputWeights=net.IW{1,1} ;
inputbias=net.b{1}; 
layerWeights=net.LW{2,1} ;
layerbias=net.b{2,1} ;

% 避免过拟合，划分训练，测试和验

net.trainParam.show = 50; 
net.trainParam.lr = 0.1; 
net.trainParam.mc = 0.04; 
net.trainParam.epochs = 1000; 
net.trainParam.goal = 1e-5; 


%训练网络
[net,tr] = train(net,P,T);

%% 根据图表判断拟合好坏
yn=net(P);
errors=T-yn;

figure(1)
plot(T,'b-');
hold on
plot(yn,'r--')
legend('股价真实值','BP网络输出值')
title('训练数据的测试结果');

% 显示均方误差
mse1 = mse(errors);
fprintf('    mse = \n     %f\n', mse1)
figure(2)

title('训练数据测试结果的残差')



% 显示相对误差
disp('    相对误差：')
fprintf('%f  ', (T - yn)./T );
fprintf('\n')

%% 预测
% 2.显示测试数据的测试结果

%预测
fn=57;  %预测步数为fn。

f_in=iinput(n-lag+1:end)';
f_out=zeros(1,fn);  %预测输出
% 多步预测时，用下面的循环将网络
for i=1:fn
    f_out(i)=net(f_in);
    f_in=[f_in(2:end);f_out(i)];
end
figure(3)
% 显示真实值
x2=1:length(price2');
plot(price2,'b-');
hold on
% 显示神经网络的输出值
plot(f_out,'r--')
legend('initial data','prediction result');
title('BP prediction');
hold off
grid on

% 显示残差
figure(4)
errors2=price2'-f_out;
mse2=mse(errors2);
plot(mse2)
title('测试数据测试结果的残差')

% 显示均方误差
mse2 = mse(errors2);
fprintf('    mse = \n     %f\n', mse2)

% 显示相对误差
disp('    相对误差：')
fprintf('%f  ', errors2./price2' );
fprintf('\n')



