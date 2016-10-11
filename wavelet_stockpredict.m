clc
clear
close all
%con=yahoo;
%myday=today;
%myday=datestr(today,23);
%
%stday=datestr(today-2000,23);
%data=fetch(con,'600016.ss','close',stday,myday);
data = zeros(337,2);
load elm_stock
for i = 1:337
    data(i,1) = i;
    data(i,2) = price(i);
end
TS=data(end:-1:1,:);
num=length(data);
starttime=datestr(TS(1,1),23);
endtime=datestr(TS(num,1),23);
testdata=TS(1:280,2);
inputest=testdata(1:1+4)';
aa=ones(1,5);
for i=2:275
    aa=testdata(i:i+4)';
    inputest=[inputest;aa];
end
outputest=testdata(6:280);

%????
M=4;
N=1;
n=6;
lr1=0.01;
lr2=0.001;
maxgen=200;
wjk=randn(n,M);
wij=randn(N,n);
a=randn(1,n);
b=randn(1,n);
d_wjk=zeros(n,M);
d_wij=zeros(N,n);
d_a=zeros(1,n);
d_b=zeros(1,n);
net=zeros(1,n);
net_ab=zeros(1,n);
y=zeros(1,N);
[inputn,inputps]=mapminmax(inputest');
[outputn,outputps]=mapminmax(outputest');
inputn=inputn';
outputn=outputn';
error=0;
%???????
for i=1:maxgen
for kk=1:size(inputest,1)
    x=inputn(kk,:);
    yqw=outputn(kk,:);
    for j=1:n
    for k=1:M
        net(j)=net(j)+wjk(j,k)*x(k);
        net_ab(j)=(net(j)-b(j))/a(j);
    end
    temp=mymorlet(net_ab(j));
    for k=1:N
        y=y+wij(k,j)*temp;
    end
    end
    error=error+sum(abs(yqw-y));
    for j=1:n
        temp=mymorlet(net_ab(j));
     for k=1:N
         d_wij(k,j)=d_wij(k,j)-(yqw(k)-y(k))*temp;
     end
     temp=d_mymorlet(net_ab(j));
    for i=1:M
        for l=1:N
            d_wjk(j,k)=d_wjk(j,k)+(yqw(l)-y(l))*wij(l,j);
        end
        d_wjk(j,k)=- d_wjk(j,k)*temp*x(k)/a(j);
    end
    for k=1:N
        d_b(j)=d_b(j)*(yqw(k)-y(k))*wij(k,j);
    end
    d_b(j)=d_b(j)*temp/a(j);
    for k=1:N
        d_a(j)=d_a(j)+(yqw(k)-y(k))*wij(k,j);
    end
    d_a(j)=d_a(j)*temp*((net(j)-b(j))/b(j))/a(j);
    end
    wij=wij-lr1*d_wij;
    wjk=wjk-lr1*d_wjk;
    b=b-lr2*d_b;
    a=a-lr2*d_a;
    
    d_wjk=zeros(n,M);
    d_wij=zeros(N,n);
    d_a=zeros(1,n);
    d_b=zeros(1,n);
    
    y=zeros(1,N);
    net=zeros(1,n);
    net_ab=zeros(1,n);
end
end

test=TS(1:end,2);
input_test=[];
input_test(1,:)=test(280:284)';
aa=ones(1,5);
for i=281:num-5
    aa=test(i:i+4)';
    input_test=[input_test;aa];
end

%????
x=mapminmax('apply',input_test',inputps);
x=x';
yuce=[];
for i=1:length(x)
x_test=x(i,:);
for j=1:n
    for k=1:M
    net(j)=net(j)+wjk(j,k)*x_test(k);
    net_ab(j)=(net(j)-b(j))/a(j);
    end
    temp=mymorlet(net_ab(j));
    for k=1:N
        y(k)=y(k)+wij(k,j)*temp;
    end
end
yuce(i)=y(k);
y=zeros(1,N);
net=zeros(1,n);
net_ab=zeros(1,n);
end
ynn=mapminmax('reverse',yuce,outputps);
figure(1)
plot(data(281:end,2),'b-');
hold on
plot(ynn,'r--');
legend('initial data','prediction result');
title('wavelet prediction');
hold off
grid on
%plot(test(281:end),'b--')