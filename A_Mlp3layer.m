%MLP 3 layers
clc;
close all;
clear all;
%% Read and Making Data_Tarian
x=xlsread('Train');


%% Normalize Dataset
data_min = min(x);
data_max = max(x);
n=size(x,1);
m=size(x,2);
for i = 1:n
    for j = 1:m-1
        x(i, j) = (x(i, j) - data_min(1,j)) / (data_max(1,j) - data_min(1,j));
    end
end
%%
%data=[Data,Target];
data_train=x(:,:);

%% Read and Making Data_Test
xT=xlsread('Test');

%% Normalize Dataset
data_minT = min(xT);
data_maxT = max(xT);
nT=size(xT,1);
mT=size(xT,2);
for i = 1:nT
    for j = 1:mT-1
        xT(i, j) = (xT(i, j) - data_minT(1,j)) / (data_maxT(1,j) - data_minT(1,j));
    end
end

%% dataT=[DataT,TargetT];
data_test=xT(:,:);
%%
eta=0.1;
max_epoch=3;

n1=m-1;
n2=16;
n3=8;
n4=3;
%%%%
lowerb=-1;
upperb=1;

w1=unifrnd(lowerb,upperb,[n2 n1]);
net1=zeros(n2,1);
o1=zeros(n2,1);

w2=unifrnd(lowerb,upperb,[n3 n2]);
net2=zeros(n3,1);
o2=zeros(n3,1);

w3=unifrnd(lowerb,upperb,[n4 n3]);
net3=zeros(n4,1);
o3=zeros(n4,1);

num_of_train1=size(data_train);
num_of_test1=size(data_test);

num_of_train=num_of_train1(1);
num_of_test=num_of_test1(1);

error_train=zeros(num_of_train-2,1);
error_test=zeros(num_of_test-2,1);

output_train=zeros(num_of_train,1);
output_test=zeros(num_of_test,1);

mse_train=zeros(max_epoch,1);
mse_test=zeros(max_epoch,1);

acurcy_train=zeros(max_epoch,1);
acurcy_test=zeros(max_epoch,1);
%%
for i=1:max_epoch
    fffff=i
  for j=1:num_of_train-2
      input=data_train(j:j+2,1:m-1);
      target=data_train(j,m);
      net1=w1*input';
      o1=logsig(net1);
      net2=w2*o1;
      o2=logsig(net2);
      net3=w3*o2;
      o3=softmax(net3);
    
      e = crossentropy(o3,target);
      [ma,ind]=max(max(o3));
      output_train(j,1)=ind;
      error_train(j,1)=e;
      
      A=diag(diag(o2*(1-o2)'));
      B=diag(diag(o1*(1-o1)'));
      
      w1=w1-eta*e*-1*1*(w3*A*w2*B)'*input;
      w2=w2-eta*e*-1*1*(w3*A)'*o1';
      w3=w3-eta*e*-1*1*o2';
           
  end
  mse_train(i,1)=error_train(1,1);
  
  Target=x(:,35);

  output_train(369289,1)=1;
  output_train(369288,1)=1;
  [c_train,cm_train ]= confusionmat(categorical(Target(:)), categorical(output_train(:)));
  acurcy_train(i,1)=((c_train(1,1)+c_train(2,2)+c_train(3,3))/(c_train(1,2)+c_train(1,3)+c_train(2,1)+c_train(2,3)+c_train(3,1)+c_train(3,2)));
  
  for j=1:num_of_test-2
      
      input=data_test(j:j+2,1:m-1);
      target=data_test(j,m);
      net1=w1*input';
      o1=logsig(net1);
      net2=w2*o1;
      o2=logsig(net2);
      net3=w3*o2;
      o3=softmax(net3);
      
      e=crossentropy(o3,target);
      
      [ma,ind]=max(max(o3));
      output_test(j,1)=ind;
      error_test(j,1)=e;
     
  end 
  
  mse_test(i,1)=error_test(1,1);
  
  TargetT=xT(:,35);
  output_test(41033,1)=1;
  output_test(41032,1)=1;
  [c_test,cm_test ]= confusionmat(categorical(TargetT(:)), categorical(output_test(:)));
  
  acurcy_test(i,1)=((c_test(1,1)+c_test(2,2)+c_test(3,3))/(c_test(1,2)+c_test(1,3)+c_test(2,1)+c_test(2,3)+c_test(3,1)+c_test(3,2)));
  
 
  %%
  figure(1);
  subplot(2,2,1),semilogy(mse_train(1:i,1),'-r');
  title('Eror Train')
  hold off;
  
  subplot(2,2,3),plot(mse_test(1:i,1),'-r');
  title('Eror Test')
  hold off;
  
  subplot(2,2,4),semilogy(acurcy_test(1:i,1),'-r');
  title('Acurcy Test')
  hold off;
  
  subplot(2,2,2),semilogy(acurcy_train(1:i,1),'-r');
  title('Acurcy Train')
  hold off;

  pause(0.001);
  fprintf('Eror train = %1.16g, Eror test = %1.16g \n', error_train(max_epoch,1), error_test(max_epoch,1))
end
fprintf('Eror train = %1.16g, Eror test = %1.16g \n', error_train(max_epoch,1), error_test(max_epoch,1))






