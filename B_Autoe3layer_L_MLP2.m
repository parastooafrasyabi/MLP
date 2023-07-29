%Autoencoder_Local_MLP2
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

%% Initialize Parameters
train_rate=0.1;
eta_e1 = 0.05;
eta_e2 = 0.03;
eta_e3 = 0.01;
eta_p = 0.001;
epochs_ae = 3;
max_epoch_p = 3;


n0_neurons = m-1;
n1_neurons =32;
n2_neurons =16;
n3_neurons=8;

%MLP
l1_neurons=4;
l2_neurons=3;

lowerb=-1;
upperb=1;

%% Initialize Autoencoder weigths
w_e1 = unifrnd(lowerb,upperb,[n1_neurons n0_neurons]);
w_e2 = unifrnd(lowerb,upperb,[n2_neurons n1_neurons]);
w_e3 = unifrnd(lowerb,upperb,[n3_neurons n2_neurons]);
w_d1 = unifrnd(lowerb,upperb,[n0_neurons n1_neurons]);
w_d2 = unifrnd(lowerb,upperb,[n1_neurons n2_neurons]);
w_d3 = unifrnd(lowerb,upperb,[n2_neurons n3_neurons]);
net_e1=zeros(n1_neurons,1);
h1=zeros(n1_neurons,1);
net_d1=zeros(n0_neurons,1);
x_hat=zeros(n0_neurons,1);

net_e2=zeros(n2_neurons,1);
h2=zeros(n2_neurons,1);
net_d2=zeros(n1_neurons,1);
h1_hat=zeros(n1_neurons,1);

net_e3=zeros(n3_neurons,1);
h3=zeros(n3_neurons,1);
net_d3=zeros(n2_neurons,1);
h2_hat=zeros(n2_neurons,1);
%% Encoder 1 Local Train
for i=1:epochs_ae
    fff=i
    for j=1:n
        % Feed-Forward

        % Encoder1
        net_e1 = w_e1* data_train(j,1:m-1)';  % n1*1
        h1 = logsig(net_e1);  % n1*1

        % Decoder1
        net_d1 = w_d1 * h1;  % n0*1
        x_hat = logsig(net_d1);  % n0*1

        % Error
        err = data_train(j,1:m-1) - x_hat';  % 1*n0

        % Back propagation
        f_driviate_d = diag(x_hat.*(1-x_hat));  % n0*n0
        f_driviate_e = diag(h1.*(1-h1));  % n1*n1
        delta_w_d1 = (eta_e1 * h1 * err * f_driviate_d)';  % n0*n1 = (1*1 * n1*1 * 1*n0 * n0*n0)'
        delta_w_e1 = (eta_e1 * data_train(j,1:m-1)' * err * f_driviate_d * w_d1 * f_driviate_e)';  % n1*n0 = (1*1 * n0*1 * 1*n0 * n0*n0 * n0*n1 * n1*n1)'
        w_d1 = w_d1 + delta_w_d1;  % n0*n1
        w_e1 = w_e1 + delta_w_e1;  % n1*n0
    end
end

%% Encoder 2 Local Train
for i=1:epochs_ae
    for j=1:n
        % Feed-Forward

        % Encoder1
        net_e1 = w_e1* data_train(j,1:m-1)';  % n1*1
        h1 = logsig(net_e1);  % n1*1
        
        % Encoder2
        net_e2 = w_e2* h1;  % n2*1
        h2 = logsig(net_e2);  % n2*1

        % Decoder2
        net_d2 = w_d2 * h2;  % n1*1
        h1_hat = logsig(net_d2);  % n1*1

        % Error
        err = (h1 - h1_hat)';  % 1*n1

        % Back propagation
        f_driviate_d = diag(h1_hat.*(1-h1_hat));  % n1*n1
        f_driviate_e = diag(h2.*(1-h2));  % n2*n2
        delta_w_d2 = (eta_e2 * h2 * err * f_driviate_d)';  % n1*n2 = (1*1 * n2*1 * 1*n1 * n1*n1)'
        delta_w_e2 = (eta_e2 * h1 * err * f_driviate_d * w_d2 * f_driviate_e)';  % n2*n1 = (1*1 * n1*1 * 1*n1 * n1*n1 * n1*n2 * n2*n2)'
        w_d2 = w_d2 + delta_w_d2;  % n1*n2
        w_e2 = w_e2 + delta_w_e2;  % n2*n1
    end
end
%% Encoder 3 Local Train
for i=1:epochs_ae
    for j=1:n
        % Feed-Forward

        % Encoder1
        net_e1 = w_e1* data_train(j,1:m-1)';  % n1*1
        h1 = logsig(net_e1);  % n1*1
        
        % Encoder2
        net_e2 = w_e2* h1;  % n2*1
        h2 = logsig(net_e2);  % n2*1
        % Encoder3
        net_e3 = w_e3* h2;  % n3*1
        h3 = logsig(net_e3);  % n3*1
        
        % Decoder3
        net_d3 = w_d3 * h3;  % n2*1
        h2_hat = logsig(net_d3);  % n2*1

        % Error
        err = (h2 - h2_hat)';  % 1*n2

        % Back propagation
        f_driviate_d = diag(h2_hat.*(1-h2_hat));  % n2*n2
        f_driviate_e = diag(h3.*(1-h3));  % n3*n3
        delta_w_d3 = (eta_e3 * h3 * err * f_driviate_d)';  % n2*n3 = (1*1 * n3*1 * 1*n2 * n2*n2)'
        delta_w_e3 = (eta_e3 * h2 * err * f_driviate_d * w_d3 * f_driviate_e)';  % n3*n2 = (1*1 * n2*1 * 1*n2 * n2*n2 * n2*n3 * n3*n3)'
        w_d3 = w_d3 + delta_w_d3;  % n2*n3
        w_e3 = w_e3 + delta_w_e3;  % n3*n2
    end
end

%% Initialize Perceptron weigths
w1=unifrnd(lowerb,upperb,[l1_neurons n3_neurons]);
net1=zeros(l1_neurons,1);
o1=zeros(l1_neurons,1);

w2=unifrnd(lowerb,upperb,[l2_neurons l1_neurons]);
net2=zeros(l2_neurons,1);
o2=zeros(l2_neurons,1);

num_of_train1=size(data_train);
num_of_test1=size(data_test);

num_of_train=num_of_train1(1);
num_of_test=num_of_test1(1);

error_train=zeros(num_of_train,1);
error_test=zeros(num_of_test,1);

output_train=zeros(num_of_train,1);
output_test=zeros(num_of_test,1);

mse_train=zeros(max_epoch_p,1);
mse_test=zeros(max_epoch_p,1);

acurcy_train=zeros(max_epoch_p,1);
acurcy_test=zeros(max_epoch_p,1);
%% 2 Layer Perceptron
% Train
for i=1:max_epoch_p
  for j=1:num_of_train-2
      
      input=data_train(j:j+2,1:m-1);
      target=data_train(j,m);

      % Feed-Forward

      % Encoder1
      net_e1 = w_e1* input';  % n1*1
      h1 = logsig(net_e1);  % n1*1

      % Encoder2
      net_e2 = w_e2* h1;  % n2*1
      h2 = logsig(net_e2);  % n2*1

      % Encoder3
      net_e3 = w_e3* h2;  % n3*1
      h3 = logsig(net_e3);  % n3*1
      
      % Layer 1
      net1=w1*h3;  % l1*1
      o1=logsig(net1);  % l1*1
      
      % Layer 2
      net2=w2*o1;  % l2*1
      o2=softmax(net2);  % l2*1
      
      % Predicted Output
      e = crossentropy(o2,target);
      [ma,ind]=max(max(o2));
      output_train(j,1)=ind;
      error_train(j,1)=e;
      
      % Back Propagation
      f_driviate=diag(diag(o1*(1-o1)')); % l1*l1 
      w1=w1-eta_p*e*-1*1*(w2*f_driviate)'*h3';  % l1*n0 = l1*n0 - 1*1 * 1*1 * (1*l1 * l1*l1)' * 1*n0
      w2=w2-eta_p*e*-1*1*o1';  % 1*l1 = 1*l1 - 1*1 * 1*1 * 1*l1
           
  end
    mse_train(i,1)=error_train(1,1);
    Target=x(:,35);
    output_train(369289,1)=1;
    output_train(369288,1)=1;
    [c_train,cm_train ]= confusionmat(categorical(Target(:)), categorical(output_train(:)));
  
     acurcy_train(i,1)=((c_train(1,1)+c_train(2,2)+c_train(3,3))/(c_train(1,2)+c_train(1,3)+c_train(2,1)+c_train(2,3)+c_train(3,1)+c_train(3,2)));
  
  % Test
  for j=1:num_of_test-2
      
      input=data_test(j:j+2,1:m-1);
      target=data_test(j,m);
      
      % Feed-Forward

      % Encoder1
      net_e1 = w_e1* input';  % n1*1
      h1 = logsig(net_e1);  % n1*1

      % Encoder2
      net_e2 = w_e2* h1;  % n2*1
      h2 = logsig(net_e2);  % n2*1
      
       % Encoder3
      net_e3 = w_e3* h2;  % n3*1
      h3 = logsig(net_e3);  % n3*1
      
      % Layer 1
      net1=w1*h3;  % l1*1
      o1=logsig(net1);  % l1*1
      
      % Layer 2
      net2=w2*o1;  % l2*1
      o2=softmax(net2);  % l2*1
      
      % Predicted Output
      e=crossentropy(o2,target);      
      [ma,ind]=max(max(o2));
      output_test(j,1)=ind;
      error_test(j,1)=e;  
      
  end 
  
  mse_test(i,1)=error_test(1,1);
  % Confusion Test Error
  TargetT=xT(:,35);
  output_test(41033,1)=1;
  output_test(41032,1)=1;
  [c_test,cm_test ]= confusionmat(categorical(TargetT(:)), categorical(output_test(:)));
   acurcy_test(i,1)=((c_test(1,1)+c_test(2,2)+c_test(3,3))/(c_test(1,2)+c_test(1,3)+c_test(2,1)+c_test(2,3)+c_test(3,1)+c_test(3,2)));
  
  %% Plot Results
 figure(1);
  subplot(2,2,1),semilogy(mse_train(1:i,1),'-r');
  title('Eror Train')
  hold off;
  
  subplot(2,2,2),plot(mse_test(1:i,1),'-r');
  title('Eror Test')
  hold off;
  
  subplot(2,2,3),semilogy(acurcy_test(1:i,1),'-r');
  title('Acurcy Test')
  hold off;
  
  subplot(2,2,4),semilogy(acurcy_train(1:i,1),'-r');
  title('Acurcy Train')
  hold off;

  
  pause(0.001);
  
  fprintf('Eror train = %1.16g, Eror test = %1.16g \n', error_train(max_epoch_p,1), error_test(max_epoch_p,1))
end

fprintf('Eror train = %1.16g, Eror test = %1.16g \n', error_train(max_epoch_p,1), error_test(max_epoch_p,1))
