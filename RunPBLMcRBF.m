%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          McRBF - Classification code                                    %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc
warning('off')

%% Load the data set for training..........................................
load seg_data;
nxi = 19:-1:1;
[~,nyi] = max(UY(nyi,:));       % Generate class label.....................
UY      = UY(nxi,:);            % Generate features........................

par = [0.28673	1.3374	0.35869	0.081297	0.84956	24	0.58834	0.073355	0.16295	0.70435];
kmax = 15*ones(1,7);%Seg


%% McRBF Model development.................................................
tic
[Mu,Sig,Beta,A,B,K] = PBLMcRBF(UY,nxi,nyi,kmax,par);
toc
%% Testing with same training data.........................................
ny          = max(nyi);         % No. of classes
conf_tra    = zeros(ny,ny);     % Confusion matrix
% Find class label based on model.
for m = 1 : size(uy1,2)
    x   = uy1(nxi,m);
    Phi = zeros(1,K);
    for i = 1 : K
        xmusq    = (x - Mu(:,i))'*(x - Mu(:,i));
        Phi(1,i) = exp(-xmusq/(Sig(i)^2));
    end
    yhat     = Phi*Beta;        % Output
    [~,chat] = max(yhat);       % Estimated class label
    cact     = uy2(1,m);        % Actual class label
    % Confusion matrix update
    conf_tra(cact,chat) = conf_tra(cact,chat) + 1;
end

% Performance matrix......................................................
tra_ova = sum(diag(conf_tra))/sum(conf_tra(:));  % Overall
tra_avg = mean(diag(conf_tra)./sum(conf_tra,2)); % Average

%% Testing with same training data.........................................
conf_tes    = zeros(ny,ny);     % Confusion matrix

% Find class label based on model.
for m = 1 : size(UY1,2)
    x   = UY1(nxi,m);
    Phi = zeros(1,K);
    for i = 1 : K
        xmusq    = (x - Mu(:,i))'*(x - Mu(:,i));
        Phi(1,i) = exp(-xmusq/(Sig(i)^2));
    end
    yhat     = Phi*Beta;        % Output
    [~,chat] = max(yhat);       % Estimated class label
    cact     = UY2(1,m);        % Actual class label
    % Confusion matrix update
    conf_tes(cact,chat) = conf_tes(cact,chat) + 1;
end

% Performance matrix......................................................
tes_ova = sum(diag(conf_tes))/sum(conf_tes(:));  % Overall
tes_avg = mean(diag(conf_tes)./sum(conf_tes,2)); % Average

%% Display the result......................................................
disp('Final Training/Testing Performance');
disp([tra_ova tra_avg tes_ova tes_avg])