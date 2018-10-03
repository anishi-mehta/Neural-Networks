function [mu,sig,bet,A,B,K] = PBLMcRBF(UY,nxi,nyi,kmax,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  The training code for PBL-McRBF Algorithm...                           %
%  The input parameters to the function:                                  %
%           UY  - containes input samples for training (nx X muy)         %
%           nxi - features used for training                              % 
%           nyi - class label for training samples  (1 X muy)             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize the network parameters.......................................
nx  = length(nxi);      % No. of input features............................
ny  = max(nyi);         % No. of outputs...................................
nh  = sum(kmax);        % Maximum hidden neurons...........................
mu  = zeros(nx,nh);     % Centers of neurons...............................
sig = zeros(1,nh);      % width of neurons.................................
bet = zeros(nh,ny);     % Output weight....................................
A   = zeros(nh,nh);     % Projection matrix................................
B   = zeros(nh,ny);     % Output matrix....................................
K   = 0;                % No. of hidden neurons............................
w1  = zeros(1,nh);      % Class association of neurons.....................

%% Initialize the control parameters.......................................
Ed      = par(1);    %skip threshold
Eadd    = par(2);    %initial adding error threshold
Elearn  = par(3);    %initial learning error threshold
r       = par(4);    %decay factor
kp      = par(5);    %overlap factor
Pmax    = par(6);    %limit for reserve samples
P1      = par(7);    %spherical potential threshold
zeta    = par(8);    %center shifting factor
kp1     = par(9);    %overlap factor for first neuron in each class
kp2     = par(10);   %Overlap factor for shifted neuron

%% Check the input data....................................................
muy     = size(UY,2);
clab    = 2*eye(ny)-1.0;
flag    = ones(1,muy);  % status of samples................................
kk      = zeros(1,ny);  % Neurons in each class............................
dflag   = zeros(1,muy); % status of deleted samples........................
rflag   = 1 : muy;      % Reserve samples..................................
uflag   = [];           % Used samples flags...............................

% Find first samples in all class..........................................
F = zeros(nx,ny);
for i = 1 : ny
    L = find(nyi==i,1);
    F(:,i) = UY(:,L);
end
FF = dist(F',F); 
FF(1:(ny+1):(ny*ny))= 1000; % Assign large value for diagonal..............
clear F;
ME = 1;                 % Exit condition is 1

%% Training phase start....................................................
while ME && sum(flag) > Pmax
    pre_sample = sum(flag);         % total samples for learning
    for rr = 1 : size(rflag,2)
        m = rflag(1);              % sample number
        if flag(m) ~= 0 
            x = UY(:,m);          % Current sample.......................
            yhat = zeros(1,ny);
            if K ~= 0
                xmusq = zeros(1,K);
                Phi   = zeros(1,K);
                for i = 1 : K
                    xmusq(i) = (x-mu(:,i))'*(x-mu(:,i));
                    Phi(1,i) =  exp(-xmusq(i)/(sig(i)^2));
                end
                yhat = Phi(1,1:K)*bet(1:K,:);    % Predicted output........
            end
            
            % Calculate ccap and hinge error...............................
            cact    = nyi(m);       % Actual class label
            [~,chat]= max(yhat);    % Predicted class label
            yact    = clab(cact,:); % Coded class label
            
            % Hinge error calculation.....    
            E = zeros(ny,1);
            for jj = 1 : ny
                if yact(jj)*yhat(jj) >= 1.0
                   E(jj,1) = 0;
                else
                   E(jj,1) = yact(jj)-yhat(jj);    
                end
            end
            err = sqrt((max(E(:,1).^2)));   % Maximum Hinge Error..........
            
            % Sample Deletion Criterion....................................
            if cact == chat && err < Ed,%Delete the sample
                flag(m)=0; dflag(m) = 1; rflag = rflag(2:end);
                continue;
            end
            
            % Sample Learning Strategy.....................................
            if kk(cact)==0      % Zero neurons in that class
                K = K + 1;      % Increment the neuron.....................
                kk(cact) = kk(cact) + 1; %Increment the neuron class.......
                flag(m) = 0;                    % Sample used
                rflag = rflag(2:end);           % sample remove from reserve
                uflag = [uflag m];              % Used samples.............
                % Compute width factor.....................................           
%                  sK = max(0.0001,kp1*sqrt((min(FF(cact,:)))));
                 sK = sqrt(kp1*(min(FF(cact,:))));
                if K == 1,    % First neuron..........
                    A(1,1)    = 1;
                    B(1,:)    = 1*yact;
                    bet(1,:)  = B(1,:);
                    mu(:,K)   = x;      % Assign new center
                    sig(1,K)  = sK;     % Assign new width
                    w1(K)     = cact;   % Associate class label
                else 
                    mu(:,K)   = x;      % Assign new center
                    sig(1,K)  = sK;     % Assign new width
                    w1(K)     = cact;   % Associate class label
                    
                    % Update A and B Matrix
                    A(1:(K-1),1:(K-1)) = A(1:(K-1),1:(K-1)) + Phi(1:K-1)'*Phi(1:K-1);
                    B(1:(K-1),:) = B(1:(K-1),:) + Phi(1:K-1)'*yact; 
                    B(K,:)      = 0;
                    
                    % Past samples for new neusons
                    Phit = zeros(1,K);
                    for i = 1 : K
                        for j = 1 : K,
                            Phit(1,j) = exp(-sum((mu(:,i)-mu(:,j)).^2)/(sig(j)^2));
                        end 
                        A(1:K,K) = A(1:K,K) + Phit(1,1:K)'*Phit(K);
                        yacti   = -1*ones(ny, 1);
                        yacti(w1(i)) = 1;
                        B(K,:) = B(K,:)+ Phit(1,K)*yacti';  
                    end                     
                    A(K, 1:K-1) = A(1:K-1, K);
                    clear Phit;
                    
                    % Find the output weight
                    bet(1:K,:) = A(1:K,1:K)\B(1:K,:);
%                    bet(1:K,:) = inv(A(1:K,1:K))*B(1:K,:);
                end
                Eadd    =  r*err+(1-r)*Eadd;    % Update selfregulating threshold

            else  % Sample learning based on novelty, max. hinge error.....
             
                % Novelty calculation.....................
                Ps = sum(Phi(w1(1:K)==cact))/(kk(cact));
            
                %Find nearest neuron of the same class...........................
                L = find(w1(1:K)==cact); nrS = []; nrI = [];
                if ~isempty(L)
                    [tem,nrS] = min(xmusq(L));
                    nrS = L(nrS);          % neareast neuron in the same class....
                    tnearS = sqrt(tem);
                end
                % Find nearest neuron in inter class..............................
                L = find(w1(1:K)~=cact);
                if ~isempty(L)
                    [tem,nrI] = min(xmusq(L));
                    nrI = L(nrI);          % neareast neuron in the Inter class....
                    tnearI = sqrt(tem);
                end
                
                % Neuron Addition strategy.................................
                if (cact ~= chat || err > Eadd) && Ps <= P1 && kk(cact) < kmax(cact),
                    Eadd =  r*err+(1-r)*Eadd;   % Update selfregulating threshold
                    kk(cact) = kk(cact)+1;      % Incremet associated neuron class
                    flag(m) = 0;                % Sample used for neuron addition......
                    rflag = rflag(2:end);       % Remove the samples from reserve
                    uflag = [uflag m];          % Used samples.............
                    if (kp*tnearS > 4*sig(nrS) && kp*tnearI > 4*sig(nrI))                
                        center = x;
                        sK = max(0.0001,kp1*sqrt(x'*x));
                    else
                        if(1 < tnearS/tnearI)
                            center = x + zeta*(mu(:,nrS)-mu(:,nrI));
                            sK = max(0.0001,kp2*dist(center',mu(:,nrS)));
                        else                    
                            center = x;
                            sK = max(0.0001,kp*tnearS);
                        end
                    end
                    K = K+1;                    % Increment Neuron
                    mu(:,K)   = center; % Assign new center
                    sig(1,K)  = sK;     % Assign new width
                    w1(K)     = cact;   % Associate class label
                    % Update A and B Matrix
                    A(1:(K-1),1:(K-1)) = A(1:(K-1),1:(K-1)) + Phi(1:K-1)'*Phi(1:K-1);
                    B(1:(K-1),:) = B(1:(K-1),:) + Phi(1:K-1)'*yact; 
                    B(K,:)      = 0;
                    
                    % Past samples for new neusons
                    Phit = zeros(1,K);
                    for i = 1 : K
                        for j = 1 : K,
                            Phit(1,j) = exp(-sum((mu(:,i)-mu(:,j)).^2)/(sig(j)^2));
                        end 
                        A(1:K,K) = A(1:K,K) + Phit(1,1:K)'*Phit(K);
                        yacti   = -1*ones(ny, 1);
                        yacti(w1(i)) = 1;
                        B(K,:) = B(K,:)+ Phit(1,K)*yacti';  
                    end                          
                    A(K, 1:K-1) = A(1:K-1, K);
                    clear Phit;                    
                    % Find the output weight
                    bet(1:K,:) = A(1:K,1:K)\B(1:K,:);
%                    bet(1:K,:) = inv(A(1:K,1:K))*B(1:K,:);
                                       
                elseif  cact == chat && err > Elearn ,      % Update Network Parameters        
                    Elearn     =  r*err+(1-r)*Elearn;
                    A(1:K,1:K) = A(1:K,1:K) + Phi(1:K)'*Phi(1:K);
                    B(1:K,:)   = B(1:K,:)+Phi'*yact;
                    bet(1:K,:) = bet(1:K,:) + A(1:K,1:K)\(Phi(1:K)' * E');
%                    bet(1:K,:) = bet(1:K,:) + inv(A(1:K,1:K))*(Phi(1:K)' * E');
                    flag(m) = 0;    % Sample used in learning
                    rflag = rflag(2:end);
                    uflag = [uflag m];  % Sample used......................
                else                                     % Preserve Samples for future Learning
                    flag(m) = 1;    % Sample is not used...................
                    rflag = [rflag(2:end) rflag(1)];
                end                 % end of sample learning strategy    
                
            end                     % end of current sample
        end                         % check for unused sample
    end                             % end of sample loop 
    if sum(flag) == pre_sample
        ME = 0;                     % Exit condition
    end
end                                 % end of while loop

%% Prepare the final results
mu  = mu(:,1:K);
sig = sig(1,1:K);
bet = bet(1:K,:);
%% End of Model development................................................