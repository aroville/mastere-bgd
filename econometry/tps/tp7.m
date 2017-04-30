load phillips.raw
[n,k]=size(phillips)
% On trace inf
y=phillips(:,3);
plot(y)

% On trace ACF et PACF
acf=sacf(y,20)
pacf=spacf(y,20)

%Test du porte manteau

%[qstat,pval]=qstat2(y,1)


% On teste AR1:

y_=y(2:n)
y_lag=y(1:n-1)
X=y_lag
[n,k]=size(X)
beta=inv(X'*X)*X'*y_
u=y_-X*beta;
sig2=u'*u/(n-k) 

AIC1 =  log(sig2) + (2.*(1))./n;


% On teste AR2:

y_=y(3:n)
y_lag=y(2:n-1)
y_lag2=y(1:n-2)
X=[y_lag y_lag2]
[n,k]=size(X)
beta=inv(X'*X)*X'*y_
u=y_-X*beta;
sig2=u'*u/(n-k) 

AIC2 =  log(sig2) + (2.*(2))./n;


%%%%% Stationnarité %%%%%%

y1=y(1:floor(n/2))
y2=y(floor(n/2):n)
m1=mean(y1)
m2=mean(y2)
v1=var(y1)
v2=var(y2)


y1=y(1:floor(n/3))
y2=y(floor(n/3):floor(2*n/3))
y3=y(floor(2*n/3):n)
m1=mean(y1)
m2=mean(y2)
m3=mean(y3)
v1=var(y1)
v2=var(y2)
v3=var(y3)




%%%%% Test de Dickey Fuller %%%%% 

y_=y(2:n)
y_lag=y(1:n-1)
Delta_y=y_-y_lag
[n,k]=size(Delta_y)
X=[ones(n,1) y_lag]
[n,k]=size(X)
beta=inv(X'*X)*X'*Delta_y
u=Delta_y-X*beta;
sig2=u'*u/(n-k);
std=sqrt(diag(sig2*inv(X'*X))) 
t=beta./std 

% test augmenté

y_=y(2:n)
y_lag=y(1:n-1)
Delta_y=y_-y_lag

[n,k]=size(Delta_y)
Delta_y_=Delta_y(1:n-4)
Delta_y_lag=Delta_y(2:n-3)
Delta_y_lag2=Delta_y(3:n-2)
Delta_y_lag3=Delta_y(4:n-1)
Delta_y_lag4=Delta_y(5:n)

y_lag_=y(1:n-4)

[n,k]=size(Delta_y_)
X=[ones(n,1) y_lag_ Delta_y_lag Delta_y_lag2 Delta_y_lag3 Delta_y_lag4]
[n,k]=size(X)
beta=inv(X'*X)*X'*Delta_y_
u=Delta_y_-X*beta;
sig2=u'*u/(n-k);
std=sqrt(diag(sig2*inv(X'*X))) 
t=beta./std 

%%%%% Tests de changement de structure %%%%% 

% Test de Chow

[n,k]=size(phillips)
t0= 34 %année 1981
D_tau=[zeros(34,1) ; ones(n-34,1)]
D_tau=D_tau(1:n-1)
y_=y(2:n)
y_lag=y(1:n-1)
X=[ones(n-1,1) y_lag D_tau.*ones(n-1,1) D_tau.*y_lag]
[n,k]=size(X)

%modèle non contraint

beta0=inv(X'*X)*X'*y_
u0=y_-X*beta0
SSR0=u0'*u0

%modèle contraint

X=X(:,[1,2])
beta1=inv(X'*X)*X'*y_
u1=y_-X*beta1
SSR1=u1'*u1

F=((SSR1-SSR0)/SSR0)*((n-k)/1)
p=fdis_prb(F,1,n-k)


% QLR statistics


[n,k]=size(phillips);
tau0=floor(0.15*n);
tau1=floor(0.85*n);
all_chows= [];

y_=y(2:n)
y_lag=y(1:n-1)
k=4

for t=tau0:tau1;
    D_tau=[zeros(t,1) ; ones(n-t,1)]
    D_tau=D_tau(1:n-1)

    X=[ones(n-1,1) y_lag D_tau.*ones(n-1,1) D_tau.*y_lag]
    
    beta0=inv(X'*X)*X'*y_
    u0=y_-X*beta0
    SSR0=u0'*u0
    
    X=X(:,[1,2])
    beta1=inv(X'*X)*X'*y_
    u1=y_-X*beta1
    SSR1=u1'*u1

    F=((SSR1-SSR0)/SSR0)*((n-k)/1)
    all_chows(end+1) = F
end;

[QLR, tau]=max(all_chows)
 
