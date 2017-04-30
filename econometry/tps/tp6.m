
%%%%% AR(1) %%%%%

%simulation

y=zeros(1000,1);
for i=2:1000;
    y(i)=0.6*y(i-1)+randn;
end;
f=figure;
plot(y)
title('AR(1)')
f=figure;
acf=sacf(y,20);
title('acf of AR(1)')
f=figure;
pacf=spacf(y,20);
title('pacf of AR(1)')


%%%%% MA(1) %%%%%

%simulation

n=1000; 
z=zeros(n,1);
e=randn(n,1);
for i=2:1000;
    z(i)=e(i)+0.8*e(i-1);
end;
f=figure;
plot(z)
title('MA(1)')
f=figure;
acf=sacf(z,20);
title('acf of MA(1)')
f=figure;
pacf=spacf(z,20);
title('pacf of MA(1)')


%%%%% AR(2) %%%%%

%simulation

n=1000; 
y=zeros(n,1);
for i=3:1000;
    y(i)=0.6*y(i-1)+0.2*y(i-2)+randn;
end;
f=figure;
plot(y)
title('AR(2)')
f=figure;
acf=sacf(y,20);
title('acf of AR(2)')
f=figure;
pacf=spacf(y,20);
title('pacf of AR(2)')


%%%%% ARMA(2) %%%%%

%simulation

y=zeros(1000,1);
for i=3:1000;
    y(i)=0.5*y(i-1)-0.8*y(i-2)+e(i)+0.6*e(i-1)+0.2*e(i-2);
end;
f=figure;
plot(y)
title('ARMA(2,2)')
f=figure;
acf=sacf(y,20)
title('acf of ARMA(2,2)')
f=figure;
pacf=spacf(y,20);
title('pacf of ARMA(2,2)')



%%%%% simulation d'autocorrelation %%%%%

n=1000
y=zeros(n,1);
u=randn(n,1);
for i=2:1000;
    y(i) = 0.8*y(i-1) + u(i)+ 0.8*u(i-1);
end;

% regression de yt sur y_(t-1)
y_=y(2:n); 
y_lag=y(1:n-1); 
[n,k]=size(y_lag)
beta=inv(y_lag'*y_lag)*y_lag'*y_




load intdef.raw

[n,k]=size(intdef)

%%%%% AR(1) %%%%%

% delta_inft | delta_inft-1 (avec constante)

inf=intdef(1:n,3);
delta_inft=inf(2:n);
delta_inft_lag=inf(1:n-1);

y=delta_inft
[n,k]=size(y)
X=[ones(n,1) delta_inft_lag]
[n,k]=size(X)
beta=inv(X'*X)*X'*y 
u=y-X*beta;
sig2=u'*u/(n-k) 
std=sqrt(diag(sig2*inv(X'*X))) 
t=beta./std 

% test de significativité 
t=t(2)
p=tdis_prb(t,n-k)


%prévision
delta_inft_pr=beta(1)*X(:,1)+beta(2)*X(:,2)
diff=delta_inft_pr-y;
RMSE=norm(diff)
