
load intdef.raw

[n,k]=size(intdef) 

%%%%% Test d'autocorrelation %%%%%


% i3 | inf, def

y=intdef(:,2); 
[n,k]=size(intdef) 
X=[ones(n,1),intdef(:, [3,6])];
[n,k]=size(X)
beta=inv(X'*X)*X'*y 
u=y-X*beta;
sig2=u'*u/(n-k) 
std=sqrt(diag(sig2*inv(X'*X))) 
t=beta./std 


% calcul de u_(t-1) puis régression de u_t sur u_(t-1)

u_=[u(2:n)]
u_lag= [u(1:n-1)]
y=u_
X=[u_lag]; %sans constante!
[n,k]=size(X)
rho=inv(X'*X)*X'*y 
u=y-X*rho;
sig2=u'*u/(n-k) 
std=sqrt(diag(sig2*inv(X'*X))) 
t=rho./std 


% test de student :on teste rho=0 et on rejette.
p=tdis_prb(t,n-k)



% transformation des donnees 

% a la main:
y=intdef(:,2); 
[n,k]=size(intdef) 
X=[ones(n,1),intdef(:, [3,6])]; 

y_=[y(2:n)]
y_lag= [y(1:n-1)]
new_y=[sqrt(1-rho^2)*y(1); y_-rho*y_lag]

X_=[X(2:n,:)]
X_lag= [X(1:n-1,:)]
new_X=[sqrt(1-rho^2)*X(1,:); X_-rho*X_lag]

% avec la matrice P:
P=zeros(n,n);
v=repmat(1,1,n)
P=P+ diag(v,0)
v=repmat(-rho,1,n-1)
P=P+ diag(v,-1)
P(1,1)=sqrt(1-rho^2)

new_y_bis=P*y
new_X_bis=P*X

% Regression MCO sur les donnees transformees

[n,k]=size(new_X)
beta=inv(new_X'*new_X)*new_X'*new_y 
u=new_y-new_X*beta;
sig2=u'*u/(n-k) 
std=sqrt(diag(sig2*inv(X'*X))) 
t=beta./std 

%%%%% Delais distribues %%%%%


y=intdef(:,2); 
[n,k]=size(intdef) ;
X=[intdef(:, [3,6])]; %sans constantes car modele dynamique (sinon ca crée un terme de tendance)

y=intdef(3:n,2); 
X_lag= [X(2:n-1,:)];
X_lag2= [X(1:n-2,:)];
X=[X_lag X_lag2];
[n,k]=size(X);
beta=inv(X'*X)*X'*y ;
u=y-X*beta;
sig2=u'*u/(n-k) ;
std=sqrt(diag(sig2*inv(X'*X))) ;
t=beta./std ;

%représentation graphique des coeff

bar(beta)
set(gca,'XTickLabel',{'inf(t-1)', 'def(t-1)', 'inf(t-2)', 'def(t-2)'})

%%% test de granger de causalite %%%%

% modele non contraint
SSR0=u'*u;

% modèle contraint (on enleve l'inflation)
X=X(:,[2,4]);
beta1=inv(X'*X)*X'*y;
u1=y-X*beta1;
SSR1=u1'*u1;

F=((SSR1-SSR0)/SSR0)*((n-k)/2);
p=fdis_prb(F,2,n-k) % -> on rejette

% modele contraint (on enleve le deficit)
X=[X_lag X_lag2];
X=X(:,[1,3]);
beta1=inv(X'*X)*X'*y;
u1=y-X*beta1;
SSR1=u1'*u1;

F=((SSR1-SSR0)/SSR0)*((n-k)/2);
p=fdis_prb(F,2,n-k) % -> on rejette



